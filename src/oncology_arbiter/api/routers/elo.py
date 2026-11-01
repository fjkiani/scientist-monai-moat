"""Route module split out of api/app.py."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__
from oncology_arbiter.api import deps
from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.runtime import (
    _ARTIFACT_ROOTS,
    _elo_reason_for_modifier,
    _run_cbis_ddsm_probe_on_bytes,
)
from oncology_arbiter.api.schemas import *  # noqa: F403
from oncology_arbiter.api.services.helpers import failed_stage_receipt as _failed_stage_receipt
from oncology_arbiter.api.services.helpers import pipeline_status as _pipeline_status
from oncology_arbiter.api.services.helpers import skipped_stage_receipt as _skipped_stage_receipt
from oncology_arbiter.arbiter.manski import (
    RESEARCH_BLIND_INFERENCE_SCOPE,
    ManskiGateError,
)
from oncology_arbiter.auth import APIKey, require_api_key, require_scope

router = APIRouter()

_to_schema_gate_report = deps.to_schema_gate_report
_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg
_score_arbiter = deps.score_arbiter
_is_env_true = deps.is_env_true
_demo_samples_dir = deps.demo_samples_dir
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


def _app():
    """Late binding so tests can patch helpers on oncology_arbiter.api.app."""
    import oncology_arbiter.api.app as app_module
    return app_module



@router.post("/v1/elo/rank", response_model=EloRankResponse)
def elo_rank(
    req: EloRankRequest,
) -> EloRankResponse:
    """L5 Co-Scientist: re-rank a caller-supplied set of therapy candidates.

    Two tournaments are run over the same set of drugs:

    1. **baseline_ranking** — the drug's own confidence + evidence + honesty
       markers, exactly as `run_co_scientist` scores hypotheses derived from
       stage envelopes. No disease context leaks in.
    2. **enriched_ranking** — same tournament but with per-drug score
       deltas from `req.modifiers` added to the Elo scorer. This is how
       HRD status boosts PARP inhibitors, PD-L1 CPS boosts pembrolizumab,
       and prior-lines burden penalises redosing options.

    The diff between the two rankings is returned as `matches[]` so the
    UI can render *why the order changed*. No LLM. Deterministic given
    `seed`. Free-tier safe (no model weights loaded).

    Honesty contract
    ----------------
    - `model_state`: `PROXY_CO_SCIENTIST` (deterministic, no live model).
    - Evidence URLs in each drug are echoed through the honesty gate:
      the response envelope's `evidence[]` field is the *union* of URLs
      registered on the input drugs, verbatim, so a caller can prove
      which sources fed the tournament.
    - `applied_modifiers` and `disease_context` are echoed verbatim on
      the response — no server-side re-derivation, no hidden bumps.
    """
    request_id = new_request_id()

    # Guard: reject duplicate drug_ids so the tournament stays well-defined.
    seen_ids: set[str] = set()
    for d in req.drugs:
        if d.drug_id in seen_ids:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate drug_id: {d.drug_id!r}",
            )
        seen_ids.add(d.drug_id)

    # Warn on modifier keys that don't map to any drug — surface as
    # response warnings, don't fail the request.
    warnings: list[str] = []
    unknown_mods = set(req.modifiers.keys()) - seen_ids
    for k in sorted(unknown_mods):
        warnings.append(f"unknown_modifier_drug_id:{k}")

    # Convert EloDrugCandidate → Hypothesis for the tournament.
    from oncology_arbiter.orchestrator.co_scientist import (
        Hypothesis,
        rank_hypotheses,
    )

    def _to_hyp(d: EloDrugCandidate) -> Hypothesis:
        return Hypothesis(
            hyp_id=d.drug_id,
            stage="therapy",
            statement=f"{d.regimen} (line {d.line})",
            confidence=d.confidence,
            evidence=[e.model_dump() for e in d.evidence],
            honesty_markers=dict(d.honesty_markers),
        )

    hyps = [_to_hyp(d) for d in req.drugs]

    # -- Baseline tournament ------------------------------------------------
    baseline_entries = rank_hypotheses(
        hyps, k_factor=req.k_factor, seed=req.seed,
    )

    # -- Enriched tournament: apply modifiers by re-ranking with adjusted
    # confidence values. Modifiers act as additive deltas on the
    # `_score_hypothesis` value; since score = confidence + evidence +
    # markers, adding to confidence is byte-equivalent to adding to
    # the raw score. Confidence is clamped to [0,1] for schema honesty.
    # NB: modifier deltas are added to the raw score, not clamped, so
    # the tournament preserves the relative order induced by the delta.
    # For the response schema, EloRankedEntry.confidence is clamped to
    # [0,1] so a caller can never see "confidence = 1.15" on the wire.
    enriched_hyps: list[Hypothesis] = []
    for d in req.drugs:
        delta = float(req.modifiers.get(d.drug_id, 0.0))
        raw_conf = d.confidence + delta
        enriched_hyps.append(Hypothesis(
            hyp_id=d.drug_id,
            stage="therapy",
            statement=f"{d.regimen} (line {d.line})",
            confidence=raw_conf,
            evidence=[e.model_dump() for e in d.evidence],
            honesty_markers=dict(d.honesty_markers),
        ))
    enriched_entries = rank_hypotheses(
        enriched_hyps, k_factor=req.k_factor, seed=req.seed,
    )

    # -- Serialize both rankings --------------------------------------------
    drug_by_id: dict[str, EloDrugCandidate] = {d.drug_id: d for d in req.drugs}

    def _to_ranked(entries: list[Any]) -> list[EloRankedEntry]:
        out: list[EloRankedEntry] = []
        for rank, entry in enumerate(entries, start=1):
            h = entry.hypothesis
            src = drug_by_id[h.hyp_id]
            out.append(EloRankedEntry(
                rank=rank,
                drug_id=h.hyp_id,
                regimen=src.regimen,
                line=src.line,
                rating=round(entry.rating, 4),
                wins=entry.wins,
                losses=entry.losses,
                draws=entry.draws,
                confidence=max(0.0, min(1.0, h.confidence)),
                honesty_markers=dict(h.honesty_markers),
                n_evidence=len(h.evidence),
            ))
        return out

    baseline_ranking = _to_ranked(baseline_entries)
    enriched_ranking = _to_ranked(enriched_entries)

    # -- Match records: diff by drug_id, ordered by enriched rank -----------
    baseline_by_id = {e.drug_id: e for e in baseline_ranking}
    matches: list[EloMatchRecord] = []
    for e in enriched_ranking:
        b = baseline_by_id[e.drug_id]
        delta = float(req.modifiers.get(e.drug_id, 0.0))
        reason = _elo_reason_for_modifier(e.drug_id, delta, req.disease_context)
        matches.append(EloMatchRecord(
            drug_id=e.drug_id,
            regimen=e.regimen,
            line=e.line,
            baseline_rank=b.rank,
            enriched_rank=e.rank,
            baseline_rating=b.rating,
            enriched_rating=e.rating,
            rank_delta=b.rank - e.rank,
            rating_delta=round(e.rating - b.rating, 4),
            applied_modifier=delta,
            reason=reason,
        ))

    # -- Provenance + evidence ---------------------------------------------
    prov = Provenance(
        model_name="oa/co_scientist_elo@v0.3.0-alpha",
        model_version="v0.3.0-alpha",
        model_state=ModelState.PROXY_CO_SCIENTIST,
        request_id=request_id,
    )

    # Union of evidence URLs across all drugs, dedup by URL, order-preserving.
    seen_urls: set[str] = set()
    merged_evidence: list[EvidenceRecord] = []
    for d in req.drugs:
        for e in d.evidence:
            if e.url not in seen_urls:
                seen_urls.add(e.url)
                merged_evidence.append(e)

    gate = HonestyGateReport(
        seen_urls_count=len(seen_urls),
        evidence_kept=len(merged_evidence),
        evidence_dropped=0,
    )

    log_event(
        request_id, "/v1/elo/rank",
        model_state=ModelState.PROXY_CO_SCIENTIST.value,
        patient_id_hash=None,
        extra={
            "n_candidates": len(req.drugs),
            "n_modifiers": len(req.modifiers),
            "n_unknown_modifiers": len(unknown_mods),
            "seed": req.seed,
            "k_factor": req.k_factor,
            "disease_context_keys": sorted(req.disease_context.keys()),
        },
        tenant_id=None,  # public route: no authenticated tenant
    )

    return EloRankResponse(
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        provenance=prov,
        honesty_gate=gate,
        evidence=merged_evidence,
        warnings=warnings,
        baseline_ranking=baseline_ranking,
        enriched_ranking=enriched_ranking,
        matches=matches,
        disease_context=req.disease_context,
        applied_modifiers=dict(req.modifiers),
        n_candidates=len(req.drugs),
    )

# ----------------------------------------------------------------------- #
# /v1/co_scientist/run — standalone 4-phase loop (PLAN §5 PR #5)
#
# Surfaces `run_co_scientist(...)` as a first-class endpoint so callers
# can run the honesty tournament without piping through /v1/case/full.
# Deterministic, no live LLM, free-tier safe.
#
# The load-bearing bit is REFLECT: any evidence URL not in
# `req.seed_urls` is dropped by `reflect_hypotheses`. A hostile input
# with N fake URLs will see `urls_dropped_hallucinated >= N` and
# matching warnings in the response envelope.

@router.post("/v1/offline_ranker/run", response_model=CoScientistRunResponse)
def co_scientist_run(
    req: CoScientistRunRequest,
) -> CoScientistRunResponse:
    """Run the 4-phase Co-Scientist loop over caller-supplied envelopes.

    Phases: generate → reflect → rank → evolve → rank.

    Honesty contract
    ----------------
    - `req.seed_urls` is the ONLY authority on what the tool-loop
      actually fetched. Any evidence URL NOT in this set is stripped
      by REFLECT before it can win Elo points.
    - `urls_dropped_hallucinated` is the count of URLs stripped.
    - `hypotheses_dropped` is the count of hypotheses whose evidence
      list was emptied by REFLECT (they're kept in `hypotheses[]` but
      a caller SHOULD treat them as untrusted).
    - Deterministic given identical inputs — no LLM sampling, no
      time-dependent randomness beyond the fixed-seed pair-order
      shuffle inside `rank_hypotheses`.
    """
    request_id = new_request_id()

    # Also parse hallucination markers from `screening.evidence[].url`,
    # etc. — anything already in the input envelopes counts as seen
    # unless the caller explicitly limits `seed_urls`. This mirrors the
    # behaviour of /v1/case/full's Co-Scientist call.
    seen_urls: set[str] = set(req.seed_urls or [])

    # Import inside the handler so cold-start of the module doesn't
    # depend on orchestrator imports (which pull in numpy).
    from oncology_arbiter.orchestrator.co_scientist import run_co_scientist

    cs_out = run_co_scientist(
        screening=req.screening,
        biopsy=req.biopsy,
        therapy=req.therapy,
        seen_urls=seen_urls,
        top_n_evolve=req.top_n_evolve,
        n_variants=req.n_variants,
        return_top=req.return_top,
        mode="offline",  # scalar path only — live clinical reasoning is /v1/co_scientist/run
    )

    # Count what REFLECT actually removed. Two metrics matter:
    #
    # 1. `urls_dropped_hallucinated`: the number of DISTINCT URLs that
    #    appeared in the input envelopes but were NOT in `seed_urls`.
    #    This is what the "5 fake URLs → all dropped by REFLECT" test
    #    checks — the honesty gate must strip every unseen URL, no
    #    matter how many hypotheses reference it.
    # 2. `hypotheses_dropped`: the number of hypotheses whose evidence
    #    list was emptied by REFLECT. Read from the warnings channel
    #    (`no_evidence_after_reflect:<hyp_id>`), which
    #    `reflect_hypotheses` emits verbatim.
    #
    # Both are exposed on the wire so a reviewer can prove the honesty
    # gate did its job on hostile input.
    proposed_urls: set[str] = set()
    for env in (req.screening, req.biopsy, req.therapy):
        if not env:
            continue
        for e in env.get("evidence") or []:
            if isinstance(e, dict) and e.get("url"):
                proposed_urls.add(e["url"])
        for opt in env.get("recommended_options") or []:
            for e in (opt or {}).get("evidence") or []:
                if isinstance(e, dict) and e.get("url"):
                    proposed_urls.add(e["url"])
    urls_dropped_total = len(proposed_urls - seen_urls)
    hyps_dropped = sum(
        1 for w in (cs_out.get("warnings") or [])
        if str(w).startswith("no_evidence_after_reflect:")
    )

    # Provenance / honesty envelope. We echo the seen_urls set on
    # `honesty_gate` so a caller can prove what was in scope; the
    # evidence field is left empty because run_co_scientist doesn't
    # produce a merged evidence list of its own (each hypothesis
    # carries its own `evidence[]`).
    prov = Provenance(
        model_name="oa/co_scientist@v0.4.0-alpha",
        model_version="v0.4.0-alpha",
        model_state=ModelState.PROXY_CO_SCIENTIST,
        request_id=request_id,
    )
    gate = HonestyGateReport(
        seen_urls_count=len(seen_urls),
        evidence_kept=0,  # Co-Scientist doesn't build a merged evidence list
        evidence_dropped=urls_dropped_total,
    )

    log_event(
        request_id, "/v1/offline_ranker/run",
        model_state=ModelState.PROXY_CO_SCIENTIST.value,
        patient_id_hash=None,
        extra={
            "has_screening": req.screening is not None,
            "has_biopsy": req.biopsy is not None,
            "has_therapy": req.therapy is not None,
            "n_seed_urls": len(seen_urls),
            "initial_count": cs_out.get("initial_count", 0),
            "after_reflect": cs_out.get("after_reflect", 0),
            "after_evolve": cs_out.get("after_evolve", 0),
            "urls_dropped_hallucinated": urls_dropped_total,
            "hypotheses_dropped": hyps_dropped,
            "top_n_evolve": req.top_n_evolve,
            "n_variants": req.n_variants,
            "return_top": req.return_top,
        },
        tenant_id=None,  # public route, matches /v1/elo/rank
    )

    return CoScientistRunResponse(
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        provenance=prov,
        honesty_gate=gate,
        evidence=[],  # per-hypothesis evidence lives on `hypotheses[]`
        warnings=list(cs_out.get("warnings") or []),
        phases=list(cs_out.get("phases") or []),
        hypotheses=list(cs_out.get("hypotheses") or []),
        initial_count=int(cs_out.get("initial_count", 0)),
        after_reflect=int(cs_out.get("after_reflect", 0)),
        after_evolve=int(cs_out.get("after_evolve", 0)),
        urls_dropped_hallucinated=urls_dropped_total,
        hypotheses_dropped=hyps_dropped,
        engine=str(cs_out.get("engine") or "offline_scalar_ranker"),
        tournament_transcript=[],
        patient_context=None,
    )
