"""Auto-extracted router: elo."""
from __future__ import annotations

from typing import Any


from fastapi import APIRouter, HTTPException

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api import deps
from oncology_arbiter.api.services import runtime_hooks
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.schemas import (
    ModelState, EloRankRequest, EloRankResponse, EloDrugCandidate,
    EloMatchRecord, EloRankedEntry, Provenance, HonestyGateReport, EvidenceRecord,
)

_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg
_score_arbiter = deps.score_arbiter
_to_schema_gate_report = deps.to_schema_gate_report
_is_env_true = deps.is_env_true
_demo_samples_dir = deps.demo_samples_dir
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report
_medgemma_biopsy_parse = PathologyParseOrchestrator._parse_medgemma

# Patchable via oncology_arbiter.api.app.* in unit tests
_run_medsiglip_on_preprocessed = runtime_hooks.run_medsiglip_on_preprocessed
_run_cbis_ddsm_probe_on_bytes = runtime_hooks.run_cbis_ddsm_probe_on_bytes
_elo_reason_for_modifier = runtime_hooks.elo_reason_for_modifier
_run_clinicalbert_parse = runtime_hooks.run_clinicalbert_parse
_score_explicit_therapy_triage = runtime_hooks.score_explicit_therapy_triage


router = APIRouter()

@router.post("/v1/elo/rank", response_model=EloRankResponse)
def elo_rank(
    req: EloRankRequest,
) -> EloRankResponse:
    """Deterministically re-rank a caller-supplied set of therapy candidates.

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
    - `model_state`: `DETERMINISTIC_OFFLINE_RANKER` (no live model).
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
        model_name="oa/offline_evidence_elo_ranker",
        model_version="v1.0.0",
        model_state=ModelState.DETERMINISTIC_OFFLINE_RANKER,
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
        model_state=ModelState.DETERMINISTIC_OFFLINE_RANKER.value,
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
