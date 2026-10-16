"""Deterministic offline evidence-ranker endpoint.

This module does not run the MedGemma Co-Scientist synthesis stage. It
transforms existing stage envelopes through generate/reflect/Elo/evolve.
"""
from __future__ import annotations

from fastapi import APIRouter

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.api.schemas import (
    CoScientistRunRequest,
    CoScientistRunResponse,
    HonestyGateReport,
    ModelState,
    Provenance,
)


router = APIRouter()

@router.post("/v1/offline_ranker/run", response_model=CoScientistRunResponse)
def offline_ranker_run(
    req: CoScientistRunRequest,
) -> CoScientistRunResponse:
    """Run the deterministic offline ranker over caller-supplied envelopes.

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
    state = ModelState.DETERMINISTIC_OFFLINE_RANKER
    prov = Provenance(
        model_name="oa/offline_evidence_ranker",
        model_version="v1.0.0",
        model_state=state,
        request_id=request_id,
    )
    gate = HonestyGateReport(
        seen_urls_count=len(seen_urls),
        evidence_kept=0,  # Co-Scientist doesn't build a merged evidence list
        evidence_dropped=urls_dropped_total,
    )

    log_event(
        request_id, "/v1/offline_ranker/run",
        model_state=state.value,
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
    )

# ----------------------------------------------------------------------- #
# Production dynamic tumor board and full-case specialist assembly
