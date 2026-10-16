"""Co-Scientist and offline ranker routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api.schemas import (
    CoScientistRunRequest,
    CoScientistRunResponse,
    EvidenceRecord,
    HonestyGateReport,
    ModelState,
    Provenance,
    RealCoScientistRequest,
    RealCoScientistResponse,
)
from oncology_arbiter.auth import APIKey, require_verified_api_key

router = APIRouter()


@router.post("/v1/co-scientist/run", response_model=RealCoScientistResponse)
def real_co_scientist_run(
    req: RealCoScientistRequest,
    tenant: APIKey = Depends(require_verified_api_key),
) -> RealCoScientistResponse:
    """Run the evidence-grounded MedGemma Co-Scientist and pairwise Elo loop.

    This endpoint never falls back to the deterministic offline ranker or an
    alternate LLM. Every tournament pair must carry a MedGemma request receipt.
    """
    endpoint_url = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint_url:
        raise HTTPException(
            status_code=503,
            detail=(
                "MEDGEMMA_MODAL_URL is required; alternate LLMs and "
                "deterministic Elo are not substitutes"
            ),
        )

    request_id = new_request_id()
    try:
        from oncology_arbiter.agents.supervisor import execute_stage
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
            assert_pinned_chat_url,
            verify_artifact_identity,
        )

        identity = verify_artifact_identity()
        assert_pinned_chat_url(endpoint_url)
        result = execute_stage(
            req.stage,
            req.context,
            n_hypotheses=req.n_hypotheses,
            n_evidence_top_k=req.n_evidence_top_k,
            seed_urls=[str(url) for url in req.seed_urls],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MedGemma Co-Scientist execution failed: {type(exc).__name__}: {exc}",
        ) from exc

    output = result.as_dict()
    if result.model_state != "executed":
        raise HTTPException(
            status_code=503,
            detail=(
                "MedGemma Co-Scientist did not complete: "
                f"model_state={result.model_state}; notes={result.notes}"
            ),
        )
    if output.get("llm_provider") != "medgemma_modal":
        raise HTTPException(status_code=502, detail="Unexpected Co-Scientist LLM provider")
    if (
        output.get("llm_model") != identity["model_id"]
        or output.get("llm_app_version") != identity["app_version"]
        or output.get("llm_revision") != identity["revision_sha256"]
    ):
        raise HTTPException(status_code=502, detail="MedGemma response identity mismatch")
    if not output.get("llm_request_ids"):
        raise HTTPException(status_code=502, detail="MedGemma request receipts are missing")

    reflections_expected = int(output.get("reflection_calls_expected") or 0)
    reflections_completed = int(output.get("reflection_calls_completed") or 0)
    reflection_receipts = list(output.get("reflection_receipts") or [])
    if (
        reflections_expected < 1
        or reflections_completed != reflections_expected
        or len(reflection_receipts) != reflections_expected
        or any(
            not item.get("completed") or not item.get("model_request_id")
            for item in reflection_receipts
        )
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "MedGemma REFLECT incomplete: "
                f"{reflections_completed}/{reflections_expected}"
            ),
        )

    expected = int(output.get("tournament_pairs_expected") or 0)
    completed = int(output.get("tournament_pairs_completed") or 0)
    matches = list(output.get("tournament_matches") or [])
    if (
        expected < 1
        or completed != expected
        or len(matches) != expected
        or any(not match.get("completed") or not match.get("model_request_id") for match in matches)
    ):
        raise HTTPException(
            status_code=503,
            detail=f"MedGemma Elo tournament incomplete: {completed}/{expected}",
        )

    hypotheses = list(output.get("hypotheses") or [])
    if len(hypotheses) < 2:
        raise HTTPException(status_code=503, detail="Fewer than two ranked hypotheses survived")
    completed_reflect_ids = {
        str(item.get("hypothesis_id"))
        for item in reflection_receipts
        if item.get("completed") and item.get("model_request_id")
    }
    hypothesis_ids = {str(item.get("hypothesis_id") or "") for item in hypotheses}
    if (
        reflections_expected != len(hypotheses)
        or completed_reflect_ids != hypothesis_ids
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Every ranked hypothesis must complete evidence-backed MedGemma "
                "REFLECT before pairwise Elo"
            ),
        )
    if (
        not str(output.get("meta_review") or "").strip()
        or not output.get("meta_review_request_id")
        or not output.get("meta_review_response_sha256")
    ):
        raise HTTPException(
            status_code=503,
            detail="MedGemma META_REVIEW text or model receipt is missing",
        )

    evidence = [
        EvidenceRecord(
            url=str(item.get("url") or ""),
            quoted_text=str(item.get("snippet") or item.get("title") or "").strip(),
            source=str(item.get("source") or "unknown"),
        )
        for item in output.get("evidence") or []
        if item.get("url") and (item.get("snippet") or item.get("title"))
    ]
    if not evidence:
        raise HTTPException(status_code=503, detail="No fetched evidence survived the honesty gate")

    provenance = Provenance(
        model_state=ModelState.LOADED_MEDGEMMA_1P5_4B,
        model_name=identity["model_id"],
        model_version=identity["revision_sha256"],
        request_id=request_id,
        model_endpoint_url=endpoint_url,
    )
    honesty_gate = HonestyGateReport(
        seen_urls_count=int(output.get("seen_urls_count") or 0),
        evidence_kept=int(output.get("evidence_kept") or 0),
        evidence_dropped=int(output.get("evidence_dropped") or 0),
    )
    log_event(
        request_id,
        "/v1/co-scientist/run",
        model_state=ModelState.LOADED_MEDGEMMA_1P5_4B.value,
        patient_id_hash=None,
        extra={
            "stage": req.stage,
            "hypotheses": len(hypotheses),
            "evidence_kept": len(evidence),
            "reflection_calls_expected": reflections_expected,
            "reflection_calls_completed": reflections_completed,
            "tournament_pairs_expected": expected,
            "tournament_pairs_completed": completed,
            "model_request_ids": len(output["llm_request_ids"]),
        },
        tenant_id=tenant.tenant_id,
    )
    return RealCoScientistResponse(
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        provenance=provenance,
        honesty_gate=honesty_gate,
        evidence=evidence,
        warnings=list(output.get("llm_honesty_warnings") or []),
        stage=req.stage,
        supervisor_version=str(output.get("supervisor_version") or ""),
        hypotheses=hypotheses,
        meta_review=str(output["meta_review"]),
        seen_urls=list(output.get("seen_urls") or []),
        reflection_calls_expected=reflections_expected,
        reflection_calls_completed=reflections_completed,
        reflection_receipts=reflection_receipts,
        tournament_pairs_expected=expected,
        tournament_pairs_completed=completed,
        tournament_matches=matches,
        llm_calls=int(output.get("llm_calls") or 0),
        tokens_in=int(output.get("llm_prompt_tokens") or 0),
        tokens_out=int(output.get("llm_completion_tokens") or 0),
        cost_usd=float(output.get("llm_cost_usd") or 0.0),
        provider=str(output["llm_provider"]),
        provider_chain=list(output.get("llm_provider_chain") or []),
        model_id=str(output["llm_model"]),
        app_name=str(output["llm_app_version"]),
        model_revision=str(output["llm_revision"]),
        model_request_ids=list(output["llm_request_ids"]),
        meta_review_request_id=str(output["meta_review_request_id"]),
        meta_review_response_sha256=str(output["meta_review_response_sha256"]),
        latency_s=float(output.get("latency_s") or 0.0),
    )

@router.post("/v1/offline_ranker/run", response_model=CoScientistRunResponse)
def co_scientist_run(
    req: CoScientistRunRequest,
) -> CoScientistRunResponse:
    """Run the deterministic offline evidence ranker over caller-supplied envelopes.

    This is not the MedGemma Co-Scientist. Phases: generate → reflect → rank →
    evolve → rank over caller envelopes only.

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
    prov = Provenance(
        model_name="oa/offline_evidence_ranker",
        model_version="v0.4.0-alpha",
        model_state=ModelState.DETERMINISTIC_OFFLINE_RANKER,
        request_id=request_id,
    )
    gate = HonestyGateReport(
        seen_urls_count=len(seen_urls),
        evidence_kept=0,  # Co-Scientist doesn't build a merged evidence list
        evidence_dropped=urls_dropped_total,
    )

    log_event(
        request_id, "/v1/offline_ranker/run",
        model_state=ModelState.DETERMINISTIC_OFFLINE_RANKER.value,
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
    )

