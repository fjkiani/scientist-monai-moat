"""Identity-locked MedGemma Co-Scientist routes (hard path).

* ``POST /v1/co-scientist/run`` — supervisor.execute_stage (GENERATE→REFLECT→Elo)
* ``POST /v1/co_scientist/run`` — live MedGemma pairwise judge over stage envelopes
* Offline scalar stays at ``POST /v1/offline_ranker/run`` in ``elo.py``
"""
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
from oncology_arbiter.auth import APIKey, require_api_key, require_verified_api_key

router = APIRouter()


@router.post("/v1/co_scientist/run", response_model=CoScientistRunResponse)
def live_envelope_co_scientist_run(
    req: CoScientistRunRequest,
    tenant: APIKey = Depends(require_api_key),
) -> CoScientistRunResponse:
    """Live MedGemma pairwise clinical reasoning over caller stage envelopes.

    Rejects the offline scalar path. Requires ``MEDGEMMA_MODAL_URL`` pinned.
    """
    endpoint_url = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint_url:
        raise HTTPException(
            status_code=503,
            detail="MEDGEMMA_MODAL_URL is required for live /v1/co_scientist/run",
        )
    request_id = new_request_id()
    try:
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import assert_pinned_chat_url
        from oncology_arbiter.orchestrator.co_scientist import run_co_scientist

        assert_pinned_chat_url(endpoint_url)
        seen_urls: set[str] = set(req.seed_urls or [])
        for env in (req.screening, req.biopsy, req.therapy):
            if not env:
                continue
            for e in env.get("evidence") or []:
                if isinstance(e, dict) and e.get("url"):
                    seen_urls.add(str(e["url"]))
            for opt in env.get("recommended_options") or []:
                for e in (opt or {}).get("evidence") or []:
                    if isinstance(e, dict) and e.get("url"):
                        seen_urls.add(str(e["url"]))
        cs_out = run_co_scientist(
            screening=req.screening,
            biopsy=req.biopsy,
            therapy=req.therapy,
            seen_urls=seen_urls,
            top_n_evolve=req.top_n_evolve,
            n_variants=req.n_variants,
            return_top=req.return_top,
            mode="live",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"live co-scientist failed: {type(exc).__name__}: {exc}",
        ) from exc

    if cs_out.get("engine") != "medgemma_pairwise_judge":
        raise HTTPException(status_code=502, detail="live engine did not engage MedGemma judge")
    transcript = list(cs_out.get("tournament_transcript") or [])
    if not transcript:
        raise HTTPException(status_code=503, detail="empty MedGemma tournament transcript")
    if any(m.get("clinical_rationale") == "offline_scalar_score" for m in transcript):
        raise HTTPException(status_code=502, detail="scalar score leaked into live transcript")

    proposed_urls: set[str] = set()
    for env in (req.screening, req.biopsy, req.therapy):
        if not env:
            continue
        for e in env.get("evidence") or []:
            if isinstance(e, dict) and e.get("url"):
                proposed_urls.add(str(e["url"]))
    urls_dropped_total = len(proposed_urls - seen_urls)
    hyps_dropped = sum(
        1 for w in (cs_out.get("warnings") or [])
        if str(w).startswith("no_evidence_after_reflect:")
    )
    prov = Provenance(
        model_name="oa/co_scientist_medgemma_pairwise",
        model_version="l5-live-v1",
        model_state=ModelState.LOADED_MEDGEMMA_1P5_4B,
        request_id=request_id,
        model_endpoint_url=endpoint_url,
    )
    gate = HonestyGateReport(
        seen_urls_count=len(seen_urls),
        evidence_kept=0,
        evidence_dropped=urls_dropped_total,
    )
    log_event(
        request_id,
        "/v1/co_scientist/run",
        model_state=ModelState.LOADED_MEDGEMMA_1P5_4B.value,
        patient_id_hash=None,
        extra={
            "engine": cs_out.get("engine"),
            "n_matches": len(transcript),
            "n_hypotheses": len(cs_out.get("hypotheses") or []),
            "tenant_id": tenant.tenant_id,
        },
        tenant_id=tenant.tenant_id,
    )
    return CoScientistRunResponse(
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        provenance=prov,
        honesty_gate=gate,
        evidence=[],
        warnings=list(cs_out.get("warnings") or []),
        phases=list(cs_out.get("phases") or []),
        hypotheses=list(cs_out.get("hypotheses") or []),
        initial_count=int(cs_out.get("initial_count", 0)),
        after_reflect=int(cs_out.get("after_reflect", 0)),
        after_evolve=int(cs_out.get("after_evolve", 0)),
        urls_dropped_hallucinated=urls_dropped_total,
        hypotheses_dropped=hyps_dropped,
        engine=str(cs_out.get("engine")),
        tournament_transcript=transcript,
        patient_context=cs_out.get("patient_context"),
    )


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

    meta_review = str(output.get("meta_review") or "").strip()
    meta_rid = str(output.get("meta_review_request_id") or "").strip()
    meta_sha = str(output.get("meta_review_response_sha256") or "").strip()
    if not meta_review or not meta_rid or not meta_sha:
        raise HTTPException(status_code=503, detail="MedGemma META_REVIEW incomplete")

    hypotheses = list(output.get("hypotheses") or [])
    if len(hypotheses) < 2:
        raise HTTPException(status_code=503, detail="Co-Scientist returned fewer than 2 hypotheses")

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
        meta_review=meta_review,
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
        meta_review_request_id=meta_rid,
        meta_review_response_sha256=meta_sha,
        latency_s=float(output.get("latency_s") or 0.0),
    )
