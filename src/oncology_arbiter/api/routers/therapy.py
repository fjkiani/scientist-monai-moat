"""Route module split out of api/app.py — L4c therapy via MedGemma (TxGemma retired)."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends

from oncology_arbiter.api import deps
from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.schemas import *  # noqa: F403
from oncology_arbiter.api.services.helpers import failed_stage_receipt as _failed_stage_receipt
from oncology_arbiter.api.services.helpers import pipeline_status as _pipeline_status
from oncology_arbiter.api.services.helpers import skipped_stage_receipt as _skipped_stage_receipt
from oncology_arbiter.arbiter.manski import ManskiGateError
from oncology_arbiter.auth import APIKey, require_api_key

router = APIRouter()

_envelope = deps.envelope
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


def _app():
    """Late binding so tests can patch helpers on oncology_arbiter.api.app."""
    import oncology_arbiter.api.app as app_module
    return app_module


def _biopsy_summary(req: TherapyRequest) -> dict[str, Any]:
    panel = req.receptors_override
    if panel is None and req.biopsy_output is not None:
        panel = req.biopsy_output.receptor_panel
    grade = None
    subtype = None
    if req.biopsy_output is not None:
        grade = req.biopsy_output.grade
        subtype = req.biopsy_output.subtype_prediction
    return {
        "receptor_panel": panel.model_dump(mode="json") if panel is not None else None,
        "grade": grade,
        "subtype_prediction": subtype,
        "patient_context": req.patient_context.model_dump(mode="json"),
    }


@router.post("/v1/therapy/reason", response_model=TherapyResponse)
def therapy_reason(
    req: TherapyRequest,
    tenant: APIKey = Depends(require_api_key),
) -> TherapyResponse:
    """MedGemma therapy reasoning + PubMed evidence for the primary drug.

    TxGemma is not called. SL-bridge remains an optional enrichment when
    mutations are supplied. Research triage stays optional.
    """
    request_id = new_request_id()
    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    evidence: list[EvidenceRecord] = []
    if req.legacy_therapy_benefit_model_name is not None:
        warnings.append("deprecated_prognostic_model_alias_remapped")

    bridge_output: dict[str, Any] | None = None
    recommended: list[TherapyOption] = []
    not_recommended: list[TherapyOption] = []
    model_state = ModelState.UNAVAILABLE
    model_name: str | None = None
    medgemma_meta: dict[str, Any] | None = None

    # --- L4c: MedGemma therapy ---
    try:
        from oncology_arbiter.nlp.medgemma_therapy_client import (
            MedGemmaTherapyError,
            reason_therapy,
        )

        t0 = time.perf_counter()
        medgemma_meta = reason_therapy(
            patient_context=req.patient_context.model_dump(mode="json"),
            biopsy_summary=_biopsy_summary(req),
            disease=req.disease,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        for item in medgemma_meta.get("recommended") or []:
            recommended.append(
                TherapyOption(
                    regimen=str(item["regimen"]),
                    line_of_therapy=int(item["line_of_therapy"]),
                    rationale=str(item["rationale"]),
                )
            )
        for item in medgemma_meta.get("not_recommended") or []:
            not_recommended.append(
                TherapyOption(
                    regimen=str(item["regimen"]),
                    line_of_therapy=int(item["line_of_therapy"]),
                    rationale=str(item["rationale"]),
                )
            )
        model_state = ModelState.LOADED_MEDGEMMA_THERAPY
        model_name = str(medgemma_meta.get("model_id") or "google/medgemma-1.5-4b-it")
        receipts.append(
            {
                "stage": "medgemma_therapy_reason",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "medgemma-1-5-4b",
                "endpoint_label": medgemma_meta.get("chat_url"),
                "app_version": medgemma_meta.get("app_version"),
                "model_name": model_name,
                "model_version": medgemma_meta.get("revision_sha256"),
                "input_reference": f"transport:{medgemma_meta.get('transport')}",
                "latency_ms": latency_ms,
                "warnings": [
                    w
                    for w in [
                        medgemma_meta.get("honesty_warning"),
                        "ruo_generative_therapy_not_clinical_decision",
                    ]
                    if w
                ],
                "error": None,
            }
        )
        if medgemma_meta.get("honesty_warning"):
            warnings.append(str(medgemma_meta["honesty_warning"]))
    except Exception as exc:  # noqa: BLE001 — fail-closed on MedGemma
        receipts.append(
            _failed_stage_receipt(
                "medgemma_therapy_reason",
                True,
                request_id,
                getattr(exc, "code", "medgemma_therapy_failed"),
                f"{type(exc).__name__}: {exc}",
                service_name="medgemma-1-5-4b",
            )
        )
        warnings.append(f"medgemma_therapy_required_stage_failed:{type(exc).__name__}")

    # --- L2: PubMed on primary drug ---
    primary_drug = None
    if medgemma_meta is not None:
        primary_drug = medgemma_meta.get("primary_drug")
    if not primary_drug and recommended:
        primary_drug = recommended[0].regimen
    if primary_drug:
        try:
            from oncology_arbiter.tools.pubmed_sync import evidence_records_from_pubmed

            t0 = time.perf_counter()
            query = f"{primary_drug} {req.disease} therapy"
            raw_ev = evidence_records_from_pubmed(query, max_results=3)
            for row in raw_ev:
                evidence.append(EvidenceRecord(**row))
            # Attach top evidence onto the primary TherapyOption as well.
            if recommended and evidence:
                recommended[0] = TherapyOption(
                    regimen=recommended[0].regimen,
                    line_of_therapy=recommended[0].line_of_therapy,
                    rationale=recommended[0].rationale,
                    evidence=list(evidence[:3]),
                    contraindications=list(recommended[0].contraindications),
                )
            receipts.append(
                {
                    "stage": "pubmed_evidence_fetch",
                    "required": False,
                    "status": "succeeded" if evidence else "failed_optional",
                    "request_id": request_id,
                    "service_name": "ncbi_eutils",
                    "model_name": "pubmed_search",
                    "input_reference": f"query:{query[:120]}",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "warnings": [] if evidence else ["pubmed_returned_zero_hits"],
                    "error": None
                    if evidence
                    else {"code": "empty_pubmed", "message": "no PubMed hits"},
                }
            )
            if not evidence:
                warnings.append("pubmed_evidence_empty")
        except Exception as exc:  # noqa: BLE001
            receipts.append(
                _failed_stage_receipt(
                    "pubmed_evidence_fetch",
                    False,
                    request_id,
                    "pubmed_failed",
                    f"{type(exc).__name__}: {exc}",
                    service_name="ncbi_eutils",
                )
            )
            warnings.append(f"pubmed_evidence_failed:{type(exc).__name__}")
    else:
        receipts.append(
            _skipped_stage_receipt(
                "pubmed_evidence_fetch",
                request_id,
                "no primary_drug from MedGemma; PubMed skipped",
            )
        )

    # --- Optional SL bridge enrichment ---
    if req.mutations or req.germline_mutations:
        try:
            from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

            payload = {
                "disease": req.disease,
                "cancer_type": req.cancer_type,
                "mutations": req.mutations,
                "germline_mutations": req.germline_mutations,
                "ranked_drugs": req.ranked_drugs,
                "include_explanations": False,
            }
            call = SLTherapyBridgeClient().run(payload, request_id=request_id, required=False)
            bridge_output = call.output
            receipts.append(call.receipt)
            for item in bridge_output.get("sl_indicated_drugs") or []:
                if not isinstance(item, dict) or not item.get("drug_name"):
                    continue
                name = str(item["drug_name"])
                if any(r.regimen.lower() == name.lower() for r in recommended):
                    continue
                recommended.append(
                    TherapyOption(
                        regimen=name,
                        line_of_therapy=1,
                        rationale=str(
                            item.get("sl_rationale") or "Synthetic-lethality actionability hypothesis"
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            receipts.append(
                _failed_stage_receipt(
                    "synthetic_lethality_therapy_bridge",
                    False,
                    request_id,
                    getattr(exc, "code", "sl_bridge_failed"),
                    f"{type(exc).__name__}: {exc}",
                    service_name="crispro-backend-v2",
                )
            )
            warnings.append("sl_bridge_optional_stage_failed")
    else:
        receipts.append(
            _skipped_stage_receipt(
                "synthetic_lethality_therapy_bridge",
                request_id,
                "no patient mutation profile supplied",
            )
        )

    triage: ArbiterScore | None = None
    try:
        triage = _app()._score_explicit_therapy_triage(req.therapy_features)
    except ManskiGateError:
        raise
    except Exception as exc:
        receipts.append(
            _failed_stage_receipt(
                "therapy_research_triage",
                False,
                request_id,
                "invalid_triage_features",
                f"{type(exc).__name__}: {exc}",
                service_name="oncology-arbiter",
            )
        )
    if triage is None and not any(r["stage"] == "therapy_research_triage" for r in receipts):
        receipts.append(
            _skipped_stage_receipt(
                "therapy_research_triage",
                request_id,
                "no explicit triage vector supplied; nothing to identify",
            )
        )
    elif triage is not None:
        receipts.append(
            {
                "stage": "therapy_research_triage",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": triage.model_name,
                "model_version": "template-v0",
                "input_reference": (
                    "explicit_complete_patient_features"
                    if not triage.provenance_warnings
                    else "partially_unobserved_panel"
                ),
                "latency_ms": 0.0,
                "warnings": [
                    "illustrative_research_triage_not_recommendation_or_treatment_benefit",
                    *triage.provenance_warnings,
                ],
                "error": None,
            }
        )

    status = _pipeline_status(receipts)
    if model_state == ModelState.UNAVAILABLE and any(
        r["status"] == "failed_required" for r in receipts
    ):
        model_state = ModelState.UNAVAILABLE
    env = _envelope(request_id, model_state=model_state, model_name=model_name)
    env["warnings"] = warnings
    env["evidence"] = evidence
    # Honesty gate: count PubMed URLs that survived.
    env["honesty_gate"] = HonestyGateReport(
        seen_urls_count=len(evidence),
        evidence_kept=len(evidence),
        evidence_dropped=0,
    )
    log_event(
        request_id,
        "/v1/therapy/reason",
        model_state=model_state.value,
        patient_id_hash=None,
        extra={
            "pipeline_status": status,
            "n_recommended": len(recommended),
            "n_evidence": len(evidence),
            "primary_drug": primary_drug,
        },
        tenant_id=tenant.tenant_id,
    )
    return TherapyResponse(
        **env,
        pipeline_status=status,
        stage_receipts=receipts,
        therapy_bridge=bridge_output,
        therapy_triage=triage,
        recommended_options=recommended,
        not_recommended=not_recommended,
        prognostic_model_executed=False,
        prognostic_score=None,
        arbiter_score=triage,
    )
