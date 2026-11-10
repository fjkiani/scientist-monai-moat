"""LUNA16 NSCLC branch of POST /v1/case/full?cancer=nsclc."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from oncology_arbiter.api import deps
from oncology_arbiter.api.audit import log_event
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.schemas import FullCaseResponse, ModelState, NsclcResponse, ReportParseBlock
from oncology_arbiter.api.services.helpers import failed_stage_receipt as _failed_stage_receipt
from oncology_arbiter.api.services.helpers import pipeline_status as _pipeline_status
from oncology_arbiter.api.services.helpers import skipped_stage_receipt as _skipped_stage_receipt

router = APIRouter()

_envelope = deps.envelope
_medgemma_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


def run_nsclc_case(req: Any, *, request_id: str, tenant: Any) -> FullCaseResponse:
    from .nlp import _run_medgemma_co_scientist

    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    case_manifest: dict[str, Any] | None = None
    luna_output: dict[str, Any] | None = None
    report_parse: ReportParseBlock | None = None
    therapy_bridge: dict[str, Any] | None = None

    if not req.case_id:
        receipts.append(_failed_stage_receipt(
            "case_storage", True, request_id, "case_id_required",
            "NSCLC production inference requires a verified case-storage case_id.",
            service_name="case-storage",
        ))
        receipts.append(_failed_stage_receipt(
            "luna16_detection", True, request_id, "blocked_by_missing_case_id",
            "LUNA16 cannot run without a verified case-storage case_id.",
            service_name="luna16-infer",
        ))
        # Surface the suppression at the top level too. stage_receipts and
        # pipeline_status already carry it, but an all-None NSCLC envelope
        # with an empty warnings[] can be misread by a downstream consumer
        # as a completed inference that simply found nothing.
        warnings.append(
            "nsclc_required_stages_not_executed:no_verified_case_id_supplied:"
            "empty_result_is_not_a_negative_finding"
        )
    else:
        try:
            from oncology_arbiter.models.specialist_clients import CaseStorageClient, Luna16Client

            manifest_call = CaseStorageClient().manifest(req.case_id, request_id=request_id, required=True)
            case_manifest = manifest_call.output
            receipts.append(manifest_call.receipt)
            luna_call = Luna16Client().detect(req.case_id, request_id=request_id, required=True)
            luna_output = luna_call.output
            luna_call.receipt["input_reference"] = (
                f"case_id:{req.case_id};manifest_sha256:{case_manifest['manifest_sha256']}"
            )
            receipts.append(luna_call.receipt)
        except Exception as exc:
            failed_stage = "luna16_detection" if case_manifest is not None else "case_storage"
            receipts.append(_failed_stage_receipt(
                failed_stage, True, request_id,
                getattr(exc, "code", f"{failed_stage}_failed"), f"{type(exc).__name__}: {exc}",
                service_name="luna16-infer" if failed_stage == "luna16_detection" else "case-storage",
                input_reference=f"case_id:{req.case_id}",
            ))
            if failed_stage == "case_storage":
                receipts.append(_failed_stage_receipt(
                    "luna16_detection", True, request_id, "blocked_by_case_storage",
                    "LUNA16 was not called because case-storage manifest verification failed.",
                    service_name="luna16-infer", input_reference=f"case_id:{req.case_id}",
                ))

    if req.biopsy_input and req.biopsy_input.report_text:
        report_parse, _panel, _grade, receipt = _medgemma_biopsy_parse(
            req.biopsy_input.report_text, request_id=request_id
        )
        receipts.append(receipt)
    else:
        receipts.append(_skipped_stage_receipt(
            "medgemma_pathology_parse", request_id, "no report_text supplied"
        ))

    mutation_payload = [item.model_dump(exclude_none=True) for item in req.mutations]
    germline_payload = [item.model_dump(exclude_none=True) for item in req.germline_mutations]
    if mutation_payload or germline_payload:
        try:
            from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

            call = SLTherapyBridgeClient().run({
                "disease": "non-small cell lung cancer",
                "cancer_type": "nsclc",
                "mutations": mutation_payload,
                "germline_mutations": germline_payload,
                "ranked_drugs": [item.model_dump(exclude_none=True) for item in req.ranked_drugs],
                "include_explanations": False,
            }, request_id=request_id, required=True)
            therapy_bridge = call.output
            receipts.append(call.receipt)
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "synthetic_lethality_therapy_bridge", True, request_id,
                getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                service_name="crispro-backend-v2",
            ))
    else:
        receipts.append(_skipped_stage_receipt(
            "synthetic_lethality_therapy_bridge", request_id,
            "no patient mutation profile supplied; no therapy recommendation computed",
        ))

    co_scientist = _run_medgemma_co_scientist(
        context={
            "cancer": "nsclc",
            "case_manifest": case_manifest,
            "luna16_detection": luna_output,
            "report_parse": report_parse.model_dump(mode="json") if report_parse else None,
            "therapy_bridge": therapy_bridge,
            "patient_context": req.therapy_context.model_dump(mode="json"),
        },
        request_id=request_id,
        requested=req.run_co_scientist,
        has_patient_signal=any((luna_output, report_parse, therapy_bridge)),
        receipts=receipts,
    )

    status = _pipeline_status(receipts)
    state = ModelState.LOADED_LUNA16_RETINANET if luna_output is not None else ModelState.UNAVAILABLE
    luna_block = None
    if luna_output is not None:
        luna_block = {
            "bundle_version": luna_output["bundle_version"],
            "n_detections": int(luna_output.get("n_detections", len(luna_output.get("detections") or []))),
            "top_score": float(luna_output.get("top_score", 0.0)),
            "detections": list(luna_output.get("detections") or []),
            "inference_seconds": float(luna_output.get("inference_seconds", 0.0)),
            "preprocessing_summary": dict(luna_output.get("preprocessing_summary") or {}),
        }
    nsclc = NsclcResponse(
        model_state=state,
        model_name=(luna_output or {}).get("model_name", "monai/lung_nodule_ct_detection@0.6.9"),
        warnings=warnings,
        luna16=luna_block,
        parsed_report=report_parse.parsed_entities if report_parse else None,
        parsed_report_provenance=(
            {
                "parser_id": report_parse.parser_id,
                "app_version": report_parse.app_version,
                "model_sha256": report_parse.model_sha256,
                "metrics_sha256": report_parse.metrics_sha256,
                "window_tokens": report_parse.window_tokens,
                "overlap_tokens": report_parse.overlap_tokens,
                "window_aggregation": report_parse.window_aggregation,
            }
            if report_parse else None
        ),
    )
    log_event(
        request_id, "/v1/case/full", model_state=state.value,
        patient_id_hash=None,
        extra={"cancer": "nsclc", "pipeline_status": status},
        tenant_id=tenant.tenant_id,
    )
    return FullCaseResponse(
        **_envelope(request_id, model_state=state, model_name="specialist-stack-composite"),
        warnings=warnings,
        pipeline_status=status,
        cancer="nsclc",
        stage_receipts=receipts,
        case_manifest=case_manifest,
        report_parse=report_parse,
        luna16_detection=luna_output,
        therapy_bridge=therapy_bridge,
        co_scientist=co_scientist,
        screening=None,
        biopsy=None,
        therapy=None,
        nsclc=nsclc,
        elo_ranked_hypotheses=[],
    )
