"""Auto-extracted router: case."""
from __future__ import annotations

import hashlib
import os
from typing import Any


from fastapi import APIRouter, Depends, HTTPException, Query

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.api import deps
from oncology_arbiter.api.services import runtime_hooks
from oncology_arbiter.api.services.helpers import (
    failed_stage_receipt as _failed_stage_receipt,
    skipped_stage_receipt as _skipped_stage_receipt,
    pipeline_status as _pipeline_status,
)
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.routers.screening import screening_analyze
from oncology_arbiter.api.schemas import (
    ModelState, ReportParseBlock,
    DynamicTumorBoardRequest, DynamicTumorBoardResponse,
    FullCaseRequest, FullCaseResponse,
    TumorBoardBundle, TumorBoardBundleResponse,
    NsclcResponse,
)
from oncology_arbiter.api.services.tumor_board import (
    TumorBoardAssembler,
    SUPPORTED_CANCERS,
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

@router.post("/v1/tumor_board/dynamic", response_model=DynamicTumorBoardResponse)
def dynamic_tumor_board(
    req: DynamicTumorBoardRequest,
    tenant: APIKey = Depends(require_api_key),
) -> DynamicTumorBoardResponse:
    request_id = new_request_id()
    response = TumorBoardAssembler.assemble_dynamic(
        req, request_id=request_id
    )
    log_event(
        request_id, "/v1/tumor_board/dynamic",
        model_state=response.provenance.model_state.value,
        patient_id_hash=None,
        extra={
            "cancer": req.cancer,
            "pipeline_status": response.pipeline_status,
            "stage_statuses": [receipt.status for receipt in response.stage_receipts],
        },
        tenant_id=tenant.tenant_id,
    )
    return response

@router.post("/v1/case/full", response_model=FullCaseResponse)
def case_full(
    req: FullCaseRequest,
    cancer: str = Query(default="breast", description="Cancer track: breast, nsclc, or hgsoc."),
    tenant: APIKey = Depends(require_api_key),
) -> FullCaseResponse:
    cancer_norm = cancer.lower().strip()
    if cancer_norm not in SUPPORTED_CANCERS:
        raise HTTPException(400, f"Unsupported cancer={cancer!r}; supported={sorted(SUPPORTED_CANCERS)}")
    request_id = new_request_id()

    if cancer_norm in {"breast", "hgsoc"}:
        biopsy_input = req.biopsy_input
        dynamic_request = DynamicTumorBoardRequest(
            cancer=cancer_norm,
            case_id=req.case_id,
            report_text=biopsy_input.report_text if biopsy_input else None,
            pathology_image_b64=biopsy_input.wsi_bytes_b64 if biopsy_input else None,
            dss_features=(biopsy_input.dss_features if biopsy_input and cancer_norm == "breast" else None),
            therapy_features=req.therapy_features if cancer_norm == "breast" else None,
            mutations=req.mutations,
            germline_mutations=req.germline_mutations,
            ranked_drugs=req.ranked_drugs,
            clinical_baseline=req.therapy_context.model_dump(mode="json"),
            run_co_scientist=req.run_co_scientist,
        )
        dynamic = TumorBoardAssembler.assemble_dynamic(
            dynamic_request, request_id=request_id,
        )
        screening = screening_analyze(req.screening_input, tenant=tenant) if req.screening_input else None
        log_event(
            request_id, "/v1/case/full",
            model_state=dynamic.provenance.model_state.value,
            patient_id_hash=None,
            extra={"cancer": cancer_norm, "pipeline_status": dynamic.pipeline_status},
            tenant_id=tenant.tenant_id,
        )
        return FullCaseResponse(
            **_envelope(
                request_id,
                model_state=dynamic.provenance.model_state,
                model_name="specialist-stack-composite",
            ),
            warnings=list(dynamic.warnings),
            pipeline_status=dynamic.pipeline_status,
            cancer=cancer_norm,
            stage_receipts=dynamic.stage_receipts,
            ovarian_arbiter=dynamic.ovarian_arbiter,
            case_manifest=dynamic.case_manifest,
            report_parse=dynamic.report_parse,
            phikon_embedding=dynamic.phikon_embedding,
            dss_prognosis=dynamic.dss_prognosis,
            therapy_bridge=dynamic.therapy_bridge,
            therapy_triage=dynamic.therapy_triage,
            co_scientist=dynamic.co_scientist,
            screening=screening,
            biopsy=None,
            therapy=None,
            nsclc=None,
            elo_ranked_hypotheses=[],
        )

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
        report_parse, _panel, _grade, receipt = _clinicalbert_biopsy_parse(
            req.biopsy_input.report_text, request_id=request_id
        )
        receipts.append(receipt)
    else:
        receipts.append(_skipped_stage_receipt(
            "clinicalbert_pathology_parse", request_id, "no report_text supplied"
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

    co_scientist = TumorBoardAssembler.run_medgemma_co_scientist(
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

# ----------------------------------------------------------------------- #
# /v1/tumor_board/bundle — v0.4.0-alpha AK MBD4-LOF tumor board upload
#
# Accepts a TumorBoardBundle (contract
# 'tumor_board.v3.multimodal-with-manuscript-claims') from an authorised
# caller and echoes it back with a computed bundle sha256 + envelope. Two
# invariants:
#
#   1. The route is a NULL-OP with respect to disk state — no PHI
#      persistence, no dossier writes, no side effects other than
#      the audit ledger. Real dossier writes happen in
#      crispro-backend-v2, not here. This route lets clinicians *upload*
#      a bundle for downstream Modal enrichment or Elo re-ranking
#      without the SPA needing to shove multi-megabyte JSON through
#      /v1/case/full.
#
#   2. HIPAA_MODE=true fails closed until a real redactor is wired. Returning
#      the bundle verbatim while labelling it redacted is forbidden.

@router.post(
    "/v1/tumor_board/bundle",
    response_model=TumorBoardBundleResponse,
    response_model_exclude_unset=True,
)
def tumor_board_bundle(
    bundle: TumorBoardBundle,
    tenant: APIKey = Depends(require_api_key),
) -> TumorBoardBundleResponse:
    """Accept a tumor board bundle, compute its SHA-256, return envelope.

    The bundle is validated by pydantic against the
    `tumor_board.v3.multimodal-with-manuscript-claims` contract. Any
    deviation (missing field, wrong claim type, wrong manuscript sha)
    raises 422 before we get here.
    """
    import json as _json

    # Reject bundles that don't match the pinned contract version — the
    # SPA has no fallback renderer and mis-versioned bundles must not
    # silently downgrade to a stale UI.
    expected = "tumor_board.v3.multimodal-with-manuscript-claims"
    if bundle.contract_version != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"contract_version={bundle.contract_version!r} "
                f"does not match server-pinned {expected!r}. "
                "See docs/audit/AK_MBD4_INTEGRATION.md."
            ),
        )

    # Canonical bundle hash: pydantic dumps with sort_keys+ensure_ascii
    # so an identical bundle round-tripped through v1 matches the
    # audit ledger entry emitted by crispro-backend-v2 side-by-side.
    canonical = _json.dumps(
        bundle.model_dump(mode="json", exclude_unset=True),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    bundle_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    request_id = new_request_id()
    hipaa_mode = os.environ.get("HIPAA_MODE", "").lower() in {"1", "true", "yes"}

    if hipaa_mode:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "hipaa_redactor_unavailable",
                "detail": (
                    "HIPAA_MODE requires an implemented redaction transform; "
                    "the server refuses to echo an unredacted tumor-board bundle."
                ),
            },
        )
    state = ModelState.LOADED_AK_BUNDLE

    log_event(
        request_id,
        "/v1/tumor_board/bundle",
        model_state=state.value,
        patient_id_hash=hashlib.sha256(
            bundle.patient_id.encode("utf-8")
        ).hexdigest()[:12],
        extra={
            "contract_version": bundle.contract_version,
            "backend_head_sha": bundle.synthetic_lethality
                .provenance.backend_head_sha[:12],
            "manuscript_sha": bundle.synthetic_lethality
                .provenance.manuscript_repo_sha_at_audit[:12],
            "n_rows": len(
                bundle.synthetic_lethality.provenance.evidence_matrix.rows
            ),
            "n_drugs": len(bundle.synthetic_lethality.recommended_drugs),
            "hipaa_mode": hipaa_mode,
        },
        tenant_id=tenant.tenant_id,
    )

    return TumorBoardBundleResponse(
        **_envelope(
            request_id,
            model_state=state,
            model_name="tumor_board_v3_multimodal",
        ),
        bundle=bundle,
        bundle_sha256=bundle_sha,
        persisted_path=None,  # no disk persistence yet — see PR #6
    )
