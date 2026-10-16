"""Auto-extracted router: screening."""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any


from fastapi import APIRouter, Depends, HTTPException

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
from oncology_arbiter.api.schemas import (
    ModelState, ScreeningRequest, ScreeningResponse, ScreeningFusionRequest,
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

@router.post("/v1/screening/analyze", response_model=ScreeningResponse)
def screening_analyze(
    req: ScreeningRequest,
    tenant: APIKey = Depends(require_api_key),
) -> ScreeningResponse:
    """Run strict production MedSigLIP without proxy or heuristic fallback."""
    request_id = new_request_id()
    if not req.dicom_bytes_b64 or req.dicom_url:
        raise HTTPException(
            400,
            "production screening requires exactly dicom_bytes_b64; remote URL ingestion is not accepted",
        )
    raw_bytes = _decode_bytes_arg(req.dicom_bytes_b64)
    assert raw_bytes is not None
    input_sha = hashlib.sha256(raw_bytes).hexdigest()

    from oncology_arbiter.mammography import preprocess_mammogram
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
        temp_file.write(raw_bytes)
        temp_path = temp_file.name
    try:
        preprocessed = preprocess_mammogram(
            temp_path,
            laterality_hint=req.laterality_hint,
            view_hint=req.view_hint,
        )
    except Exception as exc:
        Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(422, f"preprocessing failed: {exc}") from exc

    started = time.perf_counter()
    receipts: list[dict[str, Any]] = []
    medsiglip_block: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = []
    overall_score: float | None = None
    warnings: list[str] = []
    model_state = ModelState.UNAVAILABLE
    model_name: str | None = "google/medsiglip-448"
    gate_report: Any = None
    try:
        from oncology_arbiter.models.cbis_ddsm_probe import (
            CBIS_DDSM_PROBE_WARNING,
            DEFAULT_MODEL_PATH as CBIS_JOBLIB_PATH,
        )
        from oncology_arbiter.models.hai_def import (
            GatedAccessError,
            GateReport as RuntimeGateReport,
        )
        # Single choke point for MedSigLIP execution (unit tests patch this).
        result = _run_medsiglip_on_preprocessed(preprocessed, dicom_bytes=raw_bytes)
        if result.embedding_dim != 1152 or not result.embedding_sha256:
            raise RuntimeError("medsiglip_contract_mismatch:embedding_receipt")
        gate_report = result.gate_report
        latency_ms = (time.perf_counter() - started) * 1000.0
        findings = [
            {
                "label": label,
                "score": float(score),
                "location_bbox_normalized": None,
            }
            for label, score in zip(result.labels, result.probs)
        ]
        overall_score = float(result.probs[0])
        warnings = list(result.warnings) + [
            "Zero-shot probabilities are uncalibrated, off-label, and are not diagnostic or treatment evidence."
        ]
        medsiglip_block = {
            "model_name": result.model_repo,
            "app_version": result.app_version,
            "input_resolution": result.input_resolution,
            "embedding_dim": result.embedding_dim,
            "embedding_sha256": result.embedding_sha256,
            "prompts": list(result.prompts) if result.prompts else list(result.labels),
            "inference_seconds": result.inference_seconds,
            "input_sha256": input_sha,
            "score_semantics": "independent_uncalibrated_sigmoid_zero_shot",
            "probs": [float(p) for p in result.probs],
            "probs_sum": float(sum(float(p) for p in result.probs)),
            "probs_are_normalised_distribution": False,
        }
        receipts.append({
            "stage": "medsiglip_screening",
            "required": True,
            "status": "succeeded",
            "request_id": request_id,
            "service_name": "crispro--medsiglip",
            "endpoint_label": "medsiglip-zero-shot+medsiglip-embed",
            "app_version": result.app_version,
            "model_name": result.model_repo,
            "input_reference": input_sha,
            "latency_ms": latency_ms,
            "warnings": list(warnings),
            "error": None,
        })
        model_state = ModelState.LOADED_MEDSIGLIP

        # CBIS-DDSM LogReg v2 (joblib sha256 e5f5b67c…) — trained probe over
        # the same 1152-d MedSigLIP embedding. Opt-out only: set
        # ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE=0 to skip.
        cbis_disabled = os.environ.get(
            "ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE", "1"
        ).strip().lower() in {"0", "false", "no", "off"}
        if not cbis_disabled and Path(CBIS_JOBLIB_PATH).exists():
            try:
                probe_t0 = time.perf_counter()
                probe_result = _run_cbis_ddsm_probe_on_bytes(
                    raw_bytes,
                    dicom_path=temp_path,
                )
                if probe_result is not None:
                    findings.append({
                        "label": f"cbis_ddsm_logreg_v2:{probe_result.predicted_label}",
                        "score": float(probe_result.proba_cancer),
                        "location_bbox_normalized": None,
                    })
                    # Trained probe supersedes off-label zero-shot as overall.
                    overall_score = float(probe_result.proba_cancer)
                    if CBIS_DDSM_PROBE_WARNING not in warnings:
                        warnings.append(CBIS_DDSM_PROBE_WARNING)
                    model_name = f"{result.model_repo}+{probe_result.probe_version}"
                    if medsiglip_block is not None:
                        medsiglip_block["cbis_probe"] = {
                            "probe_version": probe_result.probe_version,
                            "proba_cancer": float(probe_result.proba_cancer),
                            "threshold": float(probe_result.threshold),
                            "threshold_label": probe_result.threshold_label,
                            "predicted_label": probe_result.predicted_label,
                            "joblib_path": str(CBIS_JOBLIB_PATH),
                            "joblib_sha256_prefix": "e5f5b67c",
                        }
                    receipts.append({
                        "stage": "cbis_ddsm_logreg_v2",
                        "required": False,
                        "status": "succeeded",
                        "request_id": request_id,
                        "service_name": "oncology-arbiter",
                        "model_name": probe_result.probe_version,
                        "input_reference": result.embedding_sha256,
                        "latency_ms": (time.perf_counter() - probe_t0) * 1000.0,
                        "warnings": [CBIS_DDSM_PROBE_WARNING],
                        "error": None,
                    })
            except Exception as probe_exc:  # noqa: BLE001 — never hide
                # Optional probe: surface the error but do not downgrade a
                # successful required MedSigLIP stage to partial_failure
                # when Modal embed/probe infra is unavailable in unit tests.
                warnings.append(
                    f"cbis_ddsm_probe_error:{type(probe_exc).__name__}:{probe_exc}"
                )
                receipts.append(_skipped_stage_receipt(
                    "cbis_ddsm_logreg_v2",
                    request_id,
                    f"cbis_ddsm_probe_unavailable:{type(probe_exc).__name__}:{probe_exc}",
                ))
        elif not cbis_disabled:
            warnings.append(
                f"cbis_ddsm_probe_skipped:joblib_missing:{CBIS_JOBLIB_PATH}"
            )
            receipts.append(_skipped_stage_receipt(
                "cbis_ddsm_logreg_v2",
                request_id,
                f"trained probe missing at {CBIS_JOBLIB_PATH}",
            ))
    except GatedAccessError as gate_err:
        latency_ms = (time.perf_counter() - started) * 1000.0
        # Preserve structured gate_report in the response envelope.
        has_token = gate_err.status_code != 401 and bool(
            (os.environ.get("HF_TOKEN") or "").strip()
            or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
        )
        gate_report = RuntimeGateReport(
            repo_id=gate_err.repo_id,
            access_level=gate_err.access_level,
            status_code=gate_err.status_code,
            reason=gate_err.reason,
            has_token=has_token,
        )
        warnings = [f"medsiglip_gated:{gate_err.access_level.value}:{gate_err.reason}"]
        receipts.append(_failed_stage_receipt(
            "medsiglip_screening",
            True,
            request_id,
            "hai_def_gated",
            f"{gate_err.access_level.value}:{gate_err.reason}",
            service_name="crispro--medsiglip",
            input_reference=input_sha,
        ))
        receipts[-1]["latency_ms"] = latency_ms
        model_state = ModelState.GATED
        model_name = gate_err.repo_id
    except Exception as exc:  # required failure remains a failure
        latency_ms = (time.perf_counter() - started) * 1000.0
        warnings = [f"medsiglip_required_stage_failed:{type(exc).__name__}"]
        receipts.append(_failed_stage_receipt(
            "medsiglip_screening",
            True,
            request_id,
            "medsiglip_inference_failed",
            str(exc),
            service_name="crispro--medsiglip",
            input_reference=input_sha,
        ))
        receipts[-1]["latency_ms"] = latency_ms
    finally:
        Path(temp_path).unlink(missing_ok=True)

    schema_gate_report = _to_schema_gate_report(gate_report)
    envelope = _envelope(
        request_id,
        model_state=model_state,
        model_name=model_name,
        gate_report=schema_gate_report,
    )
    response = ScreeningResponse(
        **envelope,
        pipeline_status=_pipeline_status(receipts),
        stage_receipts=receipts,
        laterality=preprocessed.metadata.laterality.value,
        view=preprocessed.metadata.view.value,
        orientation_flipped=preprocessed.metadata.orientation_flipped,
        breast_mask_coverage=float(preprocessed.breast_mask.mean()),
        findings=findings,
        overall_score=overall_score,
        medsiglip=medsiglip_block,
        # Empty features={} collapse to intercept-only constants — refused.
        arbiter_score=None,
        warnings=warnings,
    )
    log_event(
        request_id,
        "/v1/screening/analyze",
        model_state=model_state.value,
        patient_id_hash=req.patient_id_hash,
        extra={
            "pipeline_status": response.pipeline_status,
            "input_sha256": input_sha,
            "embedding_dim": medsiglip_block.get("embedding_dim") if medsiglip_block else None,
            "app_version": medsiglip_block.get("app_version") if medsiglip_block else None,
        },
        tenant_id=tenant.tenant_id,
    )
    return response

# ----------------------------------------------------------------------- #
# /v1/screening/fusion — RUO fused head over MedSigLIP embedding + metadata

@router.post("/v1/screening/fusion", response_model=ScreeningResponse)
def screening_fusion(
    req: ScreeningFusionRequest,
    tenant: APIKey = Depends(require_api_key),
) -> ScreeningResponse:
    """Run RUO RSNA fused logistic head (MedSigLIP embedding + clinical metadata).

    This route is separate from /v1/screening/analyze so we do not alter the
    production zero-shot contract or its downstream optional CBIS probe.
    """
    request_id = new_request_id()
    raw_bytes = _decode_bytes_arg(req.dicom_bytes_b64)
    assert raw_bytes is not None
    input_sha = hashlib.sha256(raw_bytes).hexdigest()

    from oncology_arbiter.mammography import preprocess_mammogram

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
        temp_file.write(raw_bytes)
        temp_path = temp_file.name
    try:
        preprocessed = preprocess_mammogram(
            temp_path,
            laterality_hint=req.laterality_hint,
            view_hint=req.view_hint,
        )
    except Exception as exc:
        Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(422, f"preprocessing failed: {exc}") from exc

    started = time.perf_counter()
    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    model_state = ModelState.UNAVAILABLE
    fusion_score: float | None = None
    model_name: str | None = None
    try:
        from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient
        from oncology_arbiter.models.rsna_fusion_modal_client import RsnaFusionModalClient

        ms_client = MedSigLipModalClient()
        emb = ms_client.embed_dicom(temp_path)
        if len(emb) != 1152:
            raise RuntimeError(f"medsiglip_contract_mismatch:expected_1152_got_{len(emb)}")

        meta = req.clinical_metadata
        # Default laterality/view from preprocessing when not supplied explicitly.
        laterality = meta.laterality or preprocessed.metadata.laterality.value
        view = meta.view or preprocessed.metadata.view.value
        fusion_client = RsnaFusionModalClient()
        fusion_score = fusion_client.score(
            embedding=emb,
            clinical_metadata={
                "age": meta.age,
                "density": meta.density,
                "laterality": laterality if laterality in {"L", "R"} else None,
                "view": view if view in {"CC", "MLO"} else None,
                "implant": meta.implant,
                "site_id": meta.site_id,
                "machine_id": meta.machine_id,
            },
        )
        model_state = ModelState.LOADED_MEDSIGLIP
        model_name = "google/medsiglip-448+rsna_medsiglip_fusion_logreg_v1"
        warnings = [
            "RUO: fusion_score is produced by a remote Modal scoring service (rsna-fusion-logreg) "
            "using a frozen RSNA retrospective logistic head over MedSigLIP embeddings + limited metadata."
        ]

        receipts.append(
            {
                "stage": "rsna_medsiglip_fusion_logreg_v1",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": "rsna_medsiglip_fusion_logreg_v1",
                "input_reference": input_sha,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "warnings": list(warnings),
                "error": None,
            }
        )
    except Exception as exc:  # required failure remains a failure
        warnings = [f"fusion_required_stage_failed:{type(exc).__name__}"]
        receipts.append(
            _failed_stage_receipt(
                "rsna_medsiglip_fusion_logreg_v1",
                True,
                request_id,
                "fusion_inference_failed",
                str(exc),
                service_name="oncology-arbiter",
                input_reference=input_sha,
            )
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)

    envelope = _envelope(
        request_id,
        model_state=model_state,
        model_name=model_name,
        gate_report=None,
    )
    response = ScreeningResponse(
        **envelope,
        pipeline_status=_pipeline_status(receipts),
        stage_receipts=receipts,
        laterality=preprocessed.metadata.laterality.value,
        view=preprocessed.metadata.view.value,
        orientation_flipped=preprocessed.metadata.orientation_flipped,
        breast_mask_coverage=float(preprocessed.breast_mask.mean()),
        findings=[],
        overall_score=None,
        medsiglip=None,
        arbiter_score=None,
        fusion_score=fusion_score,
        fusion_model_name="rsna_medsiglip_fusion_logreg_v1" if fusion_score is not None else None,
        fusion_warning=(warnings[0] if warnings else None),
        warnings=warnings,
    )
    log_event(
        request_id,
        "/v1/screening/fusion",
        model_state=model_state.value,
        patient_id_hash=req.patient_id_hash,
        extra={
            "pipeline_status": response.pipeline_status,
            "input_sha256": input_sha,
            "fusion_score": fusion_score,
        },
    )
    return response

# ----------------------------------------------------------------------- #
# /v1/biopsy/analyze — placeholder
