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
_medgemma_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


def _app():
    """Late binding so tests can patch helpers on oncology_arbiter.api.app."""
    import oncology_arbiter.api.app as app_module
    return app_module


# ----------------------------------------------------------------------- #
# /v1/biopsy/analyze — placeholder

@router.post("/v1/biopsy/analyze", response_model=BiopsyResponse)
def biopsy_analyze(
    req: BiopsyRequest,
    tenant: APIKey = Depends(require_api_key),
) -> BiopsyResponse:
    """Run MedGemma pathology extraction and Phikon probe. ClinicalBERT is not called."""
    request_id = new_request_id()
    if not any(
        (
            req.wsi_url,
            req.wsi_bytes_b64,
            req.report_text,
            req.dss_features is not None,
            req.phikon_embedding is not None,
        )
    ):
        raise HTTPException(
            400,
            "must provide pathology image, phikon_embedding, report_text, or dss_features",
        )

    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    report_parse_block: ReportParseBlock | None = None
    receptor_panel = BiopsyReceptorPanel()
    parsed_grade: int | None = None
    phikon_embedding: dict[str, Any] | None = None
    model_state = ModelState.PLACEHOLDER
    model_name: str | None = None

    if req.report_text:
        report_parse_block, receptor_panel, parsed_grade, receipt = _medgemma_biopsy_parse(
            req.report_text, request_id=request_id
        )
        receipts.append(receipt)
        if receipt["status"] == "succeeded":
            model_state = ModelState.LOADED_MEDGEMMA_PATHOLOGY
            model_name = "google/medgemma-1.5-4b-it"
        else:
            warnings.append(f"medgemma_pathology_required_stage_failed:{receipt['error']['code']}")
    else:
        receipts.append(_skipped_stage_receipt(
            "medgemma_pathology_parse", request_id, "no report_text supplied"
        ))

    subtype_prediction: str | None = None
    confidence: float | None = None
    phikon_product_ok = False

    from oncology_arbiter.models.tissue_microscopy_product import (
        PRODUCT_TISSUE_HONESTY_NOTE,
        product_tissue_model_name,
    )

    if req.wsi_bytes_b64 or req.wsi_url:
        warnings.append(PRODUCT_TISSUE_HONESTY_NOTE)

    # Secondary research: MedSigLIP + biopsy_probe_v1 (BACH). Primary tissue
    # product provenance is Phikon + phikon_probe_wiring (see below).
    biopsy_probe_disabled = os.environ.get(
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1"
    ).strip().lower() in {"0", "false", "no", "off"}
    if not biopsy_probe_disabled and (req.wsi_bytes_b64 or req.wsi_url):
        try:
            from oncology_arbiter.models.biopsy_probe_v1_wiring import build_biopsy_probe
            from oncology_arbiter.models.hai_def import GatedAccessError

            probe_t0 = time.perf_counter()
            probe = build_biopsy_probe(embedding_client=_app()._get_medsiglip())
            image_bytes = _decode_bytes_arg(req.wsi_bytes_b64)
            image_url = str(req.wsi_url) if req.wsi_url else None
            probe_result = probe.run(
                image_bytes=image_bytes,
                image_url=image_url,
            )
            subtype_prediction = probe_result.subtype
            confidence = float(probe_result.subtype_probs[probe_result.subtype])
            if not phikon_product_ok and model_state in (
                ModelState.PLACEHOLDER,
                ModelState.LOADED_CLINICALBERT_PARSER,
            ):
                model_state = ModelState.LOADED_BIOPSY_PROBE
                model_name = probe_result.model_name
            warnings.extend(probe_result.warnings)
            receipts.append({
                "stage": "biopsy_medsiglip_probe_v1_research",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": probe_result.model_name,
                "input_reference": (
                    hashlib.sha256(image_bytes).hexdigest()
                    if image_bytes is not None else image_url
                ),
                "latency_ms": (time.perf_counter() - probe_t0) * 1000.0,
                "warnings": list(probe_result.warnings),
                "error": None,
            })
        except GatedAccessError as gate_err:
            if not phikon_product_ok:
                model_state = ModelState.GATED
                model_name = gate_err.repo_id
            warnings.append(
                f"biopsy_medsiglip_gated:{gate_err.access_level.value}:{gate_err.reason}"
            )
            receipts.append(_failed_stage_receipt(
                "biopsy_medsiglip_probe_v1_research",
                False,
                request_id,
                "hai_def_gated",
                f"{gate_err.access_level.value}:{gate_err.reason}",
                service_name="medsiglip",
            ))
        except Exception as probe_exc:  # noqa: BLE001
            warnings.append(
                f"biopsy_medsiglip_error:{type(probe_exc).__name__}:{probe_exc}"
            )
            receipts.append(_failed_stage_receipt(
                "biopsy_medsiglip_probe_v1_research",
                False,
                request_id,
                "biopsy_probe_failed",
                f"{type(probe_exc).__name__}: {probe_exc}",
                service_name="oncology-arbiter",
            ))
    elif not biopsy_probe_disabled:
        receipts.append(_skipped_stage_receipt(
            "biopsy_medsiglip_probe_v1_research",
            request_id,
            "no pathology image supplied for biopsy probe",
        ))

    from oncology_arbiter.models.phikon_probe_wiring import (
        predict_tissue_class as phikon_predict_tissue_class,
    )

    def _apply_phikon_probe(emb_vec: list[float] | None) -> None:
        nonlocal phikon_embedding, phikon_product_ok, model_state, model_name, subtype_prediction, confidence
        if emb_vec is None:
            return
        probe_out = phikon_predict_tissue_class(emb_vec)
        base = phikon_embedding if isinstance(phikon_embedding, dict) else {}
        phikon_embedding = {
            **base,
            "embedding_dim": 768,
            "probe": probe_out,
        }
        phikon_product_ok = True
        subtype_prediction = str(probe_out.get("pred_label") or subtype_prediction)
        proba = probe_out.get("proba") or {}
        if subtype_prediction and isinstance(proba, dict) and subtype_prediction in proba:
            confidence = float(proba[subtype_prediction])
        receipts.append({
            "stage": "phikon_probe_v1",
            "required": True,
            "status": "succeeded",
            "request_id": request_id,
            "service_name": "oncology-arbiter",
            "model_name": "phikon_probe_v1",
            "pred_label": probe_out.get("pred_label"),
            "artifact_sha256": probe_out.get("artifact_sha256"),
            "error": None,
        })
        model_state = ModelState.LOADED_PHIKON
        model_name = product_tissue_model_name()

    if req.phikon_embedding is not None:
        try:
            emb_vec = [float(x) for x in req.phikon_embedding]
            if len(emb_vec) != 768:
                raise ValueError(f"phikon_embedding must be 768-d; got {len(emb_vec)}")
            phikon_embedding = {
                "embedding": emb_vec,
                "embedding_dim": 768,
                "source": "client_supplied_embedding",
            }
            receipts.append({
                "stage": "phikon_embedding",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "client",
                "model_name": "phikon_embedding_client_supplied",
                "input_reference": f"dim:{len(emb_vec)}",
                "latency_ms": 0.0,
                "warnings": ["client_supplied_768d_embedding_no_image_encode"],
                "error": None,
            })
            _apply_phikon_probe(emb_vec)
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "phikon_probe_v1", True, request_id,
                "phikon_probe_failed", f"{type(exc).__name__}: {exc}",
                service_name="oncology-arbiter",
            ))
            warnings.append("phikon_required_stage_failed")
    elif req.wsi_bytes_b64:
        try:
            from oncology_arbiter.models.specialist_clients import PhikonClient
            image_bytes = _decode_bytes_arg(req.wsi_bytes_b64)
            call = PhikonClient().embed(image_bytes or b"", request_id=request_id, required=True)
            phikon_embedding = call.output
            receipts.append(call.receipt)
            emb_vec = None
            if isinstance(phikon_embedding, dict):
                emb_vec = phikon_embedding.get("embedding") or (
                    (phikon_embedding.get("embeddings") or [None])[0]
                )
            try:
                _apply_phikon_probe(emb_vec if isinstance(emb_vec, list) else None)
            except Exception as probe_exc:
                receipts.append(_failed_stage_receipt(
                    "phikon_probe_v1", True, request_id,
                    "phikon_probe_failed", f"{type(probe_exc).__name__}: {probe_exc}",
                    service_name="oncology-arbiter",
                ))
                warnings.append("phikon_required_stage_failed")
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "phikon_embedding", True, request_id,
                getattr(exc, "code", "phikon_failed"), f"{type(exc).__name__}: {exc}",
                service_name="phikon",
            ))
            warnings.append("phikon_required_stage_failed")
    elif req.wsi_url:
        receipts.append(_failed_stage_receipt(
            "phikon_embedding", True, request_id, "direct_url_disabled",
            "Remote pathology-image fetching is disabled; submit de-identified image bytes or phikon_embedding.",
            service_name="phikon",
        ))
        warnings.append("phikon_required_stage_failed:direct_url_disabled")
    else:
        receipts.append(_skipped_stage_receipt(
            "phikon_embedding", request_id, "no pathology image or phikon_embedding supplied"
        ))

    dss_prognosis: BreastDssPrognosis | None = None
    if req.dss_features is not None:
        from oncology_arbiter.models.breast_dss_arbiter import score_breast_dss
        f = req.dss_features
        dss_result = score_breast_dss({
            "age": f.age, "tumor_size_mm": f.tumor_size_mm,
            "nodes_positive": f.nodes_positive, "grade": f.grade,
            "er_pos": f.er_positive, "pr_pos": f.pr_positive,
            "her2_pos": f.her2_positive,
        })
        dss_prognosis = BreastDssPrognosis(
            model_name=dss_result.model_name,
            disease_specific_mortality_score=dss_result.score,
            logit=dss_result.logit,
            term_contributions=dss_result.term_contributions,
            artifact_sha256=dss_result.artifact_sha256,
            n_training=dss_result.n_training,
            events=dss_result.events,
            oof_auroc=dss_result.oof_auroc,
            caveats=list(dss_result.caveats),
        )
        receipts.append({
            "stage": "breast_dss_prognosis", "required": False, "status": "succeeded",
            "request_id": request_id, "service_name": "oncology-arbiter",
            "model_name": dss_result.model_name, "model_version": "v3",
            "artifact_sha256": dss_result.artifact_sha256,
            "input_reference": "explicit_complete_seven_feature_vector",
            "latency_ms": 0.0,
            "warnings": ["prognosis_under_metabric_treatment_mix_not_treatment_benefit"],
            "error": None,
        })
        warnings.append("breast_dss_prognosis:frozen_metabric:explicit_features:no_imputation")
    else:
        receipts.append(_skipped_stage_receipt(
            "breast_dss_prognosis", request_id, "complete explicit seven-feature vector not supplied"
        ))

    pipeline_status = _pipeline_status(receipts)
    # PLACEHOLDER means "no model wired yet, stub response". It is the wrong
    # label for a route that required a specialist, called it, and got a
    # failure: the caller reading `placeholder` would conclude nothing had
    # been attempted. /v1/therapy/reason already reports UNAVAILABLE in the
    # identical situation; make the two routes agree. Only a request where
    # every stage was *skipped* (nothing was asked for) stays PLACEHOLDER.
    if model_state == ModelState.PLACEHOLDER and any(
        r["status"] == "failed_required" for r in receipts
    ):
        model_state = ModelState.UNAVAILABLE
    log_event(
        request_id, "/v1/biopsy/analyze", model_state=model_state.value,
        patient_id_hash=req.patient_id_hash,
        extra={"pipeline_status": pipeline_status, "stage_statuses": [r["status"] for r in receipts]},
        tenant_id=tenant.tenant_id,
    )
    env = _envelope(request_id, model_state=model_state, model_name=model_name)
    env["warnings"] = warnings
    return BiopsyResponse(
        **env,
        pipeline_status=pipeline_status,
        stage_receipts=receipts,
        phikon_embedding=phikon_embedding,
        subtype_prediction=subtype_prediction,
        receptor_panel=receptor_panel,
        grade=parsed_grade,
        confidence=confidence,
        # Empty features={} collapse to intercept-only constants — refused.
        arbiter_score=None,
        report_parse=report_parse_block,
        dss_prognosis=dss_prognosis,
    )

# ----------------------------------------------------------------------- #
# /v1/stage/biopsy/score — explicit biopsy_arbiter_v1 (stage-biopsy)
