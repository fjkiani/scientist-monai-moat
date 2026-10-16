"""Auto-extracted router: biopsy."""
from __future__ import annotations

import hashlib
import os
import time
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
    ModelState, BiopsyRequest, BiopsyResponse, BiopsyReceptorPanel,
    BreastDssPrognosis,
    ReportParseBlock,
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
_get_medsiglip = runtime_hooks.get_medsiglip
_run_medsiglip_on_preprocessed = runtime_hooks.run_medsiglip_on_preprocessed
_run_cbis_ddsm_probe_on_bytes = runtime_hooks.run_cbis_ddsm_probe_on_bytes
_elo_reason_for_modifier = runtime_hooks.elo_reason_for_modifier
_run_clinicalbert_parse = runtime_hooks.run_clinicalbert_parse
_score_explicit_therapy_triage = runtime_hooks.score_explicit_therapy_triage


router = APIRouter()

@router.post("/v1/biopsy/analyze", response_model=BiopsyResponse)
def biopsy_analyze(
    req: BiopsyRequest,
    tenant: APIKey = Depends(require_api_key),
) -> BiopsyResponse:
    """Run required ClinicalBERT and Phikon stages without proxy substitution."""
    request_id = new_request_id()
    if not any((req.wsi_url, req.wsi_bytes_b64, req.report_text, req.dss_features is not None)):
        raise HTTPException(400, "must provide pathology image, report_text, or dss_features")

    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    report_parse_block: ReportParseBlock | None = None
    receptor_panel = BiopsyReceptorPanel()
    parsed_grade: int | None = None
    phikon_embedding: dict[str, Any] | None = None
    model_state = ModelState.PLACEHOLDER
    model_name: str | None = None

    if req.report_text:
        report_parse_block, receptor_panel, parsed_grade, receipt = _clinicalbert_biopsy_parse(
            req.report_text, request_id=request_id
        )
        receipts.append(receipt)
        if receipt["status"] == "succeeded":
            # PathologyParseOrchestrator is MedGemma-primary; only the legacy
            # ClinicalBERT NER path is reachable under ONCOLOGY_FORCE_CLINICALBERT_NER=1.
            stg = str(receipt.get("stage") or "")
            if stg == "medgemma_pathology_parse":
                model_state = ModelState.LOADED_MEDGEMMA_1P5_4B
                model_name = receipt.get("model_name") or "google/medgemma-1.5-4b-it"
            elif stg == "clinicalbert_pathology_parse":
                model_state = ModelState.LOADED_CLINICALBERT_PARSER
                model_name = "emilyalsentzer/Bio_ClinicalBERT+v0.5.2-sliding-window"
            else:
                model_state = ModelState.LOADED
                model_name = receipt.get("model_name") or model_name
        else:
            warnings.append(f"pathology_parse_required_stage_failed:{receipt['error']['code']}")
    else:
        receipts.append(_skipped_stage_receipt(
            "clinicalbert_pathology_parse", request_id, "no report_text supplied"
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
            probe = build_biopsy_probe(embedding_client=_get_medsiglip())
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

    if req.wsi_bytes_b64:
        try:
            from oncology_arbiter.models.specialist_clients import PhikonClient
            from oncology_arbiter.models.phikon_probe_wiring import (
                predict_tissue_class as phikon_predict_tissue_class,
            )
            image_bytes = _decode_bytes_arg(req.wsi_bytes_b64)
            call = PhikonClient().embed(image_bytes or b"", request_id=request_id, required=True)
            phikon_embedding = call.output
            receipts.append(call.receipt)
            # Product tissue head: 768-d Phikon embedding + NCT-CRC probe v1 joblib
            try:
                emb_vec = None
                if isinstance(phikon_embedding, dict):
                    emb_vec = phikon_embedding.get("embedding") or (
                        (phikon_embedding.get("embeddings") or [None])[0]
                    )
                if emb_vec is not None:
                    probe_out = phikon_predict_tissue_class(emb_vec)
                    if isinstance(phikon_embedding, dict):
                        phikon_embedding = {**phikon_embedding, "probe": probe_out}
                    phikon_product_ok = True
                    receipts.append({
                        "stage": "phikon_probe_v1",
                        "required": False,
                        "status": "succeeded",
                        "request_id": request_id,
                        "service_name": "oncology-arbiter",
                        "pred_label": probe_out.get("pred_label"),
                        "artifact_sha256": probe_out.get("artifact_sha256"),
                        "error": None,
                    })
            except Exception as probe_exc:
                receipts.append(_failed_stage_receipt(
                    "phikon_probe_v1", False, request_id,
                    "phikon_probe_failed", f"{type(probe_exc).__name__}: {probe_exc}",
                    service_name="oncology-arbiter",
                ))
            if phikon_product_ok:
                model_state = ModelState.LOADED_PHIKON
                model_name = product_tissue_model_name()
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
            "Remote pathology-image fetching is disabled; submit de-identified image bytes.",
            service_name="phikon",
        ))
        warnings.append("phikon_required_stage_failed:direct_url_disabled")
    else:
        receipts.append(_skipped_stage_receipt(
            "phikon_embedding", request_id, "no pathology image supplied"
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

# MOVED TO routers/stage.py
