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

from .biopsy import biopsy_analyze
from .nlp import _run_medgemma_co_scientist
from .screening import screening_analyze
from .therapy import therapy_reason

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
# Production dynamic tumor board and full-case specialist assembly

_SUPPORTED_CANCERS = {"breast", "nsclc", "hgsoc"}

def _breast_dss_from_features(features: Any) -> BreastDssPrognosis:
    from oncology_arbiter.models.breast_dss_arbiter import score_breast_dss

    result = score_breast_dss({
        "age": features.age,
        "tumor_size_mm": features.tumor_size_mm,
        "nodes_positive": features.nodes_positive,
        "grade": features.grade,
        "er_pos": features.er_positive,
        "pr_pos": features.pr_positive,
        "her2_pos": features.her2_positive,
    })
    return BreastDssPrognosis(
        model_name=result.model_name,
        disease_specific_mortality_score=result.score,
        logit=result.logit,
        term_contributions=result.term_contributions,
        artifact_sha256=result.artifact_sha256,
        n_training=result.n_training,
        events=result.events,
        oof_auroc=result.oof_auroc,
        caveats=list(result.caveats),
    )

def _ovarian_retirement(request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    block = {
        "model_name": "ovarian_mortality_arbiter_v3_tcga",
        "model_state": "retired",
        "endpoint": "2-year overall mortality",
        "retirement_reason": (
            "FIGO stage did not demonstrate incremental discrimination over age; "
            "the paired OOF AUROC increment was +0.02475247524752467 with an "
            "approximate 95% interval spanning zero."
        ),
        "retirement_evidence": {
            "cohort": "TCGA-OV",
            "n": 271,
            "age_only_oof_auroc": 0.658057110058832,
            "age_plus_figo_oof_auroc": 0.6828095853063567,
            "delta_oof_auroc": 0.02475247524752467,
            "delta_ci95_approx": [-0.041, 0.095],
            "external_validation": "no_feature_compatible_external_cohort_available",
        },
        "warning": "Retired research model; no patient probability, risk bucket, or driving feature is emitted.",
    }
    receipt = {
        "stage": "ovarian_prognostic_arbiter",
        "required": False,
        "status": "retired",
        "request_id": request_id,
        "service_name": "oncology-arbiter",
        "model_name": "ovarian_mortality_arbiter_v3_tcga",
        "model_version": "v3-retired",
        "warnings": ["retired_no_incremental_figo_discrimination_over_age"],
        "error": None,
    }
    return block, receipt


def _assemble_dynamic_tumor_board(
    req: DynamicTumorBoardRequest,
    *,
    request_id: str,
) -> DynamicTumorBoardResponse:
    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    ovarian_arbiter: dict[str, Any] | None = None
    report_parse: ReportParseBlock | None = None
    phikon_embedding: dict[str, Any] | None = None
    case_manifest: dict[str, Any] | None = None
    dss_prognosis: BreastDssPrognosis | None = None
    therapy_bridge: dict[str, Any] | None = None
    therapy_triage: ArbiterScore | None = None

    if req.cancer == "hgsoc":
        ovarian_arbiter, retired_receipt = _ovarian_retirement(request_id)
        receipts.append(retired_receipt)
    else:
        receipts.append(_skipped_stage_receipt(
            "ovarian_prognostic_arbiter", request_id, "not applicable to breast cancer"
        ))

    if req.case_id:
        try:
            from oncology_arbiter.models.specialist_clients import CaseStorageClient

            call = CaseStorageClient().manifest(req.case_id, request_id=request_id, required=True)
            case_manifest = call.output
            receipts.append(call.receipt)
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "case_storage", True, request_id,
                getattr(exc, "code", "case_storage_failed"), f"{type(exc).__name__}: {exc}",
                service_name="case-storage", input_reference=f"case_id:{req.case_id}",
            ))
    else:
        receipts.append(_skipped_stage_receipt(
            "case_storage", request_id, "no case_id supplied"
        ))

    if req.report_text:
        report_parse, _panel, _grade, receipt = _medgemma_biopsy_parse(
            req.report_text, request_id=request_id
        )
        receipts.append(receipt)
    else:
        receipts.append(_skipped_stage_receipt(
            "medgemma_pathology_parse", request_id, "no report_text supplied"
        ))

    if req.pathology_image_b64:
        try:
            from oncology_arbiter.models.specialist_clients import PhikonClient

            image_bytes = _decode_bytes_arg(req.pathology_image_b64) or b""
            call = PhikonClient().embed(image_bytes, request_id=request_id, required=True)
            phikon_embedding = call.output
            receipts.append(call.receipt)
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "phikon_embedding", True, request_id,
                getattr(exc, "code", "phikon_failed"), f"{type(exc).__name__}: {exc}",
                service_name="phikon",
            ))
    else:
        receipts.append(_skipped_stage_receipt(
            "phikon_embedding", request_id, "no pathology image supplied"
        ))

    if req.cancer == "breast" and req.dss_features is not None:
        dss_prognosis = _breast_dss_from_features(req.dss_features)
        receipts.append({
            "stage": "breast_dss_prognosis",
            "required": False,
            "status": "succeeded",
            "request_id": request_id,
            "service_name": "oncology-arbiter",
            "model_name": dss_prognosis.model_name,
            "model_version": "v3",
            "artifact_sha256": dss_prognosis.artifact_sha256,
            "input_reference": "explicit_complete_seven_feature_vector",
            "latency_ms": 0.0,
            "warnings": ["prognosis_under_metabric_treatment_mix_not_treatment_benefit"],
            "error": None,
        })
    else:
        reason = (
            "not applicable to HGSOC"
            if req.cancer == "hgsoc"
            else "complete explicit seven-feature vector not supplied"
        )
        receipts.append(_skipped_stage_receipt(
            "breast_dss_prognosis", request_id, reason
        ))

    mutation_payload = [item.model_dump(exclude_none=True) for item in req.mutations]
    germline_payload = [item.model_dump(exclude_none=True) for item in req.germline_mutations]
    ranked_payload = [item.model_dump(exclude_none=True) for item in req.ranked_drugs]
    if mutation_payload or germline_payload:
        try:
            from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

            call = SLTherapyBridgeClient().run({
                "disease": "high-grade serous ovarian cancer" if req.cancer == "hgsoc" else "breast cancer",
                "cancer_type": req.cancer,
                "mutations": mutation_payload,
                "germline_mutations": germline_payload,
                "ranked_drugs": ranked_payload,
                "rs_features": req.clinical_baseline,
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

    if req.cancer == "breast" and req.therapy_features is not None:
        therapy_triage = _app()._score_explicit_therapy_triage(
            req.therapy_features.model_dump(),
        )
        if therapy_triage is not None:
            receipts.append({
                "stage": "therapy_research_triage",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": therapy_triage.model_name,
                "model_version": "template-v0",
                "input_reference": "explicit_complete_patient_features",
                "latency_ms": 0.0,
                "warnings": ["illustrative_research_triage_not_recommendation_or_treatment_benefit"],
                "error": None,
            })
        else:
            receipts.append(_skipped_stage_receipt(
                "therapy_research_triage", request_id,
                "complete explicit triage vector required; missing nodes never use unknown=0.5",
            ))
    else:
        receipts.append(_skipped_stage_receipt(
            "therapy_research_triage", request_id,
            "not applicable to HGSOC" if req.cancer == "hgsoc" else "no explicit triage features supplied",
        ))

    has_signal = any((
        report_parse is not None,
        phikon_embedding is not None,
        case_manifest is not None,
        dss_prognosis is not None,
        therapy_bridge is not None,
        bool(req.clinical_baseline),
    ))
    co_scientist = _run_medgemma_co_scientist(
        context={
            "cancer": req.cancer,
            "clinical_baseline": req.clinical_baseline,
            "report_parse": report_parse.model_dump(mode="json") if report_parse else None,
            "phikon_embedding": phikon_embedding,
            "dss_prognosis": dss_prognosis.model_dump(mode="json") if dss_prognosis else None,
            "therapy_bridge": therapy_bridge,
        },
        request_id=request_id,
        requested=req.run_co_scientist,
        has_patient_signal=has_signal,
        receipts=receipts,
    )

    status = _pipeline_status(receipts)
    state = ModelState.UNAVAILABLE if status == "failed_required_stage" else ModelState.LOADED
    env = _envelope(request_id, model_state=state, model_name="specialist-stack-composite")
    env["warnings"] = warnings
    return DynamicTumorBoardResponse(
        **env,
        pipeline_status=status,
        cancer=req.cancer,
        stage_receipts=receipts,
        ovarian_arbiter=ovarian_arbiter,
        report_parse=report_parse,
        phikon_embedding=phikon_embedding,
        case_manifest=case_manifest,
        dss_prognosis=dss_prognosis,
        therapy_bridge=therapy_bridge,
        therapy_triage=therapy_triage,
        co_scientist=co_scientist,
    )

@router.post("/v1/tumor_board/dynamic", response_model=DynamicTumorBoardResponse)
def dynamic_tumor_board(
    req: DynamicTumorBoardRequest,
    tenant: APIKey = Depends(require_api_key),
) -> DynamicTumorBoardResponse:
    if req.cancer == "hgsoc":
        raise HTTPException(
            status_code=501,
            detail={
                "error": "hgsoc_tumor_board_dynamic_not_implemented",
                "cancer": "hgsoc",
                "detail": (
                    "HGSOC is not implemented on /v1/tumor_board/dynamic. "
                    "Ovarian requests are not scored by breast screening, "
                    "breast DSS, or breast therapy."
                ),
            },
        )
    request_id = new_request_id()
    response = _assemble_dynamic_tumor_board(
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
    if cancer_norm not in _SUPPORTED_CANCERS:
        raise HTTPException(400, f"Unsupported cancer={cancer!r}; supported={sorted(_SUPPORTED_CANCERS)}")
    if cancer_norm == "hgsoc":
        raise HTTPException(
            status_code=501,
            detail={
                "error": "hgsoc_case_full_not_implemented",
                "cancer": "hgsoc",
                "detail": (
                    "HGSOC is not implemented on /v1/case/full. "
                    "Ovarian requests are not scored by breast screening, "
                    "breast DSS, or breast therapy."
                ),
            },
        )
    request_id = new_request_id()

    if cancer_norm == "breast":
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
        dynamic = _assemble_dynamic_tumor_board(
            dynamic_request, request_id=request_id,
        )
        screening = screening_analyze(req.screening_input, tenant=tenant) if req.screening_input else None
        biopsy = biopsy_analyze(biopsy_input, tenant=tenant) if biopsy_input else None
        therapy = None
        if biopsy is not None or req.therapy_features is not None or req.mutations or req.germline_mutations:
            therapy_req = TherapyRequest(
                biopsy_output=biopsy,
                patient_context=req.therapy_context,
                therapy_features=req.therapy_features,
                mutations=req.mutations,
                germline_mutations=req.germline_mutations,
                ranked_drugs=req.ranked_drugs,
            )
            therapy = therapy_reason(therapy_req, tenant=tenant)
        composite_state = dynamic.provenance.model_state
        if therapy is not None and therapy.provenance.model_state == ModelState.LOADED_MEDGEMMA_THERAPY:
            composite_state = ModelState.LOADED_MEDGEMMA_THERAPY
        elif biopsy is not None and biopsy.provenance.model_state == ModelState.LOADED_PHIKON:
            composite_state = ModelState.LOADED_PHIKON
        elif screening is not None and screening.provenance.model_state in (
            ModelState.LOADED_MEDSIGLIP_MODAL,
            ModelState.LOADED_MEDSIGLIP,
        ):
            composite_state = screening.provenance.model_state
        merged_evidence = list(therapy.evidence) if therapy is not None else []
        merged_warnings = list(dynamic.warnings)
        if screening is not None:
            merged_warnings.extend(screening.warnings)
        if biopsy is not None:
            merged_warnings.extend(biopsy.warnings)
        if therapy is not None:
            merged_warnings.extend(therapy.warnings)
        log_event(
            request_id, "/v1/case/full",
            model_state=composite_state.value,
            patient_id_hash=None,
            extra={
                "cancer": cancer_norm,
                "pipeline_status": dynamic.pipeline_status,
                "has_screening": screening is not None,
                "has_biopsy": biopsy is not None,
                "has_therapy": therapy is not None,
                "n_therapy_evidence": len(merged_evidence),
            },
            tenant_id=tenant.tenant_id,
        )
        env = _envelope(
            request_id,
            model_state=composite_state,
            model_name="specialist-stack-composite",
        )
        env["warnings"] = merged_warnings
        env["evidence"] = merged_evidence
        if therapy is not None:
            env["honesty_gate"] = therapy.honesty_gate
        return FullCaseResponse(
            **env,
            pipeline_status=dynamic.pipeline_status,
            cancer=cancer_norm,
            stage_receipts=dynamic.stage_receipts,
            ovarian_arbiter=dynamic.ovarian_arbiter,
            case_manifest=dynamic.case_manifest,
            report_parse=dynamic.report_parse or (biopsy.report_parse if biopsy else None),
            phikon_embedding=(
                dynamic.phikon_embedding
                or (biopsy.phikon_embedding if biopsy else None)
            ),
            dss_prognosis=dynamic.dss_prognosis or (biopsy.dss_prognosis if biopsy else None),
            therapy_bridge=dynamic.therapy_bridge or (therapy.therapy_bridge if therapy else None),
            therapy_triage=dynamic.therapy_triage or (therapy.therapy_triage if therapy else None),
            co_scientist=dynamic.co_scientist,
            screening=screening,
            biopsy=biopsy,
            therapy=therapy,
            nsclc=None,
            elo_ranked_hypotheses=[],
        )

    from .lung import run_nsclc_case
    return run_nsclc_case(req, request_id=request_id, tenant=tenant)


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
#   2. If HIPAA_MODE=true, patient_id and name fields are redacted
#      before the bundle is echoed. model_state=LOADED_HIPAA_REDACTOR
#      is set only after that redaction runs.

@router.post("/v1/tumor_board/bundle", response_model=TumorBoardBundleResponse)
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
    import hashlib
    import json as _json
    import os

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
    request_id = new_request_id()
    hipaa_mode = os.environ.get("HIPAA_MODE", "").lower() in {"1", "true", "yes"}
    payload = bundle.model_dump(mode="json")
    if hipaa_mode:
        from oncology_arbiter.api.hipaa_redact import redact_phi

        try:
            payload = redact_phi(payload)
            bundle = TumorBoardBundle.model_validate(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "hipaa_redactor_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        state = ModelState.LOADED_HIPAA_REDACTOR
    else:
        state = ModelState.LOADED_AK_BUNDLE

    canonical = _json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    bundle_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
