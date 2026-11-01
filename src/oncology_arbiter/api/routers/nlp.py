"""MedGemma pathology extraction and Co-Scientist stage runner."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.services.helpers import failed_stage_receipt as _failed_stage_receipt
from oncology_arbiter.api.services.helpers import skipped_stage_receipt as _skipped_stage_receipt
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.nlp.medgemma_pathology_client import (
    PATHOLOGY_CHAT_URL,
    MedGemmaPathologyError,
    extract_pathology,
)
from oncology_arbiter.nlp.pathology_extraction_schema import EXTRACTION_SYSTEM_PROMPT

router = APIRouter()

medgemma_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


class PathologyExtractRequest(BaseModel):
    report_text: str = Field(min_length=1)


def _extract(req: PathologyExtractRequest) -> dict[str, Any]:
    try:
        result = extract_pathology(req.report_text)
    except MedGemmaPathologyError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result["prompt_contract"] = "PathologyExtraction"
    result["chat_url"] = PATHOLOGY_CHAT_URL
    result["system_prompt_sha256_note"] = "EXTRACTION_SYSTEM_PROMPT keys: er_status, pr_status, her2_status, nottingham_grade, tumor_size_mm"
    if "er_status" not in EXTRACTION_SYSTEM_PROMPT or "her2_status" not in EXTRACTION_SYSTEM_PROMPT:
        raise HTTPException(status_code=500, detail="pathology prompt contract missing receptor keys")
    return result


@router.post("/v1/pathology/extract")
@router.post("/v1/predict/clinicalbert")
def pathology_extract(
    req: PathologyExtractRequest,
    tenant: APIKey = Depends(require_api_key),
) -> dict[str, Any]:
    """Extract ER, PR, HER2, grade, and size with MedGemma. ClinicalBERT is not called."""
    del tenant
    return _extract(req)


def _run_medgemma_co_scientist(
    *,
    context: dict[str, Any],
    request_id: str,
    requested: bool,
    has_patient_signal: bool,
    receipts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not requested:
        receipts.append(_skipped_stage_receipt(
            "medgemma_co_scientist", request_id, "Co-Scientist was not requested"
        ))
        return None
    if not has_patient_signal:
        receipts.append(_skipped_stage_receipt(
            "medgemma_co_scientist", request_id,
            "No validated patient stage output was available for synthesis",
        ))
        return None
    endpoint_url = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint_url:
        # Prefer the identity-locked chat host when the env is unset so case/full
        # does not silently skip Co-Scientist after therapy already ran MedGemma.
        try:
            from oncology_arbiter.models.medgemma_1_5_4b_wiring import verify_artifact_identity

            endpoint_url = str(verify_artifact_identity()["chat_url"]).rstrip("/")
            os.environ["MEDGEMMA_MODAL_URL"] = endpoint_url
        except Exception as exc:  # noqa: BLE001
            receipts.append(_failed_stage_receipt(
                "medgemma_co_scientist", False, request_id,
                "medgemma_not_configured",
                f"MEDGEMMA_MODAL_URL required and identity chat_url unavailable: {exc}",
                service_name="medgemma-1-5-4b",
            ))
            return None
    try:
        from oncology_arbiter.agents.supervisor import execute_stage
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
            assert_pinned_chat_url,
            verify_artifact_identity,
        )

        identity = verify_artifact_identity()
        assert_pinned_chat_url(endpoint_url)
        result = execute_stage(
            "case_full",
            context,
            n_hypotheses=4,
            n_evidence_top_k=2,
        )
        output = result.as_dict()
        if (
            result.model_state != "executed"
            or output.get("llm_provider") != "medgemma_modal"
            or output.get("llm_model") != identity["model_id"]
            or output.get("llm_app_version") != identity["app_version"]
            or output.get("llm_revision") != identity["revision_sha256"]
            or not output.get("llm_request_ids")
        ):
            raise RuntimeError(
                "MedGemma Co-Scientist did not complete with required provider provenance: "
                f"state={result.model_state} notes={result.notes}"
            )
        receipts.append({
            "stage": "medgemma_co_scientist",
            "required": False,
            "status": "succeeded",
            "request_id": request_id,
            "service_name": "medgemma-1-5-4b",
            "endpoint_label": endpoint_url,
            "app_version": output["llm_app_version"],
            "model_name": output["llm_model"],
            "model_version": identity["revision_sha256"][:16],
            "latency_ms": round(float(output.get("latency_s") or 0.0) * 1000.0, 3),
            "warnings": list(output.get("llm_honesty_warnings") or []),
            "error": None,
        })
        return output
    except Exception as exc:
        receipts.append(_failed_stage_receipt(
            "medgemma_co_scientist", False, request_id,
            "medgemma_unavailable", f"{type(exc).__name__}: {exc}",
            service_name="medgemma-1-5-4b",
        ))
        return None
