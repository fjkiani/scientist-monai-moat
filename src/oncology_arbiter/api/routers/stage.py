"""Stage-specific arbiter scoring endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from ..schemas import ModelState, StageBiopsyScoreRequest, StageBiopsyScoreResponse
from .. import deps

router = APIRouter(prefix="/v1/stage")


@router.post("/biopsy/score", response_model=StageBiopsyScoreResponse)
def stage_biopsy_score(
    req: StageBiopsyScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> StageBiopsyScoreResponse:
    """Score biopsy_arbiter_v1 from an explicit feature panel only.

    Calcification-only morphology without mass descriptors fails closed
    (HTTP 422, ``unvalidated_lesion_stratum``). No report parsing or
    imaging inference is performed on this route.
    """
    request_id = new_request_id()
    arbiter_score = deps.score_arbiter(
        "biopsy",
        req.features,
        stage="stage-biopsy",
    )
    env = deps.envelope(
        request_id,
        model_state=ModelState.LOADED,
        model_name=arbiter_score.model_name,
    )
    log_event(
        request_id,
        "/v1/stage/biopsy/score",
        model_state=ModelState.LOADED.value,
        patient_id_hash=req.patient_id_hash,
        extra={"n_training": arbiter_score.n_training},
        tenant_id=tenant.tenant_id,
    )
    return StageBiopsyScoreResponse(**env, arbiter_score=arbiter_score)
