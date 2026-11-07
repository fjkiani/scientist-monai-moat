"""Stage-specific arbiter scoring endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from ..schemas import (
    ModelState,
    StageBiopsyScoreRequest,
    StageBiopsyScoreResponse,
)
from .. import deps

router = APIRouter(prefix="/v1/stage")


def _score_stage(
    name: str,
    stage: str,
    req: StageBiopsyScoreRequest,
    tenant: APIKey,
) -> StageBiopsyScoreResponse:
    """Score the named L2 arbiter from the request JSON panel.

    ``require_complete`` refuses omitted ER/PR/HER2 and every other declared
    input. The scorer never receives a ``features={}`` stand-in.
    """
    request_id = new_request_id()
    arbiter_score = deps.score_arbiter(
        name,
        req.features,
        stage=stage,
        require_complete=True,
    )
    env = deps.envelope(
        request_id,
        model_state=ModelState.LOADED,
        model_name=arbiter_score.model_name,
    )
    log_event(
        request_id,
        f"/v1/stage/{name}/score",
        model_state=ModelState.LOADED.value,
        patient_id_hash=req.patient_id_hash,
        extra={"n_training": arbiter_score.n_training},
        tenant_id=tenant.tenant_id,
    )
    return StageBiopsyScoreResponse(**env, arbiter_score=arbiter_score)


@router.post("/screening/score", response_model=StageBiopsyScoreResponse)
def stage_screening_score(
    req: StageBiopsyScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> StageBiopsyScoreResponse:
    """Score screening_arbiter_v1 from the posted feature object only."""
    return _score_stage("screening", "stage-screening", req, tenant)


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
    return _score_stage("biopsy", "stage-biopsy", req, tenant)


@router.post("/therapy/score", response_model=StageBiopsyScoreResponse)
def stage_therapy_score(
    req: StageBiopsyScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> StageBiopsyScoreResponse:
    """Score therapy_arbiter_v1 from the posted feature object only.

    Missing ER, PR, or HER2 is HTTP 422. Those inputs are not stored as
    ER-negative, PR-negative, or HER2-negative.
    """
    return _score_stage("therapy", "stage-therapy", req, tenant)
