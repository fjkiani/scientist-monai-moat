"""POST /v1/ovarian/arbiter/score — two-door v3 (RUO)."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.orchestrator.biological_elo_bridge import build_ovarian_elo_bridge
from oncology_arbiter.ovarian.two_door_arbiter import OvarianInputError, score_ovarian_case

router = APIRouter(prefix="/v1/ovarian", tags=["ovarian-two-door"])


class OvarianArbiterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timepoint: Literal["pre_treatment", "post_nact"] = "post_nact"
    patient_id: str | None = None
    post_H_clean: float | None = None
    post_DDR: float | None = None
    post_S_sen: float | None = None
    post_BCL2A1_z: float | None = None
    delta_MCL1_z: float | None = None
    post_EMT_z: float | None = None
    z_stress_34_clean: float | None = None
    z_proliferation_106: float | None = None
    z_BCL2L1: float | None = None
    z_PMAIP1: float | None = None
    z_MCL1: float | None = None
    z_FAP: float | None = None
    z_CXCL10: float | None = None
    mfap4_expr: float | None = None
    mbd4_lof: bool | None = None
    mbd4_hgvs_c: str | None = None
    mbd4_consequence: str | None = None
    tp53_mut: bool | None = None
    brca_wt: bool | None = None
    hrd: float | None = Field(default=None)


@router.post("/arbiter/score")
def ovarian_arbiter_score(
    req: OvarianArbiterRequest,
    tenant: APIKey = Depends(require_api_key),
) -> dict[str, Any]:
    request_id = new_request_id()
    try:
        body = score_ovarian_case(req.model_dump())
    except OvarianInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_required_inputs", "missing": exc.missing},
        ) from exc
    body["elo_bridge"] = build_ovarian_elo_bridge(body)
    body["request_id"] = request_id
    log_event(
        request_id,
        "/v1/ovarian/arbiter/score",
        model_state="loaded",
        patient_id_hash=None,
        extra={"timepoint": body["timepoint"], "decision": body.get("decision")},
        tenant_id=tenant.tenant_id,
    )
    return body
