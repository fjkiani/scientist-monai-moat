"""POST /v1/immunotherapy/arbiter/score — CAR-T / ICI v3 (RUO)."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.immunotherapy.cart_ici_v3 import ImmunoInputError, score_immunotherapy

router = APIRouter(prefix="/v1/immunotherapy", tags=["immunotherapy-v3"])


class ImmunoArbiterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specimen: Literal["cart_infusion_bag", "cart_in_vivo_blood", "ici_tumor_biopsy"]
    bag_gate: str | None = None
    n_CD8: float | None = None
    M_ratio: float | None = None
    M_expr: float | None = None
    f_cyto_marker: float | None = None
    n_memory: float | None = None
    n_cyto: float | None = None
    n_exhausted: float | None = None
    log_baseline_spd: float | None = None
    prior_regimens: float | None = None
    prior_autologous_transplant: float | None = None
    load_ZNF683: float | None = None
    load_GNLY: float | None = None
    cyto_absent: float | None = None
    mode: Literal["bulk", "single_cell"] | None = None
    context: str | None = None
    expression: dict[str, float] | None = None
    n_TCF7_positive: float | None = None
    n_TOX_positive: float | None = None


@router.post("/arbiter/score")
def immunotherapy_arbiter_score(
    req: ImmunoArbiterRequest,
    tenant: APIKey = Depends(require_api_key),
) -> dict[str, Any]:
    request_id = new_request_id()
    payload = req.model_dump(exclude_none=True)
    try:
        body = score_immunotherapy(payload)
    except ImmunoInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_required_inputs", "missing": exc.missing},
        ) from exc
    body["request_id"] = request_id
    log_event(
        request_id,
        "/v1/immunotherapy/arbiter/score",
        model_state="loaded",
        patient_id_hash=None,
        extra={"specimen": body.get("specimen")},
        tenant_id=tenant.tenant_id,
    )
    return body
