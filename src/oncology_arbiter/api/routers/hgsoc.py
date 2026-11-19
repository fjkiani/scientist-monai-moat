"""Dedicated HGSOC track — MFAP4 platinum-resistance (GSE63885).

Does NOT reopen the retired ovarian prognostic / case/full HGSOC path.
Scores Affymetrix GPL570 probe 212713_at expression only.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api import deps
from oncology_arbiter.api.schemas import (
    ArbiterScore,
    ModelState,
    StageBiopsyScoreResponse,
)
from oncology_arbiter.arbiter import stage_mfap4_wiring as mfap4
from oncology_arbiter.auth import APIKey, require_api_key

router = APIRouter(prefix="/v1/hgsoc", tags=["hgsoc-mfap4"])


class Mfap4ScoreRequest(BaseModel):
    """Single-feature panel: GPL570 212713_at log2 expression."""

    model_config = ConfigDict(extra="forbid")
    mfap4_expr: float = Field(
        ...,
        description=(
            "Affymetrix GPL570 probe 212713_at log2 expression for MFAP4 "
            "(GSE63885 scale). No toy threshold gates."
        ),
    )
    patient_id_hash: str | None = Field(
        default=None, min_length=64, max_length=64
    )


@router.get("/mfap4/health")
def mfap4_health() -> dict[str, Any]:
    """Identity + meter for the MFAP4 platinum-resistance capability."""
    try:
        model = mfap4.load_stage_mfap4_arbiter()
        proof = mfap4.load_proof_receipt()
    except Exception as exc:  # noqa: BLE001 — surface load failure honestly
        return {
            "status": "degraded",
            "model_state": ModelState.UNAVAILABLE.value,
            "capability": mfap4.CAPABILITY,
            "error": str(exc),
            "case_full": False,
            "note": (
                "MFAP4 score route is independent of retired "
                "ovarian_prognostic_arbiter / hgsoc case/full."
            ),
        }
    held = float(model.performance["held_out_test_auroc"])
    return {
        "status": "ok",
        "model_state": ModelState.LOADED.value,
        "capability": mfap4.CAPABILITY,
        "model_name": model.model_name,
        "geo": mfap4.GEO,
        "probe": mfap4.PROBE,
        "gene": mfap4.GENE,
        "n_training": model.n_training,
        "auroc": held,
        "production_auroc": mfap4.PRODUCTION_AUROC,
        "proof_auroc": mfap4.PROOF_AUROC,
        "artifact_sha256": mfap4.ARTIFACT_SHA256,
        "proof_status": proof.get("status"),
        "forbidden": proof.get("forbidden"),
        "case_full": False,
        "endpoints": [
            "GET /v1/hgsoc/mfap4/health",
            "GET /v1/hgsoc/mfap4/metrics",
            "POST /v1/hgsoc/mfap4/score",
        ],
        "note": (
            "Dedicated HGSOC MFAP4 platinum-resistance track. "
            "Does not enable POST /v1/case/full?cancer=hgsoc (still 501)."
        ),
    }


@router.get("/mfap4/metrics")
def mfap4_metrics() -> dict[str, Any]:
    """Return the locked GSE63885 MFAP4 proof receipt."""
    try:
        proof = mfap4.load_proof_receipt()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "ruo": True,
        "capability": mfap4.CAPABILITY,
        "data": proof,
    }


@router.post("/mfap4/score", response_model=StageBiopsyScoreResponse)
def mfap4_score(
    req: Mfap4ScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> StageBiopsyScoreResponse:
    """Score platinum-resistance lean from MFAP4 expression (RUO)."""
    request_id = new_request_id()
    features = {"mfap4_expr": req.mfap4_expr}
    arbiter_score: ArbiterScore = deps.score_arbiter(
        "mfap4",
        features,
        stage="hgsoc-mfap4-platinum",
        require_complete=True,
    )
    env = deps.envelope(
        request_id,
        model_state=ModelState.LOADED,
        model_name=arbiter_score.model_name,
    )
    log_event(
        request_id,
        "/v1/hgsoc/mfap4/score",
        model_state=ModelState.LOADED.value,
        patient_id_hash=req.patient_id_hash,
        extra={
            "n_training": arbiter_score.n_training,
            "probe": mfap4.PROBE,
            "auroc": mfap4.PROOF_AUROC,
        },
        tenant_id=tenant.tenant_id,
    )
    return StageBiopsyScoreResponse(**env, arbiter_score=arbiter_score)
