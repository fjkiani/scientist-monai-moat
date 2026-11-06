"""POST /v1/ranker/8d/score — 8D/7D PATH A Mechanism-Fit + Contraindication Ranker (RUO)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.arbiter.ml_8d_trial_decoder import (
    BilinearDecoderInputError,
    score_bilinear_ml_loto,
)
from oncology_arbiter.arbiter.pathway_8d_ranker import (
    Pathway8DRankerInputError,
    score_pathway_8d_case,
)
from oncology_arbiter.auth import APIKey, require_api_key

router = APIRouter(prefix="/v1/ranker", tags=["ranker-8d"])


class Ranker8DScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = None
    domain: str | None = "crc"
    mode: str | None = "iii_c"
    profile_id: str | None = None

    # Input Style 1: Direct 8D (or 7D for io_core) patient vector in [0.0, 1.0]^D
    patient_vector: list[Any] | None = None

    # Input Style 2: Clinical features / contraindication context dictionaries
    clinical_features: dict[str, Any] | None = None
    contraindications: dict[str, Any] | None = None
    context: dict[str, Any] | None = None

    # Optional top-level clinical & contraindication attributes
    tmb: float | None = None
    hrd: str | float | None = None
    hrd_proxy: str | float | None = None
    fga: float | None = None
    fraction_genome_altered: float | None = None
    aneuploidy: float | None = None
    aneuploidy_score: float | None = None
    kras_mut: bool | None = None
    nras_mut: bool | None = None
    braf_mut: bool | None = None
    brca_mut: bool | None = None
    brca_somatic: bool | None = None
    egfr_mut: bool | None = None
    pten_null: bool | None = None
    mss: bool | None = None
    liver_met: bool | None = None
    liver_metastasis: bool | None = None
    tgfb_high: bool | None = None
    tgfb_barrier_high: bool | None = None
    immune_z: float | None = None
    gate2_z: float | None = None
    immune_score_rna: float | None = None
    gate2_hot: bool | None = None
    gate2_immune_hot: bool | None = None
    mapk_mode: str | None = None
    mapk_convention: str | None = None
    d1_trafficking: float | None = None
    D1_trafficking: float | None = None
    D1: float | None = None
    d2_presentation: float | None = None
    D2_presentation: float | None = None
    D2: float | None = None
    d1_d2_vaccine_effector_index: float | None = None
    D1_D2_Vaccine_Effector_Index: float | None = None
    apm_intact: bool | None = None
    APM_intact: bool | None = None
    b2m_truncating_lof: bool | None = None

    eligibility: float | None = None
    apply_gate: bool | None = None
    use_axis_mask_for_ranking: bool | None = None

    # engine omitted or "path_a_gated" keeps the existing PATH A ranker.
    # "bilinear_ml_loto" loads frozen signed-axis weights and never falls back.
    engine: str | None = None
    p_signed_8d: list[Any] | None = None
    moa_weights: dict[str, Any] | None = None


@router.post("/8d/score")
def ranker_8d_score(
    req: Ranker8DScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> dict[str, Any]:
    request_id = new_request_id()
    # Keep only explicitly supplied fields so empty `{}` body fails-closed with HTTP 422
    payload = req.model_dump(exclude_unset=True)
    engine = payload.get("engine") or "path_a_gated"
    try:
        if engine == "path_a_gated":
            path_payload = {
                k: v
                for k, v in payload.items()
                if k not in {"engine", "p_signed_8d", "moa_weights"}
            }
            body = score_pathway_8d_case(path_payload)
            body["engine"] = "path_a_gated"
        elif engine == "bilinear_ml_loto":
            body = score_bilinear_ml_loto(payload)
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "unknown_engine",
                    "engine": engine,
                    "allowed": ["path_a_gated", "bilinear_ml_loto"],
                    "fallback": False,
                    "message": "engine must be path_a_gated or bilinear_ml_loto",
                },
            )
    except Pathway8DRankerInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_or_missing_inputs",
                "engine": "path_a_gated",
                "missing": exc.missing,
                "message": str(exc),
                "fallback": False,
            },
        ) from exc
    except BilinearDecoderInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": exc.code,
                "engine": "bilinear_ml_loto",
                "missing": exc.missing,
                "message": str(exc),
                "fallback": False,
            },
        ) from exc
    body["request_id"] = request_id
    log_event(
        request_id,
        "/v1/ranker/8d/score",
        model_state="loaded",
        patient_id_hash=None,
        extra={
            "domain": body.get("domain"),
            "mode": body.get("mode"),
            "top_trial": body.get("top_trial"),
            "stc_1010_rank_1": body.get("stc_1010_rank_1"),
        },
        tenant_id=tenant.tenant_id,
    )
    return body
