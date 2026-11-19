"""POST /v1/crc/stc1010/score — STC-1010 6-Domain MSS CRC Biomarker Lock v2 & 4-Stratum Regimen Engine (RUO)."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.crc.stc1010_arbiter_v2 import (
    Stc1010InputError,
    score_stc1010_crc_case,
)

router = APIRouter(prefix="/v1/crc", tags=["crc-stc1010-v2"])


class CrcStc1010ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = None
    reference: Literal["TCGA", "GSE39582"] = "TCGA"
    mss: bool = True

    # Input Mode 1: Full 42-gene platform log2 expression
    log_expression: dict[str, float] | None = None
    # Input Mode 2: Full 42-gene platform z-scores (ddof=0)
    gene_z: dict[str, float] | None = None
    # Input Mode 3: Precomputed domain/module dictionary or top-level domain fields
    domains: dict[str, Any] | None = None

    D1_trafficking: float | None = None
    D2_presentation: float | None = None
    D3_effector: float | None = None
    D4_triad: float | None = None
    D5_myeloid: float | None = None
    D6_CMS1like: float | None = None
    D1: float | None = None
    D2: float | None = None
    D3: float | None = None
    D4: float | None = None
    D5: float | None = None
    D6: float | None = None

    D1_D2_Vaccine_Effector_Index: float | None = None
    D1_D2_D3_D4_Immune_Competence_Index: float | None = None
    STC1010_Readiness_Score: float | None = None
    classifier_score: float | None = None
    classifier_logit: float | None = None

    TGFB_CAF_Barrier: float | None = None
    CXCR3_Trafficking: float | None = None
    APM_TAP_Score: float | None = None
    VEGF_TOX_Exhaustion: float | None = None
    Treg_Load: float | None = None
    S_triad: float | None = None
    KRAS_IRF2_CXCL3_Axis: float | None = None
    SPP1_Core_Barrier: float | None = None

    APM_intact: bool | None = None
    apm_intact: bool | None = None
    AACR_A065_Positive_MSS: bool | None = None
    aacr_a065_positive: bool | None = None

    z_FAP: float | None = None
    z_CXCL10: float | None = None
    z_TAP1: float | None = None
    z_TAP2: float | None = None
    z_PSMB8: float | None = None
    z_B2M: float | None = None

    # Somatic & clinical context for 8D PATH A and 4-stratum regimen plan
    kras_mut: bool = False
    kras_g12d: bool = False
    nras_mut: bool = False
    braf_mut: bool = False
    braf_v600e: bool = False
    b2m_truncating_lof: bool = False
    liver_metastasis: bool = False

    hrd_proxy: str = "HRD-Intermediate"
    brca_somatic: bool = False
    tmb: float | None = None
    aneuploidy_score: float | None = None
    fraction_genome_altered: float | None = None
    mapk_convention: Literal["f07", "ws5v2"] = "f07"
    use_axis_mask_for_ranking: bool = False


@router.post("/stc1010/score")
def crc_stc1010_score(
    req: CrcStc1010ScoreRequest,
    tenant: APIKey = Depends(require_api_key),
) -> dict[str, Any]:
    request_id = new_request_id()
    payload = req.model_dump(exclude_none=True)
    try:
        body = score_stc1010_crc_case(payload)
    except Stc1010InputError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_required_inputs", "missing": exc.missing},
        ) from exc
    body["request_id"] = request_id
    log_event(
        request_id,
        "/v1/crc/stc1010/score",
        model_state="loaded",
        patient_id_hash=None,
        extra={
            "reference": body.get("reference"),
            "primary_stratum": body.get("break_crc001_regimen_plan", {}).get(
                "primary_stratum"
            ),
            "top_trial": body.get("pathway_8d", {}).get("top_trial"),
        },
        tenant_id=tenant.tenant_id,
    )
    return body
