"""Schemas for /v1/screening/*."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

from .common import ApiEnvelope, ArbiterScore, StageReceipt



# --------------------------------------------------------------------------- #
# /v1/screening/analyze


class ScreeningRequest(BaseModel):
    """Input for /v1/screening/analyze.

    Real ingestion supports either:
      * `dicom_url` — a URL our fetcher can reach (respects SSRF guard)
      * `dicom_bytes_b64` — base64-encoded DICOM bytes (small studies only)

    NOT both. NOT neither. `laterality_hint` and `view_hint` are optional
    overrides for our preprocessing pipeline's metadata detection.
    """
    dicom_url: HttpUrl | None = None
    dicom_bytes_b64: str | None = None
    laterality_hint: Literal["L", "R"] | None = None
    view_hint: Literal["CC", "MLO"] | None = None
    patient_id_hash: str | None = Field(
        default=None,
        description="SHA256 hex of patient MRN or study UID (never the raw id)",
        min_length=64,
        max_length=64,
    )



class ScreeningFusionMetadata(BaseModel):
    """Clinical metadata required for RSNA MedSigLIP fusion head (RUO).

    This mirrors the fields used in the retrospective RSNA training run.
    Callers must at minimum provide age and density.
    """

    age: float = Field(..., ge=18.0, le=120.0, description="Patient age in years.")
    density: Literal["A", "B", "C", "D"] = Field(
        ...,
        description="BI-RADS breast density category (A..D).",
    )
    laterality: Literal["L", "R"] | None = Field(default=None)
    view: Literal["CC", "MLO"] | None = Field(default=None)
    implant: Literal["0", "1"] | None = Field(default=None, description="0/1; optional.")
    site_id: str | None = Field(default=None, description="Optional site identifier (stringified).")
    machine_id: str | None = Field(default=None, description="Optional machine identifier (stringified).")



class ScreeningFusionRequest(BaseModel):
    """Input for /v1/screening/fusion (RUO).

    Strictly requires DICOM bytes + clinical metadata. URL ingestion is not accepted.
    """

    dicom_bytes_b64: str = Field(..., description="base64-encoded DICOM bytes")
    laterality_hint: Literal["L", "R"] | None = None
    view_hint: Literal["CC", "MLO"] | None = None
    patient_id_hash: str | None = Field(
        default=None,
        description="SHA256 hex of patient MRN or study UID (never the raw id)",
        min_length=64,
        max_length=64,
    )
    clinical_metadata: ScreeningFusionMetadata



class ScreeningFinding(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    location_bbox_normalized: list[float] | None = Field(
        default=None,
        description="[x0, y0, x1, y1] in [0,1] frame coordinates, or None if not localized",
    )



class ScreeningResponse(ApiEnvelope):
    pipeline_status: Literal["complete", "partial_failure", "failed_required_stage"] = "complete"
    stage_receipts: list[StageReceipt] = Field(default_factory=list)
    laterality: Literal["L", "R", "U"]
    view: Literal["CC", "MLO", "U"]
    orientation_flipped: bool
    breast_mask_coverage: float = Field(..., ge=0.0, le=1.0)
    findings: list[ScreeningFinding] = Field(default_factory=list)
    overall_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Uncalibrated off-label MedSigLIP zero-shot score; null on required-stage failure.",
    )
    medsiglip: dict[str, Any] | None = Field(
        default=None,
        description="Strict production MedSigLIP-448 inference provenance and embedding receipt.",
    )
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="Optional downstream arbiter output; never calculated from a missing MedSigLIP stage.",
    )
    fusion_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "RUO: supervised fusion probability (MedSigLIP embedding + clinical metadata) "
            "from rsna_medsiglip_fusion_logreg_v1; null when not computed."
        ),
    )
    fusion_model_name: str | None = Field(
        default=None,
        description="RUO: name of the fused head used to compute fusion_score.",
    )
    fusion_warning: str | None = Field(
        default=None,
        description="RUO warning string for fusion_score; callers MUST surface this.",
    )
