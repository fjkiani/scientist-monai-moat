"""Health, demo, model-cards, artifacts, and audit export schemas."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ModelState



# --------------------------------------------------------------------------- #
# /v1/model-cards, /v1/artifacts


class ModelCardSummary(BaseModel):
    """Summary of a single model card served by /v1/model-cards."""
    slug: str = Field(..., description="Stem of the .md file, e.g. 'medsiglip_448'")
    title: str = Field(..., description="First H1 heading of the card")
    n_bytes: int = Field(..., ge=0)
    honesty_markers: dict[str, bool] = Field(
        ...,
        description="Which honesty markers this card contains (auroc_caveat, ruo_disclaimer, ...)",
    )



class ModelCardsIndex(BaseModel):
    """/v1/model-cards response — index of every model card shipped with the API."""
    disclaimer: str
    caveat: str
    cards: list[ModelCardSummary]



class ArtifactCategory(str, Enum):
    docs = "docs"
    reports = "reports"
    data = "data"
    models = "models"



class WeightsProvenance(BaseModel):
    """Per-checkpoint provenance for fine-tuned weights.

    Ships with any model whose weights were trained in-house (mammo MONAI,
    LUNA16 refine, biopsy probe). CI gate `weights_meet_floor` refuses to
    green-light a release where achieved_metric < floor_metric.

    Floors per PLAN §7 bullet 6:
      * mammo MONAI RetinaNet: AUROC ≥ 0.85 (floor from published CBIS-DDSM CNN baseline)
      * biopsy MedSigLIP probe: AUROC ≥ 0.85
      * LUNA16 refine: ΔFROC@2 ≥ +5% over v0.6.9 baseline
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    weights_meet_floor: bool
    achieved_metric: float
    floor_metric: float
    floor_source: str = Field(
        ...,
        description=(
            "Documented source of the floor, e.g. "
            "'docs/proofs/cbis_ddsm_logreg_v1_metrics.json (typical fine-tuned CNN baseline 0.85-0.90)'."
        ),
    )
    metric_name: str = Field(
        ...,
        description="e.g. 'AUROC', 'FROC_at_2_FPs_per_scan', 'delta_FROC_at_2'.",
    )
    weights_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the saved weights file when disk-loaded; None for Modal-loaded weights.",
    )



class AuditExportResponse(BaseModel):
    """Envelope for GET /v1/audit/export."""
    since_iso: str
    until_iso: str
    n_events: int
    events_ndjson_url: str | None = None
    events_inline: list[dict[str, Any]] | None = None



# --------------------------------------------------------------------------- #
# /health


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    disclaimer: str
    caveat: str
    endpoints: list[str]
    models_loaded: dict[str, ModelState]
    # cancers advertises which cancer tracks the current build exposes via
    # `/v1/case/full?cancer=<name>`. Each entry MUST include `state`
    # (mirror of ModelState) and `endpoints` (list of sub-routes wired for
    # that cancer). The SPA reads this to decide which cancer-selector
    # options to enable and which panels to render.
    # Untyped dict so we can bolt on extra metadata (e.g. `case_full: bool`)
    # per cancer without a schema migration; the SPA only reads the keys it
    # knows.
    cancers: dict[str, dict] = {}
    # v0.3.0-alpha demo deployment gate. When true, all POST routes return
    # HTTP 403 with a contact placeholder, and pre-computed sample outputs
    # are served under /v1/demo/samples/*. Read-only GET endpoints stay
    # open. `contact_url` is where the frontend routes "Run on your own
    # data" CTA clicks.
    demo_mode: bool = False
    contact_url: str | None = None
    demo_samples: list[str] = []



# --------------------------------------------------------------------------- #
# /v1/demo/case (v0.2.2)


class DemoCaseResponse(BaseModel):
    """Fully-formed sample case returned by ``GET /v1/demo/case``.

    The DICOM is a public CBIS-DDSM training image (CC-BY-NC 4.0); the
    pathology text and patient context are synthetic. See the endpoint
    docstring in ``app.py`` for the full source citation and license.
    """
    model_config = ConfigDict(extra="forbid")

    dicom_bytes_b64: str = Field(
        ..., description="Base64-encoded DICOM ready to POST to /v1/screening/analyze."
    )
    dicom_source: str = Field(
        ..., description="Human-readable citation for the DICOM."
    )
    dicom_sha256: str = Field(
        ..., description="SHA-256 of the DICOM bytes (before base64 encoding)."
    )
    dicom_size_bytes: int = Field(
        ..., ge=0, description="Raw DICOM size before base64 encoding."
    )
    report_text: str = Field(
        ..., description="Canned luminal-A pathology report matching the frontend example."
    )
    patient_context: dict[str, Any] = Field(
        default_factory=dict, description="Patient context to send to /v1/therapy/reason."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Honesty caveats — 'not a real patient', 'RUO', 'synthetic report', etc.",
    )
