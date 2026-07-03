"""Pydantic schemas for the oncology-arbiter API.

Every response envelope carries the RUO disclaimer and AUROC caveat inline
so downstream consumers cannot strip them without noticing. Every response
also carries a `provenance` block indicating where the results came from —
placeholder, cached, live model — so a consumer can distinguish a stub from
a real inference at wire level.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# --------------------------------------------------------------------------- #
# Shared


class ModelState(str, Enum):
    PLACEHOLDER = "placeholder"       # no model wired yet, stub response
    LOADED = "loaded"                 # model in memory, inference performed
    LOADING = "loading"               # model warming up
    UNAVAILABLE = "unavailable"       # model failed to load; error path
    CACHED = "cached"                 # result served from cache
    GATED = "gated"                   # HAI-DEF access denied (401/403)
    PROXY_SIGLIP = "proxy_siglip"     # ungated general-domain SigLIP fallback (NOT MedSigLIP output)


class EvidenceRecord(BaseModel):
    """A Co-Scientist-style piece of evidence. URL must have been seen
    during the run (enforced by reflection.py honesty gate)."""
    model_config = ConfigDict(str_strip_whitespace=True)
    url: str = Field(..., description="Fetched URL")
    quoted_text: str = Field(..., description="Verbatim quote used from that URL")
    source: str = Field(default="unknown",
                        description="Source module (pubmed, arxiv, europe_pmc, web_fetch)")


class HonestyGateReport(BaseModel):
    seen_urls_count: int = Field(..., ge=0)
    evidence_kept: int = Field(..., ge=0)
    evidence_dropped: int = Field(..., ge=0)


class ArbiterScore(BaseModel):
    """L3 calibrated logistic arbiter output, matching progression_arbiter shape.

    Every stage endpoint (screening/biopsy/therapy) attaches one of these blocks
    when the L2 logistic arbiter runs. The `term_contributions` dict is required
    to be a linear decomposition of the logit so downstream 'why did the model
    say that?' UI can attribute the prediction to individual features.

    PLAN.md §4a: 'Every response body carries: stage_output, arbiter_score,
    term_contributions, driving_feature, evidence[]'.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    model_name: str = Field(..., description="e.g. screening_arbiter_template_v0")
    p_positive: float = Field(..., ge=0.0, le=1.0)
    logit: float
    risk_bucket: Literal["LOW", "MID", "HIGH"]
    recommendation: str
    term_contributions: dict[str, float]
    driving_feature: str
    driving_feature_contribution: float
    positive_class: str
    n_training: int = Field(..., ge=0)
    model_state: Literal["template", "frozen"] = Field(
        default="template",
        description="'template' when n_training==0 (illustrative), 'frozen' after Phase 3 fit.",
    )
    caveat: str = Field(..., description="AUROC caveat from the frozen JSON")


class Provenance(BaseModel):
    """Where did this response come from?"""
    model_config = ConfigDict(str_strip_whitespace=True)
    model_state: ModelState
    model_name: str | None = Field(
        default=None,
        description="Backend model identifier when loaded, e.g. google/medsiglip-448",
    )
    model_version: str | None = None
    request_id: str = Field(..., description="Trace id for correlation with the audit ledger")


class ApiEnvelope(BaseModel):
    """All response bodies extend this so disclaimers never get dropped."""
    disclaimer: str = Field(..., description="RUO disclaimer, verbatim from src/__init__.py")
    caveat: str = Field(..., description="AUROC caveat, verbatim from src/__init__.py")
    provenance: Provenance
    honesty_gate: HonestyGateReport
    evidence: list[EvidenceRecord] = Field(default_factory=list)


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


class ScreeningFinding(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    location_bbox_normalized: list[float] | None = Field(
        default=None,
        description="[x0, y0, x1, y1] in [0,1] frame coordinates, or None if not localized",
    )


class ScreeningResponse(ApiEnvelope):
    laterality: Literal["L", "R", "U"]
    view: Literal["CC", "MLO", "U"]
    orientation_flipped: bool
    breast_mask_coverage: float = Field(..., ge=0.0, le=1.0)
    findings: list[ScreeningFinding] = Field(default_factory=list)
    overall_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Overall malignancy suspicion score; None if model not wired.",
    )
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="L3 screening arbiter output (recall vs. routine follow-up).",
    )


# --------------------------------------------------------------------------- #
# /v1/biopsy/analyze


class BiopsyRequest(BaseModel):
    wsi_url: HttpUrl | None = None
    wsi_bytes_b64: str | None = None
    report_text: str | None = Field(
        default=None,
        description="Free-text pathology report; TxGemma will read this alongside the WSI.",
    )
    patient_id_hash: str | None = Field(default=None, min_length=64, max_length=64)


class BiopsyReceptorPanel(BaseModel):
    """Standard breast biopsy receptor panel."""
    er_positive: bool | None = None
    pr_positive: bool | None = None
    her2_status: Literal["negative", "equivocal", "positive"] | None = None
    ki67_percent: float | None = Field(default=None, ge=0.0, le=100.0)


class BiopsyResponse(ApiEnvelope):
    subtype_prediction: str | None = Field(
        default=None,
        description="One of: DCIS, IDC, ILC, mucinous, tubular, other. None if model not wired.",
    )
    receptor_panel: BiopsyReceptorPanel = Field(default_factory=BiopsyReceptorPanel)
    grade: int | None = Field(default=None, ge=1, le=3, description="Nottingham grade 1-3")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="L3 biopsy arbiter output (proceed to core-needle biopsy vs. follow-up).",
    )


# --------------------------------------------------------------------------- #
# /v1/therapy/reason


class TherapyPatientContext(BaseModel):
    age: int | None = Field(default=None, ge=18, le=120)
    menopausal_status: Literal["pre", "peri", "post", "unknown"] | None = None
    prior_therapies: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    genomic_markers: dict[str, Any] = Field(default_factory=dict)


class TherapyRequest(BaseModel):
    biopsy_output: BiopsyResponse | None = Field(
        default=None,
        description="If provided, we skip re-running biopsy analysis and use this directly.",
    )
    patient_context: TherapyPatientContext = Field(default_factory=TherapyPatientContext)


class TherapyOption(BaseModel):
    regimen: str
    line_of_therapy: int = Field(..., ge=1)
    rationale: str
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)


class TherapyResponse(ApiEnvelope):
    recommended_options: list[TherapyOption] = Field(default_factory=list)
    not_recommended: list[TherapyOption] = Field(default_factory=list)
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="L3 therapy arbiter output (escalate to neoadjuvant chemo vs. surgery-first).",
    )


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


# --------------------------------------------------------------------------- #
# /v1/case/full


class FullCaseRequest(BaseModel):
    screening_input: ScreeningRequest | None = None
    biopsy_input: BiopsyRequest | None = None
    therapy_context: TherapyPatientContext = Field(default_factory=TherapyPatientContext)


class FullCaseResponse(ApiEnvelope):
    screening: ScreeningResponse | None = None
    biopsy: BiopsyResponse | None = None
    therapy: TherapyResponse | None = None
    elo_ranked_hypotheses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Co-Scientist Elo tournament output over all stage hypotheses.",
    )


# --------------------------------------------------------------------------- #
# /health


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    disclaimer: str
    caveat: str
    endpoints: list[str]
    models_loaded: dict[str, ModelState]
