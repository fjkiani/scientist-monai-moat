"""Schemas for /v1/biopsy/* and /v1/stage/biopsy/score."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .common import ApiEnvelope, ArbiterScore, StageReceipt



# --------------------------------------------------------------------------- #
# /v1/biopsy/analyze


class BreastDssFeatures(BaseModel):
    """Complete feature vector for the frozen METABRIC DSS model.

    Every field is required. Report-parser values are never silently used to
    fill missing clinical inputs because age and positive-node count are not
    reliably recoverable from a pathology report.
    """

    age: float = Field(..., ge=18.0, le=120.0)
    tumor_size_mm: float = Field(..., gt=0.0, le=500.0)
    nodes_positive: int = Field(..., ge=0, le=100)
    grade: int = Field(..., ge=1, le=3)
    er_positive: bool
    pr_positive: bool
    her2_positive: bool



class BreastDssPrognosis(BaseModel):
    """Frozen disease-specific prognostic score with auditable decomposition."""

    model_name: Literal["breast_dss_arbiter_v3_metabric"]
    disease_specific_mortality_score: float = Field(..., ge=0.0, le=1.0)
    logit: float
    term_contributions: dict[str, float]
    artifact_sha256: str = Field(..., min_length=64, max_length=64)
    n_training: int = Field(..., ge=1)
    events: int = Field(..., ge=1)
    oof_auroc: float = Field(..., ge=0.0, le=1.0)
    endpoint_label: Literal["disease-specific mortality"] = "disease-specific mortality"
    model_state: Literal["frozen"] = "frozen"
    caveats: list[str] = Field(default_factory=list)



class BiopsyRequest(BaseModel):
    wsi_url: HttpUrl | None = None
    wsi_bytes_b64: str | None = None
    report_text: str | None = Field(
        default=None,
        description="Free-text pathology report; TxGemma will read this alongside the WSI.",
    )
    dss_features: BreastDssFeatures | None = Field(
        default=None,
        description=(
            "Complete seven-feature vector for breast_dss_arbiter_v3_metabric. "
            "When absent, the endpoint does not fabricate or impute a DSS score."
        ),
    )
    patient_id_hash: str | None = Field(default=None, min_length=64, max_length=64)



class BiopsyReceptorPanel(BaseModel):
    """Standard breast biopsy receptor panel.

    ``parse_state`` surfaces per-field provenance to the UI so the clinician
    can see whether each value came from a confident ClinicalBERT span, an
    ambiguous mention that needs review, or was never mentioned in the report
    at all. When the UI submits an override, the state becomes
    ``user_supplied``. No regex parser is wired on any route.
    """
    er_positive: bool | None = None
    pr_positive: bool | None = None
    her2_status: Literal["negative", "equivocal", "positive"] | None = None
    ki67_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    parse_state: dict[
        Literal["er", "pr", "her2", "grade"],
        Literal["matched", "ambiguous", "no_match", "user_supplied"],
    ] | None = Field(
        default=None,
        description="Per-field provenance from the ClinicalBERT v0.5.2 "
                    "sliding-window pathology parser. Absent when biopsy "
                    "analysis did not run a text parser.",
    )



class ExtendedReceptorField(BaseModel):
    """One extended field parsed by the ClinicalBERT v0.5.2 pathology parser.

    Carries the extended entities beyond the receptor core:
    ki67_pct, tumor_size_mm, T/N/M stage, margin, LVI. The four core fields
    (er/pr/her2/grade) remain on ``BiopsyReceptorPanel`` for
    backwards-compatibility with existing frontends.
    """
    value: Any = Field(default=None,
        description="Canonicalised value (int/float/str/bool) or null.")
    match_state: Literal["matched", "ambiguous", "no_match"] = "no_match"
    matched_text: str | None = None
    span: tuple[int, int] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # NOTE: This field is consumed by the UI to show provenance badges.
    # `medgemma_pathology` is the 72h primary structured extraction path.
    source: Literal[
        "regex",
        "clinicalbert",
        "medgemma_pathology",
        "fused",
        "disagreement",
        "none",
    ] = "none"



class ReportParseBlock(BaseModel):
    """v0.3.0 wire block describing HOW the pathology report was parsed.

    Attached to ``BiopsyResponse`` when a report was actually parsed. The
    UI reads ``parser_id`` and ``fusion_mode`` to render the source badge,
    and ``extended_fields`` to show fields regex could not extract.
    """
    parser_id: str = Field(
        ...,
        description=(
            "Parser identifier. Currently either the 72h primary MedGemma "
            "structured JSON extractor or the legacy ClinicalBERT v0.5.2 "
            "sliding-window NER. Regex extraction is not wired."
        ),
    )
    fusion_mode: Literal["clinicalbert", "medgemma_pathology"] = "clinicalbert"
    per_field_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence per field (0..1) as reported by the parser.",
    )
    per_field_source: dict[
        str,
        Literal[
            "regex",
            "clinicalbert",
            "medgemma_pathology",
            "fused",
            "disagreement",
            "none",
        ],
    ] = Field(default_factory=dict)
    extended_fields: dict[str, ExtendedReceptorField] = Field(default_factory=dict)
    parsed_entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Canonical ClinicalBERT entity output, without regex fusion.",
    )
    n_tokens: int | None = Field(default=None, ge=0)
    n_windows: int | None = Field(default=None, ge=0)
    window_tokens: int | None = Field(default=None, ge=1)
    overlap_tokens: int | None = Field(default=None, ge=0)
    window_aggregation: str | None = None
    app_version: str | None = None
    model_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    metrics_sha256: str | None = Field(default=None, min_length=64, max_length=64)



class BiopsyResponse(ApiEnvelope):
    pipeline_status: Literal["complete", "partial_failure", "failed_required_stage"] = "complete"
    stage_receipts: list[StageReceipt] = Field(default_factory=list)
    phikon_embedding: dict[str, Any] | None = None
    subtype_prediction: str | None = Field(
        default=None,
        description=(
            "BACH probe labels: benign_or_normal, in_situ_carcinoma, or "
            "invasive_carcinoma. Legacy caller-provided pathology subtypes may "
            "still appear; broad BACH classes must not be interpreted as DCIS/IDC."
        ),
    )
    receptor_panel: BiopsyReceptorPanel = Field(default_factory=BiopsyReceptorPanel)
    grade: int | None = Field(default=None, ge=1, le=3, description="Nottingham grade 1-3")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="L3 biopsy arbiter output (proceed to core-needle biopsy vs. follow-up).",
    )
    report_parse: ReportParseBlock | None = Field(
        default=None,
        description=(
            "v0.3.0: describes which parser produced the receptor panel + any "
            "extended fields (ki67_pct, tumor_size_mm, T/N/M, margin, LVI). "
            "Absent for requests that did not include free-text report_text."
        ),
    )
    dss_prognosis: BreastDssPrognosis | None = Field(
        default=None,
        description=(
            "Frozen METABRIC disease-specific mortality prognosis. Populated "
            "only when all seven explicit dss_features were supplied; never imputed."
        ),
    )



# --------------------------------------------------------------------------- #
# /v1/stage/biopsy/score — explicit L2 biopsy arbiter (stage-biopsy capability)


class StageBiopsyScoreRequest(BaseModel):
    """Caller-supplied feature panel for biopsy_arbiter_v1 (RUO)."""

    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Explicit lesion morphology and covariates for biopsy_arbiter_v1. "
            "Calcification-only morphology without mass descriptors is rejected "
            "with HTTP 422 (unvalidated lesion stratum)."
        ),
    )
    patient_id_hash: str | None = Field(default=None, min_length=64, max_length=64)



class StageBiopsyScoreResponse(ApiEnvelope):
    arbiter_score: ArbiterScore
