"""Schemas for /v1/therapy/* and related triage vectors."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .biopsy import BiopsyReceptorPanel, BiopsyResponse
from .common import (
    THERAPY_PROGNOSIS_MODEL_NAME,
    THERAPY_PROGNOSIS_UNCERTAINTY,
    ApiEnvelope,
    ArbiterScore,
    EvidenceRecord,
    StageReceipt,
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
    legacy_therapy_benefit_model_name: Literal["therapy_benefit_metabric_v1"] | None = Field(
        default=None,
        deprecated=True,
        exclude=True,
        description=(
            "Deprecated input alias accepted for one release and remapped to "
            "therapy_prognosis_metabric_v1. It is never emitted in a response."
        ),
    )
    biopsy_output: BiopsyResponse | None = Field(
        default=None,
        description="If provided, we skip re-running biopsy analysis and use this directly.",
    )
    patient_context: TherapyPatientContext = Field(default_factory=TherapyPatientContext)
    # v0.2.1: user-confirmed receptor panel from the frontend Confirm gate.
    # When present, this overrides whatever is in biopsy_output.receptor_panel
    # for the purpose of branch selection. This is the honesty contract for
    # the tumor-board demo: parser output is a *suggestion*, the pathologist
    # confirms (or corrects) before therapy runs.
    receptors_override: BiopsyReceptorPanel | None = Field(
        default=None,
        description="User-confirmed receptor panel. If provided, replaces "
                    "biopsy_output.receptor_panel for branch selection. "
                    "Frontend sends this after the Confirm gate.",
    )
    # v0.2: opt in to strict input validation in the rules-lite branch.
    # When true and the rules-lite fallback fires, receptor_status / grade /
    # stage / menopausal_status are validated and 400 is returned on drift.
    # Default false to preserve the existing wire contract.
    strict: bool = Field(default=False, deprecated=True)
    disease: str = Field(default="breast cancer", min_length=1, max_length=128)
    cancer_type: str = Field(default="breast", min_length=1, max_length=64)
    mutations: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    germline_mutations: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    ranked_drugs: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    therapy_features: dict[str, Any] | None = Field(
        default=None,
        description="Explicit patient features for the research triage decomposition.",
    )



class TherapyOption(BaseModel):
    regimen: str
    line_of_therapy: int = Field(..., ge=1)
    rationale: str
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)



class TherapyResponse(ApiEnvelope):
    pipeline_status: Literal["complete", "partial_failure", "failed_required_stage"] = "complete"
    stage_receipts: list[StageReceipt] = Field(default_factory=list)
    therapy_bridge: dict[str, Any] | None = None
    therapy_triage: ArbiterScore | None = None
    recommended_options: list[TherapyOption] = Field(default_factory=list)
    not_recommended: list[TherapyOption] = Field(default_factory=list)
    prognostic_model_name: Literal["therapy_prognosis_metabric_v1"] = (
        THERAPY_PROGNOSIS_MODEL_NAME
    )
    prognostic_model_executed: bool = Field(
        default=False,
        description=(
            "False until a complete METABRIC prognosis feature vector is wired "
            "to this therapy route; prevents the model name from implying inference."
        ),
    )
    prognostic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Null unless the canonical prognosis model actually executed.",
    )
    prognostic_uncertainty: str = THERAPY_PROGNOSIS_UNCERTAINTY
    arbiter_score: ArbiterScore | None = Field(
        default=None,
        description="L3 therapy arbiter output (escalate to neoadjuvant chemo vs. surgery-first).",
    )
    # v0.2: rules-lite fallback pins the on-disk ruleset via SHA-256 so
    # auditors comparing two runs can detect silent guideline drift.
    # Populated ONLY when model_state == PROXY_RULES_LITE; None otherwise.
    rules_sha256: str | None = Field(
        default=None,
        description="SHA-256 of therapy_rules_v0.json used to serve this call. "
                    "None when model_state != proxy_rules_lite.",
    )
    rules_model_id: str | None = Field(
        default=None,
        description="model_id inside the rules JSON (e.g. 'nccn-lite-v0'). "
                    "None when model_state != proxy_rules_lite.",
    )
    branch_id: str | None = Field(
        default=None,
        description="Which NCCN-lite branch the input landed in "
                    "(dcis | metastatic | her2_positive | triple_negative | "
                    "hr_positive_her2_negative | fallthrough). "
                    "None when model_state != proxy_rules_lite.",
    )



class PatientMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    gene: str = Field(..., min_length=1, max_length=32)
    hgvs_p: str | None = Field(default=None, max_length=128)
    hgvs_c: str | None = Field(default=None, max_length=128)
    chrom: str | None = Field(default=None, max_length=16)
    pos: int | None = Field(default=None, ge=1)
    ref: str | None = Field(default=None, max_length=512)
    alt: str | None = Field(default=None, max_length=512)



class RankedDrug(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)
    name: str = Field(..., min_length=1, max_length=128)
    moa: str | None = Field(default=None, max_length=256)
    therapy_class: str | None = Field(default=None, max_length=128)
    efficacy_score: float | None = Field(default=None, ge=0.0, le=1.0)



class TherapyTriageFeatures(BaseModel):
    """Explicit complete inputs for the illustrative therapy triage decomposition."""

    histology: Literal[
        "invasive_ductal",
        "invasive_lobular",
        "ductal_carcinoma_in_situ",
        "lobular_carcinoma_in_situ",
        "OTHER_OR_UNCLEAR",
    ] | None = None
    grade: Literal[1, 2, 3] | None = None
    er_positive: bool | None = None
    pr_positive: bool | None = None
    her2_positive: bool | None = None
    ki67_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    tumor_size_mm: float | None = Field(default=None, gt=0.0, le=500.0)
    lymph_nodes_pos: int | None = Field(default=None, ge=0, le=100)
    brca_pathogenic: bool | None = None
    age_years: float | None = Field(default=None, ge=18.0, le=120.0)
