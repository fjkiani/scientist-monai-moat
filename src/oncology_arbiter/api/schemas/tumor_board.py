"""Schemas for dynamic tumor board and AK MBD4 bundle contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .biopsy import (
    BreastDssFeatures,
    BreastDssPrognosis,
    ReportParseBlock,
)
from .common import ApiEnvelope, ArbiterScore, StageReceipt
from .therapy import PatientMutation, RankedDrug, TherapyTriageFeatures



class DynamicTumorBoardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cancer: Literal["breast", "hgsoc"]
    case_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    report_text: str | None = Field(default=None, min_length=1, max_length=20_000)
    pathology_image_b64: str | None = None
    dss_features: BreastDssFeatures | None = None
    therapy_features: TherapyTriageFeatures | None = None
    mutations: list[PatientMutation] = Field(default_factory=list, max_length=64)
    germline_mutations: list[PatientMutation] = Field(default_factory=list, max_length=64)
    ranked_drugs: list[RankedDrug] = Field(default_factory=list, max_length=100)
    clinical_baseline: dict[str, Any] = Field(default_factory=dict)
    run_co_scientist: bool = True



class DynamicTumorBoardResponse(ApiEnvelope):
    pipeline_status: Literal["complete", "partial_failure", "failed_required_stage"]
    cancer: Literal["breast", "hgsoc"]
    stage_receipts: list[StageReceipt] = Field(default_factory=list)
    ovarian_arbiter: dict[str, Any] | None = None
    report_parse: ReportParseBlock | None = None
    phikon_embedding: dict[str, Any] | None = None
    case_manifest: dict[str, Any] | None = None
    dss_prognosis: BreastDssPrognosis | None = None
    therapy_bridge: dict[str, Any] | None = None
    therapy_triage: ArbiterScore | None = None
    co_scientist: dict[str, Any] | None = None



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# v0.4.0-alpha — AK MBD4-LOF tumor board bundle contract (PLAN §2A)
#
# Consumes the `tumor_board.v3.multimodal-with-manuscript-claims` contract
# shipped by crispro-backend-v2 branch `fix/mbd4-atr-strong-tier` at HEAD
# `bfd6d11fc872c11a13365b0682cea776a136c7f3` (PR #11).  Field names, enums,
# and defaults mirror
#   api/services/synthetic_lethality/v3/multimodal/models.py
# on the backend side — additive only, no renames, no removals.
#
# Manuscript SHA of record: d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1.
# Datasets_used per bundle provenance: ["GDSC2", "DepMap 24Q2"].
# Reference clinical case: AK (real patient; rendered with redacted
# patient_id="MBD4-LOF-DEMO-01" per backend v2 HIPAAPIIMiddleware discipline).


class ModalityEvidence(BaseModel):
    """One modality row (or auxiliary_evidence entry) on an EvidenceMatrixRow.

    Mirrors backend v2 `EvidenceRow` / `AuxiliaryEvidence` (post-PR #11 additions).
    Every field is optional except `modality`, `status`, and `origin_system`
    because different modalities carry different fields (e.g. `stress_test`
    entries carry `stratifier` + `effect_size`; `expression_association`
    entries carry `pmids` + `summary` but no numeric p).
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    modality: Literal[
        "crispr_dependency",
        "expression_association",
        "prism_pharmacologic",
        "gdsc_pharmacologic",
        "clinical",
        "stress_test",
        "axis_partner",
        "falsification_arm",
        "depmap_pool",
        "manuscript",
    ]
    status: Literal["positive", "missing", "mixed", "negative"]
    delta_dep: float | None = None
    p_value: float | None = None
    fdr: float | None = None
    effect_size: float | None = None
    n_mut: int | None = None
    n_wt: int | None = None
    delta_auc: float | None = None
    delta_ln_ic50: float | None = None
    drug_screen_dataset: str | None = None
    stratifier: str | None = None
    metric: str | None = None
    summary: str | None = None
    pmids: list[str] = Field(default_factory=list)
    is_confound_flagged: bool = False
    notes: str | None = None
    origin_system: Literal[
        "live_crispr",
        "live_gdsc",
        "live_prism",
        "manuscript_receipt",
    ]



class EvidenceMatrixRow(BaseModel):
    """One row of the 6-axis synthetic-lethality evidence matrix.

    Post-PR #11 fields (backend v2 `bfd6d11f`):
      * manuscript_claim_type: enum tag driving falsification badge + alpha-stat callout
      * falsification_narrative: 500-750 char verbatim narrative when claim=falsified_mechanism
      * auxiliary_evidence: [] for most rows; ATR row has 6 entries (4 stress + 1 axis_partner + 1 falsification_arm)
      * tier_fusion_rule_id / _detail: which branch of the fusion ladder produced the tier
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    axis: Literal[
        "cytidine_analogs",
        "atr_wee1",
        "parp_inhibitors",
        "immunotherapy",
        "pkmyt1",
        "wrn",
    ]
    axis_label: str
    mechanism: str
    crispr: ModalityEvidence
    expression: ModalityEvidence
    prism: ModalityEvidence
    gdsc: ModalityEvidence
    depmap_pool: ModalityEvidence | None = None
    manuscript: ModalityEvidence | None = None
    clinical: ModalityEvidence | None = None
    recommendation_tier: str = Field(
        ...,
        description=(
            "Human-readable tier, e.g. 'Strong candidate dependency axis', "
            "'Mechanistic candidate only', 'Validated SL therapeutic lever', "
            "'Not supported / negative'."
        ),
    )
    manuscript_claim_type: Literal[
        "validated_benchmark",
        "primary_new_candidate_axis",
        "falsified_mechanism",
    ] | None = Field(
        default=None,
        description=(
            "PR #11 field. Drives the frontend MechanismFalsifiedBadge "
            "(falsified_mechanism) and AlphaStatCallout eligibility "
            "(primary_new_candidate_axis)."
        ),
    )
    falsification_narrative: str | None = Field(
        default=None,
        description=(
            "PR #11 field. Verbatim narrative (~500-750 chars) shown in "
            "MechanismFalsifiedBadge expanded panel when claim="
            "falsified_mechanism. None otherwise."
        ),
    )
    auxiliary_evidence: list[ModalityEvidence] = Field(
        default_factory=list,
        description=(
            "PR #11 field. Stress tests, axis partners, and falsification "
            "arms attached to this row. The ATR row on the AK bundle has "
            "6 entries: 4 stress_test (MSI_purge, TP53_mutant_only, "
            "leave_one_out_LOF, non_bowel_lineage), 1 axis_partner "
            "(MBD4_LOF_vs_WT for adavosertib), 1 falsification_arm "
            "(PARP1_expression_LOF_vs_comparator)."
        ),
    )
    tier_fusion_rule_id: str | None = Field(
        default=None,
        description=(
            "PR #11 field. Machine identifier for the fusion-ladder branch "
            "that produced this tier, e.g. "
            "'base_strong_crispr_or_pharma_multi_positive', "
            "'base_fallback_mechanistic_candidate_else_branch'."
        ),
    )
    tier_fusion_rule_detail: str | None = Field(
        default=None,
        description=(
            "PR #11 field. Longer-form 'Why this tier?' explanation for "
            "ScoringBreakdown 'Why this tier?' expandable UI."
        ),
    )
    bridge_recommended_drugs_policy: str | None = None
    overall_evidence_level: str | None = None
    interpretation: str | None = None



class EvidenceMatrix(BaseModel):
    """The 6-axis evidence matrix root."""
    query_gene: str
    cancer_type: str
    depmap_release: str
    rows: list[EvidenceMatrixRow]



class RecommendedDrug(BaseModel):
    """One drug row in the tumor board bundle's `recommended_drugs` list.

    Note per bundle `recommended_drugs_provenance_note`: this projection is
    materialized by the payload-build step, not by any single existing
    backend drug-bridge function. Drug names come from the canonical audit
    JSON, not invented.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    drug_name: str
    target: str
    role: Literal[
        "validated_benchmark",
        "axis_partner",
        "alternate",
        "lead",
        "falsified_on_transcriptional_basis",
    ]
    axis: str
    axis_label: str
    recommendation_tier: str
    tier_rank: int
    manuscript_claim_type: str | None = None
    bridge_recommended_drugs_policy: str
    surface_status: Literal[
        "RECOMMENDED",
        "NOT_RECOMMENDED_ON_THIS_MECHANISM",
    ]



class SlProvenance(BaseModel):
    """Bundle provenance sub-block.

    Every AK bundle carries:
      * manuscript_repo_sha_at_audit == 'd33f6403fb11b314c86fa74d9c56e07b7ac3d7b1'
      * backend_branch == 'fix/mbd4-atr-strong-tier'
      * backend_head_sha == 'bfd6d11fc872c11a13365b0682cea776a136c7f3'
      * datasets_used == ['GDSC2', 'DepMap 24Q2']
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    evidence_matrix: EvidenceMatrix
    manuscript_repo_sha_at_audit: str
    backend_branch: str
    backend_head_sha: str
    datasets_used: list[str]
    statistical_test: str = "Mann-Whitney U one-sided (alternative=less)"
    effect_size_metric: str = "Cohen's d (pooled)"
    audit_artifact: str | None = None



class SyntheticLethalityBundle(BaseModel):
    """The synthetic_lethality sub-envelope of a TumorBoardBundle."""
    model_config = ConfigDict(str_strip_whitespace=True)
    patient_id: str
    disease: str
    query_gene: str
    recommended_drugs: list[RecommendedDrug] = Field(
        ...,
        min_length=1,
        description=(
            "At least one recommended drug required. Bundles with zero drugs "
            "have no SL surface to render and must be rejected upstream."
        ),
    )
    provenance: SlProvenance
    recommended_drugs_provenance_note: str | None = None



class PrimaryAlteration(BaseModel):
    gene: str
    alteration_type: str
    germline_or_somatic: str = "unspecified_in_demo"
    vaf: float | None = None
    zygosity: str = "unspecified_in_demo"



class Diagnosis(BaseModel):
    primary_site: str
    histology: str
    stage: str = "demo"
    msi_status: str = "unknown_in_demo"
    tp53_status: str = "assumed_mutant_per_HGSOC_prior"



class LineagePriors(BaseModel):
    tp53_mutant_prevalence_in_hgsoc_pct_lit: int | None = None
    tp53_priors_source: str | None = None



class PatientContext(BaseModel):
    primary_alteration: PrimaryAlteration
    diagnosis: Diagnosis
    lineage_priors: LineagePriors | None = None



class TumorBoardBundle(BaseModel):
    """AK-style tumor board bundle — the full v3-multimodal contract.

    Consumed by:
      * `GET /v1/demo/samples/ak_mbd4_lof_case` (served by existing
        /v1/demo/samples/{kind} route, which validates `kind` is
        `[A-Za-z0-9_]+`).
      * `POST /v1/tumor_board/bundle` (new in v0.4.0; validates + persists).

    Emitted by:
      * crispro-backend-v2 `api/services/synthetic_lethality/v3/multimodal/`
        (matrix_builder + modality_fuser + literature_receipts).
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    contract_version: Literal[
        "tumor_board.v3.multimodal-with-manuscript-claims"
    ]
    patient_id: str
    generated_at: str
    demo_disclaimer: str | None = None
    patient_context: PatientContext
    synthetic_lethality: SyntheticLethalityBundle



class TumorBoardBundleResponse(ApiEnvelope):
    """Envelope for POST /v1/tumor_board/bundle validation echo."""
    bundle: TumorBoardBundle
    bundle_sha256: str = Field(
        ...,
        description="SHA-256 of the JSON payload for audit-ledger reproducibility.",
    )
    persisted_path: str | None = Field(
        default=None,
        description=(
            "When HIPAA_MODE=false and staging is enabled, the redacted "
            "bundle is persisted to demo_samples/tumor_board/{patient_id}/. "
            "None when staging is disabled or in HIPAA mode."
        ),
    )
