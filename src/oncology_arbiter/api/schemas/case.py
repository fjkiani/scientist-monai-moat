"""Schemas for /v1/case/full and NSCLC specialist blocks."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .biopsy import (
    BiopsyReceptorPanel,
    BiopsyRequest,
    BiopsyResponse,
    BreastDssPrognosis,
    ReportParseBlock,
)
from .common import ApiEnvelope, ArbiterScore, ModelState, StageReceipt
from .screening import ScreeningRequest, ScreeningResponse
from .therapy import (
    PatientMutation,
    RankedDrug,
    TherapyPatientContext,
    TherapyResponse,
    TherapyTriageFeatures,
)



# --------------------------------------------------------------------------- #
# /v1/case/full


# --------------------------------------------------------------------------- #
# NSCLC-specific inputs / outputs
#
# NSCLC CT input by client-supplied server path (`NsclcCTInput.series_dir`,
# gated behind ONCOLOGY_ARBITER_ALLOW_SERIES_DIR) is RETIRED. Production NSCLC
# consumes a verified case-storage `case_id` only, so the detector runs on an
# uploaded, hash-receipted series rather than a path the caller chose.


class NsclcCandidate(BaseModel):
    """One nodule-candidate blob surfaced by the lung heuristic."""
    label: int
    voxel_count: int
    diameter_mm: float
    mean_hu: float
    centroid_zyx_vox: tuple[float, float, float]



class NsclcTherapyOption(BaseModel):
    """One NCCN-lite therapy card returned by the proxy rules."""
    name: str
    category: str
    citation_url: str
    rationale: str
    nccn_section: str



class Luna16Detection(BaseModel):
    """One 3D bounding box returned by the LUNA16 RetinaNet detector.

    Coordinates are in world millimeters (voxel index × spacing_mm).
    """
    center_z_mm: float
    center_y_mm: float
    center_x_mm: float
    width_mm: float
    height_mm: float
    depth_mm: float
    diameter_mm: float
    score: float
    # v0.3.0-alpha demo (Step 5): true when the detection sits inside the
    # z-range where lung parenchyma is visible on the source series. For
    # TCGA-24-1423 (a CAP CT), that range is inst 13-42 / z=-100 to -245mm.
    # Detections outside the chest slab would be out-of-domain for a
    # LUNA16-trained detector (LIDC-IDRI = chest-only). Default true so
    # existing NSCLC callers (LIDC-IDRI single-body-part input) are
    # unaffected.
    in_domain: bool = True
    # v0.4.0-alpha Path C (PLAN §1D + §7 bullet 5): stricter refinement on
    # top of `in_domain`. `in_domain` is z-range-based (chest slab); this is
    # HU-based (Otsu-thresholded lung parenchyma mask, HU ∈ [-1000, -400]).
    # A detection can have `in_domain=True` but `in_lung_parenchyma=False`
    # if it sits in the chest slab but outside lung tissue (skin, chest wall,
    # CT-table interface). Regression target: TCGA-24-1423 top-1 at
    # (z=-238.74, y=313.94, x=188.31)mm score=0.8962 must flip to False.
    #
    # Default False (fail-safe): when Path C did NOT run (env flag off,
    # or an exception during mask build) the frontend should treat the
    # detection as UNVERIFIED. Setting the default True would falsely
    # promote every unfiltered detection into "confirmed in parenchyma",
    # which is the exact wrong bias for a screening/triage UX.
    in_lung_parenchyma: bool = False



class Luna16DetectionBlock(BaseModel):
    """LUNA16 RetinaNet output block on the NSCLC response.

    Present only when ONCOLOGY_ARBITER_ENABLE_LUNA16_RETINANET=1 AND the
    real pipeline branch fires (not the placeholder). Absent otherwise so
    downstream clients can `if resp.nsclc.luna16 is not None:` gate on
    real inference.
    """
    bundle_version: str = Field(..., description="MONAI bundle version, e.g. '0.6.9'")
    n_detections: int
    top_score: float
    detections: list[Luna16Detection] = Field(default_factory=list)
    inference_seconds: float
    preprocessing_summary: dict[str, Any]
    # v0.3.0-alpha demo (Step 5): world-mm z-range that defines the "chest
    # slab" for a whole-body CAP scan. All detections inside this range
    # carry `in_domain=true`; detections outside carry `in_domain=false`.
    # None for classic LIDC-IDRI single-body-part input (full volume is
    # in-domain by construction).
    in_domain_z_range_mm: list[float] | None = Field(
        default=None,
        description="[z_min_mm, z_max_mm] world-frame bounds of the in-domain chest slab. "
                    "Only set when the caller filtered a whole-body CT to a lung-only slab.",
    )



class NsclcResponse(BaseModel):
    """Envelope for the real LIDC-IDRI + NCCN-lite path (or its placeholder)."""
    model_config = ConfigDict(extra="forbid")
    model_state: ModelState
    model_name: str
    warnings: list[str] = Field(default_factory=list)
    # Lung heuristic block
    lung_voxel_fraction: float | None = None
    n_candidates_total: int | None = None
    n_candidates_kept: int | None = None
    max_diameter_mm: float | None = None
    candidates: list[NsclcCandidate] = Field(default_factory=list)
    # v0.3.0: LUNA16 RetinaNet block (present when detector is enabled).
    luna16: Luna16DetectionBlock | None = None
    # Arbiter block
    risk_score: float | None = None
    risk_bucket: str | None = None
    driving_feature: str | None = None
    logit: float | None = None
    # Therapy block
    therapy_recommended: list[NsclcTherapyOption] = Field(default_factory=list)
    therapy_not_recommended: list[NsclcTherapyOption] = Field(default_factory=list)
    # Provenance. n_slices comes from the verified case manifest, not from a
    # local read; series_dir/read_seconds/heuristic_seconds belonged to the
    # retired local-CT + HU-heuristic path and were never set on this route.
    n_slices: int | None = None
    # v0.4.1: fine-tuned Bio_ClinicalBERT report parser (Modal-backed).
    # Present only when CLINICALBERT_BACKEND=modal AND the request
    # carried biopsy_input.report_text. Shape mirrors the Modal
    # `/clinicalbert-parse` response's `parsed` field: each key is
    # an entity type (KRAS, EGFR, ALK, ROS1, PD_L1_TPS, TMB, MSI,
    # HER2_AMP, BRAF, MET, plus the breast entities) and each value is
    # a dict with `surface`, `start_tok`, `end_tok`, `value` (canonical).
    parsed_report: dict[str, Any] | None = Field(
        default=None,
        description="Fine-tuned Bio_ClinicalBERT parse of biopsy_input.report_text "
                    "(populated when CLINICALBERT_BACKEND=modal AND report_text set). "
                    "Keys are entity types (KRAS, EGFR, ALK, ROS1, PD_L1_TPS, TMB, MSI, "
                    "HER2_AMP, BRAF, MET, etc.); values carry surface + canonical value. "
                    "Training corpus: SYNTHETIC-v0.3.1 (breast + NSCLC synthetic). RUO.",
    )
    parsed_report_provenance: dict[str, Any] | None = Field(
        default=None,
        description="Meta about the ClinicalBERT parse: provenance, training_seed, "
                    "test_micro_f1, seconds, n_tokens, app_version.",
    )



class FullCaseRequest(BaseModel):
    screening_input: ScreeningRequest | None = None
    biopsy_input: BiopsyRequest | None = None
    therapy_context: TherapyPatientContext = Field(default_factory=TherapyPatientContext)
    # v0.2.1: user-confirmed receptor panel from the frontend Case View
    # Confirm gate. When present, it replaces whatever the biopsy sub-call
    # extracted from report_text — this is what the pathologist actually
    # believes about the case, and it drives the therapy branch.
    receptors_confirmed: BiopsyReceptorPanel | None = Field(
        default=None,
        description="User-confirmed receptor panel from Case View Confirm gate. "
                    "Overrides biopsy receptor extraction for therapy branch selection.",
    )
    case_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{16}$",
        description="Verified case-storage identifier for production CT ingress.",
    )
    therapy_features: "TherapyTriageFeatures | None" = None
    mutations: list["PatientMutation"] = Field(default_factory=list, max_length=64)
    germline_mutations: list["PatientMutation"] = Field(default_factory=list, max_length=64)
    ranked_drugs: list["RankedDrug"] = Field(default_factory=list, max_length=100)
    run_co_scientist: bool = True



class FullCaseResponse(ApiEnvelope):
    pipeline_status: Literal["complete", "partial_failure", "failed_required_stage"] = "complete"
    cancer: Literal["breast", "nsclc", "hgsoc"] | None = None
    stage_receipts: list[StageReceipt] = Field(default_factory=list)
    ovarian_arbiter: dict[str, Any] | None = None
    case_manifest: dict[str, Any] | None = None
    report_parse: ReportParseBlock | None = None
    phikon_embedding: dict[str, Any] | None = None
    luna16_detection: dict[str, Any] | None = None
    dss_prognosis: BreastDssPrognosis | None = None
    therapy_bridge: dict[str, Any] | None = None
    therapy_triage: ArbiterScore | None = None
    co_scientist: dict[str, Any] | None = None
    screening: ScreeningResponse | None = None
    biopsy: BiopsyResponse | None = None
    therapy: TherapyResponse | None = None
    nsclc: "NsclcResponse | None" = None
    elo_ranked_hypotheses: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "MedGemma pairwise Elo ranking from execute_stage when Co-Scientist "
            "completed. Empty when synthesis was not requested or did not "
            "complete. Never filled from /v1/elo/rank caller modifiers."
        ),
    )
