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
    LOADED_MEDSIGLIP = "loaded_medsiglip"  # HAI-DEF MedSigLIP-448 inference (medical but off-label for mammography)
    LOADED_BIOPSY_PROBE = "loaded_biopsy_probe"  # L4b MedSigLIP embed + synthetic linear probe (RUO, off-label)
    LOADED_MONAI_DETECTOR = "loaded_monai_detector"  # L4a MONAI detector with trained weights (unreachable until weights ship)
    PROXY_MONAI_HEURISTIC = "proxy_monai_heuristic"  # L4a MONAI mask-gradient heuristic when weights unavailable
    PROXY_LUNG_HEURISTIC = "proxy_lung_heuristic"  # NSCLC HU-threshold + CC blobs (LIDC-IDRI) — not a trained detector
    LOADED_LUNA16_RETINANET = "loaded_luna16_retinanet"  # v0.3.0 MONAI Model Zoo lung_nodule_ct_detection@0.6.9 (LUNA16-trained)
    PROXY_RULES_LITE = "proxy_rules_lite"  # L4c NCCN-lite rules fallback when TxGemma gated
    LOADED_TXGEMMA = "loaded_txgemma"  # L4c HAI-DEF TxGemma inference (never reachable under current token)
    TEMPLATE = "template"             # L3 arbiter JSON templates loaded from disk (n_training=0)
    PROXY_REGEX_V0 = "proxy_regex_v0" # v0.2.1 pathology-report regex parser (stateless code, always available)
    PROXY_CO_SCIENTIST = "proxy_co_scientist"  # L5 orchestrator: literature-derived Elo tournament (deterministic scoring)


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


class GateReport(BaseModel):
    """Structured preflight result for a gated repo (HAI-DEF or similar).

    Populated on `Provenance.gate_report` whenever the endpoint attempted a
    live gated-model call. Absent (`None`) when no gated preflight was run
    (e.g. placeholder-only responses, or when the endpoint is fully served
    from a proxy / heuristic path that never touched the gated repo).

    Field semantics match `oncology_arbiter.models.hai_def.GateReport`. This
    schema does NOT mirror the runtime dataclass field-for-field to avoid
    accidental drift — instead, the API-layer factory in `app.py` converts
    the dataclass into this pydantic block. That way, changes to internal
    fields (e.g. adding retry_after) can happen without a public schema bump.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    repo_id: str = Field(..., description="Gated repo id preflighted, e.g. 'google/medsiglip-448'")
    access_level: Literal["allowed", "forbidden", "unauthenticated", "unknown"] = Field(
        ..., description="AccessLevel returned by check_hai_def_access(): allowed / forbidden / unauthenticated / unknown",
    )
    status_code: int | None = Field(
        default=None,
        description="Underlying HTTP status from HF hub, if the preflight completed. None on transport failure.",
    )
    reason: str = Field(
        ..., description="Human-readable one-line reason, e.g. 'accept terms at huggingface.co/google/medsiglip-448'",
    )
    has_token: bool = Field(..., description="True if an HF token was discovered (env or ~/.cache/huggingface/token)")
    allowed: bool = Field(..., description="Convenience: access_level == 'allowed'")


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
    gate_report: GateReport | None = Field(
        default=None,
        description=(
            "Structured HAI-DEF preflight result for the gated model that this endpoint "
            "attempted. None when no gated preflight was performed (placeholder / pure "
            "proxy path). Carries repo_id, access_level, status_code, reason, has_token. "
            "Consumers should surface `access_level` alongside `model_state=GATED`."
        ),
    )


class ApiEnvelope(BaseModel):
    """All response bodies extend this so disclaimers never get dropped."""
    disclaimer: str = Field(..., description="RUO disclaimer, verbatim from src/__init__.py")
    caveat: str = Field(..., description="AUROC caveat, verbatim from src/__init__.py")
    provenance: Provenance
    honesty_gate: HonestyGateReport
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal honesty warnings for this response — e.g. "
            "`medsiglip_gated:<level>:<reason>`, out-of-distribution warnings, "
            "or proxy_siglip fallback notes. Callers MUST surface these."
        ),
    )


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
    """Standard breast biopsy receptor panel.

    ``parse_state`` (v0.2.1) surfaces per-field provenance to the UI so the
    clinician can see whether each value came from a confident regex match,
    an ambiguous mention that needs review, or was never mentioned in the
    report at all. When the UI submits an override, the state becomes
    ``user_supplied``.
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
        description="Per-field provenance from the report parser (proxy_regex_v0). "
                    "Absent when biopsy analysis did not run a text parser.",
    )


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
    strict: bool = Field(
        default=False,
        description="If true, rules-lite branch runs strict input validation "
                    "and returns HTTP 400 on receptor_status / grade / stage / "
                    "menopausal_status drift. Default false (permissive).",
    )


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


# --------------------------------------------------------------------------- #
# NSCLC-specific inputs / outputs
#
# `NsclcCTInput.series_dir` points at a LIDC-IDRI CT series directory on the
# server (e.g. `/workspace/lidc_cohort/lidc_idri/LIDC-IDRI-0001/<StudyUID>/CT_<SeriesUID>`).
# Real pipeline execution is gated behind the `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1`
# env var so untrusted deployments never trust a client-controlled filesystem
# path. When gated off, the branch falls back to shape-only placeholder.


class NsclcCTInput(BaseModel):
    """Point at a CT series on disk for the real lung heuristic + NCCN rules.

    Only honored when `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` is set on the server;
    otherwise ignored to avoid client-controlled path traversal in shared
    deployments.
    """
    model_config = ConfigDict(extra="forbid")
    series_dir: str = Field(..., description="Absolute path to a CT_<SeriesUID> directory")
    patient_id: str | None = Field(default=None, description="LIDC patient id if known")
    top_n: int = Field(default=10, ge=1, le=100, description="Max candidate blobs to summarize")


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
    # Provenance
    series_dir: str | None = None
    n_slices: int | None = None
    read_seconds: float | None = None
    heuristic_seconds: float | None = None


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
    # NSCLC track: point at a CT series on disk (LIDC-IDRI layout). Real
    # pipeline is only invoked when ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1 is
    # set on the server; otherwise the request falls back to shape-only.
    nsclc_ct_input: "NsclcCTInput | None" = None


class FullCaseResponse(ApiEnvelope):
    screening: ScreeningResponse | None = None
    biopsy: BiopsyResponse | None = None
    therapy: TherapyResponse | None = None
    nsclc: "NsclcResponse | None" = None
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
    # cancers advertises which cancer tracks the current build exposes via
    # `/v1/case/full?cancer=<name>`. Each entry MUST include `state`
    # (mirror of ModelState) and `endpoints` (list of sub-routes wired for
    # that cancer). The SPA reads this to decide which cancer-selector
    # options to enable and which panels to render.
    # Untyped dict so we can bolt on extra metadata (e.g. `case_full: bool`)
    # per cancer without a schema migration; the SPA only reads the keys it
    # knows.
    cancers: dict[str, dict] = {}


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
