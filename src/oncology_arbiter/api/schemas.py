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
    LOADED_CLINICALBERT_PARSER = "loaded_clinicalbert_parser"  # v0.3.0 Bio_ClinicalBERT fine-tuned on synthetic corpus
    FUSED_REGEX_CLINICALBERT = "fused_regex_clinicalbert"  # v0.3.0 regex ∧ ClinicalBERT fusion
    PROXY_CO_SCIENTIST = "proxy_co_scientist"  # L5 orchestrator: literature-derived Elo tournament (deterministic scoring)

    # v0.4.0-alpha additions (PLAN §2A):
    LOADED_LUNA16_REFINED = "loaded_luna16_refined"        # fjkiani-luna16-refine-v1 (LUNA16+LIDC-IDRI fine-tune, target ΔFROC@2 ≥ +5% over 0.6.9)
    LOADED_MAMMO_MONAI_V1 = "loaded_mammo_monai_v1"        # fjkiani-mammo-monai-v1 RetinaNet fine-tuned on CBIS-DDSM_1024, floor AUROC ≥ 0.85
    LOADED_TXGEMMA_MODAL = "loaded_txgemma_modal"          # HAI-DEF TxGemma-9B served via Modal (cold-start ~90s, warm 2-4s)
    LOADED_MEDSIGLIP_MODAL = "loaded_medsiglip_modal"      # HAI-DEF MedSigLIP-448 served via Modal (embedding_dim=1152, A10G)
    LOADED_AK_BUNDLE = "loaded_ak_bundle"                  # AK MBD4-LOF tumor board bundle (crispro-backend-v2 bfd6d11f, manuscript d33f6403)
    LOADED_MBD4_EVIDENCE_MATRIX = "loaded_mbd4_evidence_matrix"  # PR #11 evidence matrix (manuscript_claim_type + falsification_narrative + auxiliary_evidence)
    LOADED_SL_THERAPY_BRIDGE = "loaded_sl_therapy_bridge"  # crispro-backend-v2 api/services/sl_therapy_bridge/ mirror
    LOADED_HIPAA_REDACTOR = "loaded_hipaa_redactor"        # HIPAAPIIMiddleware log-redaction (mirror of backend v2 api/middleware/hipaa_pii.py)


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
    # v0.4.0-alpha (PLAN §2A + §7 bullet 7): Co-Scientist hostile-URL test asserts
    # that when the LLM proposes hypotheses referencing URLs the run never saw,
    # reflection.py drops the hypothesis (not just the evidence). Distinct from
    # evidence_dropped because a single hostile URL fabricates the whole hypothesis.
    hypotheses_dropped: int = Field(default=0, ge=0)


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
    # v0.4.0-alpha (PLAN §2A): when a model runs on Modal, we surface the exact
    # endpoint URL that served the request (e.g. `https://crispro-test--medsiglip-embed.modal.run`).
    # None for local / disk-loaded models. Downstream consumers can echo this in
    # the audit ledger for reproducible provenance.
    model_endpoint_url: str | None = Field(
        default=None,
        description="Modal function URL (or other remote endpoint) that served the request. None for local inference.",
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


class ExtendedReceptorField(BaseModel):
    """One field parsed by the v0.3.0 fused (regex + ClinicalBERT) parser.

    Only used for extended fields the v0.2.1 regex parser could not produce:
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
    source: Literal["regex", "clinicalbert", "fused", "disagreement", "none"] = "none"


class ReportParseBlock(BaseModel):
    """v0.3.0 wire block describing HOW the pathology report was parsed.

    Attached to ``BiopsyResponse`` when a report was actually parsed. The
    UI reads ``parser_id`` and ``fusion_mode`` to render the source badge,
    and ``extended_fields`` to show fields regex could not extract.
    """
    parser_id: str = Field(
        ...,
        description="e.g. proxy_regex_v0, clinicalbert_v1, clinicalbert_v1+regex_v0",
    )
    fusion_mode: Literal["regex", "bert", "fused"] = "regex"
    per_field_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence per field (0..1) as reported by the parser.",
    )
    per_field_source: dict[
        str,
        Literal["regex", "clinicalbert", "fused", "disagreement", "none"],
    ] = Field(default_factory=dict)
    extended_fields: dict[str, ExtendedReceptorField] = Field(
        default_factory=dict,
        description=(
            "Fields the regex parser could not produce: ki67_pct, tumor_size_mm, "
            "t_stage, n_stage, m_stage, margin, lvi. Only populated when a BERT "
            "or fused parser ran."
        ),
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
    report_parse: ReportParseBlock | None = Field(
        default=None,
        description=(
            "v0.3.0: describes which parser produced the receptor panel + any "
            "extended fields (ki67_pct, tumor_size_mm, T/N/M, margin, LVI). "
            "Absent for requests that did not include free-text report_text."
        ),
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
    # Provenance
    series_dir: str | None = None
    n_slices: int | None = None
    read_seconds: float | None = None
    heuristic_seconds: float | None = None
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


class CoScientistRunResponse(ApiEnvelope):
    """Envelope for POST /v1/co_scientist/run."""
    run_id: str
    seed: int
    n_hypotheses_generated: int
    n_evidence_pulled: int
    n_reflected_kept: int
    n_reflected_dropped: int
    n_tournament_matches: int
    top_hypothesis: dict[str, Any]
    elo_leaderboard: list[dict[str, Any]]
    completion_state: Literal["completed", "partial", "failed"]


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


# --------------------------------------------------------------------------- #
# /v1/elo/rank  (v0.3.0-alpha)


class EloDrugCandidate(BaseModel):
    """One therapy candidate submitted to the Elo tournament.

    Deliberately small: the endpoint does NOT try to be a therapy engine.
    It only ranks the candidates the caller already assembled (e.g. from
    /v1/therapy/reason recommended_options, an NCCN pocket-guide branch,
    or a manually curated list). The tournament decides the *order*.

    Fields:
      - drug_id:      stable id used as the hyp_id (e.g. "olaparib_maintenance")
      - regimen:      display name (e.g. "Olaparib 300 mg BID maintenance")
      - line:         line of therapy (1 = first-line, 2 = second-line, …)
      - confidence:   caller-supplied prior in [0,1]. If unknown, use 0.5.
      - evidence:     list of {url, quoted_text, source} — same schema as
                      EvidenceRecord. Contributes to the honesty score.
      - honesty_markers: {proxy, gated, loaded} flags carried from the
                      caller's stage envelope; used verbatim by the scorer.
    """
    model_config = ConfigDict(extra="forbid")
    drug_id: str = Field(..., min_length=1, max_length=128,
                         description="Stable id used as Elo hypothesis id.")
    regimen: str = Field(..., min_length=1, max_length=256)
    line: int = Field(..., ge=1, le=5, description="Line of therapy (1..5)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    honesty_markers: dict[str, bool] = Field(
        default_factory=dict,
        description="{'proxy': bool, 'gated': bool, 'loaded': bool} — carried "
                    "from the caller's stage envelope. Missing keys default to false.",
    )


class EloRankRequest(BaseModel):
    """POST body for /v1/elo/rank.

    Runs a Co-Scientist Elo tournament twice:
      1. baseline_ranking  — the plain tournament, scores from the candidate
                             fields alone (this is what /v1/case/full does
                             for hypotheses derived from stage envelopes).
      2. enriched_ranking  — same candidates but with `modifiers` applied
                             to bump/discount specific drugs based on
                             disease context.

    `modifiers` is a free-form dict keyed by drug_id whose values are
    scalar deltas added to that drug's Elo `_score_hypothesis` before the
    tournament runs. Deltas are unclamped so callers can either boost a
    biomarker-matched drug or explicitly stress-test a low-confidence
    one. Undocumented drug_ids in `modifiers` are surfaced as warnings
    but do not fail the request.

    `disease_context` is echoed back verbatim in the response and used to
    stamp provenance — no server-side lookup, no hidden bumps.
    """
    model_config = ConfigDict(extra="forbid")
    drugs: list[EloDrugCandidate] = Field(
        ...,
        min_length=2, max_length=32,
        description="Between 2 and 32 candidates. Fewer than 2 makes no "
                    "tournament sense; 32 caps CPU cost at 496 pairs.",
    )
    modifiers: dict[str, float] = Field(
        default_factory=dict,
        description="{drug_id: score_delta}. Deltas added to the Elo score "
                    "for the enriched tournament. Not clamped.",
    )
    disease_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Echoed verbatim on the response (e.g. cancer, stage, "
                    "HRD status, PD-L1 CPS, prior_lines). Purely provenance.",
    )
    k_factor: int = Field(default=16, ge=1, le=64,
                          description="Elo K-factor. 16 mirrors run_co_scientist.")
    seed: int = Field(default=20260703, ge=0,
                      description="RNG seed. Same seed → same ranking.")


class EloMatchRecord(BaseModel):
    """One row of the ranking-diff table.

    For each drug, we report where it sat under the baseline tournament,
    where it moved to under the enriched tournament, and the reason
    (which modifier deltas applied). This is what the SPA's
    EloRankingPanel renders as the ‘why did the ranking change’ table.
    """
    model_config = ConfigDict(extra="forbid")
    drug_id: str
    regimen: str
    line: int
    baseline_rank: int = Field(..., ge=1,
                               description="1-based rank in the baseline tournament.")
    enriched_rank: int = Field(..., ge=1,
                               description="1-based rank in the enriched tournament.")
    baseline_rating: float
    enriched_rating: float
    rank_delta: int = Field(...,
        description="baseline_rank − enriched_rank. Positive → moved up (improved).")
    rating_delta: float
    applied_modifier: float = Field(default=0.0,
        description="Score delta applied to this drug from the modifiers map "
                    "(0.0 if no modifier for this drug_id).")
    reason: str = Field(default="",
        description="Human-readable reason: 'HRD+PARP boost', 'no modifier', "
                    "'unknown drug_id in modifiers', etc.")


class EloRankedEntry(BaseModel):
    """One row of a ranked list.

    Same shape as the dict rows emitted by run_co_scientist(hypotheses), but
    with the drug metadata (regimen, line) surfaced so the SPA doesn't have
    to cross-reference by hyp_id.
    """
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(..., ge=1)
    drug_id: str
    regimen: str
    line: int
    rating: float
    wins: int = Field(..., ge=0)
    losses: int = Field(..., ge=0)
    draws: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    honesty_markers: dict[str, bool] = Field(default_factory=dict)
    n_evidence: int = Field(..., ge=0)


class EloRankResponse(ApiEnvelope):
    """Response for /v1/elo/rank.

    Carries the standard ApiEnvelope fields (disclaimer/caveat/provenance/
    honesty_gate/evidence/warnings) plus the two rankings and the diff.

    The disease_context and applied_modifiers fields are echoed verbatim
    so a downstream reader can reproduce the tournament byte-for-byte.
    """
    contract_version: Literal["v0.3.0-alpha"] = "v0.3.0-alpha"
    baseline_ranking: list[EloRankedEntry] = Field(default_factory=list)
    enriched_ranking: list[EloRankedEntry] = Field(default_factory=list)
    matches: list[EloMatchRecord] = Field(
        default_factory=list,
        description="Per-drug baseline↔enriched diff, ordered by enriched_rank.",
    )
    disease_context: dict[str, Any] = Field(default_factory=dict,
        description="Echo of the request's disease_context, verbatim.")
    applied_modifiers: dict[str, float] = Field(default_factory=dict,
        description="Echo of the request's modifiers, verbatim.")
    n_candidates: int = Field(..., ge=2, le=32,
        description="Count of drugs in the tournament (echoes request len).")


# --------------------------------------------------------------------------- #
# v0.4.0-alpha — POST /v1/co_scientist/run
#
# Surfaces the 4-phase Co-Scientist loop (generate → reflect → rank → evolve →
# rank) as a first-class endpoint, so callers can run the honesty tournament
# without going through /v1/case/full. Deterministic, no live LLM.
#
# The honesty contract is the load-bearing bit: any evidence URL not in
# `seed_urls` is dropped by the REFLECT phase. A hostile caller who
# hallucinates URLs will see them stripped from the response and reported
# in `warnings[]` as `dropped N hallucinated citation(s): [...]`.


class CoScientistRunRequest(BaseModel):
    """Input for POST /v1/co_scientist/run.

    Callers pass whatever stage envelopes they already have (screening,
    biopsy, therapy), plus the union of URLs their tool-loop actually
    fetched. Anything the model returns citing an unseen URL will be
    dropped by REFLECT.
    """

    screening: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ScreeningResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed screening-track hypotheses."
        ),
    )
    biopsy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "BiopsyResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed biopsy-track hypotheses."
        ),
    )
    therapy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "TherapyResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed therapy-track hypotheses."
        ),
    )
    seed_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs the tool-loop actually fetched. Any evidence URL NOT "
            "in this set is stripped by REFLECT. This is the ONLY "
            "authority on what was actually seen — plumb your fetch "
            "list here verbatim, no client-side filtering."
        ),
    )
    top_n_evolve: int = Field(
        default=3, ge=1, le=8,
        description="Top N ranked hypotheses to feed EVOLVE.",
    )
    n_variants: int = Field(
        default=2, ge=1, le=4,
        description="EVOLVE variants per seed hypothesis.",
    )
    return_top: int = Field(
        default=8, ge=1, le=32,
        description="Cap on hypotheses returned in `hypotheses[]`.",
    )


class CoScientistRunResponse(ApiEnvelope):
    """Output for POST /v1/co_scientist/run.

    Standard ApiEnvelope fields + per-phase counts + the ranked hypotheses.
    The two drop counts are what a reviewer looks at to confirm the
    honesty gate did its job on a hostile input:

    - `urls_dropped_hallucinated`: total URL references stripped across
      all hypotheses because they weren't in `seed_urls`.
    - `hypotheses_dropped`: count of hypotheses whose evidence list was
      emptied by REFLECT (they're kept in `hypotheses[]` but flagged in
      `warnings[]` as `no_evidence_after_reflect:<hyp_id>`).
    """

    phases: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of phases executed, e.g. "
            "['generate','reflect','rank','evolve','rank']."
        ),
    )
    hypotheses: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ranked hypotheses, up to `request.return_top`. Each entry has "
            "hyp_id, stage, statement, confidence, evidence, "
            "honesty_markers, derived_from, rating, wins, losses, draws."
        ),
    )
    initial_count: int = Field(
        default=0, ge=0,
        description="Hypothesis count after GENERATE.",
    )
    after_reflect: int = Field(
        default=0, ge=0,
        description="Hypothesis count after REFLECT (may equal initial_count).",
    )
    after_evolve: int = Field(
        default=0, ge=0,
        description="Hypothesis count after EVOLVE (includes variants).",
    )
    urls_dropped_hallucinated: int = Field(
        default=0, ge=0,
        description=(
            "Total URL references stripped by REFLECT because they weren't "
            "in `seed_urls`. This is the primary honesty metric — a "
            "hostile input with N fake URLs should see this ≥ N."
        ),
    )
    hypotheses_dropped: int = Field(
        default=0, ge=0,
        description=(
            "Hypotheses whose evidence list was emptied by REFLECT. Kept "
            "in `hypotheses[]` but a caller SHOULD treat them as untrusted."
        ),
    )
