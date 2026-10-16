"""Shared wire types: model state, provenance, envelopes, arbiter scores."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field



THERAPY_PROGNOSIS_MODEL_NAME = "therapy_prognosis_metabric_v1"

LEGACY_THERAPY_BENEFIT_MODEL_NAME = "therapy_benefit_metabric_v1"

THERAPY_PROGNOSIS_UNCERTAINTY = (
    "Single-arm METABRIC prognosis under the cohort's treatment mix; this "
    "design cannot identify a causal treatment effect. Not externally "
    "calibrated for contemporary care."
)



# --------------------------------------------------------------------------- #
# Shared

# Wire values that appear in audit envelopes written before the proxy surface
# was retired. They are deliberately NOT members of :class:`ModelState`, so no
# route can emit one; the frozen set exists only so an auditor reading an
# archived receipt can still resolve the string. Each of these labelled a
# substitute standing in for a required specialist stage.
RETIRED_MODEL_STATE_VALUES: frozenset[str] = frozenset({
    "proxy_siglip",              # google/siglip-base-patch16-224, general-domain
    "proxy_monai_heuristic",     # mask-gradient heuristic, not a detector
    "proxy_lung_heuristic",      # HU threshold + connected components
    "proxy_rules_lite",          # NCCN-lite static rules
    "proxy_regex_v0",            # regex pathology parser
    "fused_regex_clinicalbert",  # regex AND ClinicalBERT fusion
})



class ModelState(str, Enum):
    PLACEHOLDER = "placeholder"       # no model wired yet, stub response
    LOADED = "loaded"                 # model in memory, inference performed
    LOADING = "loading"               # model warming up
    UNAVAILABLE = "unavailable"       # model failed to load; error path
    CACHED = "cached"                 # result served from cache
    GATED = "gated"                   # HAI-DEF access denied (401/403)
    LOADED_MEDSIGLIP = "loaded_medsiglip"  # HAI-DEF MedSigLIP-448 inference (medical but off-label for mammography)
    LOADED_BIOPSY_PROBE = "loaded_biopsy_probe"  # L4b pinned MedSigLIP pooled embed + real BACH v1 head (RUO)
    LOADED_MONAI_DETECTOR = "loaded_monai_detector"  # L4a MONAI detector with trained weights (unreachable until weights ship)
    LOADED_LUNA16_RETINANET = "loaded_luna16_retinanet"  # v0.3.0 MONAI Model Zoo lung_nodule_ct_detection@0.6.9 (LUNA16-trained)
    LOADED_TXGEMMA = "loaded_txgemma"  # L4c HAI-DEF TxGemma inference (never reachable under current token)
    LOADED_RULES_LITE = "loaded_rules_lite"  # L4c deterministic NCCN-lite rules engine (not an ML model; no LLM)
    TEMPLATE = "template"             # L3 arbiter JSON templates loaded from disk (n_training=0)
    LOADED_CLINICALBERT_PARSER = "loaded_clinicalbert_parser"  # v0.3.0 Bio_ClinicalBERT fine-tuned on synthetic corpus
    PROXY_CO_SCIENTIST = "proxy_co_scientist"  # deprecated legacy label; retained for old receipts
    DETERMINISTIC_OFFLINE_RANKER = "deterministic_offline_ranker"  # evidence-gated Elo transform; never an LLM result
    LOADED_CASE_STORAGE = "loaded_case_storage"
    LOADED_PHIKON = "loaded_phikon"
    LOADED_MEDGEMMA_1P5_4B = "loaded_medgemma_1p5_4b"
    LOADED_MEDGEMMA_27B = "loaded_medgemma_27b"
    RETIRED = "retired"
    CONFIGURED_UNVERIFIED = "configured_unverified"  # endpoint configured; successful inference not yet proven
    # NOTE: the six substitute states (proxy_siglip, proxy_monai_heuristic,
    # proxy_lung_heuristic, proxy_rules_lite, proxy_regex_v0,
    # fused_regex_clinicalbert) were removed from this enum. See
    # RETIRED_MODEL_STATE_VALUES, defined above this class.

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



class StageError(BaseModel):
    code: str
    message: str



class StageReceipt(BaseModel):
    """Additive receipt for one specialist stage; never contains raw PHI."""

    stage: str
    required: bool
    status: Literal[
        "succeeded",
        "failed_required",
        "failed_optional",
        "skipped_not_applicable",
        "retired",
    ]
    request_id: str
    service_name: str | None = None
    endpoint_label: str | None = None
    app_version: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    artifact_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    input_reference: str | None = Field(
        default=None,
        description="Content hash, case identifier, or manifest hash; never raw patient data.",
    )
    latency_ms: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    error: StageError | None = None



class HonestyGateReport(BaseModel):
    seen_urls_count: int = Field(..., ge=0)
    evidence_kept: int = Field(..., ge=0)
    evidence_dropped: int = Field(..., ge=0)
    # v0.4.0-alpha (PLAN §2A + §7 bullet 7): Co-Scientist hostile-URL test asserts
    # that when the LLM proposes hypotheses referencing URLs the run never saw,
    # reflection.py drops the hypothesis (not just the evidence). Distinct from
    # evidence_dropped because a single hostile URL fabricates the whole hypothesis.
    hypotheses_dropped: int = Field(default=0, ge=0)



class ManskiBoundsBlock(BaseModel):
    """The identified set for a stage probability, attached to every point estimate.

    These are assumption-free (Manski) bounds: each feature the caller did not
    supply is ranged over its *entire* admissible support and the endpoints are
    the resulting extremes of the sigmoid. They are NOT a confidence interval —
    there is no sampling model here, only the logical consequence of not knowing
    a covariate.

    `width == 0.0` means the panel was complete and the estimate is
    point-identified. On every public route `identified == False` is
    unreachable in a 200 response: the gate raises HTTP 422 first, and no
    header, query parameter or environment variable relaxes it. The only
    way to see `identified == False` is the authenticated research route
    `/v1/research/arbiter/identified_set`, which requires the privileged
    `research:blind_inference` scope and stamps every release with an audit
    receipt.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    stage: str
    model_name: str
    lower: float = Field(..., ge=0.0, le=1.0)
    upper: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    max_width: float = Field(..., gt=0.0)
    identified: bool = Field(
        ..., description="True when width <= max_width and a point estimate is admissible.",
    )
    missing_features: list[str] = Field(
        default_factory=list,
        description="Declared features the caller did not supply, from the artefact schema — not only the booleans.",
    )
    unobserved_unbounded: list[str] = Field(
        default_factory=list,
        description="Unobserved continuous covariates with no declared support; each forces the interval to [0, 1].",
    )
    interval_semantics: str



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
    manski: ManskiBoundsBlock | None = Field(
        default=None,
        description=(
            "Identified set for `p_positive` given the unobserved part of the feature "
            "panel. Absent only on legacy paths that predate the gate."
        ),
    )
    provenance_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-empty only when the Manski gate was bypassed via "
            "X-Allow-Blind-Inference; the first entry is PROVISIONAL_UNOBSERVED_WARNING."
        ),
    )



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
            "`medsiglip_gated:<level>:<reason>` or out-of-distribution "
            "warnings. Callers MUST surface these. A warning never "
            "accompanies a substituted result: the substitute surface was "
            "retired, so a gated or failed specialist yields a failed "
            "required stage, not a warned fallback."
        ),
    )
