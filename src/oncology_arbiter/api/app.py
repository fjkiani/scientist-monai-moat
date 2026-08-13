"""FastAPI app factory + routes.

Everything returned here today is a PLACEHOLDER. It's the *shape* of the
real response, not the real answer. We wire real models in Phase 2+.

Design invariants for these placeholders:
  1. Every response includes `disclaimer` (RUO) and `caveat` (AUROC)
     inline. If we ever accidentally strip them, tests fail loudly.
  2. Every response includes `provenance.model_state = PLACEHOLDER` so a
     downstream consumer cannot mistake a stub for a live inference.
  3. The mammography endpoint runs REAL preprocessing (readers, laterality,
     view, mask) even when the model is a placeholder — this way we get to
     exercise ~90% of the pipeline on real DICOMs end-to-end from HTTP.
     Only the classification score is placeholder.
  4. The `honesty_gate` field always reports {kept=0, dropped=0} on the
     placeholder path since no evidence was gathered.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__

from ..arbiter.manski import (
    RESEARCH_BLIND_INFERENCE_SCOPE,
    ManskiGateError,
)
from .audit import log_event, new_request_id
from ..auth import APIKey, bootstrap_from_env, require_api_key, require_scope
from ..observability import (
    RequestIdMiddleware,
    configure_logging,
    get_logger,
)
from .schemas import (
    ApiEnvelope,
    ArbiterScore,
    ManskiBoundsBlock,
    ArtifactCategory,
    BiopsyReceptorPanel,
    BiopsyRequest,
    BiopsyResponse,
    BreastDssPrognosis,
    DemoCaseResponse,
    DynamicTumorBoardRequest,
    DynamicTumorBoardResponse,
    ExtendedReceptorField,
    ReportParseBlock,
    NsclcResponse,
    NsclcCandidate,
    NsclcTherapyOption,
    EvidenceRecord,
    FullCaseRequest,
    FullCaseResponse,
    GateReport as SchemaGateReport,
    HealthResponse,
    HonestyGateReport,
    ModelCardsIndex,
    ModelCardSummary,
    ModelState,
    Provenance,
    ResearchIdentifiedSetRequest,
    ResearchIdentifiedSetResponse,
    EloDrugCandidate,
    EloMatchRecord,
    EloRankedEntry,
    EloRankRequest,
    EloRankResponse,
    ScreeningRequest,
    ScreeningResponse,
    TherapyOption,
    TherapyRequest,
    TherapyResponse,
    # v0.4.0-alpha: AK MBD4-LOF tumor board contract
    TumorBoardBundle,
    TumorBoardBundleResponse,
    # v0.4.0-alpha: standalone Co-Scientist endpoint (PLAN §5 PR #5)
    CoScientistRunRequest,
    CoScientistRunResponse,
)


# --------------------------------------------------------------------------- #
# Helpers


def _to_schema_gate_report(runtime_gr: Any) -> SchemaGateReport | None:
    """Convert the runtime hai_def.GateReport dataclass into the pydantic
    schema GateReport, or None if the input is None.

    We do NOT rely on pydantic's model_validate over the dataclass because
    the runtime access_level is an Enum (`AccessLevel`) — we serialize its
    `.value` string so the schema's Literal validator accepts it, and so
    JSON output matches the wire contract.
    """
    if runtime_gr is None:
        return None
    return SchemaGateReport(
        repo_id=runtime_gr.repo_id,
        access_level=runtime_gr.access_level.value,
        status_code=runtime_gr.status_code,
        reason=runtime_gr.reason,
        has_token=runtime_gr.has_token,
        allowed=bool(runtime_gr.allowed),
    )


def _envelope(request_id: str, model_state: ModelState = ModelState.PLACEHOLDER,
              model_name: str | None = None,
              gate_report: SchemaGateReport | None = None) -> dict[str, Any]:
    """Common envelope fields that must appear on every response body.

    `gate_report` is populated on `provenance.gate_report` when the endpoint
    ran a HAI-DEF preflight (either successfully or hit a gate). Callers on
    placeholder / pure-proxy paths pass None and it stays None on the wire.
    """
    return {
        "disclaimer": RUO_DISCLAIMER,
        "caveat": AUROC_CAVEAT,
        "provenance": Provenance(
            model_state=model_state,
            model_name=model_name,
            request_id=request_id,
            gate_report=gate_report,
        ),
        "honesty_gate": HonestyGateReport(
            seen_urls_count=0, evidence_kept=0, evidence_dropped=0,
        ),
        "evidence": [],
    }


def _decode_bytes_arg(bytes_b64: str | None) -> bytes | None:
    if bytes_b64 is None:
        return None
    try:
        return base64.b64decode(bytes_b64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 dicom_bytes: {e}")


def _score_arbiter(
    name: str,
    features: dict[str, Any],
    *,
    stage: str | None = None,
) -> ArbiterScore:
    """Load the named L2 arbiter, score `features`, and apply the Manski gate.

    Wraps :func:`oncology_arbiter.arbiter.load_arbiter` and marshals the
    :class:`ArbiterResult` into the wire-level :class:`ArbiterScore` pydantic.

    This is the single choke point through which every stage's L2 point
    estimate reaches the wire, so it is where partial identification is
    enforced. When the assumption-free interval implied by the *unobserved*
    part of the declared feature schema is wider than
    :data:`MANSKI_MAX_WIDTH`, no point estimate is marshalled at all:
    :class:`ManskiGateError` propagates to the app-level handler and the caller
    receives 422 plus the raw interval.

    There is no bypass. This function takes no argument, reads no header and
    consults no environment variable that could release an unidentified point
    estimate on a public route. The research override is a separate route with
    its own privileged scope; see ``/v1/research/arbiter/identified_set``.
    """
    from oncology_arbiter.arbiter import load_arbiter
    from oncology_arbiter.arbiter.manski import ManskiBounds, enforce_manski_gate

    arb = load_arbiter(name)
    r = arb.score(features)
    bounds = ManskiBounds.from_arbiter(
        stage=stage or name, arbiter=arb, features=features, result=r,
    )
    provenance_warnings = enforce_manski_gate(bounds)
    return ArbiterScore(
        model_name=arb.model_name,
        p_positive=r.p_positive,
        logit=r.logit,
        risk_bucket=r.risk_bucket,  # type: ignore[arg-type]
        recommendation=r.recommendation,
        term_contributions=r.term_contributions,
        driving_feature=r.driving_feature,
        driving_feature_contribution=r.driving_feature_contribution,
        positive_class=arb.positive_class,
        n_training=arb.n_training,
        model_state=r.metadata["model_state"],  # type: ignore[arg-type]
        caveat=r.caveat,
        manski=ManskiBoundsBlock(**bounds.as_dict()),
        provenance_warnings=list(provenance_warnings),
    )


def _failed_stage_receipt(stage: str, required: bool, request_id: str, code: str, message: str, *, service_name: str | None = None, input_reference: str | None = None) -> dict[str, Any]:
    return {
        "stage": stage, "required": required,
        "status": "failed_required" if required else "failed_optional",
        "request_id": request_id, "service_name": service_name,
        "input_reference": input_reference, "warnings": [],
        "error": {"code": code, "message": message[:500]},
    }


def _skipped_stage_receipt(stage: str, request_id: str, message: str, *, required: bool = False) -> dict[str, Any]:
    return {
        "stage": stage, "required": required, "status": "skipped_not_applicable",
        "request_id": request_id, "warnings": [message], "error": None,
    }


def _pipeline_status(receipts: list[dict[str, Any]]) -> str:
    if any(receipt.get("status") == "failed_required" for receipt in receipts):
        return "failed_required_stage"
    if any(receipt.get("status") == "failed_optional" for receipt in receipts):
        return "partial_failure"
    return "complete"


def _clinicalbert_biopsy_parse(report_text: str, *, request_id: str) -> tuple[ReportParseBlock | None, BiopsyReceptorPanel, int | None, dict[str, Any]]:
    """Run only pinned production ClinicalBERT; never regex/fuse/fallback."""
    from oncology_arbiter.nlp.clinicalbert_modal_client import ClinicalBertModalClient, ClinicalBertModalError

    input_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    started = time.perf_counter()
    try:
        client = ClinicalBertModalClient()
        response = client.parse(report_text)
        parsed = response.get("parsed") or {}
        if not isinstance(parsed, dict):
            raise ClinicalBertModalError("clinicalbert_contract_mismatch:parsed")

        def value(entity: str) -> Any:
            item = parsed.get(entity)
            return item.get("value") if isinstance(item, dict) else None

        er_value, pr_value = value("ER_VALUE"), value("PR_VALUE")
        her2_value, grade_value, ki67_value = value("HER2_VALUE"), value("GRADE"), value("KI67_PCT")
        panel = BiopsyReceptorPanel(
            er_positive=True if er_value == "positive" else False if er_value == "negative" else None,
            pr_positive=True if pr_value == "positive" else False if pr_value == "negative" else None,
            her2_status=her2_value if her2_value in {"positive", "negative", "equivocal"} else None,
            ki67_percent=float(ki67_value) if isinstance(ki67_value, (int, float)) and 0 <= float(ki67_value) <= 100 else None,
            parse_state={
                "er": "matched" if "ER_VALUE" in parsed else "no_match",
                "pr": "matched" if "PR_VALUE" in parsed else "no_match",
                "her2": "matched" if "HER2_VALUE" in parsed else "no_match",
                "grade": "matched" if "GRADE" in parsed else "no_match",
            },
        )
        extended: dict[str, ExtendedReceptorField] = {}
        for entity, output_name in {
            "KI67_PCT": "ki67_pct", "TUMOR_SIZE_MM": "tumor_size_mm",
            "T_STAGE": "t_stage", "N_STAGE": "n_stage", "M_STAGE": "m_stage",
            "MARGIN": "margin", "LVI": "lvi",
        }.items():
            item = parsed.get(entity)
            if isinstance(item, dict):
                extended[output_name] = ExtendedReceptorField(
                    value=item.get("value"), match_state="matched",
                    matched_text=item.get("surface"), confidence=0.0, source="clinicalbert",
                )
        block = ReportParseBlock(
            parser_id="clinicalbert_v0.5.2_sliding_window", fusion_mode="clinicalbert",
            per_field_confidence={},
            per_field_source={key: "clinicalbert" if entity in parsed else "none" for key, entity in {
                "er": "ER_VALUE", "pr": "PR_VALUE", "her2": "HER2_VALUE", "grade": "GRADE",
            }.items()},
            extended_fields=extended, parsed_entities=parsed,
            n_tokens=response.get("n_tokens"), n_windows=response.get("n_windows"),
            window_tokens=response.get("window_tokens"), overlap_tokens=response.get("overlap_tokens"),
            window_aggregation=response.get("window_aggregation"), app_version=response.get("app_version"),
            model_sha256=response.get("model_sha256"), metrics_sha256=response.get("metrics_sha256"),
        )
        receipt = {
            "stage": "clinicalbert_pathology_parse", "required": True, "status": "succeeded",
            "request_id": request_id, "service_name": "clinicalbert",
            "endpoint_label": client.endpoints.parse, "app_version": response.get("app_version"),
            "model_name": response.get("base_model"), "model_version": "v0.5.2-sliding-window",
            "artifact_sha256": response.get("model_sha256"), "input_reference": f"sha256:{input_sha}",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "warnings": [str(response.get("disclaimer") or "")], "error": None,
        }
        grade = int(grade_value) if grade_value in {1, 2, 3} else None
        return block, panel, grade, receipt
    except Exception as exc:
        return None, BiopsyReceptorPanel(), None, _failed_stage_receipt(
            "clinicalbert_pathology_parse", True, request_id,
            getattr(exc, "code", "clinicalbert_failed"), f"{type(exc).__name__}: {exc}",
            service_name="clinicalbert", input_reference=f"sha256:{input_sha}",
        )


_TRIAGE_REQUIRED_FIELDS = {
    "histology", "grade", "er_positive", "pr_positive", "her2_positive",
    "ki67_pct", "tumor_size_mm", "lymph_nodes_pos", "brca_pathogenic", "age_years",
}




def _score_explicit_therapy_triage(
    raw: dict[str, Any] | None,
) -> ArbiterScore | None:
    """Score the explicit therapy triage vector, or return None if none was asked for.

    A *partial* panel is no longer silently dropped. Previously any missing
    field returned None with a "skipped" receipt, which hid from the caller
    both the fact that the arbiter could still have produced a number and how
    little that number would have been worth. The partial panel is now scored
    and handed to the Manski gate, so the caller either gets a point estimate
    that is genuinely identified, or a 422 carrying the interval and the list
    of features that would collapse it.
    """
    if raw is None:
        return None
    from oncology_arbiter.api.schemas import TherapyTriageFeatures
    validated = TherapyTriageFeatures.model_validate(raw).model_dump()
    if all(validated[field] is None for field in _TRIAGE_REQUIRED_FIELDS):
        # Nothing at all was supplied: this is "no triage requested", not
        # "triage requested with an unobserved panel".
        return None
    nodes = validated["lymph_nodes_pos"]
    features = {
        "histology": validated["histology"],
        "grade": None if validated["grade"] is None else str(validated["grade"]),
        "er_status_positive": validated["er_positive"],
        "pr_status_positive": validated["pr_positive"],
        "her2_status_positive": validated["her2_positive"],
        "ki67_norm": validated["ki67_pct"],
        "tumor_size_norm": validated["tumor_size_mm"],
        "node_status_positive": None if nodes is None else nodes > 0,
        "brca_status_known_pathogenic": validated["brca_pathogenic"],
        "age_at_diagnosis_norm": validated["age_years"],
    }
    return _score_arbiter(
        "therapy",
        features=features,
        stage="therapy_research_triage",
    )


# --------------------------------------------------------------------------- #
# MedSigLIP / SigLIP proxy singletons + runners
#
# Precedence rules (Phase 2 wiring, 2026-07-02):
#   1. If ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP=1, try MedSigLIP first.
#      - Preflight HAI-DEF gate → on ALLOWED, run and return the honest
#        MedSigLipResult carrying ModelState.LOADED_MEDSIGLIP.
#      - On GatedAccessError → NEVER silently fall back; the endpoint
#        decides based on ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY.
#   2. If ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY=1 (opt-in, NOT default),
#      run the proxy and return a warned proxy_siglip response.
#   3. Otherwise, return the placeholder envelope (overall_score=None).


import os

_MEDSIGLIP_SINGLETON: Any = None


def _is_env_true(name: str) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _demo_samples_dir() -> Path | None:
    """Locate the demo-samples directory on disk. Returns None if it does
    not exist. Points to `src/oncology_arbiter/api/static/demo_samples`
    which ships with the package.
    """
    here = Path(__file__).resolve().parent
    candidate = here / "static" / "demo_samples"
    return candidate if candidate.exists() else None


def _compute_models_loaded() -> dict[str, ModelState]:
    """Report configuration state without claiming successful inference readiness."""
    def configured(*names: str) -> ModelState:
        return (
            ModelState.CONFIGURED_UNVERIFIED
            if any(os.environ.get(name) for name in names)
            else ModelState.UNAVAILABLE
        )

    return {
        "case_storage": configured("CASE_STORAGE_MODAL_URL"),
        "medsiglip_448": configured("MODAL_MEDSIGLIP_URL"),
        "phikon": configured("PHIKON_MODAL_URL"),
        "luna16": configured("LUNA16_MODAL_URL"),
        "clinicalbert": configured("CLINICALBERT_MODAL_URL"),
        "medgemma_27b": configured("MEDGEMMA_MODAL_URL"),
        "breast_dss_v3": ModelState.LOADED,
        "ovarian_arbiter": ModelState.RETIRED,
        "offline_deterministic_ranker": ModelState.PROXY_CO_SCIENTIST,
    }


def _get_medsiglip() -> Any:
    """Lazy-construct the MedSigLIP client. Reuses one instance per process.

    Backend is chosen by ``MEDSIGLIP_BACKEND``:
      * ``modal`` → :class:`MedSigLipModalClient` (remote A10G, no local
        weight download, PNG/DICOM in-band via base64)
      * anything else (default ``local``) → local :class:`MedSigLip`

    The factory delegates so the choice can flip without touching this
    endpoint's contract; both backends produce the same
    :class:`MedSigLipResult` shape.
    """
    global _MEDSIGLIP_SINGLETON
    if _MEDSIGLIP_SINGLETON is None:
        from oncology_arbiter.models.medsiglip_modal_client import get_medsiglip_client
        _MEDSIGLIP_SINGLETON = get_medsiglip_client()
    return _MEDSIGLIP_SINGLETON


def _run_medsiglip_on_preprocessed(
    preprocess_result: Any,
    *,
    dicom_bytes: bytes | None = None,
) -> Any:
    """Run MedSigLIP on an already-preprocessed mammogram.

    Two paths depending on backend:

    * **Local backend** (:class:`oncology_arbiter.models.medsiglip.MedSigLip`):
      uses ``_preprocess_fn`` injection so the client's PIL conversion runs
      but no second DICOM I/O happens.
    * **Modal backend** (:class:`MedSigLipModalClient`): Modal does its own
      DICOM preprocess remotely — we need the raw DICOM bytes. We write
      them to a temp file and call ``.run(path)``. If ``dicom_bytes`` is
      not supplied, the fallback re-encodes the already-preprocessed
      float32 image to an 8-bit PNG so Modal's ``pixels_b64`` branch can
      handle it (loses some fidelity but preserves the honest path).

    Raises ``GatedAccessError`` on HAI-DEF gate denial. Callers MUST NOT
    catch this and silently fall back; the endpoint decides fallback
    policy explicitly via env flag.

    Phase 2 limitation (2026-07-03): the local branch mutates
    ``ms._preprocess_fn`` on the singleton for the duration of the call.
    That is safe under FastAPI's default single-threaded async request
    handling but is NOT safe under a threadpool. Phase 3 will either
    (a) construct a fresh MedSigLip per request (weights stay cached at
    class level so no re-download) or (b) thread the preprocess through
    the ``.run()`` signature directly.
    """
    ms = _get_medsiglip()
    # Late import avoids circulars and keeps modal client optional at load time.
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    if isinstance(ms, MedSigLipModalClient):
        if dicom_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
                tf.write(dicom_bytes)
                tmp_path = tf.name
            try:
                return ms.run(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        # No raw bytes: re-encode the already-normalised float32 image as
        # a temp PNG so Modal's pixels_b64 branch can consume it.
        import io as _io
        import numpy as _np
        from PIL import Image as _Image

        arr = _np.asarray(preprocess_result.image)
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8)
        buf = _io.BytesIO()
        _Image.fromarray(arr, mode="L").save(buf, format="PNG")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(buf.getvalue())
            tmp_path = tf.name
        try:
            return ms.run(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Local backend path (unchanged): inject the preprocessed image.
    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    ms._preprocess_fn = _inject
    return ms.run("(preprocessed)")


def _run_cbis_ddsm_probe_on_bytes(
    dicom_bytes: bytes,
    preprocess_result: Any,
) -> Any | None:
    """Run the trained CBIS-DDSM supervised probe on a mammogram.

    Requires the Modal backend (only backend that returns 1152-d embeddings
    today). For the local backend this returns ``None`` and callers should
    skip the probe finding.

    On Modal:
      1. Get the 1152-d MedSigLIP-448 embedding (one POST to /embed).
      2. Score it with the trained sklearn LogReg probe (in-process).
      3. Return a :class:`CbisDdsmProbeResult`.

    Any failure is caught and logged upstream; this returns ``None`` so a
    probe failure does NOT poison the whole /v1/screening/analyze response.
    The MedSigLIP zero-shot findings above still ship.
    """
    ms = _get_medsiglip()
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    if not isinstance(ms, MedSigLipModalClient):
        # Local backend does not expose the raw embedding today — the
        # supervised probe is Modal-only for now.
        return None

    # Write DICOM to a temp file, embed via Modal, discard.
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
        tf.write(dicom_bytes)
        tmp_path = tf.name
    try:
        embedding = ms.embed_dicom(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    from oncology_arbiter.models.cbis_ddsm_probe import CbisDdsmProbe

    probe = CbisDdsmProbe.get()
    return probe.predict(embedding)


# --------------------------------------------------------------------------- #
# Artifact paths — mirrors progression_arbiter/router.py stream_artifact

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOTS: dict[ArtifactCategory, Path] = {
    ArtifactCategory.docs: _PROJECT_ROOT / "docs" / "model_cards",
    ArtifactCategory.reports: _PROJECT_ROOT / "artifacts" / "reports",
    ArtifactCategory.data: _PROJECT_ROOT / "artifacts" / "data",
    ArtifactCategory.models: _PROJECT_ROOT / "src" / "oncology_arbiter" / "arbiter" / "models",
}


# --------------------------------------------------------------------------- #
# App factory


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan handler.

    v0.2.2: eagerly fetch the demo DICOM at startup so the first user request
    to /v1/demo/case doesn't block on a ~14 MB HuggingFace download. If the
    fetch fails (offline dev, HF outage, CI without network), we swallow the
    error — the endpoint will retry at request time and either succeed or
    return a 503 with a useful message.

    Skip pre-warm entirely when the env flag ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM
    is truthy. Unit tests set this so they don't hammer /tmp on every
    create_app() call.
    """
    if not _is_env_true("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM"):
        try:
            from .demo_fixtures import prewarm_demo_case

            path = prewarm_demo_case()
            if path is not None:
                get_logger().info(
                    "demo case pre-warmed at %s (%d bytes)",
                    path, path.stat().st_size,
                )
            else:
                get_logger().info(
                    "demo case pre-warm skipped (no local fixture, HF unreachable)"
                )
        except Exception as exc:
            # Never block startup for a demo fetch. prewarm_demo_case()
            # itself already swallows; this is belt-and-suspenders in case
            # someone refactors it to re-raise.
            get_logger().warning("demo pre-warm crashed: %s", exc)
    yield
    # No shutdown work today.




# --------------------------------------------------------------------------- #
# /v1/elo/rank — Co-Scientist deterministic re-ranking helpers


def _elo_reason_for_modifier(
    drug_id: str,
    delta: float,
    disease_context: dict,
) -> str:
    """Best-effort human-readable reason for the applied modifier delta.

    Kept deliberately terse — the SPA renders `rank_delta`, `applied_modifier`
    and `reason` side-by-side, so this text is a *hint*, not the source of
    truth.
    """
    if delta == 0.0:
        return "no modifier"

    d_low = drug_id.lower()
    ctx = disease_context or {}
    hrd = bool(ctx.get("hrd_positive")) or bool(ctx.get("brca_mutated"))
    pdl1 = ctx.get("pd_l1_cps")
    prior_lines = ctx.get("prior_lines")

    parp_names = ("olaparib", "niraparib", "rucaparib", "talazoparib")
    if any(n in d_low for n in parp_names) and hrd and delta > 0:
        return f"HRD+PARP boost ({delta:+.2f})"

    if "bevacizumab" in d_low and delta > 0:
        return f"Bevacizumab GOG-0218 posture ({delta:+.2f})"

    if any(n in d_low for n in ("pembrolizumab", "atezolizumab", "durvalumab", "nivolumab")):
        try:
            cps = float(pdl1) if pdl1 is not None else None
        except (TypeError, ValueError):
            cps = None
        if delta > 0 and cps is not None and cps >= 10:
            return f"PD-L1 CPS>=10 boost ({delta:+.2f})"
        if delta < 0 and cps is not None and cps < 10:
            return f"PD-L1 CPS<10 penalty ({delta:+.2f})"

    try:
        pl = int(prior_lines) if prior_lines is not None else None
    except (TypeError, ValueError):
        pl = None
    if delta < 0 and pl is not None and pl >= 3:
        return f"prior_lines={pl} penalty ({delta:+.2f})"

    if "ceralasertib" in d_low and delta != 0:
        return f"CAPRI ATR/PARP posture ({delta:+.2f})"

    if delta > 0:
        return f"manual boost ({delta:+.2f})"
    return f"manual penalty ({delta:+.2f})"


def _run_clinicalbert_parse(
    request_id: str,
    tenant_id: str | None,
    report_text: str | None,
    log_event_fn: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Dispatch the fine-tuned ClinicalBERT report parser.

    Pure report -> entities transform. Independent of CT / series_dir, so
    both the placeholder and the real-pipeline NSCLC branches call this to
    stamp ``parsed_report`` + ``parsed_report_provenance`` on their
    response. Failure modes:
      * backend unset or report_text empty  -> (None, None)   (silent skip)
      * modal / local client raises         -> (None, {source, error})
      * successful parse                    -> (parsed, provenance-with-source)
    Never fabricates a parse; the provenance dict always carries
    ``source`` (``clinicalbert_modal`` or ``clinicalbert_local``).
    """
    if not report_text:
        return None, None
    cbert_backend = os.environ.get("CLINICALBERT_BACKEND", "").lower()
    if cbert_backend not in ("modal", "local"):
        return None, None

    import time as _time  # local rebind for wall-clock timing
    t_cb0 = _time.perf_counter()
    parsed_dict: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    if cbert_backend == "modal":
        try:
            from oncology_arbiter.nlp.clinicalbert_modal_client import (
                ClinicalBertModalClient,
            )
            client = ClinicalBertModalClient()
            resp = client.parse(report_text)
            parsed_dict = resp.get("parsed") or {}
            provenance = {
                "provenance": resp.get("provenance"),
                "base_model": resp.get("base_model"),
                "training_seed": resp.get("training_seed"),
                "test_micro_f1": resp.get("test_micro_f1"),
                "app_version": resp.get("app_version"),
                "n_tokens": resp.get("n_tokens"),
                "seconds": resp.get("seconds"),
                "n_entity_types": len(parsed_dict),
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
                "source": "clinicalbert_modal",
            }
        except Exception as exc:  # pylint: disable=broad-except
            try:
                log_event_fn(
                    request_id, "/v1/case/full",
                    model_state="clinicalbert_modal_error",
                    patient_id_hash=None,
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                    tenant_id=tenant_id,
                )
            except Exception:
                pass
            provenance = {
                "source": "clinicalbert_modal",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
            }
    else:  # local
        try:
            from oncology_arbiter.nlp.clinicalbert_local_client import (
                ClinicalBertLocalClient,
            )
            client = ClinicalBertLocalClient()
            resp = client.parse(report_text)
            parsed_dict = resp.get("parsed") or {}
            provenance = {
                "provenance": resp.get("provenance"),
                "base_model": resp.get("base_model"),
                "training_seed": resp.get("training_seed"),
                "test_micro_f1": resp.get("test_micro_f1"),
                "app_version": resp.get("app_version"),
                "n_tokens": resp.get("n_tokens"),
                "seconds": resp.get("seconds"),
                "n_entity_types": len(parsed_dict),
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
                "source": "clinicalbert_local",
            }
        except Exception as exc:  # pylint: disable=broad-except
            try:
                log_event_fn(
                    request_id, "/v1/case/full",
                    model_state="clinicalbert_local_error",
                    patient_id_hash=None,
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                    tenant_id=tenant_id,
                )
            except Exception:
                pass
            provenance = {
                "source": "clinicalbert_local",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
            }
    return parsed_dict, provenance


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncology-arbiter",
        version=__version__,
        description=RUO_DISCLAIMER + "\n\n" + AUROC_CAVEAT,
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------- #
    # SaaS middleware wiring
    #
    # Starlette applies the OUTERMOST middleware first on the inbound side;
    # so we add them in reverse of "who should run first". We want the
    # request-id middleware to run FIRST (so every downstream error carries
    # a request id on its response header), which means it must be the LAST
    # one added.

    configure_logging(os.environ.get("ONCOLOGY_ARBITER_LOG_LEVEL", "INFO"))
    _logger = get_logger()

    # 0) Partial-identification gate.
    #
    # Registered before every other handler because it must not be reachable
    # around: a stage that cannot bound its own probability to within
    # MANSKI_MAX_WIDTH has no point estimate to serve, and the correct HTTP
    # semantics for "your input is well-formed but does not identify the
    # quantity you asked for" is 422, not a 200 carrying a number.
    @app.exception_handler(ManskiGateError)
    def _manski_bounds_handler(request: Request, exc: ManskiGateError):  # type: ignore[no-untyped-def]
        payload = exc.payload()
        _logger.warning(
            "manski gate blocked %s stage=%s width=%.6f bounds=[%.6f,%.6f] missing=%s",
            request.url.path, payload["stage"], payload["width"],
            payload["bounds"][0], payload["bounds"][1],
            ",".join(payload["missing_features"]) or "none",
        )
        return JSONResponse(status_code=exc.status_code, content=payload)

    # 1) Prometheus /metrics
    try:
        from prometheus_client import CollectorRegistry
        from prometheus_fastapi_instrumentator import Instrumentator

        # Each app gets its OWN CollectorRegistry. Against the global default
        # registry, a second create_app() in the same process raises
        # "Duplicated timeseries in CollectorRegistry"; the except-clause below
        # swallowed that into a warning, so those apps exposed /metrics with no
        # HTTP metrics and no visible error. Production runs one app, so the
        # bug was invisible there and only surfaced under test.
        _metrics_registry = CollectorRegistry(auto_describe=True)
        Instrumentator(
            excluded_handlers=["/metrics"],
            should_group_status_codes=False,
            registry=_metrics_registry,
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except ImportError as exc:  # pragma: no cover - optional dependency
        _logger.warning("prometheus instrumentator unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover
        # Previously this branch swallowed genuine wiring errors (e.g. a bad
        # kwarg or a duplicated CollectorRegistry timeseries) into a warning,
        # so the app served /metrics as 404 while looking healthy. Log it at
        # error level with the exception type so it cannot pass unnoticed.
        _logger.error(
            "prometheus instrumentation FAILED (%s): %s; /metrics will not be served",
            type(exc).__name__, exc,
        )

    # 2) Rate limit
    #
    # The default slowapi key_func reads request.client.host, which behind
    # Render + Cloudflare is the internal proxy IP — every real caller then
    # shares the same handful of buckets and the bucket for any single caller
    # never fills. Use `make_key_func()` from ..api.rate_limit, which reads
    # CF-Connecting-IP (Cloudflare-signed) first, then X-Forwarded-For (first
    # hop), then falls back to request.client.host. The header path is
    # opt-in via ONCOLOGY_ARBITER_TRUST_FORWARDED_FOR=1 (production Render).
    # See src/oncology_arbiter/api/rate_limit.py for the full trust rationale.
    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware

        from .rate_limit import make_key_func

        limiter = Limiter(
            key_func=make_key_func(),
            default_limits=[os.environ.get("ONCOLOGY_ARBITER_RATE_LIMIT", "30/minute")],
        )
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

        @app.exception_handler(RateLimitExceeded)
        def _rate_limit_handler(request, exc):  # type: ignore[no-untyped-def]
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded", "limit": str(exc.detail)},
                headers={"Retry-After": "60"},
            )
    except Exception as exc:  # pragma: no cover
        _logger.warning("slowapi rate limiter disabled: %s", exc)

    # 3) CORS
    from fastapi.middleware.cors import CORSMiddleware

    _allowed = os.environ.get("ONCOLOGY_ARBITER_ALLOWED_ORIGINS", "*")
    _origins = [o.strip() for o in _allowed.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
    )

    # 4) Request-id (last add = first inbound)
    app.add_middleware(RequestIdMiddleware)

    # 4.5) v0.3.0-alpha DEMO_MODE gate.
    #
    # When ONCOLOGY_ARBITER_DEMO_MODE=1, this deployment is a read-only
    # showcase: all POST endpoints return HTTP 403 with a contact
    # placeholder, and pre-computed sample outputs are served under
    # /v1/demo/samples/*. Read-only GET endpoints (health, model-cards,
    # artifacts, demo/case, demo/samples/*) stay open.
    #
    # Rationale: the public showcase runs on modest hardware; letting
    # anonymous callers fire real Modal / ClinicalBERT / MONAI inferences
    # burns GPU credit and misrepresents throughput. In demo mode the
    # frontend renders provenance-labeled real outputs captured on our
    # workers, and routes any "run on your own data" click to the contact
    # URL.
    _contact_url = os.environ.get(
        "ONCOLOGY_ARBITER_CONTACT_URL", "https://crispro.ai/contact"
    )

    # POST routes that are safe to run even in read-only demo mode.
    # These are pure-CPU, deterministic, no GPU / no external inference,
    # so they do NOT burn Modal credit or misrepresent throughput:
    #   * /v1/elo/rank         — Co-Scientist Elo tournament (pure Python, seeded)
    #   * /v1/co_scientist/run — Co-Scientist 4-phase loop (pure Python, seeded)
    # Anything not on this list falls back to the 403 demo-mode error.
    _DEMO_MODE_POST_ALLOWLIST = frozenset({
        "/v1/elo/rank",
        "/v1/offline_ranker/run",
    })

    @app.middleware("http")
    async def _demo_mode_gate(request: Request, call_next):  # noqa: RUF029
        if not _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"):
            return await call_next(request)
        # Only POST is gated. Everything else (GET, OPTIONS preflight,
        # HEAD) flows through so the frontend can still read health,
        # samples, model-cards, and artifacts.
        if request.method != "POST":
            return await call_next(request)
        # Explicit allow-list for POSTs that are cheap + deterministic.
        if request.url.path in _DEMO_MODE_POST_ALLOWLIST:
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "This deployment is a read-only demo of Oncology Arbiter "
                    "v0.3.0-alpha. Live inference (screening, biopsy, "
                    "therapy, case/full) is disabled here. See "
                    "GET /v1/demo/samples for pre-computed sample outputs "
                    "captured on our workers, or contact us to run the API "
                    "on your own data."
                ),
                "contact_url": _contact_url,
                "demo_endpoint": "GET /v1/demo/samples",
                "demo_mode": True,
            },
        )

    # 5) One-shot auth bootstrap from env
    #
    # On a fresh container the SQLite tenants table is empty. Flipping
    # AUTH_MODE=on without a seeded tenant locks out every caller and there
    # is no shell into the free-tier Render container to mint a key by hand.
    # `bootstrap_from_env` reads a PRE-HASHED key from env (SHA256 hex only;
    # the raw key never touches deploy env) and injects one row IFF the
    # table is empty. On a second start the table has a row and this is a
    # no-op. See src/oncology_arbiter/auth/bootstrap.py for the contract.
    try:
        _bootstrap_result = bootstrap_from_env()
        if _bootstrap_result.get("fired"):
            _logger.info(
                "auth_bootstrap fired: tenant_id=%s key_prefix=%s",
                _bootstrap_result.get("tenant_id"),
                _bootstrap_result.get("key_prefix"),
            )
        elif _bootstrap_result.get("reason") not in (
            "bootstrap_env_incomplete",
            "tenants_table_not_empty",
        ):
            # Only log if it's a real config bug (e.g. malformed hash), not
            # the two silent-no-op paths that fire on every non-configured
            # local dev start.
            _logger.warning("auth_bootstrap skipped: %s", _bootstrap_result)
    except Exception as _boot_exc:  # pragma: no cover
        _logger.warning("auth_bootstrap raised: %s", _boot_exc)


    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # `cancers` mirrors the surface `/v1/case/full?cancer=…` accepts.
        # `breast` is the flagship path (real preprocessing, arbiter, etc.);
        # `nsclc` is the LIDC-IDRI expansion track that worker-2 is wiring —
        # the endpoint currently returns a shape-only placeholder so the SPA
        # can already render a working NSCLC panel end-to-end.
        _demo_active = _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE")
        _sample_files: list[str] = []
        if _demo_active:
            _sd = _demo_samples_dir()
            if _sd is not None and _sd.exists():
                _sample_files = sorted(p.stem for p in _sd.glob("*.json"))
        endpoints = [
            "POST /v1/screening/analyze",
            "POST /v1/biopsy/analyze",
            "POST /v1/therapy/reason",
            "POST /v1/case/full",
            "POST /v1/elo/rank",
            # v0.4.0-alpha: standalone Co-Scientist loop (PLAN §5 PR #5)
            "POST /v1/offline_ranker/run",
            "POST /v1/tumor_board/dynamic",
            "GET  /v1/demo/case",
            "GET  /v1/model-cards",
            "GET  /v1/artifacts/{category}/{filename}",
            "GET  /health",
        ]
        if _demo_active:
            endpoints.append("GET  /v1/demo/samples")
            endpoints.append("GET  /v1/demo/samples/{kind}")
        return HealthResponse(
            status="ok",
            version=__version__,
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            endpoints=endpoints,
            demo_mode=_demo_active,
            contact_url=_contact_url if _demo_active else None,
            demo_samples=_sample_files,
            cancers={
                "breast": {
                    "state": ModelState.CONFIGURED_UNVERIFIED.value,
                    "case_full": True,
                    "patient_probability_emitted": True,
                    "endpoints": ["screening", "biopsy", "therapy", "tumor_board/dynamic", "case/full"],
                    "notes": "Completion depends on successful specialist stage receipts; URL configuration alone is not inference readiness.",
                },
                "nsclc": {
                    "state": ModelState.CONFIGURED_UNVERIFIED.value,
                    "case_full": True,
                    "patient_probability_emitted": True,
                    "endpoints": ["case/full"],
                    "notes": "Production flow requires verified case storage followed by LUNA16 RetinaNet bundle 0.6.9; no local series path or HU heuristic is accepted.",
                },
                "hgsoc": {
                    "state": ModelState.RETIRED.value,
                    # case_full is True because /v1/case/full genuinely accepts
                    # cancer=hgsoc. It is NOT a claim that a scorer runs: the
                    # two keys below carry that, so route reachability can no
                    # longer be misread as scoring availability.
                    "case_full": True,
                    "patient_probability_emitted": False,
                    "retired_scope": "ovarian_prognostic_arbiter_patient_level_scoring",
                    "endpoints": ["tumor_board/dynamic", "tumor_board/bundle", "case/full"],
                    "notes": "Route reachable; the ovarian prognostic arbiter is retired and emits no patient probability or risk bucket. Therapy and MedGemma stages remain independently receipt-gated.",
                },
            },
            models_loaded=_compute_models_loaded(),
        )

    # ----------------------------------------------------------------------- #
    # /v1/screening/analyze — production MedSigLIP-448 only

    @app.post("/v1/screening/analyze", response_model=ScreeningResponse)
    def screening_analyze(
        req: ScreeningRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> ScreeningResponse:
        """Run strict production MedSigLIP without proxy or heuristic fallback."""
        request_id = new_request_id()
        if not req.dicom_bytes_b64 or req.dicom_url:
            raise HTTPException(
                400,
                "production screening requires exactly dicom_bytes_b64; remote URL ingestion is not accepted",
            )
        raw_bytes = _decode_bytes_arg(req.dicom_bytes_b64)
        assert raw_bytes is not None
        input_sha = hashlib.sha256(raw_bytes).hexdigest()

        from oncology_arbiter.mammography import preprocess_mammogram
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
            temp_file.write(raw_bytes)
            temp_path = temp_file.name
        try:
            preprocessed = preprocess_mammogram(
                temp_path,
                laterality_hint=req.laterality_hint,
                view_hint=req.view_hint,
            )
        except Exception as exc:
            Path(temp_path).unlink(missing_ok=True)
            raise HTTPException(422, f"preprocessing failed: {exc}") from exc

        started = time.perf_counter()
        receipt: dict[str, Any]
        medsiglip_block: dict[str, Any] | None = None
        findings: list[dict[str, Any]] = []
        overall_score: float | None = None
        warnings: list[str] = []
        model_state = ModelState.UNAVAILABLE
        model_name: str | None = "google/medsiglip-448"
        gate_report: Any = None
        try:
            from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

            result = MedSigLipModalClient().run(temp_path)
            if result.embedding_dim != 1152 or not result.embedding_sha256:
                raise RuntimeError("medsiglip_contract_mismatch:embedding_receipt")
            gate_report = result.gate_report
            latency_ms = (time.perf_counter() - started) * 1000.0
            findings = [
                {
                    "label": label,
                    "score": float(score),
                    "location_bbox_normalized": None,
                }
                for label, score in zip(result.labels, result.probs)
            ]
            overall_score = float(result.probs[0])
            warnings = list(result.warnings) + [
                "Zero-shot probabilities are uncalibrated, off-label, and are not diagnostic or treatment evidence."
            ]
            medsiglip_block = {
                "model_name": result.model_repo,
                "app_version": result.app_version,
                "input_resolution": result.input_resolution,
                "embedding_dim": result.embedding_dim,
                "embedding_sha256": result.embedding_sha256,
                "prompts": list(result.prompts),
                "inference_seconds": result.inference_seconds,
                "input_sha256": input_sha,
                "score_semantics": "independent_uncalibrated_sigmoid_zero_shot",
                "probs": [float(p) for p in result.probs],
                "probs_sum": float(sum(float(p) for p in result.probs)),
                "probs_are_normalised_distribution": False,
            }
            receipt = {
                "stage": "medsiglip_screening",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "crispro--medsiglip",
                "endpoint_label": "medsiglip-zero-shot+medsiglip-embed",
                "app_version": result.app_version,
                "model_name": result.model_repo,
                "input_reference": input_sha,
                "latency_ms": latency_ms,
                "warnings": warnings,
                "error": None,
            }
            model_state = ModelState.LOADED_MEDSIGLIP
        except Exception as exc:  # required failure remains a failure
            latency_ms = (time.perf_counter() - started) * 1000.0
            warnings = [f"medsiglip_required_stage_failed:{type(exc).__name__}"]
            receipt = _failed_stage_receipt(
                "medsiglip_screening",
                True,
                request_id,
                "medsiglip_inference_failed",
                str(exc),
                service_name="crispro--medsiglip",
                input_reference=input_sha,
            )
            receipt["latency_ms"] = latency_ms
        finally:
            Path(temp_path).unlink(missing_ok=True)

        schema_gate_report = _to_schema_gate_report(gate_report)
        envelope = _envelope(
            request_id,
            model_state=model_state,
            model_name=model_name,
            gate_report=schema_gate_report,
        )
        response = ScreeningResponse(
            **envelope,
            pipeline_status=_pipeline_status([receipt]),
            stage_receipts=[receipt],
            laterality=preprocessed.metadata.laterality.value,
            view=preprocessed.metadata.view.value,
            orientation_flipped=preprocessed.metadata.orientation_flipped,
            breast_mask_coverage=float(preprocessed.breast_mask.mean()),
            findings=findings,
            overall_score=overall_score,
            medsiglip=medsiglip_block,
            arbiter_score=None,
            warnings=warnings,
        )
        log_event(
            request_id,
            "/v1/screening/analyze",
            model_state=model_state.value,
            patient_id_hash=req.patient_id_hash,
            extra={
                "pipeline_status": response.pipeline_status,
                "input_sha256": input_sha,
                "embedding_dim": medsiglip_block.get("embedding_dim") if medsiglip_block else None,
                "app_version": medsiglip_block.get("app_version") if medsiglip_block else None,
            },
            tenant_id=tenant.tenant_id,
        )
        return response

    # ----------------------------------------------------------------------- #
    # /v1/biopsy/analyze — placeholder

    @app.post("/v1/biopsy/analyze", response_model=BiopsyResponse)
    def biopsy_analyze(
        req: BiopsyRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> BiopsyResponse:
        """Run required ClinicalBERT and Phikon stages without proxy substitution."""
        request_id = new_request_id()
        if not any((req.wsi_url, req.wsi_bytes_b64, req.report_text, req.dss_features is not None)):
            raise HTTPException(400, "must provide pathology image, report_text, or dss_features")

        receipts: list[dict[str, Any]] = []
        warnings: list[str] = []
        report_parse_block: ReportParseBlock | None = None
        receptor_panel = BiopsyReceptorPanel()
        parsed_grade: int | None = None
        phikon_embedding: dict[str, Any] | None = None
        model_state = ModelState.PLACEHOLDER
        model_name: str | None = None

        if req.report_text:
            report_parse_block, receptor_panel, parsed_grade, receipt = _clinicalbert_biopsy_parse(
                req.report_text, request_id=request_id
            )
            receipts.append(receipt)
            if receipt["status"] == "succeeded":
                model_state = ModelState.LOADED_CLINICALBERT_PARSER
                model_name = "emilyalsentzer/Bio_ClinicalBERT+v0.5.2-sliding-window"
            else:
                warnings.append(f"clinicalbert_required_stage_failed:{receipt['error']['code']}")
        else:
            receipts.append(_skipped_stage_receipt(
                "clinicalbert_pathology_parse", request_id, "no report_text supplied"
            ))

        if req.wsi_bytes_b64:
            try:
                from oncology_arbiter.models.specialist_clients import PhikonClient
                image_bytes = _decode_bytes_arg(req.wsi_bytes_b64)
                call = PhikonClient().embed(image_bytes or b"", request_id=request_id, required=True)
                phikon_embedding = call.output
                receipts.append(call.receipt)
                if model_state == ModelState.PLACEHOLDER:
                    model_state = ModelState.LOADED_PHIKON
                    model_name = "Phikon"
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "phikon_embedding", True, request_id,
                    getattr(exc, "code", "phikon_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="phikon",
                ))
                warnings.append("phikon_required_stage_failed")
        elif req.wsi_url:
            receipts.append(_failed_stage_receipt(
                "phikon_embedding", True, request_id, "direct_url_disabled",
                "Remote pathology-image fetching is disabled; submit de-identified image bytes.",
                service_name="phikon",
            ))
            warnings.append("phikon_required_stage_failed:direct_url_disabled")
        else:
            receipts.append(_skipped_stage_receipt(
                "phikon_embedding", request_id, "no pathology image supplied"
            ))

        dss_prognosis: BreastDssPrognosis | None = None
        if req.dss_features is not None:
            from oncology_arbiter.models.breast_dss_arbiter import score_breast_dss
            f = req.dss_features
            dss_result = score_breast_dss({
                "age": f.age, "tumor_size_mm": f.tumor_size_mm,
                "nodes_positive": f.nodes_positive, "grade": f.grade,
                "er_pos": f.er_positive, "pr_pos": f.pr_positive,
                "her2_pos": f.her2_positive,
            })
            dss_prognosis = BreastDssPrognosis(
                model_name=dss_result.model_name,
                disease_specific_mortality_score=dss_result.score,
                logit=dss_result.logit,
                term_contributions=dss_result.term_contributions,
                artifact_sha256=dss_result.artifact_sha256,
                n_training=dss_result.n_training,
                events=dss_result.events,
                oof_auroc=dss_result.oof_auroc,
                caveats=list(dss_result.caveats),
            )
            receipts.append({
                "stage": "breast_dss_prognosis", "required": False, "status": "succeeded",
                "request_id": request_id, "service_name": "oncology-arbiter",
                "model_name": dss_result.model_name, "model_version": "v3",
                "artifact_sha256": dss_result.artifact_sha256,
                "input_reference": "explicit_complete_seven_feature_vector",
                "latency_ms": 0.0,
                "warnings": ["prognosis_under_metabric_treatment_mix_not_treatment_benefit"],
                "error": None,
            })
            warnings.append("breast_dss_prognosis:frozen_metabric:explicit_features:no_imputation")
        else:
            receipts.append(_skipped_stage_receipt(
                "breast_dss_prognosis", request_id, "complete explicit seven-feature vector not supplied"
            ))

        pipeline_status = _pipeline_status(receipts)
        log_event(
            request_id, "/v1/biopsy/analyze", model_state=model_state.value,
            patient_id_hash=req.patient_id_hash,
            extra={"pipeline_status": pipeline_status, "stage_statuses": [r["status"] for r in receipts]},
            tenant_id=tenant.tenant_id,
        )
        env = _envelope(request_id, model_state=model_state, model_name=model_name)
        env["warnings"] = warnings
        return BiopsyResponse(
            **env,
            pipeline_status=pipeline_status,
            stage_receipts=receipts,
            phikon_embedding=phikon_embedding,
            subtype_prediction=None,
            receptor_panel=receptor_panel,
            grade=parsed_grade,
            confidence=None,
            arbiter_score=None,
            report_parse=report_parse_block,
            dss_prognosis=dss_prognosis,
        )

    # ----------------------------------------------------------------------- #
    # /v1/therapy/reason — placeholder

    @app.post("/v1/therapy/reason", response_model=TherapyResponse)
    def therapy_reason(
        req: TherapyRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> TherapyResponse:
        """Return only live SL-bridge recommendations plus an optional research triage decomposition."""
        request_id = new_request_id()
        receipts: list[dict[str, Any]] = []
        warnings: list[str] = []
        if req.legacy_therapy_benefit_model_name is not None:
            warnings.append("deprecated_prognostic_model_alias_remapped")
        bridge_output: dict[str, Any] | None = None
        recommended: list[TherapyOption] = []
        not_recommended: list[TherapyOption] = []

        if req.mutations or req.germline_mutations:
            try:
                from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient
                payload = {
                    "disease": req.disease,
                    "cancer_type": req.cancer_type,
                    "mutations": req.mutations,
                    "germline_mutations": req.germline_mutations,
                    "ranked_drugs": req.ranked_drugs,
                    "include_explanations": False,
                }
                call = SLTherapyBridgeClient().run(payload, request_id=request_id, required=True)
                bridge_output = call.output
                receipts.append(call.receipt)
                for item in bridge_output.get("sl_indicated_drugs") or []:
                    if not isinstance(item, dict) or not item.get("drug_name"):
                        continue
                    recommended.append(TherapyOption(
                        regimen=str(item["drug_name"]),
                        line_of_therapy=1,
                        rationale=str(item.get("sl_rationale") or "Synthetic-lethality actionability hypothesis"),
                    ))
                for item in bridge_output.get("policy_overlays") or []:
                    if not isinstance(item, dict) or item.get("policy_decision") != "DISALLOW":
                        continue
                    drug = item.get("drug_name") or item.get("name")
                    if drug:
                        not_recommended.append(TherapyOption(
                            regimen=str(drug), line_of_therapy=1,
                            rationale=str(item.get("policy_rationale_short") or "SL bridge policy DISALLOW"),
                        ))
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "synthetic_lethality_therapy_bridge", True, request_id,
                    getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="crispro-backend-v2",
                ))
                warnings.append("sl_bridge_required_stage_failed:no_therapy_recommendations_emitted")
        else:
            receipts.append(_skipped_stage_receipt(
                "synthetic_lethality_therapy_bridge", request_id,
                "no patient mutation profile supplied; no therapy recommendation computed",
            ))

        triage: ArbiterScore | None = None
        try:
            triage = _score_explicit_therapy_triage(
                req.therapy_features
            )
        except ManskiGateError:
            # Never downgrade a partial-identification block into a soft
            # optional-stage failure: the caller must see the 422 and the
            # interval, not a 200 with therapy_triage=null.
            raise
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "therapy_research_triage", False, request_id, "invalid_triage_features",
                f"{type(exc).__name__}: {exc}", service_name="oncology-arbiter",
            ))
        if triage is None and not any(r["stage"] == "therapy_research_triage" for r in receipts):
            receipts.append(_skipped_stage_receipt(
                "therapy_research_triage", request_id,
                "no explicit triage vector supplied; nothing to identify",
            ))
        elif triage is not None:
            receipts.append({
                "stage": "therapy_research_triage", "required": False, "status": "succeeded",
                "request_id": request_id, "service_name": "oncology-arbiter",
                "model_name": triage.model_name, "model_version": "template-v0",
                "input_reference": (
                    "explicit_complete_patient_features" if not triage.provenance_warnings
                    else "partially_unobserved_panel"
                ),
                "latency_ms": 0.0,
                "warnings": [
                    "illustrative_research_triage_not_recommendation_or_treatment_benefit",
                    *triage.provenance_warnings,
                ],
                "error": None,
            })

        status = _pipeline_status(receipts)
        state = ModelState.LOADED_SL_THERAPY_BRIDGE if bridge_output is not None else ModelState.UNAVAILABLE
        env = _envelope(request_id, model_state=state, model_name="SLTherapyBridge-v3.1" if bridge_output else None)
        env["warnings"] = warnings
        log_event(
            request_id, "/v1/therapy/reason", model_state=state.value,
            patient_id_hash=None,
            extra={"pipeline_status": status, "n_bridge_recommended": len(recommended)},
            tenant_id=tenant.tenant_id,
        )
        return TherapyResponse(
            **env,
            pipeline_status=status,
            stage_receipts=receipts,
            therapy_bridge=bridge_output,
            therapy_triage=triage,
            recommended_options=recommended,
            not_recommended=not_recommended,
            prognostic_model_executed=False,
            prognostic_score=None,
            arbiter_score=triage,
        )

    # ----------------------------------------------------------------------- #
    # /v1/research/arbiter/identified_set — the ONLY blind-inference surface
    #
    # The public gate has no bypass: no header, query parameter or environment
    # variable releases an unidentified point estimate from /v1/screening,
    # /v1/biopsy, /v1/therapy, /v1/tumor_board or /v1/case. Blind inference
    # lives here instead, behind four separate locks:
    #
    #   1. a valid API key,
    #   2. the privileged `research:blind_inference` scope — `require_scope`
    #      deliberately does NOT honour ONCOLOGY_ARBITER_AUTH_MODE=off, so the
    #      anonymous dev principal (which holds no scopes) is refused even in a
    #      local process where every other route is wide open,
    #   3. an explicit `acknowledge_not_for_clinical_use: true` in the body,
    #   4. an audit receipt naming the tenant, written before the response is
    #      marshalled, recording exactly what the public route would have
    #      refused.
    #
    # The response returns the raw interval and the unobserved-feature list
    # alongside the number, so the caller cannot receive a point estimate
    # without simultaneously receiving the evidence that it is uninformative.

    @app.post(
        "/v1/research/arbiter/identified_set",
        response_model=ResearchIdentifiedSetResponse,
    )
    def research_identified_set(
        req: ResearchIdentifiedSetRequest,
        tenant: APIKey = Depends(require_scope(RESEARCH_BLIND_INFERENCE_SCOPE)),
    ) -> ResearchIdentifiedSetResponse:
        """Return the Manski identified set for a partial panel, un-gated.

        This is a research instrument for measuring how much a missing
        covariate costs, not a clinical endpoint. When the panel is complete
        the response is identical in content to the public route; when it is
        not, the caller gets the interval that the public route converts into
        an HTTP 422.
        """
        from oncology_arbiter.arbiter import load_arbiter
        from oncology_arbiter.arbiter.manski import (
            MANSKI_ERROR_CODE,
            UNBOUNDED_ERROR_CODE,
            ManskiBounds,
            blind_inference_receipt,
            research_release,
        )

        request_id = new_request_id()
        try:
            arb = load_arbiter(req.arbiter)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, f"unknown arbiter {req.arbiter!r}: {exc}")

        result = arb.score(req.features)
        bounds = ManskiBounds.from_arbiter(
            stage=f"research_{req.arbiter}",
            arbiter=arb,
            features=req.features,
            result=result,
        )
        warnings = research_release(bounds)
        receipt = blind_inference_receipt(
            bounds,
            tenant_id=tenant.tenant_id,
            key_prefix=tenant.key_prefix,
            route="/v1/research/arbiter/identified_set",
            request_features=req.features,
        )
        error_code = (
            UNBOUNDED_ERROR_CODE
            if bounds.unbounded
            else (MANSKI_ERROR_CODE if not bounds.identified else None)
        )
        log_event(
            request_id,
            "/v1/research/arbiter/identified_set",
            model_state=str(result.metadata["model_state"]),
            tenant_id=tenant.tenant_id,
            extra={"blind_inference_receipt": receipt},
        )
        _logger.warning(
            "blind inference released stage=%s tenant=%s width=%.6f "
            "bounds=[%.6f,%.6f] public_route_would_have_returned=%s",
            bounds.stage, tenant.tenant_id, bounds.width,
            bounds.lower, bounds.upper, error_code,
        )
        return ResearchIdentifiedSetResponse(
            request_id=request_id,
            arbiter=req.arbiter,
            model_name=arb.model_name,
            point_estimate=result.p_positive,
            bounds=ManskiBoundsBlock(**bounds.as_dict()),
            would_have_been_rejected=error_code is not None,
            public_route_error_code=error_code,
            missing_features=list(bounds.missing_features),
            unobserved_unbounded=list(bounds.unobserved_unbounded),
            warnings=list(warnings),
            audit_receipt=receipt,
        )

    # ----------------------------------------------------------------------- #
    # /v1/model-cards — index every model card shipped with the API

    @app.get("/v1/model-cards", response_model=ModelCardsIndex)
    def list_model_cards(
        tenant: APIKey = Depends(require_api_key),
    ) -> ModelCardsIndex:
        """PLAN.md §5.6: `Public model card + errata page` — served as JSON
        index so a client can enumerate what cards exist without a directory
        listing. Raw markdown is served by `/v1/artifacts/docs/{filename}`.
        """
        cards_dir = _ARTIFACT_ROOTS[ArtifactCategory.docs]
        cards: list[ModelCardSummary] = []
        if cards_dir.is_dir():
            for path in sorted(cards_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace")
                first_h1 = ""
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("# "):
                        first_h1 = line[2:].strip()
                        break
                # A card is properly disclaimed if it either quotes the
                # RUO phrase verbatim OR references the RUO_DISCLAIMER
                # constant so a reader knows where the phrase lives. Both
                # patterns exist in this repo — accept either.
                ruo_ok = ("RESEARCH USE ONLY" in text) or ("RUO_DISCLAIMER" in text)
                cards.append(ModelCardSummary(
                    slug=path.stem,
                    title=first_h1 or path.stem,
                    n_bytes=len(text.encode("utf-8")),
                    honesty_markers={
                        "auroc_caveat_present": "AUROC" in text,
                        "ruo_disclaimer_present": ruo_ok,
                        "not_fda_cleared_note": "FDA" in text,
                    },
                ))
        return ModelCardsIndex(
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            cards=cards,
        )

    # ----------------------------------------------------------------------- #
    # /v1/artifacts/{category}/{filename} — path-traversal-safe streamer.
    # Mirrors org.backend/capabilities/progression_arbiter/router.py verbatim
    # for the security model: category-whitelisted + relative_to() containment.

    @app.get("/v1/artifacts/{category}/{filename}")
    def stream_artifact(
        category: str,
        filename: str,
        tenant: APIKey = Depends(require_api_key),
    ):
        try:
            cat = ArtifactCategory(category)
        except ValueError:
            raise HTTPException(400, f"invalid category: {category}")
        category_dir = _ARTIFACT_ROOTS[cat].resolve()
        # Reject empty / suspicious filenames early
        if not filename or filename in {".", ".."} or "\x00" in filename:
            raise HTTPException(400, "invalid filename")
        candidate = (category_dir / filename).resolve()
        try:
            candidate.relative_to(category_dir)
        except ValueError:
            raise HTTPException(403, "directory traversal forbidden")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, f"artifact not found: {filename}")
        # Guess a sensible media type
        media_type = "text/markdown" if candidate.suffix == ".md" else (
            "application/json" if candidate.suffix == ".json" else (
                "application/sql" if candidate.suffix == ".sql" else "text/plain"
            )
        )
        return JSONResponse(
            status_code=200,
            content={
                "category": category,
                "filename": filename,
                "media_type": media_type,
                "n_bytes": candidate.stat().st_size,
                "content": candidate.read_text(encoding="utf-8", errors="replace"),
                "disclaimer": RUO_DISCLAIMER,
                "caveat": AUROC_CAVEAT,
            },
        )

    # ----------------------------------------------------------------------- #
    # /v1/demo/case — server-hosted sample case for first-time users
    #
    # v0.2.2: previously the SPA shipped a small canned pathology string but
    # had no bundled DICOM, forcing users to hunt for one before they could
    # exercise the pipeline. The demo endpoint returns a fully-formed case
    # (real DICOM + pathology text + patient context) so a first-time user
    # can click "Load demo case" and see the full workflow run.
    #
    # DICOM provenance: CBIS-DDSM (helloerikaaa/cbis-ddsm-r on HuggingFace,
    # CC-BY-NC 4.0). Mass-Test_P_00016_LEFT_CC.dcm is the smallest of the
    # 5 fixtures we already ship (~14 MB). No auth needed to fetch.
    # Pathology text: synthetic luminal-A case (matches the frontend
    # LUMINAL_A_EXAMPLE constant). Not a real patient.
    #
    # v0.2.2 fix (2026-07-06): PUBLIC endpoint (NO require_api_key). The
    # whole point of the demo is to let an unauthenticated first-time
    # visitor click "Load demo case" and see the pipeline work without
    # first hunting for an API key. Locking this behind auth was
    # UX-hostile *and* semantically wrong: it's a static server-hosted
    # fixture (real CBIS-DDSM DICOM + synthetic luminal-A report +
    # patient context) with no tenant data, no per-call cost beyond the
    # one-time HF download that startup pre-warms into /tmp/oa-demo, and
    # no scenario where knowing a tenant identity matters. Any endpoint
    # that DOES touch tenant data (screening/biopsy/therapy/case_full)
    # continues to require the header.

    @app.get(
        "/v1/demo/case",
        response_model=DemoCaseResponse,
        summary="Return a fully-formed sample case for demoing the pipeline.",
    )
    def demo_case() -> DemoCaseResponse:
        from .demo_fixtures import DemoFixtureUnavailable, build_demo_case

        try:
            case = build_demo_case()
        except DemoFixtureUnavailable as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Demo case unavailable: {e}. "
                    "The server could not fetch the CBIS-DDSM fixture "
                    "(helloerikaaa/cbis-ddsm-r on HuggingFace). "
                    "This is transient — please retry in a moment."
                ),
            ) from e

        return DemoCaseResponse(
            dicom_bytes_b64=case.dicom_bytes_b64,
            dicom_source=case.dicom_source,
            dicom_sha256=case.dicom_sha256,
            dicom_size_bytes=case.dicom_size_bytes,
            report_text=case.report_text,
            patient_context=case.patient_context,
            warnings=case.warnings,
        )

    # ----------------------------------------------------------------------- #
    # /v1/demo/samples/{index,kind} — v0.3.0-alpha DEMO_MODE outputs.
    #
    # These serve pre-computed real inference outputs captured on our
    # workers (see src/oncology_arbiter/api/static/demo_samples/*.json).
    # Every sample envelope carries a `demo_provenance` sub-block with the
    # DICOM sha256, worker id, weights used, latency, and a plain-English
    # note explaining what was real vs. synthetic. When DEMO_MODE is off
    # these endpoints still serve the JSON — the samples are useful
    # reference outputs regardless of gate state.

    @app.get("/v1/demo/samples")
    def demo_samples_index():
        _sd = _demo_samples_dir()
        if _sd is None:
            return JSONResponse(
                status_code=200,
                content={
                    "samples": [],
                    "demo_mode": _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"),
                    "contact_url": _contact_url,
                    "note": "demo_samples directory not present in this build",
                },
            )
        samples = []
        for p in sorted(_sd.glob("*.json")):
            samples.append({
                "kind": p.stem,
                "path": f"/v1/demo/samples/{p.stem}",
                "size_bytes": p.stat().st_size,
            })
        return JSONResponse(
            status_code=200,
            content={
                "samples": samples,
                "demo_mode": _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"),
                "contact_url": _contact_url,
                "note": (
                    "Each sample is a real /v1/... response captured on our "
                    "workers with real weights loaded. See the "
                    "`demo_provenance` sub-block in each envelope for the "
                    "DICOM sha256, weights used, latency, and a plain-English "
                    "note on what was real vs. synthetic."
                ),
            },
        )

    @app.get("/v1/demo/samples/{kind}")
    def demo_samples_read(kind: str):
        # kind must be a safe identifier — reject any traversal or dotfile.
        if not kind or not kind.replace("_", "").isalnum():
            raise HTTPException(400, f"invalid sample kind: {kind!r}")
        _sd = _demo_samples_dir()
        if _sd is None:
            raise HTTPException(404, "demo_samples not present in this build")
        candidate = (_sd / f"{kind}.json").resolve()
        try:
            candidate.relative_to(_sd.resolve())
        except ValueError:
            raise HTTPException(403, "directory traversal forbidden")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(
                404,
                (
                    f"sample not found: {kind!r}. "
                    "Available: GET /v1/demo/samples"
                ),
            )
        # Serve as JSON with cache headers so the SPA can grab all four
        # once and paint quickly.
        return FileResponse(
            candidate,
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=300"},
        )


    # ----------------------------------------------------------------------- #
    # /v1/elo/rank — v0.3.0-alpha therapy candidate re-ranking
    #
    # PUBLIC endpoint (no require_api_key). Reasoning:
    #   * Pure-CPU deterministic Python tournament; no Modal / GPU cost.
    #   * On the public demo deploy the SPA's Elo panel must be usable by
    #     an anonymous browser session, matching the pattern used by
    #     /v1/demo/case and /v1/model-cards. Gating this behind an API
    #     key would silently break the read-only demo UX.
    #   * DEMO_MODE middleware still gates every other POST via its
    #     _DEMO_MODE_POST_ALLOWLIST — /v1/elo/rank is the only allow-listed
    #     entry.

    @app.post("/v1/elo/rank", response_model=EloRankResponse)
    def elo_rank(
        req: EloRankRequest,
    ) -> EloRankResponse:
        """L5 Co-Scientist: re-rank a caller-supplied set of therapy candidates.

        Two tournaments are run over the same set of drugs:

        1. **baseline_ranking** — the drug's own confidence + evidence + honesty
           markers, exactly as `run_co_scientist` scores hypotheses derived from
           stage envelopes. No disease context leaks in.
        2. **enriched_ranking** — same tournament but with per-drug score
           deltas from `req.modifiers` added to the Elo scorer. This is how
           HRD status boosts PARP inhibitors, PD-L1 CPS boosts pembrolizumab,
           and prior-lines burden penalises redosing options.

        The diff between the two rankings is returned as `matches[]` so the
        UI can render *why the order changed*. No LLM. Deterministic given
        `seed`. Free-tier safe (no model weights loaded).

        Honesty contract
        ----------------
        - `model_state`: `PROXY_CO_SCIENTIST` (deterministic, no live model).
        - Evidence URLs in each drug are echoed through the honesty gate:
          the response envelope's `evidence[]` field is the *union* of URLs
          registered on the input drugs, verbatim, so a caller can prove
          which sources fed the tournament.
        - `applied_modifiers` and `disease_context` are echoed verbatim on
          the response — no server-side re-derivation, no hidden bumps.
        """
        request_id = new_request_id()

        # Guard: reject duplicate drug_ids so the tournament stays well-defined.
        seen_ids: set[str] = set()
        for d in req.drugs:
            if d.drug_id in seen_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate drug_id: {d.drug_id!r}",
                )
            seen_ids.add(d.drug_id)

        # Warn on modifier keys that don't map to any drug — surface as
        # response warnings, don't fail the request.
        warnings: list[str] = []
        unknown_mods = set(req.modifiers.keys()) - seen_ids
        for k in sorted(unknown_mods):
            warnings.append(f"unknown_modifier_drug_id:{k}")

        # Convert EloDrugCandidate → Hypothesis for the tournament.
        from oncology_arbiter.orchestrator.co_scientist import (
            Hypothesis,
            rank_hypotheses,
        )

        def _to_hyp(d: EloDrugCandidate) -> Hypothesis:
            return Hypothesis(
                hyp_id=d.drug_id,
                stage="therapy",
                statement=f"{d.regimen} (line {d.line})",
                confidence=d.confidence,
                evidence=[e.model_dump() for e in d.evidence],
                honesty_markers=dict(d.honesty_markers),
            )

        hyps = [_to_hyp(d) for d in req.drugs]

        # -- Baseline tournament ------------------------------------------------
        baseline_entries = rank_hypotheses(
            hyps, k_factor=req.k_factor, seed=req.seed,
        )

        # -- Enriched tournament: apply modifiers by re-ranking with adjusted
        # confidence values. Modifiers act as additive deltas on the
        # `_score_hypothesis` value; since score = confidence + evidence +
        # markers, adding to confidence is byte-equivalent to adding to
        # the raw score. Confidence is clamped to [0,1] for schema honesty.
        # NB: modifier deltas are added to the raw score, not clamped, so
        # the tournament preserves the relative order induced by the delta.
        # For the response schema, EloRankedEntry.confidence is clamped to
        # [0,1] so a caller can never see "confidence = 1.15" on the wire.
        enriched_hyps: list[Hypothesis] = []
        for d in req.drugs:
            delta = float(req.modifiers.get(d.drug_id, 0.0))
            raw_conf = d.confidence + delta
            enriched_hyps.append(Hypothesis(
                hyp_id=d.drug_id,
                stage="therapy",
                statement=f"{d.regimen} (line {d.line})",
                confidence=raw_conf,
                evidence=[e.model_dump() for e in d.evidence],
                honesty_markers=dict(d.honesty_markers),
            ))
        enriched_entries = rank_hypotheses(
            enriched_hyps, k_factor=req.k_factor, seed=req.seed,
        )

        # -- Serialize both rankings --------------------------------------------
        drug_by_id: dict[str, EloDrugCandidate] = {d.drug_id: d for d in req.drugs}

        def _to_ranked(entries: list[Any]) -> list[EloRankedEntry]:
            out: list[EloRankedEntry] = []
            for rank, entry in enumerate(entries, start=1):
                h = entry.hypothesis
                src = drug_by_id[h.hyp_id]
                out.append(EloRankedEntry(
                    rank=rank,
                    drug_id=h.hyp_id,
                    regimen=src.regimen,
                    line=src.line,
                    rating=round(entry.rating, 4),
                    wins=entry.wins,
                    losses=entry.losses,
                    draws=entry.draws,
                    confidence=max(0.0, min(1.0, h.confidence)),
                    honesty_markers=dict(h.honesty_markers),
                    n_evidence=len(h.evidence),
                ))
            return out

        baseline_ranking = _to_ranked(baseline_entries)
        enriched_ranking = _to_ranked(enriched_entries)

        # -- Match records: diff by drug_id, ordered by enriched rank -----------
        baseline_by_id = {e.drug_id: e for e in baseline_ranking}
        matches: list[EloMatchRecord] = []
        for e in enriched_ranking:
            b = baseline_by_id[e.drug_id]
            delta = float(req.modifiers.get(e.drug_id, 0.0))
            reason = _elo_reason_for_modifier(e.drug_id, delta, req.disease_context)
            matches.append(EloMatchRecord(
                drug_id=e.drug_id,
                regimen=e.regimen,
                line=e.line,
                baseline_rank=b.rank,
                enriched_rank=e.rank,
                baseline_rating=b.rating,
                enriched_rating=e.rating,
                rank_delta=b.rank - e.rank,
                rating_delta=round(e.rating - b.rating, 4),
                applied_modifier=delta,
                reason=reason,
            ))

        # -- Provenance + evidence ---------------------------------------------
        prov = Provenance(
            model_name="oa/co_scientist_elo@v0.3.0-alpha",
            model_version="v0.3.0-alpha",
            model_state=ModelState.PROXY_CO_SCIENTIST,
            request_id=request_id,
        )

        # Union of evidence URLs across all drugs, dedup by URL, order-preserving.
        seen_urls: set[str] = set()
        merged_evidence: list[EvidenceRecord] = []
        for d in req.drugs:
            for e in d.evidence:
                if e.url not in seen_urls:
                    seen_urls.add(e.url)
                    merged_evidence.append(e)

        gate = HonestyGateReport(
            seen_urls_count=len(seen_urls),
            evidence_kept=len(merged_evidence),
            evidence_dropped=0,
        )

        log_event(
            request_id, "/v1/elo/rank",
            model_state=ModelState.PROXY_CO_SCIENTIST.value,
            patient_id_hash=None,
            extra={
                "n_candidates": len(req.drugs),
                "n_modifiers": len(req.modifiers),
                "n_unknown_modifiers": len(unknown_mods),
                "seed": req.seed,
                "k_factor": req.k_factor,
                "disease_context_keys": sorted(req.disease_context.keys()),
            },
            tenant_id=None,  # public route: no authenticated tenant
        )

        return EloRankResponse(
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            provenance=prov,
            honesty_gate=gate,
            evidence=merged_evidence,
            warnings=warnings,
            baseline_ranking=baseline_ranking,
            enriched_ranking=enriched_ranking,
            matches=matches,
            disease_context=req.disease_context,
            applied_modifiers=dict(req.modifiers),
            n_candidates=len(req.drugs),
        )

    # ----------------------------------------------------------------------- #
    # /v1/co_scientist/run — standalone 4-phase loop (PLAN §5 PR #5)
    #
    # Surfaces `run_co_scientist(...)` as a first-class endpoint so callers
    # can run the honesty tournament without piping through /v1/case/full.
    # Deterministic, no live LLM, free-tier safe.
    #
    # The load-bearing bit is REFLECT: any evidence URL not in
    # `req.seed_urls` is dropped by `reflect_hypotheses`. A hostile input
    # with N fake URLs will see `urls_dropped_hallucinated >= N` and
    # matching warnings in the response envelope.

    @app.post("/v1/offline_ranker/run", response_model=CoScientistRunResponse)
    def co_scientist_run(
        req: CoScientistRunRequest,
    ) -> CoScientistRunResponse:
        """Run the 4-phase Co-Scientist loop over caller-supplied envelopes.

        Phases: generate → reflect → rank → evolve → rank.

        Honesty contract
        ----------------
        - `req.seed_urls` is the ONLY authority on what the tool-loop
          actually fetched. Any evidence URL NOT in this set is stripped
          by REFLECT before it can win Elo points.
        - `urls_dropped_hallucinated` is the count of URLs stripped.
        - `hypotheses_dropped` is the count of hypotheses whose evidence
          list was emptied by REFLECT (they're kept in `hypotheses[]` but
          a caller SHOULD treat them as untrusted).
        - Deterministic given identical inputs — no LLM sampling, no
          time-dependent randomness beyond the fixed-seed pair-order
          shuffle inside `rank_hypotheses`.
        """
        request_id = new_request_id()

        # Also parse hallucination markers from `screening.evidence[].url`,
        # etc. — anything already in the input envelopes counts as seen
        # unless the caller explicitly limits `seed_urls`. This mirrors the
        # behaviour of /v1/case/full's Co-Scientist call.
        seen_urls: set[str] = set(req.seed_urls or [])

        # Import inside the handler so cold-start of the module doesn't
        # depend on orchestrator imports (which pull in numpy).
        from oncology_arbiter.orchestrator.co_scientist import run_co_scientist

        cs_out = run_co_scientist(
            screening=req.screening,
            biopsy=req.biopsy,
            therapy=req.therapy,
            seen_urls=seen_urls,
            top_n_evolve=req.top_n_evolve,
            n_variants=req.n_variants,
            return_top=req.return_top,
        )

        # Count what REFLECT actually removed. Two metrics matter:
        #
        # 1. `urls_dropped_hallucinated`: the number of DISTINCT URLs that
        #    appeared in the input envelopes but were NOT in `seed_urls`.
        #    This is what the "5 fake URLs → all dropped by REFLECT" test
        #    checks — the honesty gate must strip every unseen URL, no
        #    matter how many hypotheses reference it.
        # 2. `hypotheses_dropped`: the number of hypotheses whose evidence
        #    list was emptied by REFLECT. Read from the warnings channel
        #    (`no_evidence_after_reflect:<hyp_id>`), which
        #    `reflect_hypotheses` emits verbatim.
        #
        # Both are exposed on the wire so a reviewer can prove the honesty
        # gate did its job on hostile input.
        proposed_urls: set[str] = set()
        for env in (req.screening, req.biopsy, req.therapy):
            if not env:
                continue
            for e in env.get("evidence") or []:
                if isinstance(e, dict) and e.get("url"):
                    proposed_urls.add(e["url"])
            for opt in env.get("recommended_options") or []:
                for e in (opt or {}).get("evidence") or []:
                    if isinstance(e, dict) and e.get("url"):
                        proposed_urls.add(e["url"])
        urls_dropped_total = len(proposed_urls - seen_urls)
        hyps_dropped = sum(
            1 for w in (cs_out.get("warnings") or [])
            if str(w).startswith("no_evidence_after_reflect:")
        )

        # Provenance / honesty envelope. We echo the seen_urls set on
        # `honesty_gate` so a caller can prove what was in scope; the
        # evidence field is left empty because run_co_scientist doesn't
        # produce a merged evidence list of its own (each hypothesis
        # carries its own `evidence[]`).
        prov = Provenance(
            model_name="oa/co_scientist@v0.4.0-alpha",
            model_version="v0.4.0-alpha",
            model_state=ModelState.PROXY_CO_SCIENTIST,
            request_id=request_id,
        )
        gate = HonestyGateReport(
            seen_urls_count=len(seen_urls),
            evidence_kept=0,  # Co-Scientist doesn't build a merged evidence list
            evidence_dropped=urls_dropped_total,
        )

        log_event(
            request_id, "/v1/offline_ranker/run",
            model_state=ModelState.PROXY_CO_SCIENTIST.value,
            patient_id_hash=None,
            extra={
                "has_screening": req.screening is not None,
                "has_biopsy": req.biopsy is not None,
                "has_therapy": req.therapy is not None,
                "n_seed_urls": len(seen_urls),
                "initial_count": cs_out.get("initial_count", 0),
                "after_reflect": cs_out.get("after_reflect", 0),
                "after_evolve": cs_out.get("after_evolve", 0),
                "urls_dropped_hallucinated": urls_dropped_total,
                "hypotheses_dropped": hyps_dropped,
                "top_n_evolve": req.top_n_evolve,
                "n_variants": req.n_variants,
                "return_top": req.return_top,
            },
            tenant_id=None,  # public route, matches /v1/elo/rank
        )

        return CoScientistRunResponse(
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            provenance=prov,
            honesty_gate=gate,
            evidence=[],  # per-hypothesis evidence lives on `hypotheses[]`
            warnings=list(cs_out.get("warnings") or []),
            phases=list(cs_out.get("phases") or []),
            hypotheses=list(cs_out.get("hypotheses") or []),
            initial_count=int(cs_out.get("initial_count", 0)),
            after_reflect=int(cs_out.get("after_reflect", 0)),
            after_evolve=int(cs_out.get("after_evolve", 0)),
            urls_dropped_hallucinated=urls_dropped_total,
            hypotheses_dropped=hyps_dropped,
        )

    # ----------------------------------------------------------------------- #
    # Production dynamic tumor board and full-case specialist assembly

    _SUPPORTED_CANCERS = {"breast", "nsclc", "hgsoc"}

    def _breast_dss_from_features(features: Any) -> BreastDssPrognosis:
        from oncology_arbiter.models.breast_dss_arbiter import score_breast_dss

        result = score_breast_dss({
            "age": features.age,
            "tumor_size_mm": features.tumor_size_mm,
            "nodes_positive": features.nodes_positive,
            "grade": features.grade,
            "er_pos": features.er_positive,
            "pr_pos": features.pr_positive,
            "her2_pos": features.her2_positive,
        })
        return BreastDssPrognosis(
            model_name=result.model_name,
            disease_specific_mortality_score=result.score,
            logit=result.logit,
            term_contributions=result.term_contributions,
            artifact_sha256=result.artifact_sha256,
            n_training=result.n_training,
            events=result.events,
            oof_auroc=result.oof_auroc,
            caveats=list(result.caveats),
        )

    def _ovarian_retirement(request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        block = {
            "model_name": "ovarian_mortality_arbiter_v3_tcga",
            "model_state": "retired",
            "endpoint": "2-year overall mortality",
            "retirement_reason": (
                "FIGO stage did not demonstrate incremental discrimination over age; "
                "the paired OOF AUROC increment was +0.02475247524752467 with an "
                "approximate 95% interval spanning zero."
            ),
            "retirement_evidence": {
                "cohort": "TCGA-OV",
                "n": 271,
                "age_only_oof_auroc": 0.658057110058832,
                "age_plus_figo_oof_auroc": 0.6828095853063567,
                "delta_oof_auroc": 0.02475247524752467,
                "delta_ci95_approx": [-0.041, 0.095],
                "external_validation": "no_feature_compatible_external_cohort_available",
            },
            "warning": "Retired research model; no patient probability, risk bucket, or driving feature is emitted.",
        }
        receipt = {
            "stage": "ovarian_prognostic_arbiter",
            "required": False,
            "status": "retired",
            "request_id": request_id,
            "service_name": "oncology-arbiter",
            "model_name": "ovarian_mortality_arbiter_v3_tcga",
            "model_version": "v3-retired",
            "warnings": ["retired_no_incremental_figo_discrimination_over_age"],
            "error": None,
        }
        return block, receipt

    def _run_medgemma_co_scientist(
        *,
        context: dict[str, Any],
        request_id: str,
        requested: bool,
        has_patient_signal: bool,
        receipts: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not requested:
            receipts.append(_skipped_stage_receipt(
                "medgemma_co_scientist", request_id, "Co-Scientist was not requested"
            ))
            return None
        if not has_patient_signal:
            receipts.append(_skipped_stage_receipt(
                "medgemma_co_scientist", request_id,
                "No validated patient stage output was available for synthesis",
            ))
            return None
        if not os.environ.get("MEDGEMMA_MODAL_URL"):
            receipts.append(_failed_stage_receipt(
                "medgemma_co_scientist", False, request_id,
                "medgemma_not_configured",
                "MEDGEMMA_MODAL_URL is required; alternate LLMs and deterministic Elo are not substitutes.",
                service_name="medgemma-27b",
            ))
            return None
        try:
            from oncology_arbiter.agents.supervisor import execute_stage

            result = execute_stage(
                "case_full",
                context,
                n_hypotheses=6,
                n_evidence_top_k=3,
            )
            output = result.as_dict()
            if (
                result.model_state != "executed"
                or output.get("llm_provider") != "medgemma_modal"
                or output.get("llm_model") != "google/medgemma-27b-it"
                or output.get("llm_app_version") != "medgemma-27b-modal-v0.5.0-production-ready"
                or not output.get("llm_request_ids")
            ):
                raise RuntimeError(
                    "MedGemma Co-Scientist did not complete with required provider provenance"
                )
            receipts.append({
                "stage": "medgemma_co_scientist",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "medgemma-27b",
                "endpoint_label": os.environ.get("MEDGEMMA_MODAL_URL"),
                "app_version": output["llm_app_version"],
                "model_name": output["llm_model"],
                "model_version": "27b-it",
                "latency_ms": round(float(output.get("llm_latency_s") or 0.0) * 1000.0, 3),
                "warnings": list(output.get("llm_honesty_warnings") or []),
                "error": None,
            })
            return output
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "medgemma_co_scientist", False, request_id,
                "medgemma_unavailable", f"{type(exc).__name__}: {exc}",
                service_name="medgemma-27b",
            ))
            return None

    def _assemble_dynamic_tumor_board(
        req: DynamicTumorBoardRequest,
        *,
        request_id: str,
    ) -> DynamicTumorBoardResponse:
        receipts: list[dict[str, Any]] = []
        warnings: list[str] = []
        ovarian_arbiter: dict[str, Any] | None = None
        report_parse: ReportParseBlock | None = None
        phikon_embedding: dict[str, Any] | None = None
        case_manifest: dict[str, Any] | None = None
        dss_prognosis: BreastDssPrognosis | None = None
        therapy_bridge: dict[str, Any] | None = None
        therapy_triage: ArbiterScore | None = None

        if req.cancer == "hgsoc":
            ovarian_arbiter, retired_receipt = _ovarian_retirement(request_id)
            receipts.append(retired_receipt)
        else:
            receipts.append(_skipped_stage_receipt(
                "ovarian_prognostic_arbiter", request_id, "not applicable to breast cancer"
            ))

        if req.case_id:
            try:
                from oncology_arbiter.models.specialist_clients import CaseStorageClient

                call = CaseStorageClient().manifest(req.case_id, request_id=request_id, required=True)
                case_manifest = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "case_storage", True, request_id,
                    getattr(exc, "code", "case_storage_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="case-storage", input_reference=f"case_id:{req.case_id}",
                ))
        else:
            receipts.append(_skipped_stage_receipt(
                "case_storage", request_id, "no case_id supplied"
            ))

        if req.report_text:
            report_parse, _panel, _grade, receipt = _clinicalbert_biopsy_parse(
                req.report_text, request_id=request_id
            )
            receipts.append(receipt)
        else:
            receipts.append(_skipped_stage_receipt(
                "clinicalbert_pathology_parse", request_id, "no report_text supplied"
            ))

        if req.pathology_image_b64:
            try:
                from oncology_arbiter.models.specialist_clients import PhikonClient

                image_bytes = _decode_bytes_arg(req.pathology_image_b64) or b""
                call = PhikonClient().embed(image_bytes, request_id=request_id, required=True)
                phikon_embedding = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "phikon_embedding", True, request_id,
                    getattr(exc, "code", "phikon_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="phikon",
                ))
        else:
            receipts.append(_skipped_stage_receipt(
                "phikon_embedding", request_id, "no pathology image supplied"
            ))

        if req.cancer == "breast" and req.dss_features is not None:
            dss_prognosis = _breast_dss_from_features(req.dss_features)
            receipts.append({
                "stage": "breast_dss_prognosis",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": dss_prognosis.model_name,
                "model_version": "v3",
                "artifact_sha256": dss_prognosis.artifact_sha256,
                "input_reference": "explicit_complete_seven_feature_vector",
                "latency_ms": 0.0,
                "warnings": ["prognosis_under_metabric_treatment_mix_not_treatment_benefit"],
                "error": None,
            })
        else:
            reason = (
                "not applicable to HGSOC"
                if req.cancer == "hgsoc"
                else "complete explicit seven-feature vector not supplied"
            )
            receipts.append(_skipped_stage_receipt(
                "breast_dss_prognosis", request_id, reason
            ))

        mutation_payload = [item.model_dump(exclude_none=True) for item in req.mutations]
        germline_payload = [item.model_dump(exclude_none=True) for item in req.germline_mutations]
        ranked_payload = [item.model_dump(exclude_none=True) for item in req.ranked_drugs]
        if mutation_payload or germline_payload:
            try:
                from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

                call = SLTherapyBridgeClient().run({
                    "disease": "high-grade serous ovarian cancer" if req.cancer == "hgsoc" else "breast cancer",
                    "cancer_type": req.cancer,
                    "mutations": mutation_payload,
                    "germline_mutations": germline_payload,
                    "ranked_drugs": ranked_payload,
                    "rs_features": req.clinical_baseline,
                    "include_explanations": False,
                }, request_id=request_id, required=True)
                therapy_bridge = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "synthetic_lethality_therapy_bridge", True, request_id,
                    getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="crispro-backend-v2",
                ))
        else:
            receipts.append(_skipped_stage_receipt(
                "synthetic_lethality_therapy_bridge", request_id,
                "no patient mutation profile supplied; no therapy recommendation computed",
            ))

        if req.cancer == "breast" and req.therapy_features is not None:
            therapy_triage = _score_explicit_therapy_triage(
                req.therapy_features.model_dump(),
            )
            if therapy_triage is not None:
                receipts.append({
                    "stage": "therapy_research_triage",
                    "required": False,
                    "status": "succeeded",
                    "request_id": request_id,
                    "service_name": "oncology-arbiter",
                    "model_name": therapy_triage.model_name,
                    "model_version": "template-v0",
                    "input_reference": "explicit_complete_patient_features",
                    "latency_ms": 0.0,
                    "warnings": ["illustrative_research_triage_not_recommendation_or_treatment_benefit"],
                    "error": None,
                })
            else:
                receipts.append(_skipped_stage_receipt(
                    "therapy_research_triage", request_id,
                    "complete explicit triage vector required; missing nodes never use unknown=0.5",
                ))
        else:
            receipts.append(_skipped_stage_receipt(
                "therapy_research_triage", request_id,
                "not applicable to HGSOC" if req.cancer == "hgsoc" else "no explicit triage features supplied",
            ))

        has_signal = any((
            report_parse is not None,
            phikon_embedding is not None,
            case_manifest is not None,
            dss_prognosis is not None,
            therapy_bridge is not None,
            bool(req.clinical_baseline),
        ))
        co_scientist = _run_medgemma_co_scientist(
            context={
                "cancer": req.cancer,
                "clinical_baseline": req.clinical_baseline,
                "report_parse": report_parse.model_dump(mode="json") if report_parse else None,
                "phikon_embedding": phikon_embedding,
                "dss_prognosis": dss_prognosis.model_dump(mode="json") if dss_prognosis else None,
                "therapy_bridge": therapy_bridge,
            },
            request_id=request_id,
            requested=req.run_co_scientist,
            has_patient_signal=has_signal,
            receipts=receipts,
        )

        status = _pipeline_status(receipts)
        state = ModelState.UNAVAILABLE if status == "failed_required_stage" else ModelState.LOADED
        env = _envelope(request_id, model_state=state, model_name="specialist-stack-composite")
        env["warnings"] = warnings
        return DynamicTumorBoardResponse(
            **env,
            pipeline_status=status,
            cancer=req.cancer,
            stage_receipts=receipts,
            ovarian_arbiter=ovarian_arbiter,
            report_parse=report_parse,
            phikon_embedding=phikon_embedding,
            case_manifest=case_manifest,
            dss_prognosis=dss_prognosis,
            therapy_bridge=therapy_bridge,
            therapy_triage=therapy_triage,
            co_scientist=co_scientist,
        )

    @app.post("/v1/tumor_board/dynamic", response_model=DynamicTumorBoardResponse)
    def dynamic_tumor_board(
        req: DynamicTumorBoardRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> DynamicTumorBoardResponse:
        request_id = new_request_id()
        response = _assemble_dynamic_tumor_board(
            req, request_id=request_id
        )
        log_event(
            request_id, "/v1/tumor_board/dynamic",
            model_state=response.provenance.model_state.value,
            patient_id_hash=None,
            extra={
                "cancer": req.cancer,
                "pipeline_status": response.pipeline_status,
                "stage_statuses": [receipt.status for receipt in response.stage_receipts],
            },
            tenant_id=tenant.tenant_id,
        )
        return response

    @app.post("/v1/case/full", response_model=FullCaseResponse)
    def case_full(
        req: FullCaseRequest,
        cancer: str = Query(default="breast", description="Cancer track: breast, nsclc, or hgsoc."),
        tenant: APIKey = Depends(require_api_key),
    ) -> FullCaseResponse:
        cancer_norm = cancer.lower().strip()
        if cancer_norm not in _SUPPORTED_CANCERS:
            raise HTTPException(400, f"Unsupported cancer={cancer!r}; supported={sorted(_SUPPORTED_CANCERS)}")
        request_id = new_request_id()

        if cancer_norm in {"breast", "hgsoc"}:
            biopsy_input = req.biopsy_input
            dynamic_request = DynamicTumorBoardRequest(
                cancer=cancer_norm,
                case_id=req.case_id,
                report_text=biopsy_input.report_text if biopsy_input else None,
                pathology_image_b64=biopsy_input.wsi_bytes_b64 if biopsy_input else None,
                dss_features=(biopsy_input.dss_features if biopsy_input and cancer_norm == "breast" else None),
                therapy_features=req.therapy_features if cancer_norm == "breast" else None,
                mutations=req.mutations,
                germline_mutations=req.germline_mutations,
                ranked_drugs=req.ranked_drugs,
                clinical_baseline=req.therapy_context.model_dump(mode="json"),
                run_co_scientist=req.run_co_scientist,
            )
            dynamic = _assemble_dynamic_tumor_board(
                dynamic_request, request_id=request_id,
            )
            screening = screening_analyze(req.screening_input, tenant=tenant) if req.screening_input else None
            log_event(
                request_id, "/v1/case/full",
                model_state=dynamic.provenance.model_state.value,
                patient_id_hash=None,
                extra={"cancer": cancer_norm, "pipeline_status": dynamic.pipeline_status},
                tenant_id=tenant.tenant_id,
            )
            return FullCaseResponse(
                **_envelope(
                    request_id,
                    model_state=dynamic.provenance.model_state,
                    model_name="specialist-stack-composite",
                ),
                warnings=list(dynamic.warnings),
                pipeline_status=dynamic.pipeline_status,
                cancer=cancer_norm,
                stage_receipts=dynamic.stage_receipts,
                ovarian_arbiter=dynamic.ovarian_arbiter,
                case_manifest=dynamic.case_manifest,
                report_parse=dynamic.report_parse,
                phikon_embedding=dynamic.phikon_embedding,
                dss_prognosis=dynamic.dss_prognosis,
                therapy_bridge=dynamic.therapy_bridge,
                therapy_triage=dynamic.therapy_triage,
                co_scientist=dynamic.co_scientist,
                screening=screening,
                biopsy=None,
                therapy=None,
                nsclc=None,
                elo_ranked_hypotheses=[],
            )

        receipts: list[dict[str, Any]] = []
        warnings: list[str] = []
        case_manifest: dict[str, Any] | None = None
        luna_output: dict[str, Any] | None = None
        report_parse: ReportParseBlock | None = None
        therapy_bridge: dict[str, Any] | None = None

        if not req.case_id:
            receipts.append(_failed_stage_receipt(
                "case_storage", True, request_id, "case_id_required",
                "NSCLC production inference requires a verified case-storage case_id.",
                service_name="case-storage",
            ))
            receipts.append(_failed_stage_receipt(
                "luna16_detection", True, request_id, "blocked_by_missing_case_id",
                "LUNA16 cannot run without a verified case-storage case_id.",
                service_name="luna16-infer",
            ))
            # Surface the suppression at the top level too. stage_receipts and
            # pipeline_status already carry it, but an all-None NSCLC envelope
            # with an empty warnings[] can be misread by a downstream consumer
            # as a completed inference that simply found nothing.
            warnings.append(
                "nsclc_required_stages_not_executed:no_verified_case_id_supplied:"
                "empty_result_is_not_a_negative_finding"
            )
        else:
            try:
                from oncology_arbiter.models.specialist_clients import CaseStorageClient, Luna16Client

                manifest_call = CaseStorageClient().manifest(req.case_id, request_id=request_id, required=True)
                case_manifest = manifest_call.output
                receipts.append(manifest_call.receipt)
                luna_call = Luna16Client().detect(req.case_id, request_id=request_id, required=True)
                luna_output = luna_call.output
                luna_call.receipt["input_reference"] = (
                    f"case_id:{req.case_id};manifest_sha256:{case_manifest['manifest_sha256']}"
                )
                receipts.append(luna_call.receipt)
            except Exception as exc:
                failed_stage = "luna16_detection" if case_manifest is not None else "case_storage"
                receipts.append(_failed_stage_receipt(
                    failed_stage, True, request_id,
                    getattr(exc, "code", f"{failed_stage}_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="luna16-infer" if failed_stage == "luna16_detection" else "case-storage",
                    input_reference=f"case_id:{req.case_id}",
                ))
                if failed_stage == "case_storage":
                    receipts.append(_failed_stage_receipt(
                        "luna16_detection", True, request_id, "blocked_by_case_storage",
                        "LUNA16 was not called because case-storage manifest verification failed.",
                        service_name="luna16-infer", input_reference=f"case_id:{req.case_id}",
                    ))

        if req.biopsy_input and req.biopsy_input.report_text:
            report_parse, _panel, _grade, receipt = _clinicalbert_biopsy_parse(
                req.biopsy_input.report_text, request_id=request_id
            )
            receipts.append(receipt)
        else:
            receipts.append(_skipped_stage_receipt(
                "clinicalbert_pathology_parse", request_id, "no report_text supplied"
            ))

        mutation_payload = [item.model_dump(exclude_none=True) for item in req.mutations]
        germline_payload = [item.model_dump(exclude_none=True) for item in req.germline_mutations]
        if mutation_payload or germline_payload:
            try:
                from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

                call = SLTherapyBridgeClient().run({
                    "disease": "non-small cell lung cancer",
                    "cancer_type": "nsclc",
                    "mutations": mutation_payload,
                    "germline_mutations": germline_payload,
                    "ranked_drugs": [item.model_dump(exclude_none=True) for item in req.ranked_drugs],
                    "include_explanations": False,
                }, request_id=request_id, required=True)
                therapy_bridge = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "synthetic_lethality_therapy_bridge", True, request_id,
                    getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="crispro-backend-v2",
                ))
        else:
            receipts.append(_skipped_stage_receipt(
                "synthetic_lethality_therapy_bridge", request_id,
                "no patient mutation profile supplied; no therapy recommendation computed",
            ))

        co_scientist = _run_medgemma_co_scientist(
            context={
                "cancer": "nsclc",
                "case_manifest": case_manifest,
                "luna16_detection": luna_output,
                "report_parse": report_parse.model_dump(mode="json") if report_parse else None,
                "therapy_bridge": therapy_bridge,
                "patient_context": req.therapy_context.model_dump(mode="json"),
            },
            request_id=request_id,
            requested=req.run_co_scientist,
            has_patient_signal=any((luna_output, report_parse, therapy_bridge)),
            receipts=receipts,
        )

        status = _pipeline_status(receipts)
        state = ModelState.LOADED_LUNA16_RETINANET if luna_output is not None else ModelState.UNAVAILABLE
        luna_block = None
        if luna_output is not None:
            luna_block = {
                "bundle_version": luna_output["bundle_version"],
                "n_detections": int(luna_output.get("n_detections", len(luna_output.get("detections") or []))),
                "top_score": float(luna_output.get("top_score", 0.0)),
                "detections": list(luna_output.get("detections") or []),
                "inference_seconds": float(luna_output.get("inference_seconds", 0.0)),
                "preprocessing_summary": dict(luna_output.get("preprocessing_summary") or {}),
            }
        nsclc = NsclcResponse(
            model_state=state,
            model_name=(luna_output or {}).get("model_name", "monai/lung_nodule_ct_detection@0.6.9"),
            warnings=warnings,
            luna16=luna_block,
            parsed_report=report_parse.parsed_entities if report_parse else None,
            parsed_report_provenance=(
                {
                    "parser_id": report_parse.parser_id,
                    "app_version": report_parse.app_version,
                    "model_sha256": report_parse.model_sha256,
                    "metrics_sha256": report_parse.metrics_sha256,
                    "window_tokens": report_parse.window_tokens,
                    "overlap_tokens": report_parse.overlap_tokens,
                    "window_aggregation": report_parse.window_aggregation,
                }
                if report_parse else None
            ),
        )
        log_event(
            request_id, "/v1/case/full", model_state=state.value,
            patient_id_hash=None,
            extra={"cancer": "nsclc", "pipeline_status": status},
            tenant_id=tenant.tenant_id,
        )
        return FullCaseResponse(
            **_envelope(request_id, model_state=state, model_name="specialist-stack-composite"),
            warnings=warnings,
            pipeline_status=status,
            cancer="nsclc",
            stage_receipts=receipts,
            case_manifest=case_manifest,
            report_parse=report_parse,
            luna16_detection=luna_output,
            therapy_bridge=therapy_bridge,
            co_scientist=co_scientist,
            screening=None,
            biopsy=None,
            therapy=None,
            nsclc=nsclc,
            elo_ranked_hypotheses=[],
        )

    # ----------------------------------------------------------------------- #
    # /v1/tumor_board/bundle — v0.4.0-alpha AK MBD4-LOF tumor board upload
    #
    # Accepts a TumorBoardBundle (contract
    # 'tumor_board.v3.multimodal-with-manuscript-claims') from an authorised
    # caller and echoes it back with a computed bundle sha256 + envelope. Two
    # invariants:
    #
    #   1. The route is a NULL-OP with respect to disk state — no PHI
    #      persistence, no dossier writes, no side effects other than
    #      the audit ledger. Real dossier writes happen in
    #      crispro-backend-v2, not here. This route lets clinicians *upload*
    #      a bundle for downstream Modal enrichment or Elo re-ranking
    #      without the SPA needing to shove multi-megabyte JSON through
    #      /v1/case/full.
    #
    #   2. If HIPAA_MODE=true is set on the deploy env, the response
    #      surface is redacted via the same HIPAAPIIMiddleware pattern
    #      described in the AK integration doc. The current stub returns
    #      the bundle verbatim under model_state=LOADED_HIPAA_REDACTOR so
    #      the SPA can trip its own PHI-check assertions; live redaction
    #      wiring lands in PR #6 alongside the audit ledger tests.

    @app.post("/v1/tumor_board/bundle", response_model=TumorBoardBundleResponse)
    def tumor_board_bundle(
        bundle: TumorBoardBundle,
        tenant: APIKey = Depends(require_api_key),
    ) -> TumorBoardBundleResponse:
        """Accept a tumor board bundle, compute its SHA-256, return envelope.

        The bundle is validated by pydantic against the
        `tumor_board.v3.multimodal-with-manuscript-claims` contract. Any
        deviation (missing field, wrong claim type, wrong manuscript sha)
        raises 422 before we get here.
        """
        import hashlib
        import json as _json
        import os

        # Reject bundles that don't match the pinned contract version — the
        # SPA has no fallback renderer and mis-versioned bundles must not
        # silently downgrade to a stale UI.
        expected = "tumor_board.v3.multimodal-with-manuscript-claims"
        if bundle.contract_version != expected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"contract_version={bundle.contract_version!r} "
                    f"does not match server-pinned {expected!r}. "
                    "See docs/audit/AK_MBD4_INTEGRATION.md."
                ),
            )

        # Canonical bundle hash: pydantic dumps with sort_keys+ensure_ascii
        # so an identical bundle round-tripped through v1 matches the
        # audit ledger entry emitted by crispro-backend-v2 side-by-side.
        canonical = _json.dumps(
            bundle.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        bundle_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        request_id = new_request_id()
        hipaa_mode = os.environ.get("HIPAA_MODE", "").lower() in {"1", "true", "yes"}

        # Model state depends on whether HIPAA redaction was applied. The
        # actual redaction stub lives in a follow-up PR; today this route
        # just tags the state so the frontend surface (SlEvidenceMoat) can
        # render the PHI-safety banner.
        state = (
            ModelState.LOADED_HIPAA_REDACTOR if hipaa_mode
            else ModelState.LOADED_AK_BUNDLE
        )

        log_event(
            request_id,
            "/v1/tumor_board/bundle",
            model_state=state.value,
            patient_id_hash=hashlib.sha256(
                bundle.patient_id.encode("utf-8")
            ).hexdigest()[:12],
            extra={
                "contract_version": bundle.contract_version,
                "backend_head_sha": bundle.synthetic_lethality
                    .provenance.backend_head_sha[:12],
                "manuscript_sha": bundle.synthetic_lethality
                    .provenance.manuscript_repo_sha_at_audit[:12],
                "n_rows": len(
                    bundle.synthetic_lethality.provenance.evidence_matrix.rows
                ),
                "n_drugs": len(bundle.synthetic_lethality.recommended_drugs),
                "hipaa_mode": hipaa_mode,
            },
            tenant_id=tenant.tenant_id,
        )

        return TumorBoardBundleResponse(
            **_envelope(
                request_id,
                model_state=state,
                model_name="tumor_board_v3_multimodal",
            ),
            bundle=bundle,
            bundle_sha256=bundle_sha,
            persisted_path=None,  # no disk persistence yet — see PR #6
        )

    # ----------------------------------------------------------------------- #
    # /ui — optional static frontend mount
    #
    # Serving the SPA is opt-in via ONCOLOGY_ARBITER_SERVE_FRONTEND=1 so the
    # backend can boot in Docker/CI without a Node build. When enabled, the
    # frontend bundle lives at src/oncology_arbiter/api/static/dist/ and is
    # produced by `npm --prefix frontend run build`. Base path is /ui/ so it
    # never collides with /v1/* API routes.
    if _is_env_true("ONCOLOGY_ARBITER_SERVE_FRONTEND"):
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles

        static_root = Path(__file__).parent / "static" / "dist"
        if static_root.is_dir() and (static_root / "index.html").is_file():
            # `html=True` makes StaticFiles fall through to index.html for any
            # sub-path (SPA routing). It still 404s on missing static assets.
            app.mount("/ui", StaticFiles(directory=str(static_root), html=True), name="ui")

            # v0.2.1: bare "/" was returning 404 because there is no root
            # handler. Redirect to /ui/ so clinicians typing the base URL
            # land on the SPA instead of a JSON not-found. 307 preserves the
            # HTTP method (harmless for GET, correct for the general case).
            @app.get("/", include_in_schema=False)
            def _root_to_ui() -> RedirectResponse:
                return RedirectResponse(url="/ui/", status_code=307)

    return app
