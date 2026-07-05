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
import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__

from .audit import log_event, new_request_id
from ..auth import APIKey, bootstrap_from_env, require_api_key
from ..observability import (
    RequestIdMiddleware,
    configure_logging,
    get_logger,
)
from .schemas import (
    ApiEnvelope,
    ArbiterScore,
    ArtifactCategory,
    BiopsyReceptorPanel,
    BiopsyRequest,
    BiopsyResponse,
    NsclcCTInput,
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
    ScreeningRequest,
    ScreeningResponse,
    TherapyOption,
    TherapyRequest,
    TherapyResponse,
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


def _score_arbiter(name: str, features: dict[str, Any]) -> ArbiterScore:
    """Load the named L2 arbiter and score `features`.

    Wraps :func:`oncology_arbiter.arbiter.load_arbiter` and marshals the
    :class:`ArbiterResult` into the wire-level :class:`ArbiterScore` pydantic.
    """
    from oncology_arbiter.arbiter import load_arbiter
    arb = load_arbiter(name)
    r = arb.score(features)
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
_SIGLIP_PROXY_SINGLETON: Any = None


def _is_env_true(name: str) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_medsiglip() -> Any:
    """Lazy-construct the MedSigLIP client. Reuses one instance per process."""
    global _MEDSIGLIP_SINGLETON
    if _MEDSIGLIP_SINGLETON is None:
        from oncology_arbiter.models.medsiglip import MedSigLip
        _MEDSIGLIP_SINGLETON = MedSigLip()
    return _MEDSIGLIP_SINGLETON


def _get_siglip_proxy() -> Any:
    """Lazy-construct the SigLIP proxy client."""
    global _SIGLIP_PROXY_SINGLETON
    if _SIGLIP_PROXY_SINGLETON is None:
        from oncology_arbiter.models.siglip_baseline import SiglipBaseline
        _SIGLIP_PROXY_SINGLETON = SiglipBaseline()
    return _SIGLIP_PROXY_SINGLETON


def _run_medsiglip_on_preprocessed(preprocess_result: Any) -> Any:
    """Run MedSigLIP on an already-preprocessed mammogram.

    Uses the client's ``preprocess_fn`` injection point to bypass a second
    DICOM read — we already have the float32 [0,1] array from the
    endpoint's ``preprocess_mammogram`` call.

    Raises ``GatedAccessError`` on HAI-DEF gate denial. Callers MUST NOT
    catch this and silently fall back; the endpoint decides fallback
    policy explicitly via env flag.

    Phase 2 limitation (2026-07-03): this helper mutates
    ``ms._preprocess_fn`` on the singleton for the duration of the call.
    That is safe under FastAPI's default single-threaded async request
    handling but is NOT safe under a threadpool. Phase 3 will either
    (a) construct a fresh MedSigLip per request (weights stay cached at
    class level so no re-download) or (b) thread the preprocess through
    the ``.run()`` signature directly.
    """
    ms = _get_medsiglip()
    # Feed the already-computed preprocess result back in via the injectable
    # hook so the client's PIL conversion path runs but no second DICOM I/O happens.
    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    ms._preprocess_fn = _inject
    return ms.run("(preprocessed)")


def _run_siglip_proxy_on_preprocessed(preprocess_result: Any) -> Any:
    """Run the SigLIP proxy on an already-preprocessed mammogram.

    Same Phase 2 caveat as :func:`_run_medsiglip_on_preprocessed` — the
    proxy singleton's ``_preprocess_fn`` is mutated for the call. Safe
    under single-thread async; not safe under a threadpool. Phase 3 fix.
    """
    proxy = _get_siglip_proxy()

    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    proxy._preprocess_fn = _inject
    return proxy.run("(preprocessed)")


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


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncology-arbiter",
        version=__version__,
        description=RUO_DISCLAIMER + "\n\n" + AUROC_CAVEAT,
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

    # 1) Prometheus /metrics
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            excluded_handlers=["/metrics"],
            should_group_status_codes=False,
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as exc:  # pragma: no cover
        _logger.warning("prometheus instrumentator disabled: %s", exc)

    # 2) Rate limit
    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[os.environ.get("ONCOLOGY_ARBITER_RATE_LIMIT", "60/minute")],
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
        return HealthResponse(
            status="ok",
            version=__version__,
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            endpoints=[
                "POST /v1/screening/analyze",
                "POST /v1/biopsy/analyze",
                "POST /v1/therapy/reason",
                "POST /v1/case/full",
                "GET  /v1/model-cards",
                "GET  /v1/artifacts/{category}/{filename}",
                "GET  /health",
            ],
            cancers={
                "breast": {
                    "state": ModelState.PLACEHOLDER.value,
                    "case_full": True,
                    "endpoints": ["screening", "biopsy", "therapy", "case/full"],
                },
                "nsclc": {
                    "state": ModelState.PROXY_LUNG_HEURISTIC.value,
                    "case_full": True,
                    "endpoints": ["case/full"],
                    "notes": (
                        "LIDC-IDRI CT + HU-threshold heuristic + NCCN-lite rules. "
                        "Real pipeline requires nsclc_ct_input.series_dir and "
                        "ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1 on the server; "
                        "otherwise the same endpoint returns a shape-only "
                        "placeholder response with a warning."
                    ),
                },
            },
            models_loaded={
                "monai_screening": ModelState.PLACEHOLDER,
                "medsiglip_biopsy": ModelState.PLACEHOLDER,
                "txgemma_therapy": ModelState.PLACEHOLDER,
                "co_scientist": ModelState.PLACEHOLDER,
                # The L3 arbiter templates ARE loaded (they're JSON on disk),
                # but they carry n_training=0 → treated as placeholder at the
                # health-check level per the PLAN.md honesty rules.
                "l3_arbiter": ModelState.PLACEHOLDER,
                # NSCLC track: real heuristic + rules-lite, gated by env.
                "nsclc_pipeline": ModelState.PROXY_LUNG_HEURISTIC,
            },
        )

    # ----------------------------------------------------------------------- #
    # /v1/screening/analyze — REAL preprocessing, placeholder classifier

    @app.post("/v1/screening/analyze", response_model=ScreeningResponse)
    def screening_analyze(
        req: ScreeningRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> ScreeningResponse:
        request_id = new_request_id()

        if not req.dicom_url and not req.dicom_bytes_b64:
            raise HTTPException(400, "must provide dicom_url or dicom_bytes_b64")
        if req.dicom_url and req.dicom_bytes_b64:
            raise HTTPException(400, "provide exactly one of dicom_url or dicom_bytes_b64")

        raw_bytes = _decode_bytes_arg(req.dicom_bytes_b64)
        if raw_bytes is None and req.dicom_url is not None:
            # Placeholder: URL ingestion goes through our SSRF-guarded fetcher
            # in Phase 2. For now, refuse cleanly.
            log_event(request_id, "/v1/screening/analyze",
                      model_state="placeholder",
                      patient_id_hash=req.patient_id_hash,
                      extra={"reason": "url_ingestion_not_wired"},
                          tenant_id=tenant.tenant_id,
                      )
            raise HTTPException(
                501, "dicom_url ingestion not yet wired (Phase 2). "
                "Send dicom_bytes_b64 for now.",
            )

        # Real preprocessing on the uploaded bytes.
        from oncology_arbiter.mammography import preprocess_mammogram
        assert raw_bytes is not None  # type narrowing
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
            tf.write(raw_bytes)
            tmp_path = tf.name
        try:
            result = preprocess_mammogram(
                tmp_path,
                laterality_hint=req.laterality_hint,
                view_hint=req.view_hint,
            )
        except Exception as e:
            log_event(request_id, "/v1/screening/analyze",
                      model_state="unavailable",
                      patient_id_hash=req.patient_id_hash,
                      extra={"error": str(e)[:200]},
                          tenant_id=tenant.tenant_id,
                      )
            raise HTTPException(422, f"preprocessing failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Run the vision backbone according to Phase-2 precedence rules.
        # Default posture is placeholder — the model states below only
        # activate when the operator explicitly turned them on via env.
        backend_result: Any = None
        backend_state: ModelState = ModelState.PLACEHOLDER
        backend_name: str | None = None
        backend_warnings: list[str] = []
        # runtime_gate_report is the hai_def.GateReport dataclass; converted
        # to schema.GateReport for the wire at envelope time.
        runtime_gate_report: Any = None

        if _is_env_true("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP"):
            try:
                backend_result = _run_medsiglip_on_preprocessed(result)
                backend_state = ModelState.LOADED_MEDSIGLIP
                backend_name = backend_result.model_repo
                backend_warnings = list(backend_result.warnings)
                if backend_result.gate_report is not None:
                    runtime_gate_report = backend_result.gate_report
            except Exception as e:
                # HAI-DEF gate denial or model load failure. Import lazily
                # so the module-level `import` doesn't force transformers.
                from oncology_arbiter.models.hai_def import GatedAccessError, GateReport as _RGR
                if isinstance(e, GatedAccessError):
                    backend_state = ModelState.GATED
                    backend_name = e.repo_id
                    backend_warnings = [
                        f"medsiglip_gated:{e.access_level.value}:{e.reason}"
                    ]
                    # Build a runtime GateReport from the exception. has_token
                    # is not carried on the exception, so we discover it once
                    # here — the same source of truth used by check_hai_def_access.
                    from oncology_arbiter.models.hai_def import _discover_hf_token
                    runtime_gate_report = _RGR(
                        repo_id=e.repo_id,
                        access_level=e.access_level,
                        status_code=e.status_code,
                        reason=e.reason,
                        has_token=_discover_hf_token() is not None,
                    )
                else:
                    backend_state = ModelState.UNAVAILABLE
                    backend_warnings = [f"medsiglip_load_error:{type(e).__name__}: {e}"]
                    log_event(request_id, "/v1/screening/analyze",
                              model_state="unavailable",
                              patient_id_hash=req.patient_id_hash,
                              extra={"medsiglip_error": str(e)[:200]},
                                  tenant_id=tenant.tenant_id,
                              )

        # Proxy fallback is OPT-IN and only activates if MedSigLIP either
        # was disabled OR the request landed as GATED and the operator has
        # explicitly enabled the proxy. This is the code path that used to
        # silently fire and mis-label proxy scores as MedSigLIP — locked
        # behind an env flag now.
        if (
            backend_result is None
            and backend_state in (ModelState.PLACEHOLDER, ModelState.GATED)
            and _is_env_true("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY")
        ):
            try:
                proxy_result = _run_siglip_proxy_on_preprocessed(result)
                # If MedSigLIP was gated, keep the gate report + warning
                # alongside the proxy warning so the response is honest
                # about WHY we fell back.
                prior_warnings = list(backend_warnings)
                backend_result = proxy_result
                backend_state = ModelState.PROXY_SIGLIP
                backend_name = proxy_result.model_repo
                backend_warnings = prior_warnings + list(proxy_result.warnings)
            except Exception as e:
                backend_state = ModelState.UNAVAILABLE
                backend_warnings.append(f"proxy_siglip_load_error:{type(e).__name__}: {e}")
                log_event(request_id, "/v1/screening/analyze",
                          model_state="unavailable",
                          patient_id_hash=req.patient_id_hash,
                          extra={"proxy_error": str(e)[:200]},
                              tenant_id=tenant.tenant_id,
                          )

        # Extract overall_score + findings from whichever backend ran.
        overall_score: float | None = None
        findings_list: list[dict[str, Any]] = []
        if backend_result is not None:
            # SigLIP-family convention: probs[0] is the "malignant" label
            # and probs[1] is the "without" label (see
            # oncology_arbiter.models.siglip_baseline.DEFAULT_ZERO_SHOT_LABELS).
            probs = list(backend_result.probs)
            labels = list(backend_result.labels)
            if len(probs) >= 1:
                overall_score = float(probs[0])
            for lbl, p in zip(labels, probs):
                findings_list.append({
                    "label": lbl,
                    "score": float(p),
                    "location_bbox_normalized": None,
                })

        # ── L4a MONAI detector (mask-gradient heuristic) ──
        # Opt-in via ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR=1. When on, we
        # run the detector on the real preprocessed image + breast mask
        # from the mammography pipeline. Findings are added with
        # ``location_bbox_normalized`` set and a ``monai_heuristic:...``
        # score prefix so downstream UI can distinguish them from SigLIP-
        # family classification findings. NEVER silently upgrades the
        # response's model_state to LOADED_MONAI_DETECTOR unless real
        # trained weights load, which under Phase 3 they do not.
        if _is_env_true("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR"):
            try:
                from oncology_arbiter.models.monai_detector import (
                    MONAI_DETECTOR_WARNING,
                    MonaiDetector,
                )
                det_result = MonaiDetector().detect(
                    result.image.astype("float32"),
                    result.breast_mask,
                )
                for box in det_result.boxes:
                    findings_list.append({
                        "label": f"monai_heuristic:{box.label}",
                        "score": float(box.score),
                        "location_bbox_normalized": [box.x0, box.y0, box.x1, box.y1],
                    })
                if MONAI_DETECTOR_WARNING not in backend_warnings:
                    backend_warnings.append(MONAI_DETECTOR_WARNING)
                if backend_state == ModelState.PLACEHOLDER:
                    backend_state = ModelState.PROXY_MONAI_HEURISTIC
                    backend_name = det_result.model_name
            except Exception as e:  # noqa: BLE001 — surface, never hide
                backend_warnings.append(
                    f"monai_detector_error:{type(e).__name__}:{e}"
                )

        # Score the L3 screening arbiter with a MINIMAL feature vector.
        # Phase 2: still no BI-RADS from a real reader — the classifier
        # isn't wired — so we deliberately submit the empty feature dict.
        # That falls through to intercept-only (base rate). Phase 3 will
        # feed features extracted from the L4a detector output.
        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("screening", features={})
        except Exception as e:
            log_event(request_id, "/v1/screening/analyze",
                      model_state="unavailable",
                      patient_id_hash=req.patient_id_hash,
                      extra={"arbiter_error": str(e)[:200]},
                          tenant_id=tenant.tenant_id,
                      )
            arbiter_block = None

        # Attach the structured gate_report onto Provenance. When populated,
        # `provenance.gate_report` carries repo_id + access_level +
        # status_code + reason + has_token — enough for a downstream UI or
        # audit log to explain WHY the endpoint returned GATED / LOADED /
        # UNAVAILABLE without having to string-parse the warnings.
        schema_gate_report = _to_schema_gate_report(runtime_gate_report)
        env = _envelope(
            request_id,
            model_state=backend_state,
            model_name=backend_name,
            gate_report=schema_gate_report,
        )
        if runtime_gate_report is not None:
            log_event(request_id, "/v1/screening/analyze",
                      model_state=backend_state.value,
                      patient_id_hash=req.patient_id_hash,
                      extra={"gate_report": {
                          "repo_id": runtime_gate_report.repo_id,
                          "access_level": runtime_gate_report.access_level.value,
                          "status_code": runtime_gate_report.status_code,
                          "reason": runtime_gate_report.reason,
                          "has_token": runtime_gate_report.has_token,
                      }},
                          tenant_id=tenant.tenant_id,
                      )

        response = ScreeningResponse(
            **env,
            laterality=result.metadata.laterality.value,
            view=result.metadata.view.value,
            orientation_flipped=result.metadata.orientation_flipped,
            breast_mask_coverage=float(result.breast_mask.mean()),
            findings=[dict(f) for f in findings_list],
            overall_score=overall_score,
            arbiter_score=arbiter_block,
            warnings=backend_warnings,
        )
        log_event(request_id, "/v1/screening/analyze",
                  model_state=backend_state.value,
                  patient_id_hash=req.patient_id_hash,
                  extra={
                      "shape": [result.image.shape[0], result.image.shape[1]],
                      "laterality": result.metadata.laterality.value,
                      "view": result.metadata.view.value,
                      "mask_coverage": float(result.breast_mask.mean()),
                      "backend": backend_state.value,
                      "overall_score": overall_score,
                      "n_warnings": len(backend_warnings),
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
        """L4b: MedSigLIP-448 embed → synthetic 3-class linear probe.

        Opt-in via ``ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP=1``.

        Contract:
        * NO WSI parser (no OpenSlide) — treats ``wsi_bytes_b64`` / ``wsi_url``
          as an image the MedSigLIP vision encoder can consume. This is a
          research proxy for a real WSI patcher.
        * Preflight HAI-DEF gate first; on FORBIDDEN/UNAUTHENTICATED emits
          ``ModelState.GATED`` with a ``biopsy_medsiglip_gated:<level>``
          warning — NEVER silently fabricates a subtype.
        * Weights are synthetic (n_training=48 synthetic=True) → the
          response's ``warnings`` list surfaces this on every call.
        """
        request_id = new_request_id()
        if not req.wsi_url and not req.wsi_bytes_b64 and not req.report_text:
            raise HTTPException(400, "must provide wsi_url, wsi_bytes_b64, or report_text")

        model_state = ModelState.PLACEHOLDER
        model_name: str | None = None
        subtype_prediction: str | None = None
        confidence: float | None = None
        warnings: list[str] = []
        # Runtime dataclass (from hai_def), converted to schema at envelope time.
        runtime_gate_report = None  # type: ignore[assignment]

        if _is_env_true("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP"):
            if not (req.wsi_url or req.wsi_bytes_b64):
                warnings.append(
                    "biopsy_medsiglip_skipped:report_text_only:no_image_provided"
                )
            else:
                try:
                    from oncology_arbiter.models.biopsy_medsiglip_probe import (
                        BiopsyMedSigLipProbe,
                    )
                    from oncology_arbiter.models.hai_def import (
                        GatedAccessError,
                        GateReport as RuntimeGateReport,
                        _discover_hf_token,
                    )

                    probe = BiopsyMedSigLipProbe()
                    image_bytes = _decode_bytes_arg(req.wsi_bytes_b64)
                    image_url = str(req.wsi_url) if req.wsi_url else None
                    result = probe.run(
                        image_bytes=image_bytes,
                        image_url=image_url,
                    )
                    subtype_prediction = result.subtype
                    confidence = float(result.subtype_probs[result.subtype])
                    model_state = ModelState.LOADED_BIOPSY_PROBE
                    model_name = "google/medsiglip-448+biopsy_probe_v0"
                    warnings.extend(result.warnings)
                    # If the probe surfaced a runtime GateReport (allowed
                    # preflight), thread it through.
                    runtime_gate_report = getattr(result, "gate_report", None)
                except GatedAccessError as gate_err:
                    model_state = ModelState.GATED
                    model_name = gate_err.repo_id
                    warnings.append(
                        f"biopsy_medsiglip_gated:{gate_err.access_level.value}:{gate_err.reason}"
                    )
                    runtime_gate_report = RuntimeGateReport(
                        repo_id=gate_err.repo_id,
                        access_level=gate_err.access_level,
                        status_code=gate_err.status_code,
                        reason=gate_err.reason,
                        has_token=_discover_hf_token() is not None,
                    )
                except Exception as e:  # noqa: BLE001 — surface, never hide
                    warnings.append(f"biopsy_medsiglip_error:{type(e).__name__}:{e}")

        log_event(request_id, "/v1/biopsy/analyze",
                  model_state=model_state.value,
                  patient_id_hash=req.patient_id_hash,
                  extra={"has_wsi": bool(req.wsi_url or req.wsi_bytes_b64),
                         "has_report": bool(req.report_text),
                         "subtype": subtype_prediction,
                         "n_warnings": len(warnings)},
                             tenant_id=tenant.tenant_id,
                         )

        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("biopsy", features={})
        except Exception:
            arbiter_block = None

        schema_gate_report = _to_schema_gate_report(runtime_gate_report)
        env = _envelope(
            request_id,
            model_state=model_state,
            model_name=model_name,
            gate_report=schema_gate_report,
        )
        env["warnings"] = warnings
        return BiopsyResponse(
            **env,
            subtype_prediction=subtype_prediction,
            receptor_panel=BiopsyReceptorPanel(),
            grade=None,
            confidence=confidence,
            arbiter_score=arbiter_block,
        )

    # ----------------------------------------------------------------------- #
    # /v1/therapy/reason — placeholder

    @app.post("/v1/therapy/reason", response_model=TherapyResponse)
    def therapy_reason(
        req: TherapyRequest,
        tenant: APIKey = Depends(require_api_key),
    ) -> TherapyResponse:
        """L4c: TxGemma-preferred, NCCN-lite rules fallback.

        Precedence (matches screening's MedSigLIP → SigLIP proxy pattern):

        1. If ``ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA=1``: try TxGemma.
           * On ALLOWED preflight (never reached under current token):
             ``ModelState.LOADED_TXGEMMA``.
           * On FORBIDDEN/UNAUTHENTICATED: emit ``txgemma_gated:<level>``
             warning and FALL THROUGH to (2) only if the rules-lite proxy
             is enabled.
        2. If ``ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1``: run
           deterministic NCCN-lite rules. ``ModelState.PROXY_RULES_LITE``.
        3. Otherwise: placeholder (recommended_options=[]).
        """
        request_id = new_request_id()

        model_state = ModelState.PLACEHOLDER
        model_name: str | None = None
        recommended: list[TherapyOption] = []
        not_recommended: list[TherapyOption] = []
        warnings: list[str] = []
        runtime_gate_report = None  # populated by TxGemma preflight

        # Extract features for rules engine from biopsy_output + patient_context
        biopsy = req.biopsy_output
        subtype: str | None = biopsy.subtype_prediction if biopsy else None
        er = bool(biopsy.receptor_panel.er_positive) if biopsy else False
        pr = bool(biopsy.receptor_panel.pr_positive) if biopsy else False
        her2_status = biopsy.receptor_panel.her2_status if biopsy else None
        her2 = her2_status == "positive"
        grade = biopsy.grade if biopsy and biopsy.grade else 2
        # Stage isn't in BiopsyResponse today; use a conservative default and
        # let the frontend pass stage via patient_context.genomic_markers.
        pc = req.patient_context
        stage = str(pc.genomic_markers.get("stage", "T1N0M0")) if pc.genomic_markers else "T1N0M0"
        menopausal_map = {"pre": "premenopausal", "post": "postmenopausal",
                          "peri": "premenopausal", "unknown": None, None: None}
        menopausal_status = menopausal_map.get(pc.menopausal_status)

        # ── (1) TxGemma path ──
        txgemma_tried = False
        if _is_env_true("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA"):
            txgemma_tried = True
            try:
                from oncology_arbiter.models.txgemma_client import TxGemmaClient
                from oncology_arbiter.models.hai_def import (
                    GatedAccessError,
                    GateReport as RuntimeGateReport,
                    _discover_hf_token,
                )

                tx = TxGemmaClient()
                tx_result = tx.recommend_therapy(
                    receptor_status={"ER": er, "PR": pr, "HER2": her2},
                    grade=grade,
                    stage=stage,
                    age=pc.age,
                    menopausal_status=menopausal_status,
                    subtype=subtype,
                )
                recommended = [
                    TherapyOption(regimen=r, line_of_therapy=1, rationale="TxGemma")
                    for r in tx_result.recommendations
                ]
                warnings.extend(tx_result.warnings)
                model_state = ModelState.LOADED_TXGEMMA
                model_name = tx.repo_id
                runtime_gate_report = getattr(tx_result, "gate_report", None)
            except GatedAccessError as gate_err:
                warnings.append(
                    f"txgemma_gated:{gate_err.access_level.value}:{gate_err.reason}"
                )
                runtime_gate_report = RuntimeGateReport(
                    repo_id=gate_err.repo_id,
                    access_level=gate_err.access_level,
                    status_code=gate_err.status_code,
                    reason=gate_err.reason,
                    has_token=_discover_hf_token() is not None,
                )

        # ── (2) NCCN-lite rules fallback ──
        if model_state == ModelState.PLACEHOLDER and _is_env_true(
            "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY"
        ):
            try:
                from oncology_arbiter.models.therapy_rules_lite import (
                    apply_nccn_lite_rules,
                )
                rules_result = apply_nccn_lite_rules(
                    receptor_status={"ER": er, "PR": pr, "HER2": her2},
                    grade=grade,
                    stage=stage,
                    age=pc.age,
                    menopausal_status=menopausal_status,
                    subtype=subtype,
                    strict=bool(req.strict),
                )
                recommended = [
                    TherapyOption(
                        regimen=o.name,
                        line_of_therapy=1,
                        rationale=o.rationale,
                        evidence=[EvidenceRecord(
                            url=o.citation_url,
                            quoted_text=f"NCCN {o.nccn_section}",
                            source="nccn-guidelines",
                        )],
                    )
                    for o in rules_result.recommended_options
                ]
                not_recommended = [
                    TherapyOption(
                        regimen=o.name,
                        line_of_therapy=1,
                        rationale=o.rationale,
                        evidence=[EvidenceRecord(
                            url=o.citation_url,
                            quoted_text=f"NCCN {o.nccn_section}",
                            source="nccn-guidelines",
                        )],
                    )
                    for o in rules_result.not_recommended
                ]
                warnings.extend(rules_result.warnings)
                model_state = ModelState.PROXY_RULES_LITE
                model_name = "nccn-lite-v0"
            except Exception as e:  # noqa: BLE001
                # strict=True input drift → HTTP 400 (surfaces validation
                # errors instead of hiding them in the warnings list).
                from oncology_arbiter.models.therapy_rules_lite import (
                    InvalidInputError as _InvalidInputError,
                )
                if isinstance(e, _InvalidInputError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"therapy_rules_lite_invalid_input: {e}",
                    ) from e
                warnings.append(f"therapy_rules_lite_error:{type(e).__name__}:{e}")

        log_event(request_id, "/v1/therapy/reason",
                  model_state=model_state.value,
                  patient_id_hash=None,
                  extra={"has_biopsy_input": biopsy is not None,
                         "txgemma_tried": txgemma_tried,
                         "n_recommended": len(recommended),
                         "n_not_recommended": len(not_recommended),
                         "n_warnings": len(warnings)},
                             tenant_id=tenant.tenant_id,
                         )

        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("therapy", features={})
        except Exception:
            arbiter_block = None

        schema_gate_report = _to_schema_gate_report(runtime_gate_report)
        env = _envelope(
            request_id,
            model_state=model_state,
            model_name=model_name,
            gate_report=schema_gate_report,
        )
        env["warnings"] = warnings
        # v0.2: only surface the ruleset fingerprint when the rules-lite
        # branch actually ran (model_state == PROXY_RULES_LITE). If TxGemma
        # was reachable, or we fell through to placeholder, these stay None.
        _rules_sha: str | None = None
        _rules_model_id: str | None = None
        _rules_branch_id: str | None = None
        if model_state == ModelState.PROXY_RULES_LITE and "rules_result" in locals():
            _rules_sha = getattr(rules_result, "rules_sha256", None)
            _rules_model_id = getattr(rules_result, "rules_model_id", None)
            _rules_branch_id = getattr(rules_result, "branch_id", None)

        return TherapyResponse(
            **env,
            recommended_options=recommended,
            not_recommended=not_recommended,
            arbiter_score=arbiter_block,
            rules_sha256=_rules_sha,
            rules_model_id=_rules_model_id,
            branch_id=_rules_branch_id,
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
    # /v1/case/full — placeholder that chains sub-endpoints

    # Cancer tracks the endpoint knows how to route. The SPA reads
    # /health.cancers to know which selectors to enable. Any value NOT in
    # this set is rejected with 400 (never silently coerced to breast) so
    # the honesty caveat never masquerades as covering another cancer.
    _SUPPORTED_CANCERS = {"breast", "nsclc"}

    @app.post("/v1/case/full", response_model=FullCaseResponse)
    def case_full(
        req: FullCaseRequest,
        cancer: str = Query(
            default="breast",
            description="Cancer track: 'breast' (full pipeline) or 'nsclc' "
                        "(shape-only placeholder; LIDC-IDRI pipeline lands "
                        "from worker-2).",
        ),
        tenant: APIKey = Depends(require_api_key),
    ) -> FullCaseResponse:
        cancer_norm = cancer.lower().strip()
        if cancer_norm not in _SUPPORTED_CANCERS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported cancer={cancer!r}. "
                    f"Supported: {sorted(_SUPPORTED_CANCERS)}. "
                    f"See /health.cancers for the wired-up set."
                ),
            )

        request_id = new_request_id()

        # ------- NSCLC branch --------------------------------------------- #
        # Two paths:
        #   (1) placeholder / shape-only    (no series_dir OR feature-gate off)
        #   (2) real lung heuristic + NCCN-lite rules
        #       (nsclc_ct_input.series_dir set AND
        #        ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1)
        # The env gate prevents client-controlled filesystem paths from
        # being trusted in shared / public deployments; local dev flips
        # the gate on.
        if cancer_norm == "nsclc":
            ct_in = req.nsclc_ct_input
            allow_series_dir = _is_env_true("ONCOLOGY_ARBITER_ALLOW_SERIES_DIR")

            # ----- (1) shape-only placeholder --------------------------- #
            if ct_in is None or not allow_series_dir:
                env = _envelope(request_id, model_name="nsclc_placeholder_v0")
                warnings = [
                    "cancer=nsclc: placeholder shape only. Set "
                    "nsclc_ct_input.series_dir AND "
                    "ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1 for the real "
                    "LIDC-IDRI CT pipeline + NCCN-NSCLC-lite rules.",
                ]
                if ct_in is not None and not allow_series_dir:
                    warnings.append(
                        "nsclc_ct_input was provided but "
                        "ONCOLOGY_ARBITER_ALLOW_SERIES_DIR is not truthy on "
                        "the server; ignoring series_dir for safety."
                    )
                log_event(request_id, "/v1/case/full",
                          model_state="placeholder",
                          patient_id_hash=None,
                          extra={"cancer": "nsclc", "has_screening": False,
                                 "has_biopsy": False, "elo_n_hypotheses": 0},
                          tenant_id=tenant.tenant_id,
                          )
                return FullCaseResponse(
                    **env,
                    warnings=warnings,
                    screening=None,
                    biopsy=None,
                    therapy=None,
                    nsclc=NsclcResponse(
                        model_state=ModelState.PLACEHOLDER,
                        model_name="nsclc_placeholder_v0",
                        warnings=warnings,
                    ),
                    elo_ranked_hypotheses=[],
                )

            # ----- (2) real pipeline ------------------------------------ #
            import time
            from oncology_arbiter.lung import (
                read_ct_series,
                run_lung_heuristic,
                score_nsclc,
                NsclcArbiterFeatures,
            )
            from oncology_arbiter.models.nccn_nsclc_rules import (
                score_nsclc_therapy,
                NSCLC_RULES_PROXY_WARNING,
            )

            series_dir = str(ct_in.series_dir)
            t0 = time.perf_counter()
            try:
                ct = read_ct_series(series_dir)
            except FileNotFoundError as exc:
                raise HTTPException(400, f"series_dir not found: {exc}") from exc
            except Exception as exc:
                raise HTTPException(400, f"failed to read CT series: {exc}") from exc
            t1 = time.perf_counter()
            spacing_mm = (
                float(ct.slice_thickness_mm),
                float(ct.pixel_spacing_mm[0]),
                float(ct.pixel_spacing_mm[1]),
            )
            heur = run_lung_heuristic(
                ct.volume, spacing_mm=spacing_mm, top_n=int(ct_in.top_n)
            )
            t2 = time.perf_counter()
            arb_feats = NsclcArbiterFeatures.from_lung_output(heur)
            arb = score_nsclc(arb_feats)
            therapy = score_nsclc_therapy(
                risk_bucket=arb.risk_bucket,
                driving_feature=arb.driving_feature,
                max_diameter_mm=heur.max_diameter_mm,
            )

            model_name = "nsclc_lung_heuristic_v0+nccn_nsclc_lite_v0"
            env = _envelope(
                request_id,
                model_state=ModelState.PROXY_LUNG_HEURISTIC,
                model_name=model_name,
            )
            warnings = [
                "cancer=nsclc: PROXY pipeline. Classical HU thresholding + "
                "connected components (not a trained detector). Diameter "
                "buckets follow Fleischner-lite anchors; therapy is "
                "rules-only.",
                NSCLC_RULES_PROXY_WARNING,
            ]
            log_event(request_id, "/v1/case/full",
                      model_state="proxy",
                      patient_id_hash=None,
                      extra={
                          "cancer": "nsclc",
                          "series_dir": series_dir,
                          "n_slices": int(ct.volume.shape[0]),
                          "n_candidates_kept": heur.n_candidates_kept,
                          "max_diameter_mm": float(heur.max_diameter_mm),
                          "risk_bucket": arb.risk_bucket,
                          "has_screening": False,
                          "has_biopsy": False,
                          "elo_n_hypotheses": 0,
                      },
                      tenant_id=tenant.tenant_id,
                      )
            return FullCaseResponse(
                **env,
                warnings=warnings,
                screening=None,
                biopsy=None,
                therapy=None,
                nsclc=NsclcResponse(
                    model_state=ModelState.PROXY_LUNG_HEURISTIC,
                    model_name=model_name,
                    warnings=warnings,
                    lung_voxel_fraction=float(heur.lung_voxel_fraction),
                    n_candidates_total=int(heur.n_candidates_total),
                    n_candidates_kept=int(heur.n_candidates_kept),
                    max_diameter_mm=float(heur.max_diameter_mm),
                    candidates=[
                        NsclcCandidate(
                            label=int(c.label),
                            voxel_count=int(c.voxel_count),
                            diameter_mm=float(c.diameter_mm),
                            mean_hu=float(c.mean_hu),
                            centroid_zyx_vox=(
                                float(c.centroid_zyx_vox[0]),
                                float(c.centroid_zyx_vox[1]),
                                float(c.centroid_zyx_vox[2]),
                            ),
                        )
                        for c in heur.candidates
                    ],
                    risk_score=float(arb.prob),
                    risk_bucket=str(arb.risk_bucket),
                    driving_feature=str(arb.driving_feature),
                    logit=float(arb.logit),
                    therapy_recommended=[
                        NsclcTherapyOption(
                            name=o.name,
                            category=o.category,
                            citation_url=o.citation_url,
                            rationale=o.rationale,
                            nccn_section=o.nccn_section,
                        )
                        for o in therapy.recommended_options
                    ],
                    therapy_not_recommended=[
                        NsclcTherapyOption(
                            name=o.name,
                            category=o.category,
                            citation_url=o.citation_url,
                            rationale=o.rationale,
                            nccn_section=o.nccn_section,
                        )
                        for o in therapy.not_recommended
                    ],
                    series_dir=series_dir,
                    n_slices=int(ct.volume.shape[0]),
                    read_seconds=float(t1 - t0),
                    heuristic_seconds=float(t2 - t1),
                ),
                elo_ranked_hypotheses=[],
            )

        # ------- Breast branch (existing behaviour) ----------------------- #
        screening: ScreeningResponse | None = None
        biopsy: BiopsyResponse | None = None
        therapy: TherapyResponse | None = None
        # If sub-inputs are provided, run them through the placeholder subroutes.
        if req.screening_input:
            screening = screening_analyze(req.screening_input, tenant=tenant)
        if req.biopsy_input:
            biopsy = biopsy_analyze(req.biopsy_input, tenant=tenant)
        therapy = therapy_reason(
            TherapyRequest(biopsy_output=biopsy, patient_context=req.therapy_context),
            tenant=tenant,
        )
        # L5 Co-Scientist 4-phase loop (opt-in). When enabled, runs
        # generate → reflect → rank (Elo) → evolve → rank over the stage
        # envelopes and returns the ranked hypotheses on
        # `elo_ranked_hypotheses`. Honesty gate: only URLs found in the
        # combined evidence[] of the stage responses count as "seen"; any
        # hypothesis carrying an unseen URL is stripped of that URL by
        # `reflect_hypotheses` before it can win Elo points.
        elo_ranked: list[dict[str, Any]] = []
        cs_warnings: list[str] = []
        if _is_env_true("ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST"):
            from oncology_arbiter.orchestrator.co_scientist import run_co_scientist
            seen_urls: set[str] = set()
            for env_dict in (
                screening.model_dump() if screening is not None else None,
                biopsy.model_dump() if biopsy is not None else None,
                therapy.model_dump() if therapy is not None else None,
            ):
                if env_dict is None:
                    continue
                for e in env_dict.get("evidence") or []:
                    if isinstance(e, dict) and "url" in e:
                        seen_urls.add(e["url"])
                # Therapy option evidence lives one level deeper
                for opt in env_dict.get("recommended_options") or []:
                    for e in (opt or {}).get("evidence") or []:
                        if isinstance(e, dict) and "url" in e:
                            seen_urls.add(e["url"])
            try:
                cs_out = run_co_scientist(
                    screening=(screening.model_dump() if screening is not None else None),
                    biopsy=(biopsy.model_dump() if biopsy is not None else None),
                    therapy=(therapy.model_dump() if therapy is not None else None),
                    seen_urls=seen_urls,
                )
                elo_ranked = cs_out["hypotheses"]
                cs_warnings = cs_out["warnings"]
            except Exception as e:
                cs_warnings = [f"co_scientist_error:{type(e).__name__}:{e}"]

        log_event(request_id, "/v1/case/full",
                  model_state="placeholder",
                  patient_id_hash=None,
                  extra={
                      "cancer": "breast",
                      "has_screening": screening is not None,
                      "has_biopsy": biopsy is not None,
                      "elo_n_hypotheses": len(elo_ranked),
                  },
                      tenant_id=tenant.tenant_id,
                  )
        return FullCaseResponse(
            **_envelope(request_id),
            screening=screening,
            biopsy=biopsy,
            therapy=therapy,
            elo_ranked_hypotheses=elo_ranked,
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
        from fastapi.staticfiles import StaticFiles

        static_root = Path(__file__).parent / "static" / "dist"
        if static_root.is_dir() and (static_root / "index.html").is_file():
            # `html=True` makes StaticFiles fall through to index.html for any
            # sub-path (SPA routing). It still 404s on missing static assets.
            app.mount("/ui", StaticFiles(directory=str(static_root), html=True), name="ui")

    return app
