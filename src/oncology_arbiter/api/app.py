"""FastAPI app factory — thin wiring over modular routers + services.

Route handlers live in ``oncology_arbiter.api.routers.*``.
Service classes live in ``oncology_arbiter.api.services.*``.

Patchable module aliases (``_get_medsiglip``, ``_run_medsiglip_on_preprocessed``,
``_score_explicit_therapy_triage``, …) remain here so existing unit tests that
monkeypatch ``oncology_arbiter.api.app`` keep working.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__
from oncology_arbiter.arbiter.manski import ManskiGateError
from oncology_arbiter.auth import bootstrap_from_env
from oncology_arbiter.observability import (
    RequestIdMiddleware,
    configure_logging,
    get_logger,
)

from . import deps
from .orchestrator.pathology_parse import PathologyParseOrchestrator
from .schemas import ArtifactCategory, ArbiterScore
from .services.cbis_probe import (  # noqa: F401
    run_cbis_ddsm_probe_on_bytes as _run_cbis_ddsm_probe_on_bytes,
)
from .services.clinicalbert import (  # noqa: F401
    run_clinicalbert_parse as _run_clinicalbert_parse,
)
from .services.elo import elo_reason_for_modifier as _elo_reason_for_modifier  # noqa: F401
from .services.medsiglip import (  # noqa: F401
    get_medsiglip as _get_medsiglip,
    run_medsiglip_on_preprocessed as _run_medsiglip_on_preprocessed,
)

# --- deps aliases (AST + historical imports) ---
_to_schema_gate_report = deps.to_schema_gate_report
_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg
_score_arbiter = deps.score_arbiter
_is_env_true = deps.is_env_true
_demo_samples_dir = deps.demo_samples_dir
_compute_models_loaded = deps.compute_models_loaded

# --- service re-exports (monkeypatch targets) ---

# Pathology parse aliases used by older tests / introspection
_medgemma_biopsy_parse = PathologyParseOrchestrator._parse_medgemma
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


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

    AST gate: this is the sole ``_score_arbiter`` call site in app.py.
    """
    if raw is None:
        return None
    from oncology_arbiter.api.schemas import TherapyTriageFeatures
    validated = TherapyTriageFeatures.model_validate(raw).model_dump()
    if all(validated[field] is None for field in _TRIAGE_REQUIRED_FIELDS):
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


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOTS: dict[ArtifactCategory, Path] = {
    ArtifactCategory.docs: _PROJECT_ROOT / "docs" / "model_cards",
    ArtifactCategory.reports: _PROJECT_ROOT / "artifacts" / "reports",
    ArtifactCategory.data: _PROJECT_ROOT / "artifacts" / "data",
    ArtifactCategory.models: _PROJECT_ROOT / "src" / "oncology_arbiter" / "arbiter" / "models",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan handler — demo DICOM pre-warm."""
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
            get_logger().warning("demo pre-warm crashed: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncology-arbiter",
        version=__version__,
        description=RUO_DISCLAIMER + "\n\n" + AUROC_CAVEAT,
        lifespan=_lifespan,
    )

    configure_logging(os.environ.get("ONCOLOGY_ARBITER_LOG_LEVEL", "INFO"))
    _logger = get_logger()

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

    try:
        from prometheus_client import CollectorRegistry
        from prometheus_fastapi_instrumentator import Instrumentator

        _metrics_registry = CollectorRegistry(auto_describe=True)
        Instrumentator(
            excluded_handlers=["/metrics"],
            should_group_status_codes=False,
            registry=_metrics_registry,
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except ImportError as exc:  # pragma: no cover
        _logger.warning("prometheus instrumentator unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover
        _logger.error(
            "prometheus instrumentation FAILED (%s): %s; /metrics will not be served",
            type(exc).__name__, exc,
        )

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

    app.add_middleware(RequestIdMiddleware)

    _contact_url = os.environ.get(
        "ONCOLOGY_ARBITER_CONTACT_URL", "https://crispro.ai/contact"
    )
    _DEMO_MODE_POST_ALLOWLIST = frozenset({
        "/v1/elo/rank",
        "/v1/offline_ranker/run",
    })

    @app.middleware("http")
    async def _demo_mode_gate(request: Request, call_next):  # noqa: RUF029
        if not _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"):
            return await call_next(request)
        if request.method != "POST":
            return await call_next(request)
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
            _logger.warning("auth_bootstrap skipped: %s", _bootstrap_result)
    except Exception as _boot_exc:  # pragma: no cover
        _logger.warning("auth_bootstrap raised: %s", _boot_exc)

    from .routers import (
        artifacts,
        biopsy,
        case,
        co_scientist,
        demo,
        elo,
        health,
        research,
        screening,
        stage,
        therapy,
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(stage.router, tags=["stage"])
    app.include_router(screening.router, tags=["screening"])
    app.include_router(biopsy.router, tags=["biopsy"])
    app.include_router(therapy.router, tags=["therapy"])
    app.include_router(research.router, tags=["research"])
    app.include_router(artifacts.router, tags=["artifacts"])
    app.include_router(demo.router, tags=["demo"])
    app.include_router(elo.router, tags=["elo"])
    app.include_router(co_scientist.router, tags=["co_scientist"])
    app.include_router(case.router, tags=["case"])

    # /ui — optional static frontend mount
    if _is_env_true("ONCOLOGY_ARBITER_SERVE_FRONTEND"):
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles

        static_root = Path(__file__).parent / "static" / "dist"
        if static_root.is_dir() and (static_root / "index.html").is_file():
            app.mount("/ui", StaticFiles(directory=str(static_root), html=True), name="ui")

            @app.get("/", include_in_schema=False)
            def _root_redirect() -> RedirectResponse:
                return RedirectResponse(url="/ui/", status_code=307)

    return app


# Module-level ASGI app for uvicorn / Docker: oncology_arbiter.api.app:app
app = create_app()
