"""HTTP middleware, lifespan, and optional /ui mount.

Route modules live under api/routers. This file is the stack create_app installs
before include_router.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from oncology_arbiter.auth import bootstrap_from_env
from oncology_arbiter.observability import RequestIdMiddleware, configure_logging, get_logger
from oncology_arbiter.arbiter.manski import ManskiGateError

from . import deps

_is_env_true = deps.is_env_true


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


def install_http_stack(app: FastAPI) -> None:
    """Middleware, Manski handler, metrics, CORS, demo gate, auth bootstrap, /ui."""
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
        import slowapi.middleware as _slowapi_mw
        from starlette.routing import Match

        from .rate_limit import make_key_func

        def _find_included_route_handler(routes, scope):
            """Resolve endpoints nested under FastAPI include_router.

            slowapi's finder only sees top-level routes. Included routers are
            ``_IncludedRouter`` objects with ``path is None``, so every
            domain route was treated as exempt and the default limit never fired.
            """
            for route in routes:
                if type(route).__name__ == "_IncludedRouter":
                    matched, _ = route.matches(scope)
                    if matched != Match.FULL:
                        continue
                    found = _find_included_route_handler(
                        route.original_router.routes, scope
                    )
                    if found is not None:
                        return found
                    continue
                try:
                    matched, _ = route.matches(scope)
                except Exception:
                    continue
                if matched == Match.FULL and hasattr(route, "endpoint"):
                    return route.endpoint
            return None

        _slowapi_mw._find_route_handler = _find_included_route_handler

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

