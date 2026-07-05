# ─────────────────────────────────────────────────────────────────────────── #
# oncology-arbiter Dockerfile
#
# Multi-stage:
#
#   1. `builder`     — installs third-party deps into /opt/oa via
#                       `pip install --prefix=/opt/oa .` (this pulls the
#                       transitive dependency graph; the arbiter package
#                       itself we ship as source in the runtime stage).
#
#   2. `runtime`     — minimal Python 3.11-slim, non-root user, HEALTHCHECK
#                       against /health. Ships the repo layout under /app/
#                       so runtime code can resolve docs/ and static/dist/
#                       relative to /app/src/oncology_arbiter/api/app.py:
#
#                         Path(__file__).resolve().parents[3] == /app
#
# Notes:
#   * The default runtime installs the CORE dependency set only (fastapi +
#     preprocessing). The `[ml]` extras (torch + monai + transformers) add
#     ~5 GB and are opt-in via build arg ONCOLOGY_ARBITER_INCLUDE_ML=1.
#   * Frontend static bundle ships from src/oncology_arbiter/api/static/dist/
#     — we do NOT run `npm run build` in Docker; that is a maintainer commit.
#   * Model cards ship from docs/model_cards/ at the repo root; the API code
#     resolves them via `_PROJECT_ROOT / "docs" / "model_cards"`.
#   * Backends that need HuggingFace tokens read HUGGINGFACE_TOKEN from the
#     env at runtime. NEVER bake tokens into the image.
# ─────────────────────────────────────────────────────────────────────────── #

ARG PYTHON_VERSION=3.11-slim

# ─── Builder ─────────────────────────────────────────────────────────────── #
FROM python:${PYTHON_VERSION} AS builder

ARG ONCOLOGY_ARBITER_INCLUDE_ML=0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install --no-install-recommends -y \
        gcc \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install third-party deps into /opt/oa. We install the arbiter package too
# (harmless duplicate — the runtime stage overrides via PYTHONPATH ordering),
# because pip resolves the dep graph from pyproject.toml only when the local
# project is the install target.
RUN pip install --upgrade pip \
    && pip install --prefix=/opt/oa . \
    && if [ "$ONCOLOGY_ARBITER_INCLUDE_ML" = "1" ]; then \
           pip install --prefix=/opt/oa ".[ml]"; \
       fi

# ─── Runtime ─────────────────────────────────────────────────────────────── #
FROM python:${PYTHON_VERSION} AS runtime

# Non-root user. UID/GID 10001 avoids clashing with anything a k8s
# security-context would pin.
RUN groupadd --system --gid 10001 arbiter \
    && useradd --system --uid 10001 --gid arbiter --home-dir /app arbiter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/oa/bin:${PATH}" \
    # /app/src is FIRST so the arbiter package resolves from the source tree
    # (whose sibling /app/docs is what _PROJECT_ROOT resolves to). The
    # installed copy under /opt/oa is only there for its dependency set.
    PYTHONPATH="/app/src:/opt/oa/lib/python3.11/site-packages" \
    # Serve the built frontend by default when the image is deployed.
    ONCOLOGY_ARBITER_SERVE_FRONTEND=1 \
    # Audit ledger lives outside /app so the non-root arbiter user can
    # write it without needing to chown a subtree at container start. On
    # Render free tier the disk is ephemeral either way, so /tmp is fine
    # for the ledger; sites that want durable audit should mount a
    # persistent volume and override this env var.
    ONCOLOGY_ARBITER_AUDIT_DIR=/tmp/oa-audit \
    # Auth DB (tenant/api-key SQLite) lives in the same writable location.
    ONCOLOGY_ARBITER_AUTH_DB_PATH=/tmp/oa-audit/tenants.sqlite

# NOTE: ONCOLOGY_ARBITER_AUTH_MODE is intentionally NOT set here.
# In-code default is `on` (see auth/middleware.py::_auth_off) — safer
# for a production image. Local dev / CI test runs opt out with
# `ONCOLOGY_ARBITER_AUTH_MODE=off` in tests/conftest.py or the shell.
# Deploy operators flip on via a Render service env var; the same
# env var also feeds the auth bootstrap hook (see auth/bootstrap.py).

# curl only needed for the HEALTHCHECK; everything else came from builder.
RUN apt-get update && apt-get install --no-install-recommends -y \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages + bin from builder (dependency set).
COPY --from=builder /opt/oa /opt/oa

# Ship the repo layout under /app/ so _PROJECT_ROOT (parents[3] from
# app.py) resolves to /app and finds sibling docs/ + artifacts/.
WORKDIR /app
COPY --chown=arbiter:arbiter src /app/src
COPY --chown=arbiter:arbiter docs /app/docs
COPY --chown=arbiter:arbiter README.md /app/README.md

USER arbiter

EXPOSE 8080

# Server-side health probe. Uvicorn starts under a few seconds, so a
# 30-second grace period + 10-second interval keeps the check honest
# without flapping.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent --max-time 4 http://127.0.0.1:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "oncology_arbiter.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
