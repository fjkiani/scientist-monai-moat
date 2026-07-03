# ─────────────────────────────────────────────────────────────────────────── #
# oncology-arbiter Dockerfile
#
# Multi-stage:
#
#   1. `builder`     — installs pip deps and does `pip install --prefix=/opt/oa .`
#                       so we can copy artifacts across into a slim runtime
#                       without dragging build tools.
#
#   2. `runtime`     — minimal Python 3.11-slim, non-root user, HEALTHCHECK
#                       against /health. This is what gets shipped.
#
# Notes:
#   * The default runtime installs the CORE dependency set only (fastapi +
#     preprocessing). The `[ml]` extras (torch + monai + transformers) add
#     ~5 GB and are opt-in via build arg ONCOLOGY_ARBITER_INCLUDE_ML=1.
#   * Frontend static bundle is COPIED from the repo (already-built dist/
#     lives under src/oncology_arbiter/api/static/dist/) — we do NOT run
#     `npm run build` in Docker; the build step is a maintainer commit.
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

# Install core deps first, then optionally [ml] extras via build arg.
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
    PYTHONPATH="/opt/oa/lib/python3.11/site-packages" \
    # Serve the built frontend by default when the image is deployed.
    ONCOLOGY_ARBITER_SERVE_FRONTEND=1

# curl only needed for the HEALTHCHECK; everything else came from builder.
RUN apt-get update && apt-get install --no-install-recommends -y \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages + bin from builder.
COPY --from=builder /opt/oa /opt/oa

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
