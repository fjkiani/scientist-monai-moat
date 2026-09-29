FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY .github/oncology-gauntlet/requirements.lock /opt/guardian/requirements.lock
RUN python -m pip install --requirement /opt/guardian/requirements.lock

COPY pyproject.toml README.md /opt/guardian/
COPY src/ /opt/guardian/src/

RUN python -m pip install --no-deps --no-build-isolation /opt/guardian

RUN useradd --create-home --uid 10001 guardian
USER guardian

WORKDIR /candidate
