"""Smoke tests for the /ui frontend static mount.

The frontend bundle is produced by `npm --prefix frontend run build` and
emitted to `src/oncology_arbiter/api/static/dist/`. This test file:

  1. verifies /ui is NOT mounted when ONCOLOGY_ARBITER_SERVE_FRONTEND is unset
  2. verifies /ui IS mounted, serves index.html, and does NOT clobber /v1/*
     when the flag is set AND the bundle exists on disk

We do not run Playwright here — those live under `frontend/` and require a
Node runtime. This is a pure Python HTTP smoke test.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


@pytest.fixture
def app_with_flag(monkeypatch):
    """Enable the frontend mount; skip if the bundle is not on disk."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", "1")
    static_root = Path(
        __file__
    ).resolve().parents[2] / "src/oncology_arbiter/api/static/dist"
    if not (static_root / "index.html").is_file():
        pytest.skip(
            f"frontend bundle missing at {static_root}; "
            "run `npm --prefix frontend run build` first"
        )
    return create_app()


@pytest.fixture
def app_without_flag(monkeypatch):
    monkeypatch.delenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", raising=False)
    return create_app()


def test_ui_not_mounted_when_flag_unset(app_without_flag):
    with TestClient(app_without_flag) as client:
        resp = client.get("/ui/")
        # Without the mount, the SPA path is a plain 404 — not a static file.
        assert resp.status_code == 404


def test_health_still_available_when_flag_unset(app_without_flag):
    with TestClient(app_without_flag) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "models_loaded" in body
        assert "version" in body


def test_ui_serves_index_when_flag_set(app_with_flag):
    with TestClient(app_with_flag) as client:
        resp = client.get("/ui/")
        assert resp.status_code == 200
        # SPA shell.
        assert "<div id=\"root\">" in resp.text
        assert "text/html" in resp.headers.get("content-type", "")


def test_ui_does_not_shadow_api_routes(app_with_flag):
    """/v1/model-cards must still return JSON, not be swallowed by the SPA fallback."""
    with TestClient(app_with_flag) as client:
        resp = client.get("/v1/model-cards")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/json")


def test_ui_health_available_alongside_spa(app_with_flag):
    with TestClient(app_with_flag) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui/").status_code == 200
