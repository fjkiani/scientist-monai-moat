"""Test v0.2.1 root redirect: / -> /ui/ when SERVE_FRONTEND is enabled.

The old behaviour was that GET / returned 404 (there was no root handler).
Clinicians who type the base URL now land on the SPA.

Contract:
- SERVE_FRONTEND=1 and dist present:      GET /  ->  307  Location: /ui/
- SERVE_FRONTEND unset:                    GET /  ->  404 (no redirect)
- The redirect is 307 (temporary, preserves method), NOT 301 (permanent),
  because if the frontend is later moved this URL should not be cached
  permanently by clients.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


def test_root_redirects_to_ui_when_frontend_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", "1")
    app = create_app()
    client = TestClient(app, follow_redirects=False)
    r = client.get("/")
    assert r.status_code == 307, r.text
    assert r.headers["location"] == "/ui/"


def test_root_returns_404_when_frontend_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", raising=False)
    app = create_app()
    client = TestClient(app, follow_redirects=False)
    r = client.get("/")
    assert r.status_code == 404


def test_root_redirect_follows_to_ui_index(monkeypatch) -> None:
    """End-to-end: following the redirect lands on the SPA index.html."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", "1")
    app = create_app()
    client = TestClient(app)  # follow_redirects=True (default)
    r = client.get("/")
    assert r.status_code == 200
    # The SPA index.html has this title; if it changes, this test tells us.
    assert "<title>" in r.text and "Oncology Arbiter" in r.text


def test_ui_prefix_still_serves_directly(monkeypatch) -> None:
    """Regression: the /ui/ mount MUST still serve index.html directly."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_SERVE_FRONTEND", "1")
    app = create_app()
    client = TestClient(app)
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "<title>" in r.text
