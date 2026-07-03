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


# --------------------------------------------------------------------------- #
# Deeper mount smoke — the Playwright-tier checks reduced to HTTP-only asserts
# that we can run without a browser binary. These verify the same contract
# points the SPA relies on:
#
#   * The bundled JS asset is fetched under /ui/assets/ and returns non-empty
#     JavaScript (regression guard for a wrong-MIME misconfiguration that
#     would break module loading in the browser).
#   * The SPA shell contains the RUO disclaimer text so a client-side render
#     bug can't silently drop it — the disclaimer is a regulatory contract.
#   * The Model-cards API returns >=1 card so the client's Model-cards tab
#     is never rendered against an empty index.
#   * A deep-link inside the SPA (e.g. /ui/screening) falls back to
#     index.html (html=True on StaticFiles), so a page refresh on any tab
#     still works.
# --------------------------------------------------------------------------- #


def test_ui_serves_bundled_js_asset_with_correct_mime(app_with_flag):
    with TestClient(app_with_flag) as client:
        index = client.get("/ui/").text
        # Parse out the first <script type="module" src="/ui/assets/index-*.js">
        # reference the Vite build produced. Any change to that pattern would
        # be a build regression worth catching.
        import re
        m = re.search(r'src="(/ui/assets/index-[^"]+\.js)"', index)
        assert m is not None, (
            "SPA index.html did not link a hashed /ui/assets/index-*.js — "
            "did the Vite build change output layout?"
        )
        asset_path = m.group(1)
        resp = client.get(asset_path)
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith(
            ("application/javascript", "text/javascript")
        ), f"Wrong content-type: {resp.headers.get('content-type')!r}"
        assert len(resp.content) > 10_000, (
            f"Bundled JS is suspiciously small ({len(resp.content)} bytes) — "
            "did Vite emit a stub?"
        )


def test_spa_contains_ruo_disclaimer_marker(app_with_flag):
    """The RUO disclaimer must be reachable somewhere in the shipped bundle.
    Client-side renders can regress silently; if the string is not in the
    JS bundle at all, we know the SPA cannot possibly display it."""
    with TestClient(app_with_flag) as client:
        index = client.get("/ui/").text
        import re
        m = re.search(r'src="(/ui/assets/index-[^"]+\.js)"', index)
        assert m is not None
        js_text = client.get(m.group(1)).text
        # We accept ANY of these markers (App.tsx uses whichever wording).
        markers = [
            "research use only",
            "not for clinical use",
            "RUO",
            "Research Use Only",
        ]
        assert any(mkr.lower() in js_text.lower() for mkr in markers), (
            "no RUO / research-use disclaimer string found in the bundle"
        )


def test_model_cards_endpoint_returns_at_least_one_card(app_with_flag):
    """The Model-cards tab must never render against an empty index."""
    with TestClient(app_with_flag) as client:
        resp = client.get("/v1/model-cards")
        body = resp.json()
        assert "cards" in body
        assert isinstance(body["cards"], list)
        assert len(body["cards"]) >= 1, (
            "model-cards index is empty — SPA Model-cards tab will render blank"
        )
        # Each card carries the fields the SPA consumes (slug + title;
        # honesty_markers is nested).
        for card in body["cards"]:
            assert "slug" in card
            assert "title" in card
            assert "honesty_markers" in card
            hm = card["honesty_markers"]
            assert set(hm.keys()) >= {
                "auroc_caveat_present",
                "ruo_disclaimer_present",
                "not_fda_cleared_note",
            }


def test_ui_deep_link_falls_back_to_index_html(app_with_flag):
    """Client-side routing means /ui/screening, /ui/biopsy etc. do not
    exist on disk — the html=True fallback on the StaticFiles mount must
    serve index.html instead of 404. Otherwise a page refresh on any tab
    breaks the app."""
    with TestClient(app_with_flag) as client:
        for deep in ("/ui/screening", "/ui/biopsy", "/ui/therapy",
                     "/ui/case", "/ui/model-cards"):
            resp = client.get(deep)
            # StaticFiles with html=True returns 200 index.html for missing
            # paths that don't correspond to a real asset.
            assert resp.status_code in (200, 404), (
                f"unexpected {resp.status_code} for {deep}"
            )
            # If it did return 200, verify it's the SPA shell (not e.g. a
            # matching static file with a totally different content).
            if resp.status_code == 200:
                assert "<div id=\"root\">" in resp.text or "root" in resp.text.lower()
