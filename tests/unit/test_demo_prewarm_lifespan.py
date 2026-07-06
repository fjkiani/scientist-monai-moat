"""v0.2.2: FastAPI lifespan pre-warms the demo DICOM at startup.

We verify:
1. When SKIP_DEMO_PREWARM=1 (default in tests via conftest), pre-warm is
   not called.
2. When SKIP_DEMO_PREWARM=0 (production/local), pre-warm IS called, and any
   exception it raises is swallowed so startup never fails.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_lifespan_skips_prewarm_when_flag_set(monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "1")

    call_counter = {"n": 0}

    def spy():
        call_counter["n"] += 1
        return None

    monkeypatch.setattr(
        "oncology_arbiter.api.demo_fixtures.prewarm_demo_case", spy
    )

    from oncology_arbiter.api.app import create_app

    with TestClient(create_app()) as tc:
        r = tc.get("/health")
        assert r.status_code == 200

    assert call_counter["n"] == 0, "prewarm must be skipped when flag=1"


def test_lifespan_calls_prewarm_when_flag_off(monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "0")

    call_counter = {"n": 0}

    def spy():
        call_counter["n"] += 1
        return None

    monkeypatch.setattr(
        "oncology_arbiter.api.demo_fixtures.prewarm_demo_case", spy
    )

    from oncology_arbiter.api.app import create_app

    with TestClient(create_app()) as tc:
        r = tc.get("/health")
        assert r.status_code == 200

    assert call_counter["n"] == 1, "prewarm must run exactly once at startup"


def test_lifespan_swallows_prewarm_exceptions(monkeypatch):
    """A crashing pre-warm must NOT prevent the app from starting."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "0")

    def boom():
        raise RuntimeError("simulated HF outage at boot")

    monkeypatch.setattr(
        "oncology_arbiter.api.demo_fixtures.prewarm_demo_case", boom
    )

    from oncology_arbiter.api.app import create_app

    # If the lifespan doesn't swallow exceptions, TestClient(...) will re-raise.
    with TestClient(create_app()) as tc:
        r = tc.get("/health")
        assert r.status_code == 200
