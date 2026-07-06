"""v0.2.2: GET /v1/demo/case — PUBLIC endpoint, returns a fully-formed case."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.auth.api_key import APIKeyDB


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(tmp_path / "auth.sqlite"))
    yield


@pytest.fixture
def demo_dicom_local(tmp_path, monkeypatch):
    """Point demo_fixtures at a small local file so we don't hit HuggingFace."""
    local = tmp_path / "local_demo.dcm"
    # Simulate a small (~4 KB) DICOM. Real fixture is ~14 MB; endpoint tests
    # only need to exercise the shape, not the payload volume.
    local.write_bytes(b"\x00\x00DICM_test_fixture" + b"\x42" * 4096)
    monkeypatch.setenv("ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", str(local))
    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_CACHE_DIR", str(tmp_path / "demo-cache")
    )
    return local


@pytest.fixture
def secured_client(monkeypatch, demo_dicom_local):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    db = APIKeyDB()
    raw, rec = db.issue(tenant_name="demo-test-tenant")
    from oncology_arbiter.api.app import create_app
    with TestClient(create_app()) as tc:
        yield tc, rec.tenant_id, raw


@pytest.fixture
def anon_client(monkeypatch, demo_dicom_local):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    from oncology_arbiter.api.app import create_app
    with TestClient(create_app()) as tc:
        yield tc


# --------------------------------------------------------------- auth behaviour


def test_demo_case_is_public_even_when_auth_enforced(secured_client):
    """v0.2.2 fix (2026-07-06): /v1/demo/case is PUBLIC.

    The demo endpoint returns a static server-hosted fixture (real CBIS-DDSM
    DICOM + synthetic luminal-A report). It has no tenant data and no
    per-call cost beyond the one-time HF download that startup pre-warms.
    The whole point of a demo is to let an unauthenticated first-time
    visitor click "Load demo case" and see the pipeline work.

    Every other endpoint that touches tenant data (screening/biopsy/therapy/
    case_full) continues to require the header. Regression tests for those
    live in test_auth.py and test_case_full.py.
    """
    client, _, _ = secured_client
    r = client.get("/v1/demo/case")  # NO X-API-Key header
    assert r.status_code == 200, r.text
    body = r.json()
    # Shape check — the payload should be a real demo case, not an error.
    for k in [
        "dicom_bytes_b64", "dicom_source", "dicom_sha256",
        "dicom_size_bytes", "report_text", "patient_context", "warnings",
    ]:
        assert k in body, f"missing key {k}"


def test_demo_case_accepts_valid_key(secured_client, demo_dicom_local):
    client, _, raw = secured_client
    r = client.get("/v1/demo/case", headers={"X-API-Key": raw})
    assert r.status_code == 200, r.text
    body = r.json()

    # Basic shape
    for k in [
        "dicom_bytes_b64", "dicom_source", "dicom_sha256",
        "dicom_size_bytes", "report_text", "patient_context", "warnings",
    ]:
        assert k in body, f"missing key {k}"

    # Bytes round-trip and match the local fixture we planted.
    decoded = base64.b64decode(body["dicom_bytes_b64"])
    assert decoded == demo_dicom_local.read_bytes()
    assert body["dicom_size_bytes"] == len(decoded)
    assert body["dicom_sha256"] == hashlib.sha256(decoded).hexdigest()


def test_demo_case_works_in_anon_mode(anon_client, demo_dicom_local):
    """AUTH_MODE=off (local dev) still returns the demo case."""
    r = anon_client.get("/v1/demo/case")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["dicom_bytes_b64"]) > 0


# --------------------------------------------------------------- content checks


def test_demo_case_report_text_is_luminal_a(secured_client):
    client, _, raw = secured_client
    r = client.get("/v1/demo/case", headers={"X-API-Key": raw})
    assert r.status_code == 200
    body = r.json()
    # These substrings are what the therapy rules engine routes on to reach
    # the luminal-A branch. If they drift, the demo won't demo the pipeline.
    text = body["report_text"]
    assert "Estrogen Receptor: Positive" in text
    assert "HER2/neu: Negative" in text
    assert "Invasive ductal carcinoma" in text


def test_demo_case_warnings_and_source(secured_client):
    client, _, raw = secured_client
    r = client.get("/v1/demo/case", headers={"X-API-Key": raw})
    body = r.json()
    joined = " ".join(body["warnings"]).lower()
    assert "demo" in joined
    assert "not a real patient" in joined or "research use only" in joined
    src = body["dicom_source"].lower()
    assert "cbis-ddsm" in src


# --------------------------------------------------------------- error path


def test_demo_case_returns_503_when_fixture_unavailable(anon_client, monkeypatch):
    """No local file, no cache, HF unreachable → HTTP 503 with a useful message."""
    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", "/nonexistent/local.dcm"
    )
    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_CACHE_DIR", "/nonexistent/cache/dir/xyz"
    )
    from oncology_arbiter.api import demo_fixtures
    with patch(
        "oncology_arbiter.api.demo_fixtures._download_from_hf",
        side_effect=demo_fixtures.DemoFixtureUnavailable("simulated HF outage"),
    ):
        r = anon_client.get("/v1/demo/case")
    assert r.status_code == 503
    assert "unavailable" in r.text.lower()


# --------------------------------------------------------------- /health surface


def test_health_advertises_demo_endpoint(secured_client):
    """/health must list /v1/demo/case so clients can discover it."""
    client, _, _ = secured_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    endpoints = [e.replace(" ", "") for e in body["endpoints"]]
    assert any("GET" in e and "/v1/demo/case" in e for e in body["endpoints"]), (
        f"demo endpoint missing from /health: {body['endpoints']}"
    )
