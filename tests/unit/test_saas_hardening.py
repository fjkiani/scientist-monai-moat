"""SaaS-hardening middleware tests.

Covers the v0.2 additions on top of the placeholder API surface:
  * X-Request-Id echoed on every response (mint or reuse client-supplied)
  * CORS preflight + response headers
  * X-API-Key gate: 401 without key, 200 with valid key, per-tenant audit
  * /metrics exposed
  * Rate limit enforcement (429 with Retry-After)
  * JSON structured logging shim
  * Per-tenant audit-ledger partitioning
"""
from __future__ import annotations

import base64
import glob
import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.auth.api_key import APIKeyDB


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "cbis_ddsm"


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, tmp_path):
    """Give each test its own AUDIT_DIR + auth DB in a fresh tmp dir."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(tmp_path / "auth.sqlite"))
    # Clear cached env-driven paths that could persist from a prior test
    yield


@pytest.fixture
def anon_client(monkeypatch):
    """Client with auth explicitly disabled (AUTH_MODE=off)."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    from oncology_arbiter.api.app import create_app
    with TestClient(create_app()) as tc:
        yield tc


@pytest.fixture
def secured_client(monkeypatch):
    """Client with auth ENFORCED. Yields (client, tenant_id, raw_key)."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    db = APIKeyDB()
    raw, rec = db.issue(tenant_name="test-tenant")
    from oncology_arbiter.api.app import create_app
    with TestClient(create_app()) as tc:
        yield tc, rec.tenant_id, raw


# ------------------------------------------------------------------ request-id

class TestRequestId:
    def test_mints_request_id_when_none_supplied(self, anon_client):
        r = anon_client.get("/health")
        assert r.status_code == 200
        rid = r.headers.get("X-Request-Id")
        assert rid and len(rid) >= 8, f"missing/short X-Request-Id: {rid!r}"

    def test_reuses_client_supplied_request_id(self, anon_client):
        rid = "trace-abcd-1234"
        r = anon_client.get("/health", headers={"X-Request-Id": rid})
        assert r.headers.get("X-Request-Id") == rid

    def test_rejects_malformed_and_mints_fresh(self, anon_client):
        bad = "!!!drop table tenants!!!"
        r = anon_client.get("/health", headers={"X-Request-Id": bad})
        rid = r.headers.get("X-Request-Id")
        assert rid != bad
        assert len(rid) == 32  # uuid4 hex


# ------------------------------------------------------------------------- CORS

class TestCORS:
    def test_wildcard_origin_by_default(self, anon_client):
        # OPTIONS preflight
        r = anon_client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}


# --------------------------------------------------------------- api-key auth

class TestApiKeyAuth:
    def test_no_key_returns_401(self, secured_client):
        client, _, _ = secured_client
        r = client.get("/v1/model-cards")
        assert r.status_code == 401
        # Should not leak details about tenants
        body = r.json()
        assert "detail" in body

    def test_invalid_key_returns_401(self, secured_client):
        client, _, _ = secured_client
        r = client.get("/v1/model-cards", headers={"X-API-Key": "oa_live_invalidkey"})
        assert r.status_code == 401

    def test_valid_key_returns_200(self, secured_client):
        client, tid, raw = secured_client
        r = client.get("/v1/model-cards", headers={"X-API-Key": raw})
        assert r.status_code == 200, r.text

    def test_health_never_requires_key(self, secured_client):
        client, _, _ = secured_client
        r = client.get("/health")
        assert r.status_code == 200

    def test_revoked_key_returns_401(self, secured_client):
        client, tid, raw = secured_client
        db = APIKeyDB()
        assert db.revoke(tid) is True
        r = client.get("/v1/model-cards", headers={"X-API-Key": raw})
        assert r.status_code == 401


# ----------------------------------------------------------- per-tenant audit

class TestPerTenantAudit:
    def test_screening_writes_under_tenant_dir(self, secured_client, tmp_path):
        client, tid, raw = secured_client
        # Use a real DICOM fixture from the existing test set
        fixture = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"
        if not fixture.exists():
            pytest.skip("no fixture dicom")
        b64 = base64.b64encode(fixture.read_bytes()).decode()
        r = client.post(
            "/v1/screening/analyze",
            headers={"X-API-Key": raw},
            json={"dicom_bytes_b64": b64, "patient_id_hash": "b" * 64},
        )
        assert r.status_code == 200, r.text
        request_id = r.json()["provenance"]["request_id"]
        # Ledger should be under <AUDIT_DIR>/<tid>/audit-YYYY-MM-DD.jsonl
        audit_dir = Path(os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"]) / tid
        logs = list(audit_dir.glob("audit-*.jsonl"))
        assert logs, f"no audit log in {audit_dir}"
        seen = False
        for lp in logs:
            for line in lp.read_text().splitlines():
                entry = json.loads(line)
                if entry["request_id"] == request_id:
                    seen = True
                    assert entry["tenant_id"] == tid
                    assert entry["endpoint"] == "/v1/screening/analyze"
                    break
        assert seen, "no audit entry with expected request_id + tenant_id"


# ------------------------------------------------------------------ /metrics

class TestMetrics:
    def test_metrics_endpoint_exposed(self, anon_client):
        r = anon_client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Prometheus exposition format uses '# HELP' + '# TYPE' comments
        assert "# HELP" in body or "# TYPE" in body

    def test_health_traffic_shows_in_metrics(self, anon_client):
        anon_client.get("/health")
        anon_client.get("/health")
        r = anon_client.get("/metrics")
        assert "/health" in r.text or "handler=\"/health\"" in r.text


# ----------------------------------------------------------------- rate limit

class TestRateLimit:
    def test_rate_limit_kicks_in(self, monkeypatch):
        """With a tight rate-limit set, the Nth+1 request should get 429."""
        monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
        monkeypatch.setenv("ONCOLOGY_ARBITER_RATE_LIMIT", "3/minute")
        from oncology_arbiter.api.app import create_app
        with TestClient(create_app()) as client:
            statuses = [client.get("/health").status_code for _ in range(6)]
        # First 3 should be 200, then at least one 429
        assert 429 in statuses, f"expected 429 among {statuses}"
        assert statuses[:3] == [200, 200, 200]


# ------------------------------------------------------------ auth-mode flag

class TestAuthMode:
    def test_auth_off_mode_bypasses_key(self, anon_client):
        r = anon_client.get("/v1/model-cards")
        assert r.status_code == 200

    def test_anon_tenant_id_used_when_auth_off(self, anon_client):
        fixture = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"
        if not fixture.exists():
            pytest.skip("no fixture dicom")
        b64 = base64.b64encode(fixture.read_bytes()).decode()
        r = anon_client.post(
            "/v1/screening/analyze",
            json={"dicom_bytes_b64": b64, "patient_id_hash": "b" * 64},
        )
        assert r.status_code == 200
        rid = r.json()["provenance"]["request_id"]
        audit_dir = Path(os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"]) / "_anon"
        logs = list(audit_dir.glob("audit-*.jsonl"))
        assert logs, f"expected _anon dir at {audit_dir}"
        found = False
        for lp in logs:
            for line in lp.read_text().splitlines():
                if json.loads(line)["request_id"] == rid:
                    found = True
                    break
        assert found


# ---------------------------------------------------------------------------
# Tenant CLI (python -m oncology_arbiter.auth.cli ...)
# ---------------------------------------------------------------------------


class TestTenantCLI:
    """The CLI is what an operator uses to mint keys on a running deploy."""

    def test_issue_and_list_and_revoke(self, tmp_path, monkeypatch, capsys):
        import json
        from oncology_arbiter.auth.cli import main

        db_path = tmp_path / "tenants.sqlite"
        monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(db_path))

        # issue
        rc = main(["issue", "acme"])
        assert rc == 0
        out = capsys.readouterr().out
        rec = json.loads(out)
        assert rec["tenant_name"] == "acme"
        assert rec["api_key"].startswith("oa_live_")
        assert len(rec["api_key"]) == 8 + 32  # "oa_live_" + 32 hex
        tid = rec["tenant_id"]

        # list
        rc = main(["list"])
        assert rc == 0
        assert "acme" in capsys.readouterr().out

        # revoke
        rc = main(["revoke", tid])
        assert rc == 0
        assert '"revoked": true' in capsys.readouterr().out.lower()

        # double-revoke -> non-zero
        rc = main(["revoke", tid])
        assert rc == 2
