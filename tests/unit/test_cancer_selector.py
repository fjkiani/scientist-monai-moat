"""End-to-end tests for the multi-cancer selector.

Covers:
    /health.cancers announces the wired-up set (breast + nsclc today)
    /v1/case/full?cancer=nsclc returns a shape-only placeholder envelope
    /v1/case/full?cancer=lymphoma returns 400
    /v1/case/full?cancer=breast still works (regression)
    /v1/case/full defaults to cancer=breast (regression, no query param)
    NSCLC envelope carries a warning flagging the placeholder status
    NSCLC audit log records extra.cancer=='nsclc' under the tenant partition

These tests run auth OFF so they can hit the endpoints without minting a
tenant key; the SaaS-hardening test file already covers the auth wiring.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUDIT_DIR", str(tmp_path / "audit"))
    from oncology_arbiter.api.app import create_app
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# /health.cancers


class TestHealthCancers:

    def test_health_exposes_cancers_map(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        j = r.json()
        assert "cancers" in j, f"/health MUST expose the cancers map for the SPA; keys={sorted(j.keys())}"
        assert set(j["cancers"].keys()) == {"breast", "nsclc"}, (
            "cancers map must announce both wired-up cancer tracks"
        )

    def test_each_cancer_entry_declares_state_and_endpoints(self, client):
        j = client.get("/health").json()
        for cancer, cap in j["cancers"].items():
            assert "state" in cap, f"{cancer} missing state"
            assert cap["state"] in {
                "placeholder", "loaded", "loading", "unavailable", "cached",
                "gated", "proxy_siglip", "loaded_medsiglip",
                "loaded_biopsy_probe", "loaded_monai_detector",
                "proxy_monai_heuristic", "proxy_rules_lite", "loaded_txgemma",
            }, f"{cancer}.state is not a recognized ModelState value"
            assert "endpoints" in cap, f"{cancer} missing endpoints"
            assert "case/full" in cap["endpoints"], (
                f"{cancer} must at minimum declare 'case/full' — that's the "
                "surface the SPA cancer selector needs to render a working panel"
            )

    def test_nsclc_flagged_as_placeholder(self, client):
        j = client.get("/health").json()
        assert j["cancers"]["nsclc"]["state"] == "placeholder"
        assert "notes" in j["cancers"]["nsclc"], (
            "nsclc should carry an operator-visible 'notes' string until the "
            "LIDC-IDRI pipeline lands"
        )


# --------------------------------------------------------------------------- #
# /v1/case/full?cancer=…


class TestCancerRouting:

    def test_breast_default_still_works(self, client):
        """Not passing cancer=… must not regress the breast path."""
        r = client.post("/v1/case/full", json={})
        assert r.status_code == 200
        j = r.json()
        # Breast branch runs therapy even with no biopsy input.
        assert j["therapy"] is not None
        assert "disclaimer" in j and "provenance" in j

    def test_breast_explicit_matches_default(self, client):
        r1 = client.post("/v1/case/full", json={})
        r2 = client.post("/v1/case/full?cancer=breast", json={})
        assert r1.status_code == 200 == r2.status_code
        # Request-ids differ, but the shape should match: same fields, same
        # therapy branch fired.
        j1, j2 = r1.json(), r2.json()
        assert set(j1.keys()) == set(j2.keys())
        assert (j1["therapy"] is None) == (j2["therapy"] is None)

    def test_nsclc_returns_placeholder_envelope(self, client):
        r = client.post("/v1/case/full?cancer=nsclc", json={})
        assert r.status_code == 200
        j = r.json()
        # Placeholder branch never populates the breast sub-stages.
        assert j["screening"] is None
        assert j["biopsy"] is None
        assert j["therapy"] is None
        assert j["elo_ranked_hypotheses"] == []
        # Full envelope contract still holds.
        assert "disclaimer" in j and "provenance" in j and "honesty_gate" in j
        assert j["provenance"]["model_state"] == "placeholder"
        assert j["provenance"]["model_name"] == "nsclc_placeholder_v0"
        # Placeholder MUST self-flag via warnings — otherwise a downstream
        # consumer could confuse the empty envelope with a "no findings"
        # real inference.
        assert any(
            "nsclc" in w.lower() and "placeholder" in w.lower()
            for w in j["warnings"]
        ), f"nsclc warning missing; got warnings={j['warnings']!r}"

    def test_invalid_cancer_returns_400(self, client):
        r = client.post("/v1/case/full?cancer=lymphoma", json={})
        assert r.status_code == 400
        d = r.json()["detail"]
        assert "lymphoma" in d.lower()
        assert "breast" in d and "nsclc" in d, (
            "400 message should tell the operator which cancers ARE supported"
        )

    def test_cancer_param_case_insensitive(self, client):
        r = client.post("/v1/case/full?cancer=NSCLC", json={})
        assert r.status_code == 200
        # Same placeholder branch as lowercase 'nsclc'.
        assert r.json()["provenance"]["model_name"] == "nsclc_placeholder_v0"


# --------------------------------------------------------------------------- #
# Audit ledger records the cancer track


class TestAuditRecordsCancer:

    def test_nsclc_audit_row_carries_cancer(self, client, tmp_path):
        audit_dir = tmp_path / "audit"
        # Fire the request first, then walk the ledger.
        r = client.post("/v1/case/full?cancer=nsclc", json={})
        assert r.status_code == 200

        # Auth is off in this fixture, so the tenant partition is `_anon`.
        matches = list(glob.glob(str(audit_dir / "*" / "audit-*.jsonl")))
        assert matches, f"no audit log written under {audit_dir}"

        rows = [json.loads(l) for f in matches for l in Path(f).read_text().splitlines() if l.strip()]
        # There will typically be exactly one row for a single /v1/case/full
        # call, but keep the assertion generous in case tests share fixtures.
        case_full_rows = [r for r in rows if r.get("endpoint") == "/v1/case/full"]
        assert case_full_rows, f"no /v1/case/full row in audit log; got endpoints={[r.get('endpoint') for r in rows]}"
        # Every case/full row must carry extra.cancer so grep / dashboards
        # can partition by cancer track.
        cancers_logged = {r.get("extra", {}).get("cancer") for r in case_full_rows}
        assert "nsclc" in cancers_logged, (
            f"case/full audit rows are missing cancer=nsclc; got {cancers_logged!r}"
        )
