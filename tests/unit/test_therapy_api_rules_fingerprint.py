"""API-level tests for the v0.2 rules-lite fingerprint + strict wire contract.

Focus: prove that `/v1/therapy/reason` surfaces the ruleset SHA-256 on the
response envelope when (and only when) the NCCN-lite rules fallback ran,
and that `strict=True` propagates through to HTTP 400 on bad input.

Fully deterministic — no biopsy input required, no network.
"""
from __future__ import annotations

import hashlib
import os

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.models.therapy_rules_lite import _RULES_PATH


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    # Make sure we do NOT try TxGemma in these tests.
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", raising=False)
    return TestClient(create_app())


@pytest.fixture()
def client_no_rules(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client with rules-lite gate turned OFF, to exercise the placeholder path."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", raising=False)
    return TestClient(create_app())


def _live_rules_sha() -> str:
    return hashlib.sha256(_RULES_PATH.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Rules-lite path surfaces sha + model_id + branch_id
# --------------------------------------------------------------------------- #


def test_therapy_reason_rules_lite_surfaces_sha_and_branch(client: TestClient) -> None:
    r = client.post("/v1/therapy/reason", json={
        "patient_context": {"menopausal_status": "post", "age": 62},
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provenance"]["model_state"] == "proxy_rules_lite"
    assert d["rules_sha256"] == _live_rules_sha()
    assert d["rules_model_id"] == "nccn-lite-v0"
    # branch_id must be one of the known code paths.
    assert d["branch_id"] in {
        "dcis", "metastatic", "her2_positive",
        "triple_negative", "hr_positive_her2_negative", "fallthrough",
    }, d["branch_id"]


def test_therapy_reason_placeholder_leaves_fingerprint_fields_null(
    client_no_rules: TestClient,
) -> None:
    """When rules-lite is not enabled, the fingerprint block MUST be null."""
    r = client_no_rules.post("/v1/therapy/reason", json={"patient_context": {}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provenance"]["model_state"] == "placeholder"
    assert d["rules_sha256"] is None
    assert d["rules_model_id"] is None
    assert d["branch_id"] is None


# --------------------------------------------------------------------------- #
# 2. strict=True → HTTP 400 with a machine-readable detail
# --------------------------------------------------------------------------- #


def test_therapy_reason_strict_rejects_bad_stage_with_400(client: TestClient) -> None:
    r = client.post("/v1/therapy/reason", json={
        "patient_context": {
            "menopausal_status": "post",
            "age": 62,
            "genomic_markers": {"stage": "stage-two"},
        },
        "strict": True,
    })
    assert r.status_code == 400, r.text
    body = r.json()
    detail = body.get("detail", "")
    assert "therapy_rules_lite_invalid_input" in detail
    assert "stage" in detail
    assert "TNM" in detail or "M1" in detail


def test_therapy_reason_non_strict_still_serves_on_bad_stage(client: TestClient) -> None:
    """The default (strict=False) contract must NOT break on sloppy stage strings."""
    r = client.post("/v1/therapy/reason", json={
        "patient_context": {
            "menopausal_status": "post",
            "age": 62,
            "genomic_markers": {"stage": "stage-two"},
        },
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["provenance"]["model_state"] == "proxy_rules_lite"
    assert d["rules_sha256"] == _live_rules_sha()


# --------------------------------------------------------------------------- #
# 3. Fingerprint is stable across two calls
# --------------------------------------------------------------------------- #


def test_therapy_reason_fingerprint_is_stable_across_calls(client: TestClient) -> None:
    payload = {"patient_context": {"menopausal_status": "post", "age": 62}}
    r1 = client.post("/v1/therapy/reason", json=payload)
    r2 = client.post("/v1/therapy/reason", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    d1, d2 = r1.json(), r2.json()
    assert d1["rules_sha256"] == d2["rules_sha256"]
    assert d1["branch_id"] == d2["branch_id"]
