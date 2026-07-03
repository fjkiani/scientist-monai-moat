"""Verify biopsy + therapy endpoints emit structured ``gate_report`` on the
``Provenance`` block, matching the contract added by conductor 2.1 for the
screening endpoint.

Contract:
  * When the L4b/L4c backends are disabled → ``provenance.gate_report`` is
    ``None`` (placeholder path never touched HAI-DEF).
  * When the L4c NCCN-lite rules proxy is on but there is no HAI-DEF preflight
    → still ``None`` (proxy is offline / does not talk to a gated repo).
  * When TxGemma is enabled AND the fake client raises ``GatedAccessError`` →
    ``provenance.gate_report`` is populated with the structured fields
    (repo_id / access_level / status_code / reason / has_token / allowed).
  * The Literal for ``access_level`` MUST accept "forbidden" verbatim (regression
    guard for the enum-value → string conversion in ``_to_schema_gate_report``).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GatedAccessError,
    GateReport as RuntimeGateReport,
)


# --------------------------------------------------------------------------- #
# Biopsy endpoint

def test_biopsy_placeholder_has_no_gate_report(monkeypatch):
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", raising=False)
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/biopsy/analyze",
            json={"report_text": "specimen: benign fibroadenoma"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["gate_report"] is None
    assert body["provenance"]["model_state"] == "placeholder"


def test_biopsy_gated_response_has_structured_gate_report(monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")

    def _raise_gated(*args, **kwargs):
        raise GatedAccessError(
            repo_id="google/medsiglip-448",
            access_level=AccessLevel.FORBIDDEN,
            status_code=403,
            reason="gated repo — access request required",
        )

    from oncology_arbiter.models import biopsy_medsiglip_probe as mod

    with patch.object(mod.BiopsyMedSigLipProbe, "run", side_effect=_raise_gated):
        with TestClient(create_app()) as client:
            # Tiny JPEG-ish base64 blob is fine — GatedAccessError raised
            # before we ever look at the bytes.
            r = client.post(
                "/v1/biopsy/analyze",
                json={"wsi_bytes_b64": "iVBORw0KGgo=", "report_text": "x"},
            )
    assert r.status_code == 200, r.text
    body = r.json()
    prov = body["provenance"]
    assert prov["model_state"] == "gated"
    gr = prov["gate_report"]
    assert gr is not None
    assert gr["repo_id"] == "google/medsiglip-448"
    assert gr["access_level"] == "forbidden"
    assert gr["status_code"] == 403
    assert gr["allowed"] is False
    assert isinstance(gr["has_token"], bool)


# --------------------------------------------------------------------------- #
# Therapy endpoint

def test_therapy_placeholder_has_no_gate_report(monkeypatch):
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", raising=False)
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/therapy/reason",
            json={"patient_context": {"age": 60, "menopausal_status": "post"}},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"]["gate_report"] is None


def test_therapy_rules_proxy_has_no_gate_report(monkeypatch):
    """Rules-lite is offline; it must NOT synthesize a gate_report — that
    would misrepresent a proxy as though it had HAI-DEF preflight."""
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", raising=False)
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/therapy/reason",
            json={"patient_context": {"age": 60, "menopausal_status": "post"}},
        )
    assert r.status_code == 200
    body = r.json()
    # Rules-lite responds, but there was no HAI-DEF preflight → gate_report None
    assert body["provenance"]["gate_report"] is None
    assert body["provenance"]["model_state"] == "proxy_rules_lite"


def test_therapy_txgemma_gated_response_has_structured_gate_report(monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", raising=False)

    from oncology_arbiter.models import txgemma_client as tx_mod

    def _raise_gated(*args, **kwargs):
        raise GatedAccessError(
            repo_id="google/txgemma-9b",
            access_level=AccessLevel.FORBIDDEN,
            status_code=403,
            reason="gated repo — access request required",
        )

    with patch.object(tx_mod.TxGemmaClient, "recommend_therapy",
                      side_effect=_raise_gated):
        with TestClient(create_app()) as client:
            r = client.post(
                "/v1/therapy/reason",
                json={"patient_context": {"age": 55, "menopausal_status": "post"}},
            )
    assert r.status_code == 200, r.text
    body = r.json()
    gr = body["provenance"]["gate_report"]
    assert gr is not None
    assert gr["repo_id"] == "google/txgemma-9b"
    assert gr["access_level"] == "forbidden"
    assert gr["status_code"] == 403
    assert gr["allowed"] is False


def test_therapy_txgemma_gated_still_falls_through_to_rules_proxy(monkeypatch):
    """If TxGemma is gated AND rules-lite is on, the response still contains
    recommendations (from rules-lite) but gate_report reflects the TxGemma
    preflight — client learns why TxGemma didn't run."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")

    from oncology_arbiter.models import txgemma_client as tx_mod

    def _raise_gated(*args, **kwargs):
        raise GatedAccessError(
            repo_id="google/txgemma-9b",
            access_level=AccessLevel.FORBIDDEN,
            status_code=403,
            reason="gated",
        )

    with patch.object(tx_mod.TxGemmaClient, "recommend_therapy",
                      side_effect=_raise_gated):
        with TestClient(create_app()) as client:
            r = client.post(
                "/v1/therapy/reason",
                json={"patient_context": {"age": 55, "menopausal_status": "post"}},
            )
    assert r.status_code == 200
    body = r.json()
    # Model state is rules-lite because that produced the recommendations,
    # but gate_report carries the TxGemma preflight so the client knows why
    # the higher-tier backend didn't run.
    assert body["provenance"]["model_state"] == "proxy_rules_lite"
    gr = body["provenance"]["gate_report"]
    assert gr is not None
    assert gr["repo_id"] == "google/txgemma-9b"
    assert gr["allowed"] is False
