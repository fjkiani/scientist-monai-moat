"""API-level tests for L4b biopsy + L4c therapy endpoints.

These exercise the wired FastAPI routes end-to-end (via TestClient) without
hitting HF — we monkey-patch the probe/client construction. Deterministic,
offline, and fast.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.biopsy_medsiglip_probe import (
    MEDSIGLIP_REPO,
    BiopsyMedSigLipProbe,
    BiopsyProbeResult,
    BiopsyProbeWeights,
)
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
)
from oncology_arbiter.models.txgemma_client import TxGemmaClient


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeEncoder:
    def __init__(self, seed: int = 12345):
        self._rng = np.random.default_rng(seed)

    def embed_image(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        preprocessed_image: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._rng.standard_normal(768).astype(np.float32)


def _preflight_allowed(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="OK",
        has_token=True,
    )


def _preflight_forbidden(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.FORBIDDEN,
        status_code=403,
        reason="terms not accepted",
        has_token=True,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """FastAPI TestClient with all env flags disabled by default."""
    for flag in (
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY",
        "ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY",
    ):
        monkeypatch.delenv(flag, raising=False)
    return TestClient(create_app())


def _biopsy_output_json(subtype: str = "IDC", grade: int = 2,
                        er: bool = True, pr: bool = True,
                        her2: str = "negative") -> dict[str, Any]:
    """Realistic BiopsyResponse json shape for TherapyRequest payloads."""
    return {
        "disclaimer": "RESEARCH USE ONLY",
        "caveat": "AUROC caveat",
        "provenance": {"model_state": "loaded_biopsy_probe", "request_id": "req-t"},
        "honesty_gate": {"seen_urls_count": 0, "evidence_kept": 0, "evidence_dropped": 0},
        "evidence": [],
        "warnings": [],
        "subtype_prediction": subtype,
        "receptor_panel": {
            "er_positive": er, "pr_positive": pr, "her2_status": her2,
        },
        "grade": grade,
        "confidence": 0.7,
    }


def _install_biopsy_stub(monkeypatch, preflight_fn) -> None:
    """Replace BiopsyMedSigLipProbe.__init__ so it uses the fake encoder + preflight."""
    def _fake_init(self, *args, **kwargs) -> None:
        self.repo_id = MEDSIGLIP_REPO
        self._preflight_fn = preflight_fn
        self._weights = BiopsyProbeWeights.load()
        self._client = _FakeEncoder(seed=42)

    monkeypatch.setattr(BiopsyMedSigLipProbe, "__init__", _fake_init)


def _install_txgemma_stub(monkeypatch, preflight_fn) -> None:
    """Replace TxGemmaClient.__init__ so it uses the fake preflight."""
    def _fake_init(self, *args, **kwargs) -> None:
        self.repo_id = "google/txgemma-9b-chat"
        self._preflight_fn = preflight_fn
        self._model = None
        self._tokenizer = None

    monkeypatch.setattr(TxGemmaClient, "__init__", _fake_init)


# --------------------------------------------------------------------------- #
# BIOPSY endpoint
# --------------------------------------------------------------------------- #


def test_biopsy_placeholder_when_flag_off(client: TestClient) -> None:
    resp = client.post("/v1/biopsy/analyze", json={"wsi_bytes_b64": base64.b64encode(b"x").decode()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["subtype_prediction"] is None


def test_biopsy_needs_wsi_or_report(client: TestClient) -> None:
    resp = client.post("/v1/biopsy/analyze", json={})
    assert resp.status_code == 400


def test_biopsy_wired_returns_subtype(monkeypatch, client: TestClient) -> None:
    """When flag is on and preflight ALLOWED (stubbed), we get a real subtype."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")
    _install_biopsy_stub(monkeypatch, _preflight_allowed)

    resp = client.post(
        "/v1/biopsy/analyze",
        json={"wsi_bytes_b64": base64.b64encode(b"fake-image").decode()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["subtype_prediction"] in {"IDC", "DCIS", "benign"}
    assert body["provenance"]["model_state"] == "loaded_biopsy_probe"
    assert body["provenance"]["model_name"] == "google/medsiglip-448+biopsy_probe_v0"
    assert 0.0 <= body["confidence"] <= 1.0
    assert any("synthetic" in w.lower() for w in body["warnings"]), body["warnings"]


def test_biopsy_gated_state_when_forbidden(monkeypatch, client: TestClient) -> None:
    """Preflight FORBIDDEN → gated envelope, NO fabricated subtype."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")
    _install_biopsy_stub(monkeypatch, _preflight_forbidden)

    resp = client.post(
        "/v1/biopsy/analyze",
        json={"wsi_bytes_b64": base64.b64encode(b"fake-image").decode()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "gated"
    assert body["subtype_prediction"] is None
    assert any("biopsy_medsiglip_gated:forbidden" in w for w in body["warnings"])


# --------------------------------------------------------------------------- #
# THERAPY endpoint
# --------------------------------------------------------------------------- #


def test_therapy_placeholder_when_flags_off(client: TestClient) -> None:
    resp = client.post("/v1/therapy/reason", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["recommended_options"] == []


def test_therapy_rules_lite_hr_positive(monkeypatch, client: TestClient) -> None:
    """HR+/HER2- postmenopausal → NCCN-lite recommends AI."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    payload = {
        "biopsy_output": _biopsy_output_json(subtype="IDC", grade=2,
                                             er=True, pr=True, her2="negative"),
        "patient_context": {"age": 65, "menopausal_status": "post"},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "proxy_rules_lite"
    regimens = [o["regimen"].lower() for o in body["recommended_options"]]
    assert any("aromatase" in r for r in regimens), regimens
    # every recommendation carries NCCN evidence
    for opt in body["recommended_options"]:
        assert len(opt["evidence"]) >= 1
        assert "nccn.org" in opt["evidence"][0]["url"]


def test_therapy_txgemma_gated_falls_through_to_rules(monkeypatch, client: TestClient) -> None:
    """TxGemma FORBIDDEN + rules-lite ON → rules-lite serves, warning present."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    _install_txgemma_stub(monkeypatch, _preflight_forbidden)

    payload = {
        "biopsy_output": _biopsy_output_json(subtype="IDC", grade=3,
                                             er=False, pr=False, her2="negative"),
        "patient_context": {"age": 45},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "proxy_rules_lite"
    assert any("txgemma_gated:forbidden" in w for w in body["warnings"]), body["warnings"]


def test_therapy_txgemma_gated_no_rules_stays_placeholder(monkeypatch, client: TestClient) -> None:
    """TxGemma FORBIDDEN + rules-lite OFF → placeholder, no silent fabrication."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    _install_txgemma_stub(monkeypatch, _preflight_forbidden)

    resp = client.post("/v1/therapy/reason", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["recommended_options"] == []
    assert any("txgemma_gated:forbidden" in w for w in body["warnings"])
