"""API-level tests for L4b biopsy + L4c therapy endpoints.

These exercise the wired FastAPI routes end-to-end (via TestClient) without
hitting HF — we monkey-patch the probe/client construction. Deterministic,
offline, and fast.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import app as app_module
from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.biopsy_medsiglip_probe import (
    MEDSIGLIP_REPO,
    BiopsyMedSigLipProbe,
    BiopsyProbeWeights,
)
from oncology_arbiter.models.biopsy_probe_v1_wiring import (
    BASE_MODEL_REVISION,
)
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
)
from oncology_arbiter.models.txgemma_client import TxGemmaClient


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeEncoder:
    repo_id = MEDSIGLIP_REPO
    model_revision = BASE_MODEL_REVISION

    def __init__(self, seed: int = 12345):
        self._rng = np.random.default_rng(seed)

    def preflight(self, repo_id: str = MEDSIGLIP_REPO) -> GateReport:
        return _preflight_allowed(repo_id)

    def embed_image(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        preprocessed_image: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._rng.standard_normal(1152).astype(np.float32)


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
        "ONCOLOGY_ARBITER_ENABLE_L2_EVIDENCE",
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


def test_biopsy_placeholder_when_flag_off(monkeypatch, client: TestClient) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "0")
    resp = client.post("/v1/biopsy/analyze", json={"wsi_bytes_b64": base64.b64encode(b"x").decode()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["model_state"] in ("placeholder", "unavailable")
    assert body["subtype_prediction"] is None


def test_biopsy_needs_wsi_or_report(client: TestClient) -> None:
    resp = client.post("/v1/biopsy/analyze", json={})
    assert resp.status_code == 400


def test_biopsy_report_maps_medgemma_clinical_fields(
    monkeypatch,
    client: TestClient,
) -> None:
    """A de-identified mastectomy report maps to exact structured biomarkers."""
    report_path = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "patient_sample_01"
        / "pathology_note.txt"
    )
    report_text = report_path.read_text(encoding="utf-8")
    revision = "a" * 64

    def _extract_pathology(observed_report: str) -> dict[str, Any]:
        assert observed_report == report_text
        return {
            "model_id": "google/medgemma-1.5-4b-it",
            "revision_sha256": revision,
            "disclaimer": "Research Use Only",
            "extraction": {
                "er_status": "POSITIVE",
                "er_percent": None,
                "pr_status": "POSITIVE",
                "her2_status": "EQUIVOCAL",
                "nottingham_grade": 3,
                "tumor_size_mm": 45.0,
            },
        }

    monkeypatch.setattr(
        "oncology_arbiter.nlp.medgemma_pathology_client.extract_pathology",
        _extract_pathology,
    )
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": report_text},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pipeline_status"] == "complete"
    assert body["provenance"]["model_state"] == "loaded_medgemma_1p5_4b"
    assert body["receptor_panel"] == {
        "er_positive": True,
        "pr_positive": True,
        "her2_status": "equivocal",
        "ki67_percent": None,
        "parse_state": {
            "er": "matched",
            "pr": "matched",
            "her2": "matched",
            "grade": "matched",
        },
    }
    assert body["grade"] == 3
    assert body["report_parse"]["fusion_mode"] == "medgemma_pathology"
    assert (
        body["report_parse"]["extended_fields"]["tumor_size_mm"]["value"]
        == 45.0
    )
    receipt = next(
        row
        for row in body["stage_receipts"]
        if row["stage"] == "medgemma_pathology_parse"
    )
    assert receipt["status"] == "succeeded"
    assert receipt["artifact_sha256"] == revision
    assert body["dss_prognosis"] is None


def test_biopsy_gated_state_when_forbidden(monkeypatch, client: TestClient) -> None:
    """Preflight FORBIDDEN → gated envelope, NO fabricated subtype."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")
    # Pin embedding client identity so a leftover MEDSIGLIP_BACKEND=modal
    # singleton from other tests cannot trip repo-mismatch before the gate.
    import oncology_arbiter.api.services.medsiglip as medsiglip_svc

    medsiglip_svc._MEDSIGLIP_SINGLETON = None
    monkeypatch.setattr(app_module, "_get_medsiglip", lambda: _FakeEncoder(seed=42))
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
    assert resp.status_code == 400


def test_therapy_rules_lite_hr_positive(monkeypatch, client: TestClient) -> None:
    """HR+/HER2- postmenopausal → NCCN-lite recommends AI."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    payload = {
        "biopsy_output": _biopsy_output_json(subtype="IDC", grade=2,
                                             er=True, pr=True, her2="negative"),
        "patient_context": {
            "age": 65,
            "menopausal_status": "post",
            "genomic_markers": {"tnm_stage": "T1N0M0"},
        },
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == ModelState.LOADED_RULES_LITE.value
    regimens = [o["regimen"].lower() for o in body["recommended_options"]]
    assert any("aromatase" in r for r in regimens), regimens
    assert not any("trastuzumab" in r for r in regimens)
    assert body["branch_id"] == "hr_positive_her2_negative"
    assert body["rules_model_id"] == "nccn-lite-v0"
    assert len(body["rules_sha256"]) == 64


def test_therapy_txgemma_gated_falls_through_to_rules(monkeypatch, client: TestClient) -> None:
    """TxGemma FORBIDDEN + rules-lite ON → rules-lite serves, warning present."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    _install_txgemma_stub(monkeypatch, _preflight_forbidden)

    payload = {
        "biopsy_output": _biopsy_output_json(subtype="IDC", grade=3,
                                             er=False, pr=False, her2="negative"),
        "patient_context": {"age": 45, "genomic_markers": {"tnm_stage": "T2N1M0"}},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == ModelState.LOADED_RULES_LITE.value
    assert any("txgemma_gated:forbidden" in w for w in body["warnings"]), body["warnings"]
    assert body["branch_id"] == "triple_negative"
    recommended = [o["regimen"].lower() for o in body["recommended_options"]]
    not_recommended = [o["regimen"].lower() for o in body["not_recommended"]]
    assert any("chemotherapy" in regimen for regimen in recommended)
    assert any("endocrine therapy" in regimen for regimen in not_recommended)


def test_therapy_txgemma_gated_no_rules_stays_placeholder(monkeypatch, client: TestClient) -> None:
    """TxGemma FORBIDDEN + rules-lite OFF → unavailable, no silent fabrication."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", "1")
    _install_txgemma_stub(monkeypatch, _preflight_forbidden)

    payload = {
        "biopsy_output": _biopsy_output_json(subtype="IDC", grade=2,
                                             er=True, pr=True, her2="negative"),
        "patient_context": {"age": 55, "genomic_markers": {"tnm_stage": "T1N0M0"}},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == ModelState.UNAVAILABLE.value
    assert body["recommended_options"] == []
    assert any("txgemma_gated:forbidden" in w for w in body["warnings"])
