"""API-level tests for L4b biopsy + L4c therapy endpoints.

Biopsy image tests monkey-patch the probe. Therapy tests call the live
MedGemma 1.5 4B chat host with a stated receptor panel.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import app as app_module
from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.biopsy_medsiglip_probe import (
    BACH_BIOPSY_CLASSES,
    MEDSIGLIP_REPO,
    BiopsyMedSigLipProbe,
    BiopsyProbeResult,
    BiopsyProbeWeights,
)
from oncology_arbiter.models.biopsy_probe_v1_wiring import (
    BASE_MODEL_REVISION,
    PRODUCTION_MODEL_NAME,
)
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
)
from oncology_arbiter.models.specialist_clients import SpecialistCall
from oncology_arbiter.models.tissue_microscopy_product import product_tissue_model_name
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




class _FakePhikonClient:
    def embed(self, _image_bytes: bytes, *, request_id: str, required: bool = True):
        vec = np.linspace(0.0, 1.0, 768, dtype=np.float32).tolist()
        return SpecialistCall(
            output={"embeddings": [vec]},
            receipt={
                "stage": "phikon_embedding",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "phikon",
                "error": None,
            },
        )

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


def test_biopsy_wired_returns_subtype(monkeypatch, client: TestClient) -> None:
    """When flag is on and preflight ALLOWED (stubbed), we get a real subtype."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")
    _install_biopsy_stub(monkeypatch, _preflight_allowed)
    import oncology_arbiter.models.specialist_clients as sc

    monkeypatch.setattr(sc, "PhikonClient", _FakePhikonClient)
    monkeypatch.setattr(app_module, "_get_medsiglip", lambda: _FakeEncoder(seed=42))

    resp = client.post(
        "/v1/biopsy/analyze",
        json={"wsi_bytes_b64": base64.b64encode(b"fake-image").decode()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["subtype_prediction"] in BACH_BIOPSY_CLASSES
    assert body["provenance"]["model_state"] == "loaded_phikon"
    assert body["provenance"]["model_name"] == product_tissue_model_name()
    assert 0.0 <= body["confidence"] <= 1.0
    assert not any("synthetic" in w.lower() for w in body["warnings"])
    assert any("not IDC" in w for w in body["warnings"]), body["warnings"]


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


@pytest.mark.integration
def test_therapy_live_medgemma_hr_positive_postmenopausal(client: TestClient) -> None:
    """HR+ HER2- grade 2, age 65, postmenopausal. Live 4B, not a stubbed failure."""
    import json
    from pathlib import Path

    identity = json.loads(
        (Path(__file__).resolve().parents[2] / "models" / "medgemma_1_5_4b_identity.json").read_text()
    )
    payload = {
        "receptors_override": {
            "er_positive": True,
            "pr_positive": True,
            "her2_status": "negative",
        },
        "biopsy_output": _biopsy_output_json(
            subtype="IDC", grade=2, er=True, pr=True, her2="negative"
        ),
        "patient_context": {"age": 65, "menopausal_status": "post"},
        "disease": "breast cancer",
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "loaded_medgemma_1_5_4b_therapy"
    assert body["provenance"]["model_name"] == identity["model_id"]
    receipt = next(r for r in body["stage_receipts"] if r["stage"] == "medgemma_therapy_reason")
    assert receipt["status"] == "succeeded"
    assert receipt["model_version"] == identity["revision_sha256"]
    assert receipt["endpoint_label"] == identity["chat_url"]
    options = body["recommended_options"]
    assert options, body["warnings"]
    regimens = " ".join(opt["regimen"] for opt in options).lower()
    assert any(
        token in regimens
        for token in ("letrozole", "anastrozole", "exemestane")
    ), regimens
    assert not any(
        token in regimens
        for token in (
            "nab-paclitaxel",
            "paclitaxel",
            "doxorubicin",
            "anthracycline",
            "tamoxifen",
            "fulvestrant",
        )
    ), regimens
    blob = " ".join(body.get("warnings") or [])
    assert "RESEARCH USE ONLY" in blob


def test_subtype_router_emits_one_guideline() -> None:
    """Each receptor panel gets one constraint. The other subtypes' rules stay out."""
    from oncology_arbiter.nlp.medgemma_therapy_client import (
        MedGemmaTherapyError,
        build_therapy_user_message,
        get_subtype_prompt_constraint,
        subtype_inputs_from_case,
    )

    her2 = get_subtype_prompt_constraint(True, True, True, True)
    hr_post = get_subtype_prompt_constraint(True, True, False, True)
    hr_pre = get_subtype_prompt_constraint(True, False, False, False)
    tnbc = get_subtype_prompt_constraint(False, False, False, False)
    assert "Trastuzumab / Pertuzumab" in her2
    assert "Aromatase Inhibitor" not in her2
    assert "Letrozole, Anastrozole, or Exemestane" in hr_post
    assert "Tamoxifen" not in hr_post
    assert "Triple-Negative" not in hr_post
    assert "Tamoxifen or ovarian suppression" in hr_pre
    assert "Triple-Negative" not in hr_pre
    assert "CONTRAINDICATED" in tnbc
    assert "Aromatase Inhibitor (Letrozole, Anastrozole, or Exemestane)" not in tnbc

    def _msg(er: bool, pr: bool, her2_status: str, meno: str) -> str:
        _kind, content = build_therapy_user_message(
            patient_context={"age": 60, "menopausal_status": meno},
            biopsy_summary={
                "receptor_panel": {
                    "er_positive": er,
                    "pr_positive": pr,
                    "her2_status": her2_status,
                }
            },
            disease="breast cancer",
        )
        return content

    post = _msg(True, True, "negative", "post")
    triple = _msg(False, False, "negative", "pre")
    assert "HR-positive / HER2-negative postmenopausal" in post
    assert "Triple-Negative" not in post
    assert "Tamoxifen or ovarian suppression" not in post
    assert "Triple-Negative Breast Cancer" in triple
    assert "HR-positive / HER2-negative postmenopausal" not in triple
    assert "If menopausal_status is pre" not in triple

    with pytest.raises(MedGemmaTherapyError, match="receptor panel incomplete"):
        subtype_inputs_from_case({}, {"receptor_panel": {}})


@pytest.mark.integration
def test_therapy_live_medgemma_tnbc_rejects_endocrine(client: TestClient) -> None:
    """ER-/PR-/HER2- must not come back as tamoxifen. One TNBC guideline only."""
    payload = {
        "receptors_override": {
            "er_positive": False,
            "pr_positive": False,
            "her2_status": "negative",
        },
        "biopsy_output": _biopsy_output_json(
            subtype="IDC", grade=3, er=False, pr=False, her2="negative"
        ),
        "patient_context": {"age": 52, "menopausal_status": "pre"},
        "disease": "breast cancer",
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "loaded_medgemma_1_5_4b_therapy"
    regimens = " ".join(opt["regimen"] for opt in body["recommended_options"]).lower()
    assert any(
        token in regimens
        for token in (
            "carboplatin",
            "paclitaxel",
            "doxorubicin",
            "pembrolizumab",
            "atezolizumab",
        )
    ), regimens
    assert not any(
        token in regimens
        for token in ("tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant")
    ), regimens
