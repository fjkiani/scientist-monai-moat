"""Production screening accepts only strict MedSigLIP-448 inference."""
from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState


class _Preprocessed:
    image = np.zeros((8, 8), dtype=np.float32)
    breast_mask = np.ones((8, 8), dtype=bool)
    metadata = SimpleNamespace(
        laterality=SimpleNamespace(value="L"),
        view=SimpleNamespace(value="CC"),
        orientation_flipped=False,
    )


def _client(monkeypatch, modal_client) -> TestClient:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.setattr("oncology_arbiter.mammography.preprocess_mammogram", lambda *args, **kwargs: _Preprocessed())
    monkeypatch.setattr("oncology_arbiter.models.medsiglip_modal_client.MedSigLipModalClient", modal_client)
    return TestClient(create_app())


def _payload() -> dict[str, str]:
    return {"dicom_bytes_b64": base64.b64encode(b"deidentified-dicom").decode("ascii")}


def test_success_requires_1152_dimensional_production_receipt(monkeypatch) -> None:
    class ModalClient:
        def run(self, path):
            return SimpleNamespace(
                labels=["malignant", "without malignancy"],
                probs=[0.2, 0.8],
                warnings=["off-label"],
                model_repo="google/medsiglip-448",
                app_version="medsiglip-modal-v0.4.0-alpha",
                input_resolution=448,
                embedding_dim=1152,
                embedding_sha256="a" * 64,
                prompts=["malignant", "without malignancy"],
                inference_seconds=1.25,
                gate_report=None,
            )

    response = _client(monkeypatch, ModalClient).post("/v1/screening/analyze", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_status"] == "complete"
    assert body["provenance"]["model_state"] == ModelState.LOADED_MEDSIGLIP.value
    assert body["medsiglip"]["embedding_dim"] == 1152
    assert body["medsiglip"]["embedding_sha256"] == "a" * 64
    assert body["stage_receipts"][0]["status"] == "succeeded"
    assert body["arbiter_score"] is None


def test_failure_is_terminal_without_proxy_or_heuristic(monkeypatch) -> None:
    class ModalClient:
        def run(self, path):
            raise TimeoutError("remote inference timed out")

    response = _client(monkeypatch, ModalClient).post("/v1/screening/analyze", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_status"] == "failed_required_stage"
    assert body["stage_receipts"][0]["status"] == "failed_required"
    assert body["findings"] == []
    assert body["overall_score"] is None
    assert body["medsiglip"] is None
    assert body["arbiter_score"] is None
    assert body["provenance"]["model_state"] == ModelState.UNAVAILABLE.value


def test_url_ingestion_is_not_silently_substituted(monkeypatch) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    response = TestClient(create_app()).post(
        "/v1/screening/analyze",
        json={"dicom_url": "https://example.org/study.dcm"},
    )
    assert response.status_code == 400
