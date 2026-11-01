"""RUO screening fusion endpoint: MedSigLIP embedding + clinical metadata."""

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


def _payload() -> dict:
    return {
        "dicom_bytes_b64": base64.b64encode(b"deidentified-dicom").decode("ascii"),
        "clinical_metadata": {"age": 55.0, "density": "B"},
    }


def test_screening_fusion_returns_score(monkeypatch) -> None:
    class ModalClient:
        def embed_dicom(self, path):
            return [1.0] * 1152

    class FusionClient:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, *, embedding, clinical_metadata):
            assert len(embedding) == 1152
            assert clinical_metadata["age"] == 55.0
            assert clinical_metadata["density"] == "B"
            return 0.42

    monkeypatch.setattr("oncology_arbiter.models.rsna_fusion_modal_client.RsnaFusionModalClient", FusionClient)

    resp = _client(monkeypatch, ModalClient).post("/v1/screening/fusion", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline_status"] == "complete"
    assert body["provenance"]["model_state"] == ModelState.LOADED_MEDSIGLIP.value
    assert body["fusion_model_name"] == "rsna_medsiglip_fusion_logreg_v1"
    assert body["fusion_score"] is not None
    assert body["fusion_score"] == 0.42
    assert body["overall_score"] is None  # fusion_score is distinct from zero-shot

