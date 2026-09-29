"""Artifact, backbone, and API-route identity contract for biopsy-probe v1."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from oncology_arbiter.api import app as app_module
from oncology_arbiter.api.app import create_app
from oncology_arbiter.models.biopsy_probe_v1_wiring import (
    ARTIFACT_FILENAME,
    ARTIFACT_SHA256,
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    CLASSES,
    EMBED_DIM,
    PRODUCTION_MODEL_NAME,
    artifact_path,
    build_biopsy_probe,
)
from oncology_arbiter.models.hai_def import AccessLevel, GateReport
from oncology_arbiter.models.medsiglip_modal_client import (
    MedSigLipModalClient,
    ModalEndpointConfig,
)


class _PinnedFakeEncoder:
    repo_id = BASE_MODEL_REPO
    model_revision = BASE_MODEL_REVISION

    def preflight(self) -> GateReport:
        return GateReport(
            repo_id=self.repo_id,
            access_level=AccessLevel.ALLOWED,
            status_code=200,
            reason=f"test-pinned revision={self.model_revision}",
            has_token=True,
        )

    def embed_image(self, **_kwargs) -> np.ndarray:
        return np.linspace(-1.0, 1.0, EMBED_DIM, dtype=np.float32)


def test_biopsy_probe_v1_artifact_identity_and_lineage() -> None:
    path = artifact_path()
    assert path.name == ARTIFACT_FILENAME == "biopsy_probe_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ARTIFACT_SHA256

    payload = json.loads(path.read_text())
    assert payload["$schema_version"] == "biopsy_probe_v1"
    assert tuple(payload["classes"]) == CLASSES
    assert payload["embed_dim"] == EMBED_DIM == 1152
    assert np.asarray(payload["coefficients"]).shape == (3, 1152)
    assert np.asarray(payload["biases"]).shape == (3,)
    assert np.isfinite(np.asarray(payload["coefficients"], dtype=float)).all()
    assert payload["n_training"] == 292
    assert payload["n_training_synthetic"] is False
    assert payload["trained_on_real_patient_data"] is True
    assert payload["embedding_model_repo"] == BASE_MODEL_REPO
    assert payload["embedding_model_revision"] == BASE_MODEL_REVISION


def test_builder_binds_exact_head_and_backbone_revision() -> None:
    probe = build_biopsy_probe(embedding_client=_PinnedFakeEncoder())
    result = probe.run(image_bytes=b"fake-image")
    assert result.artifact_sha256 == ARTIFACT_SHA256
    assert result.embedding_model_revision == BASE_MODEL_REVISION
    assert result.embedding_dim == EMBED_DIM
    assert result.model_name == PRODUCTION_MODEL_NAME
    assert result.weights_n_training == 292
    assert result.weights_n_training_synthetic is False
    assert set(result.subtype_probs) == set(CLASSES)


def test_modal_deployment_source_pins_training_revision() -> None:
    deploy_source = (
        Path(__file__).resolve().parents[2] / "deploy" / "modal" / "medsiglip_app.py"
    ).read_text()
    assert f'MODEL_REVISION = "{BASE_MODEL_REVISION}"' in deploy_source
    assert "revision=MODEL_REVISION" in deploy_source
    assert '"model_revision": self.model_revision' in deploy_source


def test_modal_preflight_requires_revision_and_pooler_dimension() -> None:
    endpoints = ModalEndpointConfig(base="https://example--medsiglip")
    client = MedSigLipModalClient(endpoints=endpoints)
    client._info_cache = {
        "model_repo": BASE_MODEL_REPO,
        "model_revision": BASE_MODEL_REVISION,
        "embedding_dim": EMBED_DIM,
    }
    allowed = client.preflight()
    assert allowed.access_level is AccessLevel.ALLOWED
    assert BASE_MODEL_REVISION in allowed.reason

    drifted = MedSigLipModalClient(endpoints=endpoints)
    drifted._info_cache = {
        "model_repo": BASE_MODEL_REPO,
        "embedding_dim": EMBED_DIM,
    }
    denied = drifted.preflight()
    assert denied.access_level is AccessLevel.UNKNOWN
    assert "model_revision" in denied.reason


def test_api_production_route_exposes_v1_identity(monkeypatch) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "1")
    monkeypatch.setattr(app_module, "_get_medsiglip", lambda: _PinnedFakeEncoder())

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/biopsy/analyze",
            json={"wsi_bytes_b64": base64.b64encode(b"fake-image").decode()},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provenance"]["model_state"] == "loaded_biopsy_probe"
    assert body["provenance"]["model_name"] == PRODUCTION_MODEL_NAME
    assert body["subtype_prediction"] in CLASSES
    assert not any("synthetic" in warning.lower() for warning in body["warnings"])
    assert any("not IDC" in warning for warning in body["warnings"])
