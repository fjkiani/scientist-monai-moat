"""Artifact-identity contract for therapy_arbiter_v1.json."""
from __future__ import annotations

import hashlib
import json

from oncology_arbiter.arbiter.stage_therapy_wiring import (
    ARTIFACT_FILENAME,
    ARTIFACT_SHA256,
    artifact_path,
    load_stage_therapy_arbiter,
)


def test_stage_therapy_v1_artifact_identity_and_target() -> None:
    path = artifact_path()
    assert path.name == ARTIFACT_FILENAME == "therapy_arbiter_v1.json"
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_sha == ARTIFACT_SHA256
    payload = json.loads(path.read_text())
    assert payload["trained_on_real_patient_data"] is True
    assert payload["n_training_synthetic"] is False
    assert payload["n_training"] > 0
    assert payload["n_positive"] + payload["n_negative"] == payload["n_training"]
    assert payload["positive_class"] == "observed_chemotherapy_receipt"
    assert "not benefit" in payload["target_definition"]
    assert len(payload["features"]) == len(payload["coefficients"]) > 0
    model = load_stage_therapy_arbiter()
    assert model.model_name == "therapy_arbiter_v1"
    assert model.n_training == payload["n_training"]
