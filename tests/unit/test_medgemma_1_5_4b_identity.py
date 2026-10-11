"""Identity lock for genuine MedGemma 1.5 4B-IT tonight delivery."""
from pathlib import Path
import hashlib
import json

from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
    ARTIFACT_SHA256,
    EXPECTED_MODEL_ID,
    EXPECTED_REVISION_SHA256,
)

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "medgemma_1_5_4b_identity.json"


def test_medgemma_identity_sha_and_genuine_model_id():
    assert ARTIFACT.name == "medgemma_1_5_4b_identity.json"
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text())
    assert payload["model_id"] == EXPECTED_MODEL_ID
    assert "qwen" not in payload["model_id"].lower()
    assert payload["revision_sha256"] == EXPECTED_REVISION_SHA256
    assert payload["deployment_image_digest"].startswith("sha256:")
    assert payload["info_url"].startswith("https://")
    assert float(payload["chat_sla_seconds"]) > 0
