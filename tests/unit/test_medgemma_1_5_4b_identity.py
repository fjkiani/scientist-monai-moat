"""Identity lock for genuine MedGemma 1.5 4B-IT tonight delivery."""
from pathlib import Path
import hashlib
import json

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "medgemma_1_5_4b_identity.json"
EXPECTED = "e69dbed948f15ec0134c0315d6d6c8f648d58f63020b6b7a298838149bd2bcfb"
REVISION = "300c724c2c1fcdea39f1e21865cb1b14a605f1e3bb5ef50550faaa48da944fc8"


def test_medgemma_identity_sha_and_genuine_model_id():
    assert ARTIFACT.name == "medgemma_1_5_4b_identity.json"
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(ARTIFACT.read_text())
    assert payload["model_id"] == "google/medgemma-1.5-4b-it"
    assert "qwen" not in payload["model_id"].lower()
    assert payload["revision_sha256"] == REVISION
    assert payload["deployment_image_digest"].startswith("sha256:")
    assert payload["info_url"].startswith("https://")
    assert float(payload["chat_sla_seconds"]) > 0
