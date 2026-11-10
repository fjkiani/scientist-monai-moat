"""Identity lock for mammo-retinanet v1 (honest 0.785 AUROC / floor 0.75)."""
from pathlib import Path
import hashlib
import json

ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "mammo"
    / "mammo_retinanet_v1.safetensors"
)
EXPECTED = "691233bca51c4eb99fd6859301b8e1393ada6643b25e79622548ef80de951d32"
EVAL = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "mammo_retinanet"
    / "mammo_retinanet_evaluation_v2.json"
)


def test_mammo_retinanet_v1_sha_and_honest_auroc():
    assert ARTIFACT.name == "mammo_retinanet_v1.safetensors"
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED
    evaluation = json.loads(EVAL.read_text())
    assert evaluation["artifact_sha256"] == EXPECTED
    assert evaluation["metric"]["name"] == "auroc"
    assert abs(float(evaluation["metric"]["value"]) - 0.785453260650111) < 1e-9
    assert evaluation["honesty"]["floor_amended_to"] == 0.75
