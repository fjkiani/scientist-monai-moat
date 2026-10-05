"""Identity lock for LUNA16 authorized production baseline (.pt, pure SHA)."""
from pathlib import Path
import hashlib
import json

ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "luna16"
    / "luna16_baseline_b5e79231466a.pt"
)
EXPECTED = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
FROC = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "luna16"
    / "luna16_baseline_b5e79231466a_froc_result.json"
)
EVAL = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "luna16"
    / "luna16_delivery_evaluation_baseline_v2.json"
)


def test_luna16_baseline_sha_pure_pt_and_honest_delta_zero():
    assert ARTIFACT.suffix == ".pt"
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED
    receipt = json.loads(FROC.read_text())
    assert receipt["ckpt_sha256"] == EXPECTED
    assert float(receipt["froc_at_2"]) == 0.8267399271674979
    assert receipt["status"] == "production_fallback"
    evaluation = json.loads(EVAL.read_text())
    assert evaluation["artifact_sha256"] == EXPECTED
    assert evaluation["metric"]["name"] == "delta_froc_at_2"
    assert float(evaluation["metric"]["value"]) == 0.0
    assert evaluation["honesty"]["status_intent"] == "production_fallback"
