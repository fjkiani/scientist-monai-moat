"""Identity lock for LUNA16 authorized production baseline (production_fallback).

We keep the upstream MONAI `.pt` artifact for provenance, but validate a
`.safetensors` conversion for safer downstream validation.
"""
from pathlib import Path
import hashlib
import json

ARTIFACT_PT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "luna16"
    / "luna16_baseline_b5e79231466a.pt"
)
PT_SHA = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
ARTIFACT_SAFE = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "luna16"
    / "luna16_baseline_b5e79231466a.safetensors"
)
SAFE_SHA = "94bc1635f55e9f7241643dc5b998a28b7f278379850f3fc38fc61a8350f9047e"
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
    assert ARTIFACT_PT.suffix == ".pt"
    assert hashlib.sha256(ARTIFACT_PT.read_bytes()).hexdigest() == PT_SHA

    assert ARTIFACT_SAFE.suffix == ".safetensors"
    assert hashlib.sha256(ARTIFACT_SAFE.read_bytes()).hexdigest() == SAFE_SHA

    receipt = json.loads(FROC.read_text())
    assert receipt["ckpt_sha256"] == PT_SHA
    assert receipt["ckpt_safetensors_sha256"] == SAFE_SHA
    assert float(receipt["froc_at_2"]) == 0.8267399271674979
    assert receipt["status"] == "production_fallback"
    evaluation = json.loads(EVAL.read_text())
    assert evaluation["artifact_sha256"] == SAFE_SHA
    assert evaluation["metric"]["name"] == "delta_froc_at_2"
    assert float(evaluation["metric"]["value"]) == 0.0
    assert evaluation["honesty"]["status_intent"] == "production_fallback"
