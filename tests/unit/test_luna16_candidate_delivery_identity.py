"""Identity lock for LUNA16 candidate research artifact."""
from pathlib import Path
import hashlib
import json

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "luna16" / "luna16_candidate_90e733f34c6a.safetensors"
EXPECTED = "a1cf3fadf3bebcf14d405c75799acf9e045f13e41370f1d83368900662d4ed24"
FROC = Path(__file__).resolve().parents[2] / "models" / "luna16" / "luna16_candidate_90e733f34c6a_froc_result.json"
BASELINE = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"


def test_luna16_candidate_sha_and_floor_fail_receipt():
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED
    receipt = json.loads(FROC.read_text())
    assert receipt["baseline"]["ckpt_sha256"] == BASELINE
    assert receipt["delta"]["meets_floor_ref"] is False
    assert float(receipt["froc_at_2_refined"]) == 0.0
