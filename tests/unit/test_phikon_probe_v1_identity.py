"""Identity lock for Phikon probe v1 joblib delivery."""
from pathlib import Path
import hashlib

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "phikon_probe_v1.joblib"
EXPECTED = "7c77d0759dbcc00440dac6c9b6d99255523ff9706041ce275e8c56724ff8a713"


def test_phikon_joblib_sha256_identity():
    h = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert h == EXPECTED
    assert ARTIFACT.name == "phikon_probe_v1.joblib"
