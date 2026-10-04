"""Identity lock for CBIS-DDSM fusion v3 joblib delivery."""
from pathlib import Path
import hashlib

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "cbis_ddsm_logreg_v3.joblib"
EXPECTED = "f40f33c1cd3adfa48d1d10cc4d136795ad0642c3d1f94a543872ae1dc283aaf7"


def test_cbis_joblib_sha256_identity():
    h = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert h == EXPECTED
    assert ARTIFACT.name == "cbis_ddsm_logreg_v3.joblib"
