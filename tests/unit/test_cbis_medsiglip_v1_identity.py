"""Identity lock for CBIS-DDSM LogReg v2 joblib delivery."""
from pathlib import Path
import hashlib

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "cbis_ddsm_logreg_v2.joblib"
EXPECTED = "e5f5b67caf8ba18006e3d582e91becf95fbb20b8b34417366c7ef07b906a211b"


def test_cbis_joblib_sha256_identity():
    h = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert h == EXPECTED
    assert ARTIFACT.name == "cbis_ddsm_logreg_v2.joblib"
