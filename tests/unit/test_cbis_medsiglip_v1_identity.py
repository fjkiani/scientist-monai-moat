"""Identity lock for CBIS-DDSM LogReg v1 joblib delivery."""
from pathlib import Path
import hashlib

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "cbis_ddsm_logreg_v1.joblib"
EXPECTED = "80cd01d8724ab4fac4b6ca8ec9b755afa6db2ec3d96ba086d8fb7a83eddbd3ea"


def test_cbis_joblib_sha256_identity():
    h = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert h == EXPECTED
    assert ARTIFACT.name == "cbis_ddsm_logreg_v1.joblib"
