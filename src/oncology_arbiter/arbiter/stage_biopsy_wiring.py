"""Identity-locked production wiring for biopsy_arbiter_v1.json."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .logistic import L2LogisticArbiter

CAPABILITY = "stage-biopsy"
ARTIFACT_FILENAME = "biopsy_arbiter_v1.json"
ARTIFACT_SHA256 = "27c457ed4429d8431c86a7f29bba235ed8c8a799f4368c8c3a970b8b59324958"


def artifact_path() -> Path:
    """Return the package-local trained biopsy artifact path."""
    return Path(__file__).parent / "models" / ARTIFACT_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_biopsy_arbiter() -> L2LogisticArbiter:
    """Load only the byte-identical real-patient biopsy v1 artifact."""
    path = artifact_path()
    if not path.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {path}")
    actual = _sha256(path)
    if actual != ARTIFACT_SHA256:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: expected {ARTIFACT_SHA256}, got {actual}"
        )
    model = L2LogisticArbiter(path)
    if model.n_training <= 0:
        raise RuntimeError(f"{CAPABILITY} refuses n_training={model.n_training}")
    return model
