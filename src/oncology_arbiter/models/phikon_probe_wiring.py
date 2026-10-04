"""Identity-locked production wiring for Phikon NCT-CRC linear probe v1.

Binds ``models/phikon_probe_v1.joblib`` (sha256 below) into the pathology path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from oncology_arbiter.models.phikon_probe import EXPECTED_SHA256, PhikonProbe

CAPABILITY = "phikon"
ARTIFACT_FILENAME = "phikon_probe_v1.joblib"
ARTIFACT_PATH = "models/phikon_probe_v1.joblib"
ARTIFACT_SHA256 = EXPECTED_SHA256


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / ARTIFACT_FILENAME


def verify_artifact_identity(path: Path | None = None) -> Path:
    resolved = path or artifact_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != ARTIFACT_SHA256:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: "
            f"expected {ARTIFACT_SHA256}, got {actual}"
        )
    return resolved


def predict_tissue_class(embedding: Sequence[float]) -> dict[str, Any]:
    """Run the fitted multinomial probe on a 768-d Phikon embedding."""
    verify_artifact_identity()
    return PhikonProbe.get().predict(embedding)
