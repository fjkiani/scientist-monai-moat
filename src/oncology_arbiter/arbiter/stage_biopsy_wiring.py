"""Identity-locked production wiring for biopsy_arbiter_v1.json."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .logistic import L2LogisticArbiter

CAPABILITY = "stage-biopsy"

UNVALIDATED_LESION_STRATUM_CODE = "unvalidated_lesion_stratum"
UNVALIDATED_LESION_STRATUM_MESSAGE = "Unvalidated Lesion Stratum"

_CALC_LESION_LEVELS = frozenset({
    "calcification_pleomorphic",
    "calcification_amorphous",
})
_CALC_ONE_HOT_KEYS = frozenset({
    "lesion_type_calcification_pleomorphic",
    "lesion_type_calcification_amorphous",
})
_MASS_ONE_HOT_KEYS = frozenset({
    "lesion_type_mass_spiculated",
    "lesion_type_mass_circumscribed",
    "lesion_type_architectural_distortion",
})


class UnvalidatedLesionStratumError(Exception):
    """Calcification-only morphology is outside the validated biopsy arbiter stratum."""

    def detail(self) -> dict[str, Any]:
        return {
            "error": UNVALIDATED_LESION_STRATUM_CODE,
            "message": UNVALIDATED_LESION_STRATUM_MESSAGE,
            "stage": CAPABILITY,
            "detail": (
                "biopsy_arbiter_v1 was not validated on calcification-only lesions; "
                "supply a mass or architectural-distortion descriptor, or omit "
                "calcification morphology."
            ),
        }


def _feature_active(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def enforce_validated_lesion_stratum(features: Mapping[str, Any]) -> None:
    """Fail closed when the panel is calcification-only (no mass descriptors).

    RUO honesty: the frozen biopsy_arbiter_v1 evaluation stratum is mass- and
    distortion-heavy; calc-only inputs must not receive a calibrated score.
    """
    lesion_type = features.get("lesion_type")
    if isinstance(lesion_type, str):
        level = lesion_type.strip()
        if level in _CALC_LESION_LEVELS:
            raise UnvalidatedLesionStratumError()
        return

    has_calc = any(_feature_active(features.get(key)) for key in _CALC_ONE_HOT_KEYS)
    has_mass = any(_feature_active(features.get(key)) for key in _MASS_ONE_HOT_KEYS)
    if has_calc and not has_mass:
        raise UnvalidatedLesionStratumError()


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
