"""Identity-locked production wiring for ovarian_mfap4_arbiter_v1.json.

GSE63885 Affymetrix GPL570 probe 212713_at → MFAP4 platinum-resistance
L2 logistic. Honest cohort n=75 (34 resistant / 41 sensitive).
Forbidden: retracted n=101 AUROC 0.7625 (NA-as-sensitive).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .logistic import L2LogisticArbiter

CAPABILITY = "hgsoc-mfap4-platinum"
ARTIFACT_FILENAME = "ovarian_mfap4_arbiter_v1.json"
ARTIFACT_SHA256 = (
    "1524d1638076a6b382104b823e22080294a0b4ace216e03b8420b5ce0fa7a605"
)
PROOF_AUROC = 0.6836
PROBE = "212713_at"
GENE = "MFAP4"
GEO = "GSE63885"
HONEST_N = 75
PRODUCTION_AUROC = 0.684


def artifact_path() -> Path:
    return Path(__file__).parent / "models" / ARTIFACT_FILENAME


def proof_path() -> Path:
    """Repo-root docs/proofs receipt (not package-local)."""
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "proofs"
        / "gse63885_mfap4_metrics.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_mfap4_arbiter() -> L2LogisticArbiter:
    """Load only the byte-identical GSE63885 MFAP4 v1 artifact."""
    path = artifact_path()
    if not path.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {path}")
    actual = _sha256(path)
    if actual != ARTIFACT_SHA256:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: "
            f"expected {ARTIFACT_SHA256}, got {actual}"
        )
    model = L2LogisticArbiter(path)
    if model.n_training != HONEST_N:
        raise RuntimeError(
            f"{CAPABILITY} refuses n_training={model.n_training} "
            f"(expected honest {HONEST_N})"
        )
    held = float(model.performance.get("held_out_test_auroc", 0.0))
    if abs(held - PROOF_AUROC) > 5e-4:
        raise RuntimeError(
            f"{CAPABILITY} AUROC {held} does not match proof {PROOF_AUROC}"
        )
    if held >= 0.75:
        raise RuntimeError(
            f"{CAPABILITY} refuses AUROC>={held} — "
            "retracted n=101 0.7625 must never ship"
        )
    return model


def load_proof_receipt() -> dict:
    path = proof_path()
    if not path.is_file():
        raise FileNotFoundError(f"{CAPABILITY} proof missing: {path}")
    return json.loads(path.read_text())
