"""Identity-locked production wiring for the real BACH biopsy probe v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from oncology_arbiter.models.biopsy_medsiglip_probe import (
    BACH_BIOPSY_CLASSES,
    BIOPSY_PROBE_EMBED_DIM,
    BIOPSY_PROBE_MODEL_NAME,
    BUNDLED_BIOPSY_PROBE_V1_SHA256,
    BiopsyMedSigLipProbe,
)
from oncology_arbiter.models.medsiglip import MEDSIGLIP_REPO, MEDSIGLIP_REVISION

CAPABILITY = "biopsy-probe"
ARTIFACT_FILENAME = "biopsy_probe_v1.json"
ARTIFACT_SHA256 = BUNDLED_BIOPSY_PROBE_V1_SHA256
BASE_MODEL_REPO = MEDSIGLIP_REPO
BASE_MODEL_REVISION = MEDSIGLIP_REVISION
EMBED_DIM = BIOPSY_PROBE_EMBED_DIM
CLASSES = BACH_BIOPSY_CLASSES
PRODUCTION_MODEL_NAME = BIOPSY_PROBE_MODEL_NAME


def artifact_path() -> Path:
    """Return the package-local v1 biopsy-probe artifact path."""
    return Path(__file__).resolve().parent.parent / "arbiter" / "models" / ARTIFACT_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_identity(path: Path | None = None) -> Path:
    """Reject any artifact other than the byte-identical real-patient v1 head."""
    resolved = path or artifact_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {resolved}")
    actual = _sha256(resolved)
    if actual != ARTIFACT_SHA256:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: "
            f"expected {ARTIFACT_SHA256}, got {actual}"
        )

    payload = json.loads(resolved.read_text())
    checks = {
        "$schema_version": payload.get("$schema_version") == "biopsy_probe_v1",
        "classes": tuple(payload.get("classes", ())) == CLASSES,
        "embed_dim": payload.get("embed_dim") == EMBED_DIM,
        "n_training": int(payload.get("n_training", 0)) > 0,
        "n_training_synthetic": payload.get("n_training_synthetic") is False,
        "trained_on_real_patient_data": payload.get("trained_on_real_patient_data") is True,
        "embedding_model_repo": payload.get("embedding_model_repo") == BASE_MODEL_REPO,
        "embedding_model_revision": (
            payload.get("embedding_model_revision") == BASE_MODEL_REVISION
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"{CAPABILITY} artifact failed production lineage checks: {failed}"
        )
    return resolved


def build_biopsy_probe(
    *,
    embedding_client: Any | None = None,
    preflight_fn: Any = None,
) -> BiopsyMedSigLipProbe:
    """Build the only production biopsy route: pinned backbone plus exact head."""
    path = verify_artifact_identity()
    if embedding_client is None:
        from oncology_arbiter.models.medsiglip_modal_client import get_medsiglip_client

        embedding_client = get_medsiglip_client()

    client_repo = getattr(embedding_client, "repo_id", None)
    if client_repo != BASE_MODEL_REPO:
        raise RuntimeError(
            f"{CAPABILITY} embedding repo mismatch: "
            f"expected {BASE_MODEL_REPO!r}, got {client_repo!r}"
        )
    client_revision = getattr(
        embedding_client,
        "model_revision",
        getattr(embedding_client, "revision", None),
    )
    if client_revision != BASE_MODEL_REVISION:
        raise RuntimeError(
            f"{CAPABILITY} embedding revision mismatch: "
            f"expected {BASE_MODEL_REVISION!r}, got {client_revision!r}"
        )

    return BiopsyMedSigLipProbe(
        repo_id=BASE_MODEL_REPO,
        weights_path=path,
        preflight_fn=preflight_fn,
        _shared_client=embedding_client,
        expected_artifact_sha256=ARTIFACT_SHA256,
        expected_model_revision=BASE_MODEL_REVISION,
    )


__all__ = [
    "ARTIFACT_FILENAME",
    "ARTIFACT_SHA256",
    "BASE_MODEL_REPO",
    "BASE_MODEL_REVISION",
    "CAPABILITY",
    "CLASSES",
    "EMBED_DIM",
    "PRODUCTION_MODEL_NAME",
    "artifact_path",
    "build_biopsy_probe",
    "verify_artifact_identity",
]
