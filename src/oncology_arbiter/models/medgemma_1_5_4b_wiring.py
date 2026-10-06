"""Identity-locked wiring for genuine MedGemma 1.5 4B-IT (Modal RUO chat).

Binds ``models/medgemma_1_5_4b_identity.json`` into the oncology arbiter surface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CAPABILITY = "medgemma"
ARTIFACT_FILENAME = "medgemma_1_5_4b_identity.json"
ARTIFACT_PATH = "models/medgemma_1_5_4b_identity.json"
ARTIFACT_SHA256 = "1013eee9b36bf4468d682e86b259bfc1cd9c89f442d50e50816bf3f2526d9b1e"
EXPECTED_MODEL_ID = "google/medgemma-1.5-4b-it"
EXPECTED_REVISION_SHA256 = "300c724c2c1fcdea39f1e21865cb1b14a605f1e3bb5ef50550faaa48da944fc8"


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / ARTIFACT_FILENAME


def verify_artifact_identity(path: Path | None = None) -> dict[str, Any]:
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
    payload = json.loads(resolved.read_text())
    if payload.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError(f"{CAPABILITY}: model_id is not genuine MedGemma")
    if payload.get("revision_sha256") != EXPECTED_REVISION_SHA256:
        raise RuntimeError(f"{CAPABILITY}: revision_sha256 mismatch")
    if "qwen" in str(payload.get("model_id", "")).lower():
        raise RuntimeError(f"{CAPABILITY}: Qwen identity is forbidden")
    return payload


def info_url() -> str:
    return str(verify_artifact_identity()["info_url"])
