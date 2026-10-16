"""Identity-locked wiring for genuine MedGemma 1.5 4B-IT (Modal RUO chat).

Binds ``models/medgemma_1_5_4b_identity.json`` into the oncology arbiter surface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CAPABILITY = "medgemma"
ARTIFACT_FILENAME = "medgemma_1_5_4b_identity.json"
ARTIFACT_PATH = "models/medgemma_1_5_4b_identity.json"
ARTIFACT_SHA256 = "17c89032837d1ad671334dfce0547b83923099313f6e44a9f92493161ee3570e"
EXPECTED_MODEL_ID = "google/medgemma-1.5-4b-it"
EXPECTED_REVISION_SHA256 = "300c724c2c1fcdea39f1e21865cb1b14a605f1e3bb5ef50550faaa48da944fc8"
PINNED_CHAT_HOSTS = frozenset({
    "fjkiani--medgemma-1-5-4b-chat.modal.run",
    "crispro-test--medgemma-1-5-4b-chat.modal.run",
    "testing1235--medgemma-1-5-4b-chat.modal.run",
})


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
    chat_host = urlparse(str(payload.get("chat_url") or "")).netloc
    if chat_host not in PINNED_CHAT_HOSTS:
        raise RuntimeError(f"{CAPABILITY}: chat_url host is not identity-pinned")
    return payload


def assert_pinned_chat_url(url: str) -> str:
    """Reject a MedGemma URL whose host is not in the identity pin set."""
    verify_artifact_identity()
    normalized = (url or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or parsed.netloc not in PINNED_CHAT_HOSTS:
        raise RuntimeError(
            f"{CAPABILITY}: MEDGEMMA_MODAL_URL host is not identity-pinned: "
            f"{parsed.netloc or normalized!r}"
        )
    return normalized


def info_url() -> str:
    return str(verify_artifact_identity()["info_url"])
