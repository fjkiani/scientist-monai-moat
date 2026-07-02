"""Simple audit ledger for API requests.

Every incoming request gets a `request_id` (uuid4). We log the endpoint,
timestamp, request hash, and response summary to `artifacts/audit/`
newline-delimited JSON. This is the ledger the PLAN §4b talks about — it
lets a site data manager pull retrospective validation data without needing
raw PHI to leave the deployment.

Nothing PII goes into the ledger — we accept an optional patient_id_hash
(SHA256) from the caller and never see the underlying identifier.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(
    os.environ.get(
        "ONCOLOGY_ARBITER_AUDIT_DIR",
        str(Path.cwd() / "artifacts" / "audit"),
    )
)


def new_request_id() -> str:
    return uuid.uuid4().hex


def log_event(
    request_id: str,
    endpoint: str,
    *,
    model_state: str,
    patient_id_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a single audit event."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    log_path = AUDIT_DIR / f"audit-{day}.jsonl"
    entry = {
        "request_id": request_id,
        "ts": time.time(),
        "endpoint": endpoint,
        "model_state": model_state,
        "patient_id_hash": patient_id_hash,
        "extra": extra or {},
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
