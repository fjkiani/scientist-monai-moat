"""Simple audit ledger for API requests.

Every incoming request gets a `request_id` (uuid4). We log the endpoint,
timestamp, request hash, tenant id, and response summary to
`artifacts/audit/<tenant_id>/audit-YYYY-MM-DD.jsonl` — one file per tenant
per UTC day. This is the ledger the PLAN §4b talks about: it lets a site
data manager pull retrospective validation data without needing raw PHI to
leave the deployment.

Nothing PII goes into the ledger — we accept an optional patient_id_hash
(SHA256) from the caller and never see the underlying identifier.

If `tenant_id` is not provided (or empty), events land in
`<AUDIT_DIR>/_anon/` — that's expected on the free-tier deployment where
API-key auth is disabled.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _default_audit_dir() -> Path:
    return Path.cwd() / "artifacts" / "audit"


def _audit_dir() -> Path:
    """Resolve AUDIT_DIR each call so env changes at test time take effect."""
    return Path(os.environ.get("ONCOLOGY_ARBITER_AUDIT_DIR", str(_default_audit_dir())))


# Back-compat: module-level constant reads from env at import; tests that
# set the env before importing this module still see the right value.
AUDIT_DIR = _audit_dir()


def new_request_id() -> str:
    return uuid.uuid4().hex


def log_event(
    request_id: str,
    endpoint: str,
    *,
    model_state: str,
    tenant_id: str | None = None,
    patient_id_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a single audit event.

    Events are partitioned by tenant so a per-customer export is a simple
    `find <AUDIT_DIR>/<tenant_id>` — no scanning of other customers' rows.
    """
    tid = tenant_id or "_anon"
    day = time.strftime("%Y-%m-%d", time.gmtime())
    tenant_dir = _audit_dir() / tid
    tenant_dir.mkdir(parents=True, exist_ok=True)
    log_path = tenant_dir / f"audit-{day}.jsonl"
    entry = {
        "request_id": request_id,
        "ts": time.time(),
        "endpoint": endpoint,
        "model_state": model_state,
        "tenant_id": tid,
        "patient_id_hash": patient_id_hash,
        "extra": extra or {},
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
