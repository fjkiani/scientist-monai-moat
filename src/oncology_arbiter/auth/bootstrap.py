"""One-shot bootstrap of the SQLite tenants table from environment.

Motivation
----------
On a fresh Render container the SQLite tenants table is empty, so flipping
``ONCOLOGY_ARBITER_AUTH_MODE`` to ``on`` locks out every caller. There is no
shell into the free-tier container to mint a key by hand, and we do not want
to expose an admin HTTP endpoint that mints keys over the wire.

This module resolves that by reading a pre-hashed key from environment on
startup and injecting exactly one tenant row *iff* the table is empty. The
plaintext key never touches the container's environment — only the SHA256
hex digest does.

Contract
--------
The deploy operator generates a key locally with :func:`make_key`, hashes it
with :func:`hash_key`, and sets three env vars on the Render service::

    ONCOLOGY_ARBITER_BOOTSTRAP_TENANT_ID=<free-form id, e.g. "bootstrap-alpha">
    ONCOLOGY_ARBITER_BOOTSTRAP_TENANT_NAME=<display name, e.g. "Alpha Deploy">
    ONCOLOGY_ARBITER_BOOTSTRAP_KEY_HASH=<64 hex chars, SHA256 of raw key>

On process start :func:`bootstrap_from_env` is called. It is a no-op if:
  - any of the three env vars is missing, OR
  - the ``tenants`` table already has at least one row.

If all three vars are present and the table is empty, it inserts a single
tenant row. The insert is idempotent — a second start on the same container
sees a non-empty table and does nothing.

Return value
------------
Returns a small dict describing what happened; the API startup handler
writes this to the log so operators can confirm the bootstrap fired.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from .api_key import APIKey, APIKeyDB


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

ENV_TENANT_ID = "ONCOLOGY_ARBITER_BOOTSTRAP_TENANT_ID"
ENV_TENANT_NAME = "ONCOLOGY_ARBITER_BOOTSTRAP_TENANT_NAME"
ENV_KEY_HASH = "ONCOLOGY_ARBITER_BOOTSTRAP_KEY_HASH"
ENV_KEY_PREFIX = "ONCOLOGY_ARBITER_BOOTSTRAP_KEY_PREFIX"  # optional, for log-visible prefix


def _peek_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def bootstrap_from_env(
    db: Optional[APIKeyDB] = None,
    *,
    env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Inject one bootstrap tenant row if env is set AND table is empty.

    Parameters
    ----------
    db:
        Optional pre-built :class:`APIKeyDB`. Tests pass a temp-dir one.
    env:
        Optional env override for testing. Defaults to ``os.environ``.

    Returns
    -------
    dict with keys:
      - ``fired`` (bool): True iff a row was inserted.
      - ``reason`` (str): human-readable status.
      - ``tenant_id`` (str | None): inserted tenant id, if fired.
      - ``key_prefix`` (str | None): logged key prefix, if provided.
    """
    e = env if env is not None else os.environ

    tid = _peek_env(ENV_TENANT_ID) if env is None else (e.get(ENV_TENANT_ID) or "").strip() or None
    tname = _peek_env(ENV_TENANT_NAME) if env is None else (e.get(ENV_TENANT_NAME) or "").strip() or None
    khash = _peek_env(ENV_KEY_HASH) if env is None else (e.get(ENV_KEY_HASH) or "").strip() or None
    kprefix = _peek_env(ENV_KEY_PREFIX) if env is None else (e.get(ENV_KEY_PREFIX) or "").strip() or None

    # Missing any of the 3 required vars -> silent no-op.
    if not (tid and tname and khash):
        return {
            "fired": False,
            "reason": "bootstrap_env_incomplete",
            "tenant_id": None,
            "key_prefix": None,
        }

    # Validate the hash format loudly. A malformed hash is a config bug we
    # must surface, not silently absorb.
    if not _HEX64_RE.match(khash.lower()):
        return {
            "fired": False,
            "reason": "bootstrap_key_hash_malformed",
            "tenant_id": None,
            "key_prefix": None,
        }

    db = db or APIKeyDB()

    # Idempotency guard: any existing tenant means we've already bootstrapped
    # (or the operator minted one via a different path). Do NOT overwrite.
    existing = db.list_all()
    if existing:
        return {
            "fired": False,
            "reason": "tenants_table_not_empty",
            "tenant_id": None,
            "key_prefix": None,
            "existing_count": len(existing),
        }

    # Insert directly against the SQLite backing so we can carry a caller-
    # supplied hash (the normal `issue()` path generates its own raw key).
    prefix_display = kprefix or "oa_live_boot"
    now = time.time()
    with db._conn() as con:  # noqa: SLF001 — bootstrap is a privileged callee inside the auth package
        con.execute(
            "INSERT INTO tenants(tenant_id, tenant_name, key_hash, key_prefix, created_ts, revoked_ts) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            (tid, tname, khash.lower(), prefix_display[:16], now),
        )

    return {
        "fired": True,
        "reason": "bootstrap_ok",
        "tenant_id": tid,
        "key_prefix": prefix_display[:16],
    }


__all__ = [
    "bootstrap_from_env",
    "ENV_TENANT_ID",
    "ENV_TENANT_NAME",
    "ENV_KEY_HASH",
    "ENV_KEY_PREFIX",
]
