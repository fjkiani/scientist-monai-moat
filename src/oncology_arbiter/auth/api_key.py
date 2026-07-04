"""API key issuance + verification, backed by SQLite.

Wire format: `oa_live_<32-hex>` (36 chars total). The `oa_live_` prefix
lets a leaked key be identified in log scanners without knowing the whole
value. The 32-hex body is 128 bits of entropy from `secrets.token_hex(16)`.

On disk we store only SHA256(full_key). Verifying a request means hashing
the presented key and looking it up.

We deliberately do NOT use a slow KDF (argon2, bcrypt) here — the tokens
are already 128 bits of raw entropy, so an attacker can't shortcut the
search with a dictionary. A single SHA256 is enough.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DEFAULT_DB = Path(os.environ.get("ONCOLOGY_ARBITER_AUTH_DB_PATH", "/tmp/oa-audit/tenants.sqlite"))
KEY_PREFIX = "oa_live_"


@dataclass(frozen=True)
class APIKey:
    tenant_id: str
    tenant_name: str
    key_prefix: str        # first 12 chars of the key (oa_live_XXXX) for logs
    created_ts: float
    revoked_ts: float | None


def make_key() -> str:
    """Mint a fresh API key: `oa_live_<32-hex>`."""
    return f"{KEY_PREFIX}{secrets.token_hex(16)}"


def hash_key(raw: str) -> str:
    """SHA256 of the raw key, hex-encoded."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class APIKeyDB:
    """Very small SQLite wrapper.

    Schema:
      tenants(tenant_id TEXT PRIMARY KEY,
              tenant_name TEXT NOT NULL,
              key_hash TEXT NOT NULL UNIQUE,
              key_prefix TEXT NOT NULL,
              created_ts REAL NOT NULL,
              revoked_ts REAL)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            con.row_factory = sqlite3.Row
            yield con
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id   TEXT PRIMARY KEY,
                    tenant_name TEXT NOT NULL,
                    key_hash    TEXT NOT NULL UNIQUE,
                    key_prefix  TEXT NOT NULL,
                    created_ts  REAL NOT NULL,
                    revoked_ts  REAL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_key_hash ON tenants(key_hash)")

    # ------------------------------------------------------------- write
    def issue(self, tenant_name: str, tenant_id: str | None = None) -> tuple[str, APIKey]:
        """Mint + persist a new key. Returns (raw_key, record).

        The raw_key is shown to the operator ONCE — never again.
        """
        raw = make_key()
        tid = tenant_id or secrets.token_hex(8)
        rec = APIKey(
            tenant_id=tid,
            tenant_name=tenant_name,
            key_prefix=raw[: len(KEY_PREFIX) + 4],  # oa_live_XXXX
            created_ts=time.time(),
            revoked_ts=None,
        )
        with self._conn() as con:
            con.execute(
                "INSERT INTO tenants(tenant_id, tenant_name, key_hash, key_prefix, created_ts, revoked_ts) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (rec.tenant_id, rec.tenant_name, hash_key(raw), rec.key_prefix, rec.created_ts),
            )
        return raw, rec

    def revoke(self, tenant_id: str) -> bool:
        with self._conn() as con:
            cur = con.execute(
                "UPDATE tenants SET revoked_ts = ? WHERE tenant_id = ? AND revoked_ts IS NULL",
                (time.time(), tenant_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------- read
    def lookup(self, raw_key: str) -> APIKey | None:
        h = hash_key(raw_key)
        with self._conn() as con:
            row = con.execute(
                "SELECT tenant_id, tenant_name, key_prefix, created_ts, revoked_ts "
                "FROM tenants WHERE key_hash = ?",
                (h,),
            ).fetchone()
        if row is None:
            return None
        return APIKey(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            key_prefix=row["key_prefix"],
            created_ts=row["created_ts"],
            revoked_ts=row["revoked_ts"],
        )

    def list_all(self) -> list[APIKey]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT tenant_id, tenant_name, key_prefix, created_ts, revoked_ts "
                "FROM tenants ORDER BY created_ts"
            ).fetchall()
        return [
            APIKey(
                tenant_id=r["tenant_id"],
                tenant_name=r["tenant_name"],
                key_prefix=r["key_prefix"],
                created_ts=r["created_ts"],
                revoked_ts=r["revoked_ts"],
            )
            for r in rows
        ]


def verify_api_key(raw_key: str, db: APIKeyDB | None = None) -> APIKey | None:
    """Return the tenant record if the key is valid + not revoked, else None."""
    if not raw_key or not raw_key.startswith(KEY_PREFIX):
        return None
    db = db or APIKeyDB()
    rec = db.lookup(raw_key)
    if rec is None or rec.revoked_ts is not None:
        return None
    return rec
