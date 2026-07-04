"""API-key authentication for the arbiter.

Private-alpha SaaS bar: token-based auth via X-API-Key header. Tokens are
minted server-side, hashed on disk (SHA256), and looked up on every request
that isn't `/health` or `/metrics`.

Storage: SQLite file at `$ONCOLOGY_ARBITER_AUTH_DB_PATH`
   (default `/tmp/oa-audit/tenants.sqlite` — writable by the non-root uid,
   same partition as the audit ledger so operators find them together).

The `X-API-Key` header comes in as raw text. We SHA256 it and index the
`tenants` table on that hash. The plaintext key never touches disk.
"""
from .api_key import (
    APIKey,
    APIKeyDB,
    hash_key,
    make_key,
    verify_api_key,
)
from .middleware import ApiKeyDep, require_api_key

__all__ = [
    "APIKey",
    "APIKeyDB",
    "ApiKeyDep",
    "hash_key",
    "make_key",
    "require_api_key",
    "verify_api_key",
]
