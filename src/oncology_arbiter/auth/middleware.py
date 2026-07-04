"""FastAPI dependency + middleware helpers for API-key auth.

Usage in an endpoint:

    @app.get("/v1/foo")
    def foo(tenant: APIKey = Depends(require_api_key)):
        ...

Header: `X-API-Key: oa_live_<32-hex>`

If ONCOLOGY_ARBITER_AUTH_MODE=off, the dependency short-circuits and returns
a special `_anon` APIKey — useful for local dev. In production leave the
env var unset (default: on).
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from .api_key import APIKey, APIKeyDB, verify_api_key


API_KEY_HEADER = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _auth_off() -> bool:
    return os.environ.get("ONCOLOGY_ARBITER_AUTH_MODE", "on").lower() == "off"


def _anon() -> APIKey:
    return APIKey(
        tenant_id="_anon",
        tenant_name="_anon",
        key_prefix="oa_off_",
        created_ts=0.0,
        revoked_ts=None,
    )


def require_api_key(
    request: Request,
    header_key: Optional[str] = Depends(_api_key_scheme),
) -> APIKey:
    """FastAPI dependency: reject requests without a valid X-API-Key header.

    Only fires when `ONCOLOGY_ARBITER_AUTH_MODE != off`. Attaches
    `request.state.tenant_id` for the audit ledger.
    """
    if _auth_off():
        tenant = _anon()
        request.state.tenant_id = tenant.tenant_id
        return tenant

    if not header_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required",
            headers={"WWW-Authenticate": f"ApiKey realm=\"oncology-arbiter\""},
        )
    tenant = verify_api_key(header_key)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is invalid or has been revoked",
            headers={"WWW-Authenticate": f"ApiKey realm=\"oncology-arbiter\""},
        )
    request.state.tenant_id = tenant.tenant_id
    request.state.tenant_name = tenant.tenant_name
    return tenant


# Re-export as `ApiKeyDep` for terse annotations
ApiKeyDep = Depends(require_api_key)


__all__ = ["require_api_key", "ApiKeyDep", "APIKeyDB"]
