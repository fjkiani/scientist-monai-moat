"""Bootstrap-from-env contract for the auth subsystem.

These tests protect three invariants:

1. On a fresh container with full bootstrap env, the tenants table receives
   exactly ONE row and the pre-hashed key verifies.
2. Bootstrap is idempotent — a second call with the table non-empty is a
   no-op (does NOT create a duplicate row or overwrite the existing one).
3. Malformed / missing env is a silent no-op (missing) OR a loud no-op
   (malformed hash), never a raise. This matters because a fresh dev clone
   without env should still boot the app.

The bootstrap module is small on purpose; keep these tests small too.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from oncology_arbiter.auth import verify_api_key
from oncology_arbiter.auth.api_key import APIKeyDB
from oncology_arbiter.auth.bootstrap import (
    ENV_KEY_HASH,
    ENV_KEY_PREFIX,
    ENV_TENANT_ID,
    ENV_TENANT_NAME,
    bootstrap_from_env,
)


def _mk_db(tmp_path: Path) -> APIKeyDB:
    """Fresh SQLite DB at tmp_path/tenants.sqlite."""
    return APIKeyDB(db_path=tmp_path / "tenants.sqlite")


# ----------------------------------------------------------------- happy path
def test_bootstrap_full_env_empty_table_inserts_one_row(tmp_path):
    db = _mk_db(tmp_path)
    raw = "oa_live_" + "c" * 32
    h = hashlib.sha256(raw.encode()).hexdigest()
    env = {
        ENV_TENANT_ID: "alpha-1",
        ENV_TENANT_NAME: "Alpha Deploy",
        ENV_KEY_HASH: h,
    }
    r = bootstrap_from_env(db=db, env=env)
    assert r["fired"] is True
    assert r["reason"] == "bootstrap_ok"
    assert r["tenant_id"] == "alpha-1"
    tenants = db.list_all()
    assert len(tenants) == 1
    assert tenants[0].tenant_id == "alpha-1"
    assert tenants[0].tenant_name == "Alpha Deploy"


def test_bootstrap_yields_verifiable_key(tmp_path):
    """The whole point: after bootstrap, verify_api_key(raw) returns the tenant."""
    db = _mk_db(tmp_path)
    raw = "oa_live_" + "d" * 32
    h = hashlib.sha256(raw.encode()).hexdigest()
    bootstrap_from_env(
        db=db,
        env={
            ENV_TENANT_ID: "beta-1",
            ENV_TENANT_NAME: "Beta",
            ENV_KEY_HASH: h,
        },
    )
    tenant = verify_api_key(raw, db=db)
    assert tenant is not None
    assert tenant.tenant_id == "beta-1"


def test_bootstrap_uses_supplied_key_prefix(tmp_path):
    db = _mk_db(tmp_path)
    raw = "oa_live_" + "e" * 32
    h = hashlib.sha256(raw.encode()).hexdigest()
    bootstrap_from_env(
        db=db,
        env={
            ENV_TENANT_ID: "gamma-1",
            ENV_TENANT_NAME: "Gamma",
            ENV_KEY_HASH: h,
            ENV_KEY_PREFIX: raw[:12],  # "oa_live_eeee"
        },
    )
    tenant = verify_api_key(raw, db=db)
    assert tenant is not None
    assert tenant.key_prefix == raw[:12]


# ----------------------------------------------------------------- idempotency
def test_bootstrap_is_no_op_when_table_already_populated(tmp_path):
    db = _mk_db(tmp_path)
    # Seed one tenant via the normal issue() path.
    raw_existing, _ = db.issue("Pre-existing")
    assert len(db.list_all()) == 1

    raw_boot = "oa_live_" + "f" * 32
    h = hashlib.sha256(raw_boot.encode()).hexdigest()
    r = bootstrap_from_env(
        db=db,
        env={
            ENV_TENANT_ID: "boot",
            ENV_TENANT_NAME: "Boot",
            ENV_KEY_HASH: h,
        },
    )
    assert r["fired"] is False
    assert r["reason"] == "tenants_table_not_empty"
    assert r["existing_count"] == 1
    # Existing tenant must still verify; bootstrap key must NOT.
    assert verify_api_key(raw_existing, db=db) is not None
    assert verify_api_key(raw_boot, db=db) is None


def test_bootstrap_second_call_same_env_is_no_op(tmp_path):
    """Two starts on the same container: first fires, second no-ops."""
    db = _mk_db(tmp_path)
    raw = "oa_live_" + "a" * 32
    h = hashlib.sha256(raw.encode()).hexdigest()
    env = {ENV_TENANT_ID: "boot", ENV_TENANT_NAME: "Boot", ENV_KEY_HASH: h}
    r1 = bootstrap_from_env(db=db, env=env)
    r2 = bootstrap_from_env(db=db, env=env)
    assert r1["fired"] is True
    assert r2["fired"] is False
    assert r2["reason"] == "tenants_table_not_empty"
    # Still exactly one row.
    assert len(db.list_all()) == 1


# ----------------------------------------------------------- error / edge paths
@pytest.mark.parametrize(
    "env",
    [
        {},
        {ENV_TENANT_ID: "x"},
        {ENV_TENANT_ID: "x", ENV_TENANT_NAME: "y"},  # missing hash
        {ENV_TENANT_NAME: "y", ENV_KEY_HASH: "a" * 64},  # missing tid
        {ENV_TENANT_ID: "x", ENV_KEY_HASH: "a" * 64},  # missing name
        {ENV_TENANT_ID: "  ", ENV_TENANT_NAME: "y", ENV_KEY_HASH: "a" * 64},  # blank id
        {ENV_TENANT_ID: "x", ENV_TENANT_NAME: "  ", ENV_KEY_HASH: "a" * 64},  # blank name
    ],
)
def test_bootstrap_incomplete_env_is_silent_no_op(tmp_path, env):
    db = _mk_db(tmp_path)
    r = bootstrap_from_env(db=db, env=env)
    assert r["fired"] is False
    assert r["reason"] == "bootstrap_env_incomplete"
    assert db.list_all() == []


@pytest.mark.parametrize(
    "bad_hash",
    [
        "not-hex-at-all",
        "0" * 63,          # too short
        "0" * 65,          # too long
        "g" * 64,          # non-hex char
        "  " + "a" * 62,   # embedded whitespace
    ],
)
def test_bootstrap_malformed_hash_is_loud_no_op(tmp_path, bad_hash):
    db = _mk_db(tmp_path)
    r = bootstrap_from_env(
        db=db,
        env={
            ENV_TENANT_ID: "x",
            ENV_TENANT_NAME: "y",
            ENV_KEY_HASH: bad_hash,
        },
    )
    assert r["fired"] is False
    assert r["reason"] == "bootstrap_key_hash_malformed"
    assert db.list_all() == []


def test_bootstrap_accepts_uppercase_hash(tmp_path):
    """SHA256 hex is case-insensitive; accept uppercase input, store lower."""
    db = _mk_db(tmp_path)
    raw = "oa_live_" + "9" * 32
    h_upper = hashlib.sha256(raw.encode()).hexdigest().upper()
    r = bootstrap_from_env(
        db=db,
        env={
            ENV_TENANT_ID: "case",
            ENV_TENANT_NAME: "Case",
            ENV_KEY_HASH: h_upper,
        },
    )
    assert r["fired"] is True
    assert verify_api_key(raw, db=db) is not None


def test_bootstrap_never_raises_on_bad_env(tmp_path):
    """Even totally bogus inputs must yield a dict, not a traceback."""
    db = _mk_db(tmp_path)
    for env in [{}, {"foo": "bar"}, {ENV_KEY_HASH: ""}, {ENV_TENANT_ID: ""}]:
        r = bootstrap_from_env(db=db, env=env)
        assert isinstance(r, dict)
        assert r["fired"] is False


# ----------------------------------------------------------------- end-to-end
def test_bootstrap_then_app_startup_401_and_200(tmp_path, monkeypatch):
    """The end-to-end pattern the Render deploy will actually hit."""
    from fastapi.testclient import TestClient

    raw = "oa_live_" + "1" * 32
    h = hashlib.sha256(raw.encode()).hexdigest()

    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(tmp_path / "tenants.sqlite"))
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    monkeypatch.setenv(ENV_TENANT_ID, "e2e")
    monkeypatch.setenv(ENV_TENANT_NAME, "End-to-End")
    monkeypatch.setenv(ENV_KEY_HASH, h)

    from oncology_arbiter.api.app import create_app

    app = create_app()
    client = TestClient(app)

    # /health is public
    assert client.get("/health").status_code == 200

    # /v1/model-cards without key -> 401
    r = client.get("/v1/model-cards")
    assert r.status_code == 401
    assert "X-API-Key header is required" in r.json()["detail"]

    # /v1/model-cards with the correct key -> 200
    r = client.get("/v1/model-cards", headers={"X-API-Key": raw})
    assert r.status_code == 200

    # /v1/model-cards with a wrong key -> 401
    r = client.get("/v1/model-cards", headers={"X-API-Key": "oa_live_" + "z" * 32})
    assert r.status_code == 401
    assert "invalid or has been revoked" in r.json()["detail"]
