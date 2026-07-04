"""Tenant CLI: `python -m oncology_arbiter.auth.cli <cmd> ...`

Commands:
    issue <tenant_name>            Mint a new key. Prints the RAW key ONCE.
    revoke <tenant_id>             Revoke a tenant's key.
    list                           List all tenants (id, name, prefix, state).

The DB path is `$ONCOLOGY_ARBITER_AUTH_DB_PATH` or the default
`/tmp/oa-audit/tenants.sqlite`. Persist the DB by mounting a volume.
"""
from __future__ import annotations

import argparse
import json
import sys
from time import strftime, localtime

from .api_key import APIKeyDB


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return strftime("%Y-%m-%d %H:%M:%S", localtime(ts))


def _cmd_issue(db: APIKeyDB, args: argparse.Namespace) -> int:
    raw_key, record = db.issue(tenant_name=args.tenant_name)
    print(json.dumps({
        "tenant_id": record.tenant_id,
        "tenant_name": record.tenant_name,
        "key_prefix": record.key_prefix,
        "created_ts": _fmt_ts(record.created_ts),
        "api_key": raw_key,
        "warning": "Store this key now; it is not recoverable.",
    }, indent=2))
    return 0


def _cmd_revoke(db: APIKeyDB, args: argparse.Namespace) -> int:
    ok = db.revoke(args.tenant_id)
    if not ok:
        print(f"error: tenant_id {args.tenant_id!r} not found or already revoked", file=sys.stderr)
        return 2
    print(json.dumps({"tenant_id": args.tenant_id, "revoked": True}))
    return 0


def _cmd_list(db: APIKeyDB, _args: argparse.Namespace) -> int:
    rows = db.list_all()
    if not rows:
        print("(no tenants)")
        return 0
    print(f"{'tenant_id':<38} {'tenant_name':<24} {'prefix':<14} {'created':<20} {'revoked':<20}")
    for r in rows:
        print(f"{r.tenant_id:<38} {r.tenant_name[:24]:<24} {r.key_prefix:<14} "
              f"{_fmt_ts(r.created_ts):<20} {_fmt_ts(r.revoked_ts):<20}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oncology_arbiter.auth.cli",
                                description="Manage API keys / tenants.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_issue = sub.add_parser("issue", help="Issue a new API key for a tenant.")
    p_issue.add_argument("tenant_name", help="Human-readable tenant name")

    p_revoke = sub.add_parser("revoke", help="Revoke a tenant's key.")
    p_revoke.add_argument("tenant_id", help="Tenant UUID to revoke")

    sub.add_parser("list", help="List all tenants.")

    args = p.parse_args(argv)
    db = APIKeyDB()

    dispatch = {"issue": _cmd_issue, "revoke": _cmd_revoke, "list": _cmd_list}
    return dispatch[args.cmd](db, args)


if __name__ == "__main__":
    raise SystemExit(main())
