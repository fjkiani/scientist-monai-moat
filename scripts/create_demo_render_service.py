"""Create the oncology-arbiter-demo web service on Render via API.

Idempotent: if a service with the same name already exists in the workspace
it prints the existing service and exits without creating a duplicate.

Reads RENDER_API_KEY from env. Owner is hard-coded to the caller workspace.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# ─────────────────────────────────────────────────────────────────────────
# Config

WORKSPACE_ID = "tea-ctia5e52ng1s739ff5rg"  # fjkiani1@gmail.com
SERVICE_NAME = "oncology-arbiter-demo"
REPO_URL = "https://github.com/fjkiani/scientist-monai-moat"
BRANCH = "main"

API_KEY = os.environ.get("RENDER_API_KEY")
if not API_KEY:
    print("ERROR: RENDER_API_KEY not set in environment", file=sys.stderr)
    sys.exit(2)

BASE = "https://api.render.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict | list]:
    data = None if body is None else json.dumps(body).encode()
    req = Request(f"{BASE}{path}", data=data, headers=HEADERS, method=method)
    try:
        with urlopen(req) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except HTTPError as e:
        body_text = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body_text)
        except Exception:  # noqa: BLE001
            return e.code, {"raw": body_text}


# ─────────────────────────────────────────────────────────────────────────
# 1) Idempotency: does the service already exist?

print(f"[1/3] Checking whether service {SERVICE_NAME!r} already exists…")
status, resp = call("GET", f"/services?name={SERVICE_NAME}&limit=20")
if status != 200:
    print(f"  list failed status={status} resp={resp}", file=sys.stderr)
    sys.exit(3)

existing = None
if isinstance(resp, list):
    for entry in resp:
        svc = entry.get("service", {}) if isinstance(entry, dict) else {}
        if svc.get("name") == SERVICE_NAME and svc.get("ownerId") == WORKSPACE_ID:
            existing = svc
            break
if existing is not None:
    url = existing.get("serviceDetails", {}).get("url") or "(no url yet)"
    print(f"  service already exists id={existing['id']} url={url}")
    print(json.dumps(existing, indent=2))
    sys.exit(0)
print("  no existing service by that name — will create one.")


# ─────────────────────────────────────────────────────────────────────────
# 2) Build create payload

payload = {
    "type": "web_service",
    "name": SERVICE_NAME,
    "ownerId": WORKSPACE_ID,
    "repo": REPO_URL,
    "branch": BRANCH,
    "autoDeploy": "yes",
    "envVars": [
        {"key": "ONCOLOGY_ARBITER_DEMO_MODE", "value": "1"},
        {"key": "ONCOLOGY_ARBITER_CONTACT_URL", "value": "https://crispro.ai/contact"},
        {"key": "ONCOLOGY_ARBITER_SERVE_FRONTEND", "value": "1"},
    ],
    "serviceDetails": {
        "env": "docker",
        "plan": "free",
        "region": "oregon",
        "healthCheckPath": "/health",
        "envSpecificDetails": {
            "dockerfilePath": "./Dockerfile",
            "dockerContext": ".",
        },
        "numInstances": 1,
    },
}

print("\n[2/3] Create payload:")
print(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────
# 3) POST /v1/services

print(f"\n[3/3] Creating service on Render (workspace {WORKSPACE_ID})…")
status, resp = call("POST", "/services", body=payload)
print(f"  HTTP {status}")
print(json.dumps(resp, indent=2))

if status not in (200, 201):
    sys.exit(1)
