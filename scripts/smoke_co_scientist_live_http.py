#!/usr/bin/env python3
"""Live HTTP smoke: POST /v1/co_scientist/run with a minted API key → swarm engine."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from oncology_arbiter.auth.api_key import APIKeyDB
from oncology_arbiter.orchestrator.co_scientist import ENGINE_LIVE


def main() -> int:
    endpoint = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint:
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import verify_artifact_identity

        endpoint = str(verify_artifact_identity().get("chat_url") or "").strip()
        os.environ["MEDGEMMA_MODAL_URL"] = endpoint

    os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = "on"
    audit = ROOT / "artifacts/testing1235/_smoke_audit"
    authdb = ROOT / "artifacts/testing1235/_smoke_auth.sqlite"
    audit.mkdir(parents=True, exist_ok=True)
    os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"] = str(audit)
    os.environ["ONCOLOGY_ARBITER_AUTH_DB_PATH"] = str(authdb)

    db = APIKeyDB(db_path=authdb)
    raw, rec = db.issue(tenant_name="l5-smoke-tenant")

    from oncology_arbiter.api.app import create_app

    pubmed = "https://pubmed.ncbi.nlm.nih.gov/25176067/"
    payload = {
        "screening": {
            "provenance": {"model_state": "loaded_medsiglip"},
            "findings": [{"label": "suspicious_mass", "score": 0.9}],
            "evidence": [{"url": pubmed, "quoted_text": "evidence", "source": "pubmed"}],
        },
        "biopsy": {
            "provenance": {"model_state": "loaded_biopsy_probe"},
            "subtype_prediction": "IDC",
            "confidence": 0.9,
            "receptor_panel": {"ER": "POSITIVE", "PR": "POSITIVE", "HER2": "POSITIVE"},
            "evidence": [{"url": pubmed, "quoted_text": "evidence", "source": "pubmed"}],
        },
        "therapy": {
            "provenance": {"model_state": "loaded_therapy_arbiter_v1"},
            "recommended_options": [
                {
                    "regimen": "trastuzumab + pertuzumab + chemotherapy",
                    "line_of_therapy": 1,
                    "evidence": [{"url": pubmed, "quoted_text": "CLEOPATRA", "source": "pubmed"}],
                },
                {
                    "regimen": "aromatase inhibitor alone",
                    "line_of_therapy": 1,
                    "evidence": [{"url": pubmed, "quoted_text": "endocrine", "source": "pubmed"}],
                },
            ],
        },
        "seed_urls": [pubmed],
        "top_n_evolve": 1,
        "n_variants": 1,
        "return_top": 5,
    }

    print(f"[smoke] POST /v1/co_scientist/run tenant={rec.tenant_id} endpoint={endpoint}", flush=True)
    with TestClient(create_app()) as client:
        resp = client.post(
            "/v1/co_scientist/run",
            json=payload,
            headers={"X-API-Key": raw},
        )
    print(f"[smoke] status={resp.status_code}", flush=True)
    if resp.status_code != 200:
        print(resp.text[:2000], flush=True)
        return 1
    body = resp.json()
    engine = body.get("engine")
    transcript = body.get("tournament_transcript") or []
    out = {
        "http_status": resp.status_code,
        "engine": engine,
        "engine_ok": engine == ENGINE_LIVE,
        "n_matches": len(transcript),
        "n_hypotheses": len(body.get("hypotheses") or []),
        "scalar_leak": any(
            m.get("clinical_rationale") == "offline_scalar_score" for m in transcript
        ),
        "sample_winner": (transcript[0] or {}).get("winner") if transcript else None,
        "sample_margin": (transcript[0] or {}).get("margin") if transcript else None,
    }
    receipt = ROOT / "artifacts/testing1235/co_scientist_live_http_smoke.json"
    receipt.write_text(json.dumps({**out, "body_keys": sorted(body.keys())}, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[smoke] wrote {receipt}", flush=True)
    if not out["engine_ok"] or out["scalar_leak"] or out["n_matches"] < 1:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
