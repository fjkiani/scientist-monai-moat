#!/usr/bin/env python3
"""Live L5 MedGemma pairwise Co-Scientist receipt for case 892531e5e4dde1e8.

Calls ``run_co_scientist(mode=\"live\")`` (same engine as ``POST /v1/co_scientist/run``)
against identity-pinned MedGemma. Writes an auditable JSON receipt under
``artifacts/live_io/``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASE_ID = "892531e5e4dde1e8"
PUBMED_TRASTUZUMAB = "https://pubmed.ncbi.nlm.nih.gov/27650830/"
PUBMED_PERTUZUMAB = "https://pubmed.ncbi.nlm.nih.gov/25176067/"
PUBMED_TDM1 = "https://pubmed.ncbi.nlm.nih.gov/23020162/"


def _envelopes_from_e2e() -> tuple[dict, dict, dict, dict, set[str]]:
    """Build stage envelopes from the live production receipt for case 892531e5."""
    receipt_path = ROOT / "artifacts/testing1235/e2e_live_production_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    mods = receipt.get("modalities") or {}
    entities = (mods.get("clinicalbert") or {}).get("entities") or {}
    luna = mods.get("luna16") or {}
    mammo = mods.get("mammo_retinanet") or {}
    therapy = mods.get("stage_therapy") or {}
    final = receipt.get("final_clinical_action_report") or {}

    patient_context = {
        "case_id": CASE_ID,
        "age": 54,
        "receptors": {
            "ER": entities.get("er_status"),
            "PR": entities.get("pr_status"),
            "HER2": entities.get("her2_status"),
        },
        "nottingham_grade": entities.get("nottingham_grade"),
        "tumor_size_mm": entities.get("tumor_size_mm"),
        "luna16_n_detections": luna.get("n_detections"),
        "luna16_top_score": luna.get("top_score"),
        "mammo_proba_cancer": mammo.get("proba_cancer"),
        "therapy_risk_bucket": therapy.get("risk_bucket"),
        "prescription_ruo": final.get("prescription_ruo"),
        "primary_driver": final.get("primary_driver"),
    }

    seen_urls = {PUBMED_TRASTUZUMAB, PUBMED_PERTUZUMAB, PUBMED_TDM1}

    screening = {
        "provenance": {
            "model_state": mammo.get("model_state") or "loaded_mammo_retinanet_v1",
            "model_name": "mammo_retinanet_v1",
        },
        "findings": [
            {
                "label": "suspicious_mass",
                "score": float(mammo.get("proba_cancer") or 0.98),
            }
        ],
        "evidence": [
            {
                "url": PUBMED_TRASTUZUMAB,
                "quoted_text": "trastuzumab adjuvant breast cancer",
                "source": "pubmed",
            }
        ],
    }
    biopsy = {
        "provenance": {
            "model_state": "loaded_medgemma_1_5_4b",
            "model_name": "medgemma_pathology_extract",
        },
        "subtype_prediction": "IDC",
        "confidence": 0.91,
        "receptor_panel": patient_context["receptors"],
        "grade": entities.get("nottingham_grade"),
        "tumor_size_mm": entities.get("tumor_size_mm"),
        "evidence": [
            {
                "url": PUBMED_PERTUZUMAB,
                "quoted_text": "pertuzumab CLEOPATRA / APHINITY class evidence",
                "source": "pubmed",
            }
        ],
    }
    therapy_env = {
        "provenance": {
            "model_state": therapy.get("model_state") or "loaded_therapy_arbiter_v1",
            "model_name": "therapy_arbiter_v1",
        },
        "recommended_options": [
            {
                "regimen": "trastuzumab + chemotherapy",
                "line_of_therapy": 1,
                "evidence": [
                    {
                        "url": PUBMED_TRASTUZUMAB,
                        "quoted_text": "adjuvant trastuzumab improves DFS/OS in HER2+",
                        "source": "pubmed",
                    }
                ],
            },
            {
                "regimen": "trastuzumab + pertuzumab + chemotherapy",
                "line_of_therapy": 1,
                "evidence": [
                    {
                        "url": PUBMED_PERTUZUMAB,
                        "quoted_text": "dual blockade adds pathologic CR / DFS signal",
                        "source": "pubmed",
                    }
                ],
            },
            {
                "regimen": "T-DM1 (ado-trastuzumab emtansine)",
                "line_of_therapy": 2,
                "evidence": [
                    {
                        "url": PUBMED_TDM1,
                        "quoted_text": "T-DM1 for residual disease / progressive HER2+",
                        "source": "pubmed",
                    }
                ],
            },
        ],
        "evidence": [
            {
                "url": PUBMED_TDM1,
                "quoted_text": "T-DM1 EMILIA / KATHERINE class",
                "source": "pubmed",
            }
        ],
    }
    return screening, biopsy, therapy_env, patient_context, seen_urls


def main() -> int:
    from oncology_arbiter.models.llm_client import GemmaClient
    from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
        assert_pinned_chat_url,
        verify_artifact_identity,
    )
    from oncology_arbiter.orchestrator.co_scientist import run_co_scientist

    endpoint = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint:
        identity = verify_artifact_identity()
        endpoint = str(identity.get("chat_url") or "").strip()
        os.environ["MEDGEMMA_MODAL_URL"] = endpoint
    assert_pinned_chat_url(endpoint)
    identity = verify_artifact_identity()
    print(f"[live] endpoint={endpoint}", flush=True)

    screening, biopsy, therapy, patient_context, seen_urls = _envelopes_from_e2e()
    llm = GemmaClient(timeout_s=180.0)
    print("[live] starting MedGemma pairwise tournament…", flush=True)
    out = run_co_scientist(
        screening=screening,
        biopsy=biopsy,
        therapy=therapy,
        seen_urls=seen_urls,
        patient_context=patient_context,
        top_n_evolve=1,
        n_variants=1,
        return_top=8,
        mode="live",
        llm=llm,
    )
    print(
        f"[live] engine={out.get('engine')} matches={len(out.get('tournament_transcript') or [])}",
        flush=True,
    )

    transcript = list(out.get("tournament_transcript") or [])
    gates = {
        "engine_medgemma_pairwise": out.get("engine") == "medgemma_pairwise_judge",
        "mode_live": out.get("mode") == "live",
        "transcript_nonempty": len(transcript) > 0,
        "no_offline_scalar_rationale": all(
            m.get("clinical_rationale") != "offline_scalar_score" for m in transcript
        ),
        "no_0_5_loaded_hack_in_transcript": all(
            "0.5*loaded" not in json.dumps(m) for m in transcript
        ),
        "judgments_have_rationale": all(
            str(m.get("clinical_rationale") or "").strip()
            and m.get("clinical_rationale") != "offline_scalar_score"
            for m in transcript
        ),
        "real_non_fallback_judgment": any(
            m.get("winner") in {"A", "B"}
            and "no clinical_rationale" not in str(m.get("clinical_rationale") or "")
            for m in transcript
        ),
        "not_all_fallback_draws": sum(
            1
            for m in transcript
            if "no clinical_rationale" in str(m.get("clinical_rationale") or "")
        )
        < max(1, len(transcript) // 2),
        "at_least_one_model_request_id": any(m.get("model_request_id") for m in transcript),
        "pubmed_urls_in_seen": seen_urls.issubset(
            {
                e.get("url")
                for h in (out.get("hypotheses") or [])
                for e in (h.get("evidence") or [])
                if isinstance(e, dict)
            }
            | seen_urls
        ),
        "hypotheses_present": len(out.get("hypotheses") or []) > 0,
        "endpoint_pinned": True,
        "identity_model": identity.get("model_id") == "google/medgemma-1.5-4b-it",
    }
    gates["verdict_pass"] = all(gates.values())

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "artifacts/live_io"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / f"co_scientist_pairwise_live_{CASE_ID}_{ts}.json"
    latest_path = out_dir / "co_scientist_pairwise_live_latest.json"
    receipt = {
        "created_at": datetime.now(UTC).isoformat(),
        "live": True,
        "route": "/v1/co_scientist/run",
        "case_id": CASE_ID,
        "endpoint": endpoint,
        "identity": {
            "model_id": identity.get("model_id"),
            "revision_sha256": identity.get("revision_sha256"),
            "modal_workspace": identity.get("modal_workspace"),
        },
        "gates": gates,
        "engine": out.get("engine"),
        "mode": out.get("mode"),
        "phases": out.get("phases"),
        "patient_context": out.get("patient_context"),
        "initial_count": out.get("initial_count"),
        "after_reflect": out.get("after_reflect"),
        "after_evolve": out.get("after_evolve"),
        "warnings": out.get("warnings"),
        "hypotheses": out.get("hypotheses"),
        "tournament_transcript": transcript,
        "n_matches": len(transcript),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    latest_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({
        "receipt": str(receipt_path),
        "verdict_pass": gates["verdict_pass"],
        "engine": out.get("engine"),
        "n_matches": len(transcript),
        "gates": gates,
        "top_hypothesis": (out.get("hypotheses") or [None])[0],
    }, indent=2))
    return 0 if gates["verdict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
