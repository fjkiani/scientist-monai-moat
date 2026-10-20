#!/usr/bin/env python3
"""APOCALYPTIC: burn all 20 L5 ground-truth cases through live MedGemma swarm.

Writes ``artifacts/testing1235/l5_live_medgemma_ground_truth_scorecard.json``.
Incremental progress is flushed after every case so a mid-run kill is auditable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "artifacts/testing1235/l5_live_medgemma_ground_truth_scorecard.json"
PARTIAL_PATH = ROOT / "artifacts/testing1235/l5_live_medgemma_ground_truth_partial.json"


def main() -> int:
    from oncology_arbiter.models.llm_client import GemmaClient
    from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
        assert_pinned_chat_url,
        verify_artifact_identity,
    )
    from oncology_arbiter.orchestrator.co_scientist import ENGINE_LIVE, run_co_scientist
    from tests.clinical_eval.l5_ground_truth_cases import (
        CLINICAL_GROUND_TRUTH_CASES,
        build_envelopes_for_case,
    )
    from tests.clinical_eval.test_l5_co_scientist_ground_truth import (
        _matches_preferred,
        _mentions_contraindicated,
        _scorecard,
        _top_therapy,
    )

    # Force equal-conf landmark envelopes (no offline preferred boost).
    os.environ["L5_LIVE_EVAL"] = "1"
    endpoint = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint:
        identity = verify_artifact_identity()
        endpoint = str(identity.get("chat_url") or "").strip()
        os.environ["MEDGEMMA_MODAL_URL"] = endpoint
    assert_pinned_chat_url(endpoint)
    identity = verify_artifact_identity()
    llm = GemmaClient(timeout_s=180.0)

    print(
        f"[l5-live] endpoint={endpoint} cases={len(CLINICAL_GROUND_TRUTH_CASES)}",
        flush=True,
    )
    rows: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(CLINICAL_GROUND_TRUTH_CASES, start=1):
        case_t0 = time.time()
        print(f"[l5-live] case {i}/{len(CLINICAL_GROUND_TRUTH_CASES)} {case['case_id']}", flush=True)
        screening, biopsy, therapy, seen = build_envelopes_for_case(case)
        out = run_co_scientist(
            screening=screening,
            biopsy=biopsy,
            therapy=therapy,
            seen_urls=seen,
            patient_context=case["patient_context"],
            top_n_evolve=1,
            n_variants=1,
            return_top=8,
            mode="live",
            llm=llm,
        )
        if out.get("engine") != ENGINE_LIVE:
            raise SystemExit(f"engine cheat: got {out.get('engine')!r}")
        transcript = list(out.get("tournament_transcript") or [])
        if any(m.get("clinical_rationale") == "offline_scalar_score" for m in transcript):
            raise SystemExit(f"scalar leak in {case['case_id']}")
        top = _top_therapy(list(out.get("hypotheses") or []))
        statement = (top or {}).get("statement") or ""
        concordant = _matches_preferred(statement, case)
        contra_hit = _mentions_contraindicated(statement, case)
        landmark_ok = concordant if case["kind"] == "landmark" else None
        row = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "mode": out.get("mode"),
            "engine": out.get("engine"),
            "top_statement": statement[:240],
            "concordant": concordant,
            "contraindication_violation": contra_hit,
            "landmark_ok": landmark_ok,
            "honesty_gate": out.get("honesty_gate"),
            "n_matches": len(transcript),
            "elapsed_s": round(time.time() - case_t0, 2),
            "sample_match": transcript[0] if transcript else None,
        }
        rows.append(row)
        partial = {
            "created_at": datetime.now(UTC).isoformat(),
            "live": True,
            "endpoint": endpoint,
            "identity": {
                "model_id": identity.get("model_id"),
                "revision_sha256": identity.get("revision_sha256"),
            },
            "completed": len(rows),
            "total": len(CLINICAL_GROUND_TRUTH_CASES),
            "rows": rows,
        }
        PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PARTIAL_PATH.write_text(json.dumps(partial, indent=2))
        print(
            f"[l5-live]   → concordant={concordant} contra={contra_hit} "
            f"matches={len(transcript)} engine={out.get('engine')} "
            f"elapsed={row['elapsed_s']}s",
            flush=True,
        )

    card = _scorecard(rows)
    card.update({
        "created_at": datetime.now(UTC).isoformat(),
        "live": True,
        "cheat_banned": True,
        "endpoint": endpoint,
        "identity": {
            "model_id": identity.get("model_id"),
            "revision_sha256": identity.get("revision_sha256"),
            "modal_workspace": identity.get("modal_workspace"),
        },
        "total_elapsed_s": round(time.time() - t0, 2),
        "engine_required": ENGINE_LIVE,
        "all_rows_live": all(r.get("mode") == "live" for r in rows),
        "all_rows_swarm_engine": all(r.get("engine") == ENGINE_LIVE for r in rows),
        "no_offline_scalar_in_any_row": True,
    })
    # Fail closed if any row cheated offline.
    if not card["all_rows_live"] or not card["all_rows_swarm_engine"]:
        raise SystemExit("LIVE CHEAT DETECTED in scorecard rows")

    OUT_PATH.write_text(json.dumps(card, indent=2))
    print(json.dumps({k: card[k] for k in card if k != "rows"}, indent=2), flush=True)
    print(f"[l5-live] wrote {OUT_PATH}", flush=True)

    meters_ok = (
        card["pass"]["concordance"]
        and card["pass"]["contraindication"]
        and card["pass"]["landmark"]
    )
    # Still write the receipt even if meters miss target — honesty over theater.
    return 0 if meters_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
