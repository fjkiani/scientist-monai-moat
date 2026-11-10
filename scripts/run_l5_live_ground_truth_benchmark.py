#!/usr/bin/env python3
"""Burn all 20 L5 ground-truth cases through the live clinical reasoning swarm.

Backend-agnostic. Selected via ``CO_SCIENTIST_LLM_BACKEND=medgemma|gemini_argon``.

Writes:
  artifacts/testing1235/l5_live_{backend}_ground_truth_scorecard.json
  artifacts/testing1235/l5_live_{backend}_ground_truth_partial.json

Optional env file (gitignored): ``.env.l5-gemini`` or ``L5_ENV_FILE``.
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


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _bootstrap_env() -> None:
    explicit = (os.environ.get("L5_ENV_FILE") or "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        ROOT / ".env.l5-gemini",
        ROOT / ".env.l5",
        ROOT / "envs" / "l5-gemini-argon.env",
    ])
    for path in candidates:
        _load_env_file(path)


def main() -> int:
    _bootstrap_env()

    from oncology_arbiter.orchestrator.co_scientist import (
        build_live_llm,
        is_live_swarm_engine,
        resolve_backend,
        run_co_scientist,
    )
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
    backend = resolve_backend(os.environ.get("CO_SCIENTIST_LLM_BACKEND"))
    os.environ["CO_SCIENTIST_LLM_BACKEND"] = backend

    identity: dict = {"backend": backend}
    endpoint: str | None = None
    if backend == "medgemma":
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
            assert_pinned_chat_url,
            verify_artifact_identity,
        )

        endpoint = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
        if not endpoint:
            identity.update(verify_artifact_identity())
            endpoint = str(identity.get("chat_url") or "").strip()
            os.environ["MEDGEMMA_MODAL_URL"] = endpoint
        assert_pinned_chat_url(endpoint)
        identity.update(verify_artifact_identity())
    else:
        if not (os.environ.get("GEMINI_API_KEY") or "").strip():
            raise SystemExit("GEMINI_API_KEY required for gemini_argon backend")
        identity.update({
            "model": os.environ.get("GEMINI_ARGON_MODEL") or "gemini-4-argon",
            "agent": os.environ.get("GEMINI_ARGON_AGENT")
            or "antigravity-preview-09-2026",
            "transport": os.environ.get("CO_SCIENTIST_GEMINI_TRANSPORT") or "model",
        })

    llm, engine_id, backend_name = build_live_llm(backend=backend)
    out_path = ROOT / f"artifacts/testing1235/l5_live_{backend_name}_ground_truth_scorecard.json"
    partial_path = ROOT / f"artifacts/testing1235/l5_live_{backend_name}_ground_truth_partial.json"

    print(
        f"[l5-live] backend={backend_name} engine={engine_id} "
        f"endpoint={endpoint or identity.get('model')} "
        f"cases={len(CLINICAL_GROUND_TRUTH_CASES)}",
        flush=True,
    )
    rows: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(CLINICAL_GROUND_TRUTH_CASES, start=1):
        case_t0 = time.time()
        print(
            f"[l5-live] case {i}/{len(CLINICAL_GROUND_TRUTH_CASES)} {case['case_id']}",
            flush=True,
        )
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
            llm_backend=backend_name,
        )
        if not is_live_swarm_engine(out.get("engine")):
            raise SystemExit(f"engine cheat: got {out.get('engine')!r}")
        if out.get("engine") != engine_id:
            raise SystemExit(
                f"engine mismatch: expected {engine_id!r} got {out.get('engine')!r}"
            )
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
            "llm_backend": out.get("llm_backend"),
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
            "llm_backend": backend_name,
            "engine": engine_id,
            "endpoint": endpoint,
            "identity": identity,
            "completed": len(rows),
            "total": len(CLINICAL_GROUND_TRUTH_CASES),
            "rows": rows,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(partial, indent=2))
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
        "llm_backend": backend_name,
        "endpoint": endpoint,
        "identity": identity,
        "total_elapsed_s": round(time.time() - t0, 2),
        "engine_required": engine_id,
        "all_rows_live": all(r.get("mode") == "live" for r in rows),
        "all_rows_swarm_engine": all(
            is_live_swarm_engine(r.get("engine")) and r.get("engine") == engine_id
            for r in rows
        ),
        "no_offline_scalar_in_any_row": True,
    })
    if not card["all_rows_live"] or not card["all_rows_swarm_engine"]:
        raise SystemExit("LIVE CHEAT DETECTED in scorecard rows")

    out_path.write_text(json.dumps(card, indent=2))
    print(json.dumps({k: card[k] for k in card if k != "rows"}, indent=2), flush=True)
    print(f"[l5-live] wrote {out_path}", flush=True)

    meters_ok = (
        card["pass"]["concordance"]
        and card["pass"]["contraindication"]
        and card["pass"]["landmark"]
    )
    # Still write the receipt even if meters miss target — honesty over theater.
    return 0 if meters_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
