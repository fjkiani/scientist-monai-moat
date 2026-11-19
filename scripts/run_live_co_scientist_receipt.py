"""Run the real MedGemma Co-Scientist loop and persist an auditable JSON receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oncology_arbiter.agents.supervisor import execute_stage
from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
    assert_pinned_chat_url,
    verify_artifact_identity,
)

DEFAULT_CONTEXT: dict[str, Any] = {
    "patient": "De-identified woman in her 50s",
    "pathology": (
        "Invasive ductal carcinoma, grade 3, 22 mm; ER 95% positive, "
        "PR negative, HER2 IHC 1+ (negative), Ki-67 35%."
    ),
    "nodes": "Sentinel node with 3 mm macrometastasis in 1 of 3 nodes.",
    "clinical_question": (
        "Rank evidence-grounded hypotheses relevant to adjuvant systemic "
        "therapy planning and identify what remains unproven."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Receipt path; defaults to artifacts/medgemma_co_scientist/<timestamp>.json.",
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "biopsy", "therapy", "case_full"),
        default="biopsy",
    )
    parser.add_argument("--n-hypotheses", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--n-evidence-top-k", type=int, choices=range(1, 7), default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    endpoint = (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip()
    if not endpoint:
        raise SystemExit("MEDGEMMA_MODAL_URL is required")

    identity = verify_artifact_identity()
    assert_pinned_chat_url(endpoint)
    result = execute_stage(
        args.stage,
        DEFAULT_CONTEXT,
        n_hypotheses=args.n_hypotheses,
        n_evidence_top_k=args.n_evidence_top_k,
    )
    output = result.as_dict()

    expected = int(output.get("tournament_pairs_expected") or 0)
    completed = int(output.get("tournament_pairs_completed") or 0)
    matches = list(output.get("tournament_matches") or [])
    reflections_expected = int(output.get("reflection_calls_expected") or 0)
    reflections_completed = int(output.get("reflection_calls_completed") or 0)
    reflection_receipts = list(output.get("reflection_receipts") or [])
    gates = {
        "executed": output.get("model_state") == "executed",
        "medgemma_provider": output.get("llm_provider") == "medgemma_modal",
        "model_identity": output.get("llm_model") == identity["model_id"],
        "app_identity": output.get("llm_app_version") == identity["app_version"],
        "revision_identity": output.get("llm_revision") == identity["revision_sha256"],
        "request_receipts": bool(output.get("llm_request_ids")),
        "evidence_retained": bool(output.get("evidence")),
        "reflections_complete": (
            reflections_expected > 0
            and reflections_completed == reflections_expected
            and len(reflection_receipts) == reflections_expected
            and all(
                item.get("completed") and item.get("model_request_id")
                for item in reflection_receipts
            )
        ),
        "pair_count_complete": expected > 0 and completed == expected,
        "pair_receipts_complete": (
            len(matches) == expected
            and all(item.get("completed") and item.get("model_request_id") for item in matches)
        ),
        "meta_review_present": bool(str(output.get("meta_review") or "").strip()),
        "meta_review_receipt": bool(
            output.get("meta_review_request_id")
            and output.get("meta_review_response_sha256")
        ),
        "endpoint_pinned": True,
        "all_hypotheses_reflected": (
            len(output.get("hypotheses") or []) == reflections_expected
            and reflections_expected == reflections_completed
            and reflections_expected > 0
        ),
        "pair_reason_present": all(
            str(item.get("reason") or "").strip()
            and str(item.get("margin") or "") in {"clear", "moderate", "marginal", "tie"}
            for item in matches
        ),
    }
    gates["verdict_pass"] = all(
        passed for name, passed in gates.items() if name != "verdict_pass"
    )

    meta_review = str(output.get("meta_review") or "")
    context_canonical = json.dumps(DEFAULT_CONTEXT, sort_keys=True, separators=(",", ":"))
    receipt = {
        "created_at": datetime.now(UTC).isoformat(),
        "live": True,
        "stage": args.stage,
        "endpoint": endpoint,
        "clinical_context_sha256": hashlib.sha256(context_canonical.encode()).hexdigest(),
        "identity": {
            "model_id": output.get("llm_model"),
            "app_version": output.get("llm_app_version"),
            "revision_sha256": output.get("llm_revision"),
        },
        "verdict": "PASS" if gates.get("verdict_pass") else "FAIL",
        "gates": gates,
        "model_state": output.get("model_state"),
        "model_request_ids": output.get("llm_request_ids"),
        "fetched_urls": [
            item.get("url") for item in output.get("evidence") or [] if item.get("url")
        ],
        "evidence": output.get("evidence"),
        "reflection_calls_expected": reflections_expected,
        "reflection_calls_completed": reflections_completed,
        "reflection_receipts": reflection_receipts,
        "tournament_pairs_expected": expected,
        "tournament_pairs_completed": completed,
        "tournament_matches": matches,
        "hypotheses": output.get("hypotheses"),
        "meta_review": meta_review,
        "meta_review_sha256": hashlib.sha256(meta_review.encode()).hexdigest(),
        "meta_review_request_id": output.get("meta_review_request_id"),
        "meta_review_response_sha256": output.get("meta_review_response_sha256"),
        "usage": {
            "calls": output.get("llm_calls"),
            "prompt_tokens": output.get("llm_prompt_tokens"),
            "completion_tokens": output.get("llm_completion_tokens"),
            "cost_usd": output.get("llm_cost_usd"),
            "llm_latency_s": output.get("llm_latency_s"),
            "wall_latency_s": output.get("latency_s"),
        },
        "notes": output.get("notes"),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or (
        Path("artifacts") / "medgemma_co_scientist" / f"live_{args.stage}_{timestamp}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(destination)

    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise SystemExit(f"live Co-Scientist gates failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
