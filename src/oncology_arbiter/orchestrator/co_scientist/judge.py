"""Live MedGemma pairwise match evaluator (LLM-as-a-judge)."""
from __future__ import annotations

import json
from typing import Any

from .const import (
    EVIDENCE_ITEMS_IN_PROMPT,
    EVIDENCE_JSON_CHARS,
    FALLBACK_RATIONALE,
    JUDGE_SYSTEM,
    MARGIN_LABEL_TO_FLOAT,
    MEDGEMMA_JUDGE_MAX_TOKENS,
    MEDGEMMA_JUDGE_TEMPERATURE,
    PATIENT_CONTEXT_JSON_CHARS,
    RATIONALE_MAX_CHARS,
    WINNER_ALIASES,
)
from .json_util import chat_json_with_retry
from .models import Hypothesis


def _normalize_margin(raw: Any, *, winner: str) -> float:
    """Accept float 0..1 or legacy label strings; clamp to [0, 1]."""
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        label = raw.strip().lower()
        if label in MARGIN_LABEL_TO_FLOAT:
            return MARGIN_LABEL_TO_FLOAT[label]
        try:
            return max(0.0, min(1.0, float(label)))
        except ValueError:
            pass
    return 0.0 if winner == "DRAW" else MARGIN_LABEL_TO_FLOAT["moderate"]


def score_match(
    hyp_a: Hypothesis,
    hyp_b: Hypothesis,
    *,
    patient_context: dict[str, Any] | None,
    llm: Any,
) -> dict[str, Any]:
    """Live pairwise clinical evaluation via MedGemma on Modal.

    Returns ``{winner, margin (float 0..1), clinical_rationale, ...}``.
    """
    user = (
        "You are an expert clinical oncologist on a molecular tumor board.\n"
        "PATIENT_CLINICAL_PROFILE_JSON:\n"
        + json.dumps(patient_context or {}, ensure_ascii=False)[:PATIENT_CONTEXT_JSON_CHARS]
        + "\n\nOption A (id="
        + hyp_a.hyp_id
        + "):\n"
        + hyp_a.statement
        + "\nEvidence A: "
        + json.dumps(hyp_a.evidence[:EVIDENCE_ITEMS_IN_PROMPT], ensure_ascii=False)[
            :EVIDENCE_JSON_CHARS
        ]
        + "\n\nOption B (id="
        + hyp_b.hyp_id
        + "):\n"
        + hyp_b.statement
        + "\nEvidence B: "
        + json.dumps(hyp_b.evidence[:EVIDENCE_ITEMS_IN_PROMPT], ensure_ascii=False)[
            :EVIDENCE_JSON_CHARS
        ]
        + "\n\nCompare Option A vs Option B. Return JSON:"
    )
    resp, parsed = chat_json_with_retry(
        llm,
        system=JUDGE_SYSTEM,
        user=user,
        max_tokens=MEDGEMMA_JUDGE_MAX_TOKENS,
        temperature=MEDGEMMA_JUDGE_TEMPERATURE,
    )
    winner_raw = str(parsed.get("winner") or "DRAW").strip().upper()
    winner = WINNER_ALIASES.get(winner_raw, "DRAW")
    rationale = str(
        parsed.get("clinical_rationale") or parsed.get("reason") or ""
    ).strip()
    margin = _normalize_margin(parsed.get("margin"), winner=winner)
    if not rationale:
        rationale = FALLBACK_RATIONALE
        winner = "DRAW"
        margin = 0.0
    return {
        "winner": winner,
        "margin": margin,
        "clinical_rationale": rationale[:RATIONALE_MAX_CHARS],
        "model_request_id": getattr(resp, "request_id", None),
        "model": getattr(resp, "model", None),
        "route": getattr(resp, "route", None),
    }


# Public alias matching Alpha's strike naming.
_score_match = score_match
_judge_match_medgemma = score_match
