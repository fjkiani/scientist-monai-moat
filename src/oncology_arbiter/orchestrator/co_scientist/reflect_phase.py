"""Phase 2 — honesty gate via reflection.py."""
from __future__ import annotations

from typing import Iterable

from ..reflection import LoopResult, reflect_and_filter
from .models import Hypothesis


def reflect_hypotheses(
    hyps: Iterable[Hypothesis],
    seen_urls: set[str],
) -> tuple[list[Hypothesis], list[str]]:
    """Drop evidence URLs not in ``seen_urls``; keep hypotheses with warnings."""
    loop = LoopResult(seen_urls=set(seen_urls))
    kept: list[Hypothesis] = []
    warnings: list[str] = []
    for h in hyps:
        record, stage_warnings = reflect_and_filter(
            h.as_evidence_record(), loop, require_evidence=False,
        )
        h.evidence = list(record.get("evidence") or [])
        kept.append(h)
        for w in stage_warnings:
            warnings.append(f"reflect:{h.hyp_id}:{w}")
        if not h.evidence:
            warnings.append(f"no_evidence_after_reflect:{h.hyp_id}")
    return kept, warnings
