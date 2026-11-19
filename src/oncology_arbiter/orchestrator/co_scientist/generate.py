"""Phase 1 — generate hypotheses from stage envelopes."""
from __future__ import annotations

from typing import Any

from .const import (
    BIOPSY_ALTERNATE_MAP,
    DEFAULT_THERAPY_CONFIDENCE,
    MAX_SCREENING_FINDINGS,
    MAX_THERAPY_OPTIONS,
)
from .models import Hypothesis


def _honesty_markers(prov: dict[str, Any]) -> dict[str, bool]:
    state = prov.get("model_state") or ""
    return {
        "proxy": state.startswith("proxy_"),
        "gated": state == "gated",
        "loaded": state.startswith("loaded"),
    }


def generate_hypotheses(
    *,
    screening: dict[str, Any] | None,
    biopsy: dict[str, Any] | None,
    therapy: dict[str, Any] | None,
) -> list[Hypothesis]:
    """Enumerate hypotheses from stage envelopes (no invented findings)."""
    hyps: list[Hypothesis] = []

    if screening:
        markers = _honesty_markers(screening.get("provenance") or {})
        for f in (screening.get("findings") or [])[:MAX_SCREENING_FINDINGS]:
            hyps.append(Hypothesis(
                hyp_id=f"screening:{f.get('label', 'unknown')}",
                stage="screening",
                statement=(
                    f"Screening finding '{f.get('label')}' scored "
                    f"{float(f.get('score', 0.0)):.4f}"
                ),
                confidence=float(f.get("score", 0.0)),
                evidence=list(screening.get("evidence") or []),
                honesty_markers=markers,
            ))

    if biopsy:
        markers = _honesty_markers(biopsy.get("provenance") or {})
        subtype = biopsy.get("subtype_prediction")
        conf = biopsy.get("confidence")
        if subtype is not None:
            hyps.append(Hypothesis(
                hyp_id=f"biopsy:{subtype}",
                stage="biopsy",
                statement=(
                    f"Biopsy subtype prediction: {subtype} at "
                    f"{(conf or 0.0):.2f} confidence"
                ),
                confidence=float(conf or 0.0),
                evidence=list(biopsy.get("evidence") or []),
                honesty_markers=markers,
            ))
        if subtype in BIOPSY_ALTERNATE_MAP and conf is not None:
            alt = BIOPSY_ALTERNATE_MAP[subtype]
            hyps.append(Hypothesis(
                hyp_id=f"biopsy:{alt}:alternate",
                stage="biopsy",
                statement=f"Alternate biopsy hypothesis: {alt} instead of {subtype}",
                confidence=max(0.0, 1.0 - float(conf)),
                evidence=list(biopsy.get("evidence") or []),
                honesty_markers=markers,
            ))

    if therapy:
        markers = _honesty_markers(therapy.get("provenance") or {})
        for opt in (therapy.get("recommended_options") or [])[:MAX_THERAPY_OPTIONS]:
            conf = opt.get("confidence")
            hyps.append(Hypothesis(
                hyp_id=(
                    f"therapy:{opt.get('regimen', 'unknown')}:"
                    f"line{opt.get('line_of_therapy', 1)}"
                ),
                stage="therapy",
                statement=(
                    f"Therapy: {opt.get('regimen')} "
                    f"(line {opt.get('line_of_therapy')})"
                ),
                confidence=float(
                    conf if conf is not None else DEFAULT_THERAPY_CONFIDENCE
                ),
                evidence=list(opt.get("evidence") or []),
                honesty_markers=markers,
            ))
    return hyps
