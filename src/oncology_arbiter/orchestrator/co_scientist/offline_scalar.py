"""Offline-only scalar scoring — quarantined for unit tests / offline_ranker."""
from __future__ import annotations

from .const import (
    OFFLINE_EVIDENCE_CAP,
    OFFLINE_GATED_PENALTY,
    OFFLINE_LOADED_BONUS,
    OFFLINE_PROXY_PENALTY,
)
from .models import Hypothesis


def score_hypothesis(h: Hypothesis) -> float:
    """OFFLINE-ONLY scalar. Forbidden on the live clinical path."""
    ev = min(1.0, len(h.evidence) / OFFLINE_EVIDENCE_CAP)
    marker = 0.0
    if h.honesty_markers.get("loaded"):
        marker += OFFLINE_LOADED_BONUS
    if h.honesty_markers.get("proxy"):
        marker -= OFFLINE_PROXY_PENALTY
    if h.honesty_markers.get("gated"):
        marker -= OFFLINE_GATED_PENALTY
    return h.confidence + ev + marker


# Back-compat alias for unit tests.
_score_hypothesis = score_hypothesis
