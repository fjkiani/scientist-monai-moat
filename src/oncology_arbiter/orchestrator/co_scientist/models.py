"""Data models for the Co-Scientist loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .const import DEFAULT_THERAPY_CONFIDENCE, ELO_INITIAL_RATING


@dataclass
class Hypothesis:
    """One tumor-board hypothesis considered by the loop."""

    hyp_id: str
    stage: str
    statement: str
    confidence: float = DEFAULT_THERAPY_CONFIDENCE
    evidence: list[dict[str, Any]] = field(default_factory=list)
    honesty_markers: dict[str, bool] = field(default_factory=dict)
    derived_from: str | None = None

    def as_evidence_record(self) -> dict[str, Any]:
        return {
            "hyp_id": self.hyp_id,
            "statement": self.statement,
            "evidence": list(self.evidence),
        }


@dataclass
class EloEntry:
    hypothesis: Hypothesis
    rating: float = ELO_INITIAL_RATING
    wins: int = 0
    losses: int = 0
    draws: int = 0
