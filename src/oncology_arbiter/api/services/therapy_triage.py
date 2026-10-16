"""Explicit therapy triage scoring (identity-locked therapy_arbiter_v1)."""
from __future__ import annotations

from typing import Any

from .. import deps
from ..schemas import ArbiterScore

# Canonical implementation lives in deps (single Manski choke point).
score_explicit_therapy_triage = deps.score_explicit_therapy_triage
TRIAGE_REQUIRED_FIELDS = deps.TRIAGE_REQUIRED_FIELDS


class TherapyTriageService:
    """Class facade for explicit therapy triage scoring."""

    REQUIRED_FIELDS = TRIAGE_REQUIRED_FIELDS

    @staticmethod
    def score(raw: dict[str, Any] | None) -> ArbiterScore | None:
        return score_explicit_therapy_triage(raw)
