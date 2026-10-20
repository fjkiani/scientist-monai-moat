"""L5 Co-Scientist — live MedGemma clinical reasoning swarm.

Public surface stays import-compatible:

    from oncology_arbiter.orchestrator.co_scientist import run_co_scientist
"""
from __future__ import annotations

from .const import ENGINE_LIVE, ENGINE_OFFLINE, ELO_K_FACTOR, PHASES
from .evolve import evolve_hypotheses
from .generate import generate_hypotheses
from .judge import score_match, _judge_match_medgemma, _score_match
from .models import EloEntry, Hypothesis
from .offline_scalar import _score_hypothesis, score_hypothesis
from .rank import _expected, expected_score, rank_hypotheses
from .reflect_phase import reflect_hypotheses
from .runner import resolve_mode, run_co_scientist

__all__ = [
    "ENGINE_LIVE",
    "ENGINE_OFFLINE",
    "ELO_K_FACTOR",
    "PHASES",
    "EloEntry",
    "Hypothesis",
    "evolve_hypotheses",
    "expected_score",
    "generate_hypotheses",
    "rank_hypotheses",
    "reflect_hypotheses",
    "resolve_mode",
    "run_co_scientist",
    "score_hypothesis",
    "score_match",
    "_expected",
    "_judge_match_medgemma",
    "_score_hypothesis",
    "_score_match",
]
