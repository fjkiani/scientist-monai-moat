"""L5 Co-Scientist — live clinical reasoning swarm (LLM-backend agnostic).

Public surface stays import-compatible:

    from oncology_arbiter.orchestrator.co_scientist import run_co_scientist

Swap backend with ``CO_SCIENTIST_LLM_BACKEND=medgemma|gemini_argon``.
"""
from __future__ import annotations

from .const import (
    ENGINE_LIVE,
    ENGINE_OFFLINE,
    ELO_K_FACTOR,
    LLM_BACKEND_GEMINI_ARGON,
    LLM_BACKEND_MEDGEMMA,
    PHASES,
    ZETA_SYSTEM_INSTRUCTION,
    engine_id_for_backend,
)
from .evolve import evolve_hypotheses
from .generate import generate_hypotheses
from .judge import score_match, _judge_match_medgemma, _score_match
from .llm_backend import (
    build_live_llm,
    is_live_swarm_engine,
    resolve_backend,
    resolve_live_available,
)
from .models import EloEntry, Hypothesis
from .offline_scalar import _score_hypothesis, score_hypothesis
from .rank import _expected, expected_score, rank_hypotheses
from .reflect_phase import reflect_hypotheses
from .runner import resolve_mode, run_co_scientist

__all__ = [
    "ENGINE_LIVE",
    "ENGINE_OFFLINE",
    "ELO_K_FACTOR",
    "LLM_BACKEND_GEMINI_ARGON",
    "LLM_BACKEND_MEDGEMMA",
    "PHASES",
    "ZETA_SYSTEM_INSTRUCTION",
    "EloEntry",
    "Hypothesis",
    "build_live_llm",
    "engine_id_for_backend",
    "evolve_hypotheses",
    "expected_score",
    "generate_hypotheses",
    "is_live_swarm_engine",
    "rank_hypotheses",
    "reflect_hypotheses",
    "resolve_backend",
    "resolve_live_available",
    "resolve_mode",
    "run_co_scientist",
    "score_hypothesis",
    "score_match",
    "_expected",
    "_judge_match_medgemma",
    "_score_hypothesis",
    "_score_match",
]
