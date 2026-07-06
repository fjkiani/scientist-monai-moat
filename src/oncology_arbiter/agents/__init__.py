"""Co-Scientist-style supervisor agent.

v0.3.0 ships the **real** supervisor loop — a five-phase Co-Scientist
pipeline (generate → evidence → reflect → tournament → meta-review) that
calls a live LLM (Gemma via Google direct, OpenRouter fallbacks) and
retrieves real literature URLs through the same tool inventory the API
already uses.

Backwards compatibility
-----------------------

* :func:`plan_stage` still returns the deterministic ``StagePlan`` shape
  used by the Phase-1 tests and the ``ApiEnvelope.honesty_gate`` block.
* :func:`run_placeholder` still returns a ``StageResult`` with
  ``model_state="placeholder"`` — this is the legacy path when
  :envvar:`ONCOLOGY_ARBITER_USE_LLM_SUPERVISOR` is off. It is honest by
  construction: no fabricated evidence, no fabricated Elo.
* :func:`execute_stage` is the new real path. It requires an LLM client
  (see :class:`oncology_arbiter.models.llm_client.GemmaClient`). If the
  LLM ladder is exhausted it returns ``model_state="llm_unavailable"``
  and NEVER fabricates hypotheses or evidence.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

from .supervisor import (
    AgentPhase,
    Hypothesis,
    StagePlan,
    StageResult,
    SUPERVISOR_VERSION,
    execute_stage,
    plan_stage,
    run_placeholder,
)

__all__ = [
    "AgentPhase",
    "Hypothesis",
    "StagePlan",
    "StageResult",
    "SUPERVISOR_VERSION",
    "execute_stage",
    "plan_stage",
    "run_placeholder",
]
