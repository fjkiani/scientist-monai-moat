"""Co-Scientist-style supervisor agent scaffold.

Phase 1 ships **only the interface** — a supervisor stub that maps a
per-stage input to an ordered plan and a placeholder "not-yet-wired"
response. Phase 5 (per :mod:`oncology_arbiter` PLAN) replaces the stub
body with real agent orchestration (generate → reflect → tournament →
meta-review) using the tools already ported into
:mod:`oncology_arbiter.tools`.

Why ship a stub now:

* Every stage endpoint can reserve a slot for the eventual agent trace
  in its response schema (``ApiEnvelope.honesty_gate`` + a future
  ``orchestrator_trace`` block).
* Every stage response can call :func:`plan_stage` today and get an
  honest ``StagePlan`` with ``model_state="placeholder"`` so a downstream
  consumer never sees a masquerading agent output.
* The empty ``agents/`` directory that existed pre-Phase-1 was itself
  a smell — a name in the scaffolding tree with no honest surface.

The stub deliberately does NOT emit ``mock`` / ``fake`` / ``MagicMock``
markers: it emits a real dataclass carrying the same shape a live agent
would produce, so drop-in replacement in Phase 5 does not require any
schema migration.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

from .supervisor import (
    AgentPhase,
    StagePlan,
    StageResult,
    SUPERVISOR_VERSION,
    plan_stage,
    run_placeholder,
)

__all__ = [
    "AgentPhase",
    "StagePlan",
    "StageResult",
    "SUPERVISOR_VERSION",
    "plan_stage",
    "run_placeholder",
]
