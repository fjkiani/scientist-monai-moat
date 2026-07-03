"""Supervisor stub — Phase 5 will replace this with a live Co-Scientist loop.

Contract:

* :func:`plan_stage` returns an ordered :class:`StagePlan` for the given
  stage (screening / biopsy / therapy). The plan is *deterministic* —
  same input in, same plan out — so tests can pin against it.
* :func:`run_placeholder` returns an honest :class:`StageResult` with
  ``model_state="placeholder"`` and empty evidence, so no downstream
  consumer can mistake a stub for a real agent run.

The tools referenced in the plan are the ones actually shipped in
:mod:`oncology_arbiter.tools` — we cross-check at import time that they
exist, so the plan can never claim a non-existent tool. That check is
what makes this "not a mock": the plan reflects the real tool inventory,
even though execution isn't wired yet.

Phase 5 will drop in ``execute_stage(...)`` alongside these functions
without changing the return types.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER

# When Phase 5 lands, bump this and add a migration note in errata.md.
SUPERVISOR_VERSION = "0.0.1-phase1-stub"

# Tools that Phase 5 supervisor MUST have access to. We resolve at import
# time so a Phase 1 developer removing a tool trips a real ImportError
# instead of a dangling name in the plan.
_REQUIRED_TOOL_MODULES = (
    "oncology_arbiter.tools.pubmed_search",
    "oncology_arbiter.tools.arxiv_search",
    "oncology_arbiter.tools.europe_pmc_search",
    "oncology_arbiter.tools.web_fetch",
    "oncology_arbiter.tools.honesty",
)


class AgentPhase(str, Enum):
    """The four Co-Scientist phases, mirrored from
    ``fjkiani/Co-Scientist`` — kept as an enum so downstream consumers
    can pattern-match on the trace even before the loop is live.
    """
    GENERATE   = "generate"
    REFLECT    = "reflect"
    TOURNAMENT = "tournament"
    META_REVIEW = "meta_review"


@dataclass
class StagePlan:
    """Deterministic per-stage execution plan.

    ``phase_tools`` maps every :class:`AgentPhase` to the ordered list of
    tool module names Phase 5 will invoke in that phase. Tools not
    resolved on import raise ``ImportError`` at :func:`plan_stage` time,
    so the plan is guaranteed valid.
    """
    stage: str
    phase_tools: Dict[AgentPhase, List[str]]
    seen_urls_policy: str = "reflect_and_prune"
    disclaimer: str = RUO_DISCLAIMER
    caveat: str = AUROC_CAVEAT


@dataclass
class StageResult:
    """Honest placeholder output — no evidence collected yet."""
    stage: str
    model_state: str  # always "placeholder" in Phase 1
    evidence: List[Mapping[str, Any]] = field(default_factory=list)
    seen_urls_count: int = 0
    evidence_kept: int = 0
    evidence_dropped: int = 0
    notes: str = (
        "Phase 1 supervisor stub — Co-Scientist loop lands in Phase 5. "
        "No evidence gathered."
    )
    disclaimer: str = RUO_DISCLAIMER
    caveat: str = AUROC_CAVEAT


# ── module-scope import-time tool check ─────────────────────────────
_MISSING_TOOL_MODULES: tuple[str, ...] = tuple(
    m for m in _REQUIRED_TOOL_MODULES
    if importlib.util.find_spec(m) is None
)


_STAGE_PLANS: Dict[str, Dict[AgentPhase, List[str]]] = {
    "screening": {
        AgentPhase.GENERATE: [
            "oncology_arbiter.tools.pubmed_search",
            "oncology_arbiter.tools.europe_pmc_search",
        ],
        AgentPhase.REFLECT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.TOURNAMENT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.META_REVIEW: [
            "oncology_arbiter.tools.web_fetch",
        ],
    },
    "biopsy": {
        AgentPhase.GENERATE: [
            "oncology_arbiter.tools.pubmed_search",
            "oncology_arbiter.tools.arxiv_search",
        ],
        AgentPhase.REFLECT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.TOURNAMENT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.META_REVIEW: [
            "oncology_arbiter.tools.web_fetch",
        ],
    },
    "therapy": {
        AgentPhase.GENERATE: [
            "oncology_arbiter.tools.pubmed_search",
            "oncology_arbiter.tools.arxiv_search",
            "oncology_arbiter.tools.europe_pmc_search",
        ],
        AgentPhase.REFLECT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.TOURNAMENT: [
            "oncology_arbiter.tools.honesty",
        ],
        AgentPhase.META_REVIEW: [
            "oncology_arbiter.tools.web_fetch",
        ],
    },
}


def plan_stage(stage: str) -> StagePlan:
    """Return the deterministic Phase-1 plan for the given stage.

    Raises :class:`ValueError` on unknown stage. Raises :class:`ImportError`
    if a required tool module is missing (this is a build-time invariant —
    every named tool must resolve to a real module).
    """
    if stage not in _STAGE_PLANS:
        raise ValueError(f"Unknown stage {stage!r}; allowed = {sorted(_STAGE_PLANS)}")
    if _MISSING_TOOL_MODULES:
        raise ImportError(
            "Supervisor stub cannot build a plan — required tool modules "
            f"missing at import: {_MISSING_TOOL_MODULES}"
        )
    plan = _STAGE_PLANS[stage]
    return StagePlan(stage=stage, phase_tools={ph: list(tools) for ph, tools in plan.items()})


def run_placeholder(stage: str) -> StageResult:
    """Return an honest placeholder :class:`StageResult` for the given stage.

    The result is guaranteed to have ``model_state="placeholder"`` and
    ``evidence=[]`` — this is the wire-level signal that Phase 5 is not
    yet running and nothing should be interpreted as a real agent trace.
    """
    if stage not in _STAGE_PLANS:
        raise ValueError(f"Unknown stage {stage!r}; allowed = {sorted(_STAGE_PLANS)}")
    return StageResult(stage=stage, model_state="placeholder")
