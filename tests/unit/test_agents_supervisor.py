"""Tests for the Phase 1 supervisor stub.

The scaffold shipped in ``src/oncology_arbiter/agents/`` is deliberately a
stub — Phase 5 replaces the body with a real Co-Scientist loop. These
tests pin down the *stub contract* so that swap can happen without any
test churn on the honesty invariants.
"""
from __future__ import annotations

import importlib

import pytest

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.agents import (
    AgentPhase,
    StagePlan,
    StageResult,
    SUPERVISOR_VERSION,
    plan_stage,
    run_placeholder,
)


# ── Structural ─────────────────────────────────────────────────────────


def test_module_exports_expected_symbols() -> None:
    import oncology_arbiter.agents as pkg
    for name in ("AgentPhase", "StagePlan", "StageResult", "SUPERVISOR_VERSION",
                 "plan_stage", "run_placeholder"):
        assert hasattr(pkg, name), f"agents package missing symbol: {name}"


def test_supervisor_version_is_phase1_stub_marker() -> None:
    """Version string must clearly say it's a stub, not a live release."""
    assert "phase1" in SUPERVISOR_VERSION or "stub" in SUPERVISOR_VERSION


def test_agent_phase_enum_matches_co_scientist_shape() -> None:
    """The four Co-Scientist phases (generate / reflect / tournament / meta-review)."""
    assert set(p.value for p in AgentPhase) == {
        "generate", "reflect", "tournament", "meta_review",
    }


# ── plan_stage() ───────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy"])
def test_plan_stage_returns_full_four_phase_plan(stage: str) -> None:
    plan = plan_stage(stage)
    assert isinstance(plan, StagePlan)
    assert plan.stage == stage
    # All four phases represented
    for ph in AgentPhase:
        assert ph in plan.phase_tools, f"stage {stage} plan missing phase {ph}"
        assert plan.phase_tools[ph], f"stage {stage} phase {ph} has no tools"


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy"])
def test_plan_stage_only_names_real_tool_modules(stage: str) -> None:
    """Every tool module named in the plan must actually import.

    This is the invariant that separates the stub from a mock: the plan
    reflects the real tool inventory, not a fabricated one.
    """
    plan = plan_stage(stage)
    for phase, tools in plan.phase_tools.items():
        for module_name in tools:
            spec = importlib.util.find_spec(module_name)
            assert spec is not None, (
                f"plan for {stage}/{phase.value} names non-existent tool: {module_name}"
            )


def test_plan_stage_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        plan_stage("not_a_stage")


def test_plan_stage_carries_honesty_constants() -> None:
    plan = plan_stage("screening")
    assert plan.disclaimer == RUO_DISCLAIMER
    assert plan.caveat == AUROC_CAVEAT


def test_plan_stage_is_deterministic() -> None:
    """Same input in → same plan out, so tests and audits can pin against it."""
    a = plan_stage("biopsy")
    b = plan_stage("biopsy")
    assert a.stage == b.stage
    assert a.phase_tools == b.phase_tools


# ── run_placeholder() ──────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy"])
def test_run_placeholder_returns_honest_stub(stage: str) -> None:
    r = run_placeholder(stage)
    assert isinstance(r, StageResult)
    assert r.stage == stage
    assert r.model_state == "placeholder"
    # Evidence honesty: no real evidence has been gathered yet.
    assert r.evidence == []
    assert r.seen_urls_count == 0
    assert r.evidence_kept == 0
    assert r.evidence_dropped == 0


def test_run_placeholder_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        run_placeholder("not_a_stage")


def test_run_placeholder_carries_honesty_constants() -> None:
    r = run_placeholder("therapy")
    assert r.disclaimer == RUO_DISCLAIMER
    assert r.caveat == AUROC_CAVEAT
    # And the notes field must say this is a stub.
    assert "stub" in r.notes.lower() or "phase 1" in r.notes.lower()
