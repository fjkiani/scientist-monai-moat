"""Tests for the v0.3.0 real Co-Scientist supervisor.

The old Phase-1 stub is now a legacy branch reachable through
``run_placeholder()``; the real work lives in :func:`execute_stage`.
These tests pin down:

  * the module surface (symbols exported)
  * the version string is a live semver, no longer a stub marker
  * the five phases (generate / evidence / reflect / tournament /
    meta_review) — evidence was added in v0.3.0 because retrieval is
    a distinct phase from generation
  * the plan is deterministic and only names real tool modules
  * ``run_placeholder`` still exists, still returns the honest empty
    result, and still labels itself a stub / legacy path

The full integration test for :func:`execute_stage` lives in
``tests/integration/test_supervisor_live.py`` — it is gated on the
``@pytest.mark.live_llm`` marker because it costs real API tokens.
"""
from __future__ import annotations

import importlib

import pytest

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.agents import (
    AgentPhase,
    Hypothesis,
    StagePlan,
    StageResult,
    SUPERVISOR_VERSION,
    execute_stage,
    plan_stage,
    run_placeholder,
)


# ── Structural ─────────────────────────────────────────────────────────


def test_module_exports_expected_symbols() -> None:
    import oncology_arbiter.agents as pkg
    for name in ("AgentPhase", "Hypothesis", "StagePlan", "StageResult",
                 "SUPERVISOR_VERSION",
                 "plan_stage", "run_placeholder", "execute_stage"):
        assert hasattr(pkg, name), f"agents package missing symbol: {name}"


def test_supervisor_version_is_live_semver() -> None:
    """v0.3.0: no longer a stub. Version must be a real semver >= 1.0.0."""
    # Not a stub anymore
    assert "phase1" not in SUPERVISOR_VERSION
    assert "stub" not in SUPERVISOR_VERSION
    # Semver-ish: MAJOR.MINOR.PATCH
    parts = SUPERVISOR_VERSION.split(".")
    assert len(parts) >= 3, f"SUPERVISOR_VERSION should be semver, got {SUPERVISOR_VERSION!r}"
    for p in parts[:3]:
        assert p.isdigit(), f"SUPERVISOR_VERSION part {p!r} not numeric"
    assert int(parts[0]) >= 1, f"SUPERVISOR_VERSION MAJOR must be >= 1, got {SUPERVISOR_VERSION!r}"


def test_agent_phase_enum_matches_co_scientist_shape() -> None:
    """v0.3.0 phases: GENERATE, EVIDENCE, REFLECT, TOURNAMENT, META_REVIEW.

    EVIDENCE was split out from GENERATE in v0.3.0 because retrieval is
    a distinct, honesty-gated phase — hypotheses can name URLs during
    GENERATE but only the URLs actually fetched during EVIDENCE survive
    into REFLECT.
    """
    assert set(p.value for p in AgentPhase) == {
        "generate", "evidence", "reflect", "tournament", "meta_review",
    }


# ── plan_stage() ───────────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy", "case_full"])
def test_plan_stage_returns_full_phase_plan(stage: str) -> None:
    plan = plan_stage(stage)
    assert isinstance(plan, StagePlan)
    assert plan.stage == stage
    # All phases represented
    for ph in AgentPhase:
        assert ph in plan.phase_tools, f"stage {stage} plan missing phase {ph}"
        assert plan.phase_tools[ph], f"stage {stage} phase {ph} has no tools"


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy", "case_full"])
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


# ── run_placeholder() — legacy stub, still shipped ────────────────────


@pytest.mark.parametrize("stage", ["screening", "biopsy", "therapy"])
def test_run_placeholder_returns_honest_stub(stage: str) -> None:
    r = run_placeholder(stage)
    assert isinstance(r, StageResult)
    assert r.stage == stage
    assert r.model_state == "placeholder"
    # Evidence honesty: no real evidence has been gathered.
    assert r.evidence == []
    assert r.seen_urls_count == 0
    assert r.evidence_kept == 0
    assert r.evidence_dropped == 0
    # No fake hypotheses either.
    assert r.hypotheses == []
    # No fake LLM usage.
    assert r.llm_calls == 0
    assert r.llm_total_tokens == 0
    assert r.llm_cost_usd == 0.0


def test_run_placeholder_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        run_placeholder("not_a_stage")


def test_run_placeholder_carries_honesty_constants() -> None:
    r = run_placeholder("therapy")
    assert r.disclaimer == RUO_DISCLAIMER
    assert r.caveat == AUROC_CAVEAT
    # The notes field must clearly label this as legacy/placeholder — a
    # downstream reader must not confuse it with a real supervisor run.
    notes_lower = r.notes.lower()
    assert (
        "placeholder" in notes_lower
        or "legacy" in notes_lower
        or "stub" in notes_lower
        or "phase 1" in notes_lower
    ), f"run_placeholder notes must self-label as legacy/placeholder/stub, got: {r.notes!r}"


# ── execute_stage() — real supervisor, without an LLM (degrades honestly) ──


def test_execute_stage_without_llm_falls_back_honestly() -> None:
    """No LLM passed in and no ambient client → returns placeholder, does not fabricate.

    This is the invariant that separates the real supervisor from a mock:
    when the LLM ladder is unavailable, ``execute_stage`` returns a
    ``StageResult`` with ``model_state in {"placeholder", "llm_unavailable"}``
    and empty hypotheses/evidence, not fabricated content.
    """
    result = execute_stage(
        stage="screening",
        context={"cancer": "breast", "screening_summary": "unit-test, no LLM"},
        llm=None,
        n_hypotheses=6,
        n_evidence_top_k=3,
    )
    assert isinstance(result, StageResult)
    assert result.model_state in {"placeholder", "llm_unavailable"}, (
        f"expected honest degrade state, got {result.model_state!r}"
    )
    assert result.hypotheses == []
    assert result.evidence == []
    assert result.llm_calls == 0
    assert result.llm_cost_usd == 0.0
    assert result.disclaimer == RUO_DISCLAIMER
    assert result.caveat == AUROC_CAVEAT


def test_execute_stage_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        execute_stage(
            stage="not_a_stage",
            context={},
            llm=None,
        )


# ── Hypothesis dataclass shape ────────────────────────────────────────


def test_hypothesis_dataclass_has_expected_fields() -> None:
    """The Hypothesis shape is the contract every downstream consumer
    (API envelope, orchestrator trace, meta-review) reads from.

    Callers supply hypothesis_id (typically uuid.uuid4().hex from
    _parse_hypotheses); the dataclass just carries the field.
    """
    import uuid
    hid = uuid.uuid4().hex
    h = Hypothesis(
        hypothesis_id=hid,
        claim="test claim",
        rationale="test rationale",
        source_scope="literature",
        initial_confidence=0.5,
        evidence_urls=["https://example.org/paper"],
    )
    # Required real fields
    assert h.hypothesis_id == hid
    assert h.claim == "test claim"
    assert h.rationale == "test rationale"
    assert h.source_scope == "literature"
    assert h.initial_confidence == 0.5
    assert h.evidence_urls == ["https://example.org/paper"]
    # Default fields carry the honesty invariants
    assert h.reflection == ""
    assert h.evidence_alignment == "unknown"
    assert h.elo == 1500.0
    assert h.tournament_wins == 0
    assert h.tournament_losses == 0

    # Serialization round-trip through as_dict()
    d = h.as_dict()
    for k in ("hypothesis_id", "claim", "rationale", "source_scope",
              "initial_confidence", "evidence_urls", "reflection",
              "evidence_alignment", "elo", "tournament_wins", "tournament_losses"):
        assert k in d, f"as_dict() missing field {k!r}"
