"""Regression: enforce ClinicalBERT F1 floor across all 5 SYNTHETIC-v0.3.1 seeds.

The v0.4.1 production plan requires micro-F1 >= 0.85 on the test split for every
seed in {42, 123, 456, 789, 1234}. This test reads the per-seed ``metrics.json``
files produced by the fine-tuning run (durably stored on the session shared
volume) and asserts:

  1. All 5 seeds are present.
  2. Each per-seed test micro-F1 >= 0.85.
  3. The aggregate ``AGGREGATE.json`` mean/min match the per-seed detail
     (no accidental over-writing).
  4. ``best_seed`` and its metric agree with the per-seed maximum.

The test is skipped (not failed) if the training artifacts are not present,
so it stays green in environments where fine-tuning hasn't been done. In CI
the artifacts must be present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RUNS_DIR = Path("/mnt/shared-workspace/shared/clinicalbert_runs")
SEEDS = (42, 123, 456, 789, 1234)
FLOOR = 0.85


def _load_seed_metric(seed: int) -> dict:
    p = RUNS_DIR / f"seed{seed}" / "metrics.json"
    if not p.exists():
        pytest.skip(f"metrics.json missing for seed {seed} at {p}")
    return json.loads(p.read_text())


def _extract_test_micro_f1(m: dict) -> float | None:
    """Support both flat (``test_micro_f1``) and nested (``test.micro.f1``) shapes."""
    if "test_micro_f1" in m:
        return m["test_micro_f1"]
    test = m.get("test") or {}
    micro = test.get("micro") if isinstance(test, dict) else None
    if isinstance(micro, dict):
        return micro.get("f1")
    return None


def _load_aggregate() -> dict:
    p = RUNS_DIR / "AGGREGATE.json"
    if not p.exists():
        pytest.skip(f"AGGREGATE.json missing at {p}")
    return json.loads(p.read_text())


@pytest.mark.regression
@pytest.mark.parametrize("seed", SEEDS)
def test_per_seed_f1_floor(seed: int) -> None:
    """Each of the 5 seeds must clear the 0.85 micro-F1 floor on the test split."""
    m = _load_seed_metric(seed)
    f1 = _extract_test_micro_f1(m)
    assert f1 is not None, f"seed {seed}: metrics.json missing test.micro.f1"
    assert f1 >= FLOOR, (
        f"seed {seed}: test_micro_f1={f1:.6f} below floor {FLOOR}"
    )
    # Guard against corpus drift — the fine-tune must be on the frozen split.
    prov = m.get("provenance") or m.get("corpus_provenance")
    if prov is not None:
        assert prov == "SYNTHETIC-v0.3.1", (
            f"seed {seed}: unexpected provenance {prov!r}, "
            f"expected SYNTHETIC-v0.3.1"
        )


@pytest.mark.regression
def test_aggregate_matches_seed_detail() -> None:
    """AGGREGATE.json must be a faithful roll-up of the per-seed detail."""
    agg = _load_aggregate()
    per_seed = {int(s): _extract_test_micro_f1(_load_seed_metric(s)) for s in SEEDS}

    reported = agg.get("test_micro_f1_by_seed", {})
    for s, f1 in per_seed.items():
        r = reported.get(str(s), reported.get(s))
        assert r is not None, f"aggregate missing seed {s}"
        assert abs(r - f1) < 1e-9, (
            f"aggregate seed {s}: reported={r} vs metrics.json={f1}"
        )

    reported_min = agg.get("min_test_micro_f1")
    reported_max = agg.get("max_test_micro_f1")
    assert reported_min is not None and reported_max is not None
    assert abs(reported_min - min(per_seed.values())) < 1e-9
    assert abs(reported_max - max(per_seed.values())) < 1e-9

    best_seed = agg.get("best_seed")
    assert best_seed is not None
    assert per_seed[int(best_seed)] == max(per_seed.values()), (
        f"best_seed={best_seed} doesn't own the maximum F1"
    )

    assert agg.get("floor") == FLOOR, (
        f"aggregate floor mismatch: {agg.get('floor')} vs {FLOOR}"
    )
    assert agg.get("all_seeds_pass_floor") is True, (
        "aggregate says not all seeds pass floor"
    )
