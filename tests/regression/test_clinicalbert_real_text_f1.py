"""Regression test: real-text ClinicalBERT F1 on TCGA-242 pathologist-adjudicated gold.

Purpose
-------
- Records the honest real-text micro-F1 achieved by v0.5.0 on TCGA-242 gold
  (breast + colorectal), replacing the synthetic 0.97 as the on-wire number.
- Also verifies the aggregate JSON contains the expected keys so downstream
  (Modal deploy manifest, prod parsed_report_provenance) has a stable schema.

This test is a RECORD, not a gate. The measured F1 goes on wire regardless.
The test asserts:
    (a) the aggregate file exists and parses
    (b) it contains per-seed test_micro_f1 for all 5 seeds
    (c) the aggregate's mean F1 >= the recorded floor (documented in the
        test itself, updated each time we intentionally retrain)

If v0.5.1 drops below this floor by > 5 relative points, CI screams. If it
improves, bump the floor in this file after human review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

AGG = Path("/mnt/shared-workspace/shared/clinicalbert_runs_v05/AGGREGATE_v05.json")

# Recorded floor (updated at retrain time; see docs/audit/real_text_retrain_v05.md).
# Interpretation: v0.5.0's real-text micro-F1 baseline. Do not raise this
# without a documented rerun.
REAL_TEXT_F1_FLOOR_MEAN = 0.30
REAL_TEXT_F1_FLOOR_MIN_SEED = 0.20


def _load_agg() -> dict:
    if not AGG.exists():
        pytest.skip(f"Aggregate file not yet generated: {AGG}")
    return json.loads(AGG.read_text())


def test_aggregate_exists_and_has_all_seeds():
    d = _load_agg()
    assert "per_seed" in d
    per_seed = d["per_seed"]
    seeds_present = set(per_seed.keys())
    expected = {"42", "123", "456", "789", "1234"}
    assert seeds_present == expected, f"Missing seeds: {expected - seeds_present}"


def test_all_seeds_have_test_micro_f1():
    d = _load_agg()
    for seed, m in d["per_seed"].items():
        assert "test_micro_f1" in m, f"seed {seed} missing test_micro_f1: keys={list(m.keys())}"
        assert isinstance(m["test_micro_f1"], (int, float)), f"seed {seed}: bad type"


def test_real_text_f1_mean_at_or_above_floor():
    """The mean micro-F1 across all 5 seeds must be >= REAL_TEXT_F1_FLOOR_MEAN.

    This is the honest real-text baseline. If it drops, we investigate.
    """
    d = _load_agg()
    mean = d.get("test_micro_f1_mean")
    assert mean is not None, "aggregate lacks test_micro_f1_mean"
    assert mean >= REAL_TEXT_F1_FLOOR_MEAN, (
        f"real-text micro-F1 mean {mean:.4f} < floor {REAL_TEXT_F1_FLOOR_MEAN:.4f}. "
        f"Investigate corpus, seeds, or hyperparams."
    )


def test_no_seed_catastrophically_below_floor():
    d = _load_agg()
    for seed, m in d["per_seed"].items():
        f1 = m["test_micro_f1"]
        assert f1 >= REAL_TEXT_F1_FLOOR_MIN_SEED, (
            f"seed {seed} F1={f1:.4f} < min-seed floor {REAL_TEXT_F1_FLOOR_MIN_SEED:.4f}"
        )


def test_aggregate_records_provenance():
    """Provenance stamp must be REAL-v0.5.0 so on-wire parsed_report_provenance
    can be trusted."""
    d = _load_agg()
    prov = d.get("provenance", "")
    assert prov.startswith("REAL-v0.5.0"), f"provenance mismatch: {prov!r}"
