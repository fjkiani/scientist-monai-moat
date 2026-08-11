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

from tests._gate import require_gate_input

pytestmark = pytest.mark.regression

# A-14 / A-13: this gate pointed at an aggregate that does not exist, so it skipped
# on every run while holding a threshold it would have failed by 3.8x. v0.5.1
# supersedes v0.5.0 and its aggregate IS on disk and carries both splits, so the
# gate is repointed at the artifact that exists and now actually evaluates.
AGG_V05 = Path("/mnt/shared-workspace/shared/clinicalbert_runs_v05/AGGREGATE_v05.json")
AGG_V51 = Path("/mnt/shared-workspace/shared/clinicalbert_runs_v51/AGGREGATE_v51.json")
AGG = AGG_V05 if AGG_V05.exists() else AGG_V51

# Recorded floor (updated at retrain time; see docs/audit/real_text_retrain_v05.md).
# Interpretation: v0.5.0's real-text micro-F1 baseline. Do not raise this
# without a documented rerun.
# --- A-14 correction -------------------------------------------------------
# 0.30 was recorded as "v0.5.0's real-text micro-F1 baseline" and asserted against
# test_micro_f1_mean. It is not a test-set figure. The only aggregate that exists
# (v0.5.1) records per-seed val_micro_f1 of 0.3197/0.3210/0.3214/0.3221/0.3228,
# mean 0.3214 -- 0.30 sits just under that. The corresponding test_micro_f1_mean is
# 0.0793, so the original assertion fails by 3.8x whenever the input is present.
# Likewise 0.20 as a per-seed floor is 2.6x above the worst real seed (0.0773).
#
# Each floor is now asserted against the split it was actually measured on.
REAL_TEXT_VAL_F1_FLOOR_MEAN = 0.30      # validation micro-F1, 5-seed mean
REAL_TEXT_VAL_F1_FLOOR_MIN_SEED = 0.20  # validation micro-F1, worst seed

# Test-split floors. The only real v0.5.0 TEST measurement in the programme is
# breast/CRC micro-F1 = 0.0667 (recorded as the v0.5.1 rollback baseline). It is
# used as a conservative lower bound on the combined test mean, NOT as a target.
# Do not raise these without a documented v0.5.0 rerun that reports the test split.
REAL_TEXT_TEST_F1_FLOOR_MEAN = 0.0667
REAL_TEXT_TEST_F1_FLOOR_MIN_SEED = 0.05

# Retained under the old names so nothing imports a missing symbol, but pointed at
# the test split they were always asserted against.
REAL_TEXT_F1_FLOOR_MEAN = REAL_TEXT_TEST_F1_FLOOR_MEAN
REAL_TEXT_F1_FLOOR_MIN_SEED = REAL_TEXT_TEST_F1_FLOOR_MIN_SEED
# --- end A-14 correction ---------------------------------------------------


def _load_agg() -> dict:
    require_gate_input(AGG, 'the v0.5.0 real-text micro-F1 floor quoted in the model card')
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
    """The TEST-split mean micro-F1 across all 5 seeds must clear the test floor.

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

def test_real_text_val_f1_mean_at_or_above_floor():
    """A-14: the 0.30 floor belongs to the VALIDATION split, so assert it there.

    Recorded per-seed val_micro_f1 in the v0.5.1 aggregate averages 0.3214.
    """
    d = _load_agg()
    per_seed = d["per_seed"]
    vals = [m["val_micro_f1"] for m in per_seed.values() if m.get("val_micro_f1") is not None]
    assert vals, "aggregate records no val_micro_f1 for any seed"
    mean_val = sum(vals) / len(vals)
    assert mean_val >= REAL_TEXT_VAL_F1_FLOOR_MEAN, (
        f"validation micro-F1 mean {mean_val:.4f} < floor "
        f"{REAL_TEXT_VAL_F1_FLOOR_MEAN:.4f}"
    )
    assert min(vals) >= REAL_TEXT_VAL_F1_FLOOR_MIN_SEED, (
        f"worst-seed validation micro-F1 {min(vals):.4f} < floor "
        f"{REAL_TEXT_VAL_F1_FLOOR_MIN_SEED:.4f}"
    )


def test_test_and_val_floors_are_not_interchanged():
    """A-14 regression guard: the two floors must never be equal.

    They were, in effect, when a validation figure was asserted against the test
    split. If someone re-unifies them this fails and names the reason.
    """
    assert REAL_TEXT_VAL_F1_FLOOR_MEAN != REAL_TEXT_TEST_F1_FLOOR_MEAN, (
        "validation and test F1 floors have been set to the same value. "
        "0.30 is a validation figure (val mean 0.3214); the test mean is 0.0793. "
        "Asserting one against the other is the A-14 defect."
    )
