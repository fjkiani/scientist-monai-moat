"""Regression test: v0.5.1 real-text ClinicalBERT F1 on NSCLC + breast/CRC gold.

Purpose
-------
- Records honest real-text micro-F1 achieved by v0.5.1 on:
  * NSCLC gold (200 reports, hy3 self-consistency, kappa mean 0.737)
  * Breast/CRC gold (96 TCGA-242 pathologist-adjudicated reports, reused from v0.5.0)
- Provides the aggregate's expected schema for downstream (Modal /info,
  prod parsed_report_provenance).
- Enforces the rollback rule: if breast/CRC F1 drops >5 relative points vs
  v0.5.0 baseline mean (0.0667), CI blocks the release.

This test asserts:
    (a) aggregate JSON exists, parses, has all 5 seeds
    (b) per-cancer F1 (nsclc, breast_crc, combined) is present
    (c) v0.5.1 uses REAL provenance and class-weighted CE + label smoothing
    (d) v0.5.1 breast/CRC F1 mean does NOT drop >5 rel points vs v0.5.0
    (e) v0.5.1 records the annotator kappa, snorkel LF accuracies, and
        BIO alignment error rate (evidence trail)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

AGG = Path("/mnt/shared-workspace/shared/clinicalbert_runs_v51/AGGREGATE_v51.json")

# v0.5.0 baseline mean over 5 seeds (from AGGREGATE_v05.json).
# Rollback rule: v0.5.1 breast/CRC F1 mean must be within 5 relative points.
V050_BREAST_CRC_F1_MEAN = 0.0667
ROLLBACK_MAX_RELATIVE_DROP_PCT = 5.0

# Recorded expected floor for aggregate F1. v0.5.1 targets a meaningful
# uplift from the class-weighted CE (min=1.0 max=25.0 O_weight=1.0) and
# label smoothing 0.1. If below this floor by >5 rel points on breast/CRC,
# CI blocks release and we investigate.
V051_MIN_SEED_F1_FLOOR = 0.05


def _load_agg() -> dict:
    if not AGG.exists():
        pytest.skip(f"v0.5.1 aggregate file not yet generated: {AGG}")
    return json.loads(AGG.read_text())


def test_v51_aggregate_exists_and_has_all_seeds():
    d = _load_agg()
    assert "per_seed" in d
    per_seed = d["per_seed"]
    seeds_present = set(per_seed.keys())
    expected = {"42", "123", "456", "789", "1234"}
    assert seeds_present == expected, f"Missing seeds: {expected - seeds_present}"


def test_v51_provenance_and_hyperparams():
    d = _load_agg()
    prov = d.get("provenance", "")
    assert prov.startswith("REAL-v0.5.1"), f"provenance mismatch: {prov!r}"


def test_v51_per_cancer_metrics_present():
    """Ensures per-cancer eval writes actually landed in the aggregate."""
    d = _load_agg()
    for seed, m in d["per_seed"].items():
        # Per-cancer keys must exist (they may be None if _test_splits absent,
        # but that would be an upstream corpus bug).
        assert "nsclc_micro_f1" in m, f"seed {seed}: nsclc_micro_f1 missing"
        assert "breast_crc_micro_f1" in m, f"seed {seed}: breast_crc_micro_f1 missing"
        assert m["nsclc_micro_f1"] is not None, f"seed {seed}: nsclc_micro_f1 is None"
        assert m["breast_crc_micro_f1"] is not None, f"seed {seed}: breast_crc_micro_f1 is None"


def test_v51_rollback_rule_not_triggered():
    """v0.5.1 breast/CRC F1 must NOT drop >5 relative points vs v0.5.0."""
    d = _load_agg()
    rb = d.get("rollback")
    assert rb is not None, "aggregate lacks rollback block"
    assert rb["baseline_breast_crc_f1_mean_v050"] == pytest.approx(V050_BREAST_CRC_F1_MEAN, abs=1e-4)
    assert not rb["rollback_triggered"], (
        f"ROLLBACK TRIGGERED: v0.5.1 breast/CRC F1 mean {rb['v051_breast_crc_f1_mean']:.4f} "
        f"vs v0.5.0 baseline {V050_BREAST_CRC_F1_MEAN:.4f} "
        f"(delta {rb['relative_delta_pct']:+.2f}%). Keep v0.5.0 weights."
    )


def test_v51_no_seed_catastrophically_low():
    d = _load_agg()
    for seed, m in d["per_seed"].items():
        f1 = m.get("combined_micro_f1") or m.get("test_micro_f1") or 0.0
        assert f1 >= V051_MIN_SEED_F1_FLOOR, (
            f"seed {seed} combined F1={f1:.4f} < floor {V051_MIN_SEED_F1_FLOOR:.4f}. "
            f"Class-weighted CE may have destabilized."
        )


def test_v51_class_weighted_loss_enabled():
    """All 5 v0.5.1 seeds must be trained with class-weighted CE + label smoothing."""
    d = _load_agg()
    for seed, m in d["per_seed"].items():
        assert m.get("class_weighted_loss") is True, f"seed {seed}: class_weighted_loss not True"
        assert m.get("label_smoothing") == pytest.approx(0.1, abs=1e-4), (
            f"seed {seed}: label_smoothing={m.get('label_smoothing')} != 0.1"
        )


def test_v51_annotator_kappa_recorded():
    """Evidence: NSCLC gold annotator kappa (>=0.5 = reasonable agreement)."""
    d = _load_agg()
    q = d.get("annotator_quality_nsclc") or {}
    kappa = q.get("kappa_mean")
    assert kappa is not None, "annotator_kappa_nsclc missing from aggregate"
    assert kappa >= 0.5, f"NSCLC gold kappa {kappa:.4f} < 0.5 (poor annotator agreement)"


def test_v51_snorkel_lf_accuracies_recorded():
    """Evidence: Snorkel LFs' per-LF accuracies (from LabelModel fit)."""
    d = _load_agg()
    corpus = d.get("corpus") or {}
    snorkel = (corpus.get("snorkel_stats") or {}).get("lf_accuracies") or {}
    # Must have at least LF-regex + LF-LLM + LF-ontology + LF-section
    for lf in ("LF-regex", "LF-LLM", "LF-ontology", "LF-section"):
        assert lf in snorkel, f"missing LF accuracy: {lf}"


def test_v51_bio_alignment_error_rate_recorded():
    """Evidence: how often the BIO writer mis-aligned an entity."""
    d = _load_agg()
    corpus = d.get("corpus") or {}
    er = corpus.get("bio_alignment_error_rate")
    assert er is not None, "bio_alignment_error_rate missing"
    # Sanity: must be <5% (else annotation pipeline is broken)
    assert 0.0 <= er <= 0.05, f"BIO alignment error rate {er:.4f} outside [0, 0.05]"


def test_v51_corpus_sizes():
    d = _load_agg()
    corpus = d.get("corpus") or {}
    # v0.5.1: 2389 train + 217 val + 200 NSCLC gold + 96 breast/CRC gold
    assert corpus.get("n_train_reports") == 2389, f"n_train={corpus.get('n_train_reports')}"
    assert corpus.get("n_nsclc_gold_reports") == 200
    assert corpus.get("n_breast_crc_gold_reports") == 96
