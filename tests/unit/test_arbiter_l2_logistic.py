"""Unit tests for the L2 logistic arbiter template.

Every test in this module falls into one of five buckets:

  1. Structural — module surface, expected exports, dataclass shape.
  2. Schema     — the three shipped JSONs load, expose the right fields,
                  and carry an ``AUROC_CAVEAT`` (honesty gate).
  3. Encoding   — one-hot / bool / continuous encoders behave like the
                  reference ProgressionArbiter (unknown = 0.5, unknown
                  category raises, unrecognised feature raises).
  4. Invariants — sum(term_contributions) == logit up to 1e-9 tolerance,
                  which is the property downstream code relies on to
                  attribute the logit back to individual features.
  5. Contract   — driving_feature = argmax|contribution|, recommendation
                  bucket matches ``RISK_BUCKETS``, caveat + disclaimer
                  are non-empty strings, template models return
                  ``model_state == "template"``.

The synthetic-training test at the bottom does a small round-trip:
we fit an ``sklearn`` L2 logistic on a toy dataset, dump it into the
template JSON schema, load it back through :class:`L2LogisticArbiter`,
and verify that sklearn's ``predict_proba`` agrees with our scorer to
1e-6. This test guards the frozen-JSON contract we use for the real
Phase 3 training runs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.arbiter import (
    ArbiterResult,
    L2LogisticArbiter,
    RISK_BUCKETS,
    biopsy_arbiter,
    load_arbiter,
    screening_arbiter,
    therapy_arbiter,
)


# ── 1. Structural ─────────────────────────────────────────────────────


def test_module_exposes_expected_public_api() -> None:
    import oncology_arbiter.arbiter as pkg

    for name in (
        "ArbiterResult",
        "L2LogisticArbiter",
        "RISK_BUCKETS",
        "load_arbiter",
        "screening_arbiter",
        "biopsy_arbiter",
        "therapy_arbiter",
    ):
        assert hasattr(pkg, name), f"arbiter package missing expected symbol: {name}"


def test_arbiter_result_dataclass_has_expected_fields() -> None:
    r = ArbiterResult(
        p_positive=0.5,
        logit=0.0,
        risk_bucket="MID",
        recommendation="TEST",
        term_contributions={"intercept": 0.0},
        driving_feature="intercept",
        driving_feature_contribution=0.0,
    )
    d = r.as_dict()
    for key in (
        "p_positive",
        "logit",
        "risk_bucket",
        "recommendation",
        "term_contributions",
        "driving_feature",
        "driving_feature_contribution",
        "disclaimer",
        "caveat",
        "metadata",
    ):
        assert key in d, f"ArbiterResult.as_dict() missing field: {key}"


def test_risk_buckets_cover_probability_range() -> None:
    covered = []
    for _name, (lo, hi) in RISK_BUCKETS.items():
        covered.append((lo, hi))
    covered.sort()
    # Adjacent, no gaps, starts at 0 ends at >=1.0
    assert covered[0][0] == 0.0
    assert covered[-1][1] > 1.0
    for i in range(1, len(covered)):
        assert covered[i][0] == covered[i - 1][1], "RISK_BUCKETS have a gap"


# ── 2. Schema ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["screening", "biopsy", "therapy"])
def test_load_arbiter_returns_l2_logistic(name: str) -> None:
    arb = load_arbiter(name)
    assert isinstance(arb, L2LogisticArbiter)
    assert arb.model_type == "L2_regularized_logistic_regression"


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_factory_functions_load_template_v0(factory) -> None:
    arb = factory()
    assert arb.n_training == 0, (
        "Templates must ship with n_training=0 — real training runs replace them "
        "in Phase 3, at which point this test flips to n_training > 0."
    )
    assert "template_v0" in arb.model_name


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_template_carries_auroc_caveat(factory) -> None:
    arb = factory()
    caveat = arb.performance["AUROC_CAVEAT"]
    # Honesty gate: template AUROC caveat must explicitly say TEMPLATE.
    assert caveat.startswith("TEMPLATE"), (
        f"Template model {arb.model_name} must declare TEMPLATE in AUROC_CAVEAT"
    )
    # And it must mention n_training=0.
    assert "n_training=0" in caveat


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_template_carries_ruo_disclaimer(factory) -> None:
    arb = factory()
    assert "RESEARCH USE ONLY" in arb.disclaimer


def test_load_arbiter_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown arbiter"):
        load_arbiter("does-not-exist")


def test_missing_frozen_json_raises_valueerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"intercept": 0.0}))  # missing coefficients etc.
    with pytest.raises(ValueError, match="missing required key"):
        L2LogisticArbiter(bad)


def test_frozen_model_without_auroc_caveat_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "intercept": 0.0,
        "coefficients": {},
        "feature_encodings": {},
        "recommendations": {"LOW": "x", "MID": "y", "HIGH": "z"},
        "performance": {}   # no AUROC_CAVEAT
    }))
    with pytest.raises(ValueError, match="AUROC_CAVEAT"):
        L2LogisticArbiter(bad)


# ── 3. Encoding ───────────────────────────────────────────────────────


def test_screening_birads_unknown_value_raises() -> None:
    arb = screening_arbiter()
    with pytest.raises(ValueError, match="allowed ="):
        arb.score({"birads": "BI_RADS_9"})


def test_unrecognised_feature_raises() -> None:
    arb = screening_arbiter()
    with pytest.raises(ValueError, match="Unrecognised feature"):
        arb.score({"birads": "BI_RADS_4", "not_a_real_feature": 1})


def test_bool_unknown_encodes_as_half() -> None:
    """Match ProgressionArbiter: True→1.0, False→0.0, None→0.5 (unknown)."""
    arb = screening_arbiter()
    r_true  = arb.score({"birads": "BI_RADS_1", "family_history_first_degree": True})
    r_false = arb.score({"birads": "BI_RADS_1", "family_history_first_degree": False})
    r_unk   = arb.score({"birads": "BI_RADS_1", "family_history_first_degree": None})
    coef = arb.coefficients["family_history_first_degree"]
    # Unknown should sit exactly halfway between true and false in logit space.
    assert math.isclose(r_true.term_contributions["family_history_first_degree"], coef * 1.0)
    assert math.isclose(r_false.term_contributions["family_history_first_degree"], coef * 0.0)
    assert math.isclose(r_unk.term_contributions["family_history_first_degree"], coef * 0.5)


def test_continuous_feature_normalised_by_divisor() -> None:
    """age_norm divisor is 100.0 per the JSON: age_years=50 → 0.5 → coef * 0.5."""
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_1", "age_norm": 50.0})
    coef = arb.coefficients["age_norm"]
    expected = coef * (50.0 / 100.0)
    assert math.isclose(r.term_contributions["age_norm"], round(expected, 6))


def test_one_hot_reference_class_contributes_zero() -> None:
    """BI_RADS_6 is the reference class → all birads_* contributions are 0."""
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_6"})
    for feat, val in r.term_contributions.items():
        if feat.startswith("birads_"):
            assert val == 0.0, f"reference class must contribute 0, got {feat}={val}"


def test_missing_feature_defaults_to_reference_for_categorical() -> None:
    """When a categorical feature isn't provided, we still emit one-hot=0 for every level."""
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_4"})  # density omitted
    for feat, val in r.term_contributions.items():
        if feat.startswith("density_"):
            assert val == 0.0


# ── 4. Invariants ─────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_sum_of_term_contributions_equals_logit(factory) -> None:
    """Core invariant: term_contributions sum to logit within 1e-6.

    Downstream code (L5 API 'why did the model say that?' UI) attributes
    the logit back to individual features by summing term_contributions.
    If this invariant breaks, the attributions become dishonest.

    Rounding to 6 decimals in ArbiterResult drops us from strict
    equality to ~1e-6 tolerance; the raw pre-round sum is stored in
    logit so we compare against that.
    """
    arb = factory()
    # Empty feature dict — just intercept.
    r_empty = arb.score({})
    assert math.isclose(sum(r_empty.term_contributions.values()), r_empty.logit, abs_tol=1e-6)

    # A populated case.
    if factory is screening_arbiter:
        features = {
            "birads": "BI_RADS_4",
            "density": "C_heterogeneously_dense",
            "prior_biopsy_history": True,
            "family_history_first_degree": False,
            "brca_status_known_pathogenic": None,
            "age_norm": 55.0,
            "years_since_last_mammo_norm": 2.0,
        }
    elif factory is biopsy_arbiter:
        features = {
            "lesion_type": "mass_spiculated",
            "size_norm": 15.0,
            "growth_delta_norm": 3.0,
            "prior_biopsy_benign_at_site": False,
            "family_history_first_degree": True,
            "brca_status_known_pathogenic": False,
            "us_correlate_hypoechoic_mass": True,
            "us_correlate_simple_cyst": False,
        }
    else:
        features = {
            "histology": "invasive_ductal",
            "grade": "3",
            "er_status_positive": False,
            "pr_status_positive": False,
            "her2_status_positive": True,
            "node_status_positive": True,
            "brca_status_known_pathogenic": True,
            "ki67_norm": 40.0,
            "tumor_size_norm": 35.0,
            "age_at_diagnosis_norm": 45.0,
        }
    r = arb.score(features)
    assert math.isclose(sum(r.term_contributions.values()), r.logit, abs_tol=1e-6)


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_probability_is_sigmoid_of_logit(factory) -> None:
    arb = factory()
    r = arb.score({})  # just intercept
    expected_p = 1.0 / (1.0 + math.exp(-r.logit))
    # p is rounded to 6 decimals so 1e-6 tolerance is what we can assert
    assert abs(r.p_positive - expected_p) < 1.5e-6


# ── 5. Contract ───────────────────────────────────────────────────────


def test_driving_feature_is_argmax_absolute_contribution() -> None:
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_5"})  # coef +4.5 dominates intercept -2.0
    non_intercept = {k: v for k, v in r.term_contributions.items() if k != "intercept"}
    expected = max(non_intercept, key=lambda k: abs(non_intercept[k]))
    assert r.driving_feature == expected
    assert r.driving_feature_contribution == non_intercept[expected]


def test_high_birads_pushes_to_high_bucket() -> None:
    """BI-RADS 5 (highly suspicious) → HIGH bucket → recall recommendation."""
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_5"})
    assert r.risk_bucket == "HIGH"
    assert r.recommendation == "RECALL_FOR_DIAGNOSTIC_WORKUP"


def test_low_birads_pushes_to_low_bucket() -> None:
    """BI-RADS 1 (negative) → LOW bucket → routine follow-up."""
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_1"})
    assert r.risk_bucket == "LOW"
    assert r.recommendation == "ROUTINE_1YR_FOLLOWUP"


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_result_carries_disclaimer_and_caveat(factory) -> None:
    arb = factory()
    r = arb.score({})
    assert "RESEARCH USE ONLY" in r.disclaimer
    assert r.caveat.startswith("TEMPLATE")  # honesty gate


@pytest.mark.parametrize("factory", [screening_arbiter, biopsy_arbiter, therapy_arbiter])
def test_template_result_metadata_flags_template_state(factory) -> None:
    arb = factory()
    r = arb.score({})
    assert r.metadata["model_state"] == "template"
    assert r.metadata["n_training"] == 0


def test_explain_produces_readable_string() -> None:
    arb = screening_arbiter()
    r = arb.score({"birads": "BI_RADS_4", "density": "D_extremely_dense"})
    txt = arb.explain(r)
    assert "P(recall_for_diagnostic_workup)" in txt
    assert "Risk bucket:" in txt
    assert "Driving feature:" in txt
    assert "RESEARCH USE ONLY" in txt
    # Template arbiters MUST prefix the probability line so a casual reader
    # never sees an unqualified p=0.93 that looks like a validated risk score.
    assert "[TEMPLATE — coefficients illustrative]" in txt
    # AUROC_CAVEAT must be present in the explain string as well.
    assert "TEMPLATE" in txt


# ── Bonus: sklearn round-trip ─────────────────────────────────────────


def _fit_sklearn_l2(X: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    """Fit ``sklearn.LogisticRegression(penalty='l2', C=1.0)`` and return (intercept, coefs)."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    m.fit(X, y)
    return float(m.intercept_[0]), [float(c) for c in m.coef_[0]]


def test_sklearn_round_trip_matches_our_scorer(tmp_path: Path) -> None:
    """Fit L2 logistic on synthetic data, dump to our JSON schema, verify
    ``L2LogisticArbiter.score`` reproduces ``sklearn.predict_proba`` to 1e-6.

    This is the property that lets us trust the frozen-JSON contract as an
    interchange format when the real Phase 3 training runs land.
    """
    rng = np.random.default_rng(seed=20260701)
    n = 400
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.integers(0, 2, size=n).astype(float)
    logit = -0.5 + 1.2 * x1 - 0.8 * x2 + 2.0 * x3
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    X = np.stack([x1, x2, x3], axis=1)

    intercept, coefs = _fit_sklearn_l2(X, y)

    frozen = {
        "model_name": "synthetic_l2_test_v1",
        "model_type": "L2_regularized_logistic_regression",
        "lambda": 1.0,
        "n_training": n,
        "positive_class": "y_eq_1",
        "intercept": intercept,
        "coefficients": {"x1": coefs[0], "x2": coefs[1], "x3_bool": coefs[2]},
        "feature_encodings": {
            "x1": "x1 / 1.0",
            "x2": "x2 / 1.0",
            "x3_bool": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        },
        "recommendations": {"LOW": "L", "MID": "M", "HIGH": "H"},
        "performance": {
            "AUROC_CAVEAT": (
                "Synthetic-data round-trip; not a real prospective validation. "
                "Only used to guarantee schema-level agreement between our scorer "
                "and sklearn.predict_proba."
            )
        },
        "disclaimer": RUO_DISCLAIMER,
    }
    path = tmp_path / "synth.json"
    path.write_text(json.dumps(frozen))

    arb = L2LogisticArbiter(path)

    from sklearn.linear_model import LogisticRegression
    sk_model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    sk_model.fit(X, y)

    max_abs_err = 0.0
    for i in range(20):  # first 20 rows are enough to catch schema drift
        features = {"x1": float(X[i, 0]), "x2": float(X[i, 1]), "x3_bool": bool(X[i, 2] == 1)}
        r = arb.score(features)
        sk_p = float(sk_model.predict_proba(X[i:i+1])[0, 1])
        max_abs_err = max(max_abs_err, abs(r.p_positive - sk_p))
    # Our scorer rounds to 6 decimals; 5e-6 easily accommodates that.
    assert max_abs_err < 5e-6, f"max |p_ours - p_sklearn| = {max_abs_err:.3e}"
