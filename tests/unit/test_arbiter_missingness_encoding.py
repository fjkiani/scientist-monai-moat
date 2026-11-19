"""Missing booleans must not manufacture risk in the L2 logistic arbiter.

The regression these tests lock down: encoding an unobserved boolean at the
midpoint 0.5 injected ``0.5 * sum(coef)`` of log-odds. On the shipped screening
template that was exactly +1.15, moving an all-unknown patient from the
intercept-only base rate 0.11920292 to 0.29943286.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from oncology_arbiter.arbiter.logistic import (
    MISSINGNESS_DECLARED_INDICATOR,
    MISSINGNESS_REFERENCE_LEVEL,
    MISSING_BOOL_ENCODING,
    MISSING_INDICATOR_SUFFIX,
    L2LogisticArbiter,
)

MODELS = Path(__file__).resolve().parents[2] / "src" / "oncology_arbiter" / "arbiter" / "models"
SCREENING = MODELS / "screening_arbiter_template_v0.json"
THERAPY = MODELS / "therapy_arbiter_template_v0.json"

# sigmoid(-2.0) — the screening template's intercept-only base rate.
SCREENING_BASE_RATE = 0.11920292202211755
# 0.5 * (0.6 + 0.5 + 1.2) — the log-odds the midpoint encoding used to inject.
FORMER_MIDPOINT_INFLATION = 1.15


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_missing_bool_encoding_constant_is_zero_not_half() -> None:
    assert MISSING_BOOL_ENCODING == 0.0


def test_all_unknown_panel_lands_exactly_on_intercept_base_rate() -> None:
    """The whole point of the repair: absence returns the artefact's base rate."""
    arb = L2LogisticArbiter(SCREENING)
    result = arb.score({"birads": None, "density": None})
    assert result.p_positive == pytest.approx(SCREENING_BASE_RATE, abs=1e-6)
    assert result.logit == pytest.approx(arb.intercept, abs=1e-9)
    for feature in ("prior_biopsy_history", "family_history_first_degree", "brca_status_known_pathogenic"):
        assert result.term_contributions[feature] == 0.0
        assert feature in result.missing_features
    assert result.missingness_policy == MISSINGNESS_REFERENCE_LEVEL


def test_midpoint_inflation_is_no_longer_reachable() -> None:
    """Reproduce the old arithmetic, then prove the shipped path avoids it."""
    arb = L2LogisticArbiter(SCREENING)
    bool_coefs = [
        arb.coefficients["prior_biopsy_history"],
        arb.coefficients["family_history_first_degree"],
        arb.coefficients["brca_status_known_pathogenic"],
    ]
    assert sum(bool_coefs) == pytest.approx(2.3)
    assert 0.5 * sum(bool_coefs) == pytest.approx(FORMER_MIDPOINT_INFLATION)

    inflated_logit = arb.intercept + FORMER_MIDPOINT_INFLATION
    assert _sigmoid(inflated_logit) == pytest.approx(0.29943286, abs=1e-6)

    actual = arb.score({"birads": None, "density": None})
    assert actual.logit < inflated_logit - 1.0
    assert actual.p_positive < 0.20


def test_prevalence_valued_encoding_would_still_overshoot_the_base_rate() -> None:
    """A feature-space prevalence is not a probability-space base rate.

    Encoding the missing feature at 0.1192 still injects 0.1192 * 2.3 of
    log-odds; only the reference level reproduces sigmoid(intercept).
    """
    arb = L2LogisticArbiter(SCREENING)
    sum_coef = 2.3
    p_at_prevalence = _sigmoid(arb.intercept + 0.1192 * sum_coef)
    assert p_at_prevalence == pytest.approx(0.15112046, abs=1e-6)
    assert p_at_prevalence > SCREENING_BASE_RATE
    # The shipped path is the one that matches the base rate exactly.
    assert arb.score({"birads": None, "density": None}).p_positive == pytest.approx(
        SCREENING_BASE_RATE, abs=1e-6
    )


def test_missing_bool_is_indistinguishable_from_observed_false_in_logit() -> None:
    arb = L2LogisticArbiter(THERAPY)
    base = {"histology": None, "grade": None, "er_status_positive": True}
    missing = arb.score({**base, "node_status_positive": None})
    observed_false = arb.score({**base, "node_status_positive": False})
    assert missing.logit == observed_false.logit
    assert missing.term_contributions["node_status_positive"] == 0.0
    assert observed_false.term_contributions["node_status_positive"] == 0.0
    # ...but only one of them declares the absence.
    assert "node_status_positive" in missing.missing_features
    assert "node_status_positive" not in observed_false.missing_features


def test_assumption_free_bounds_bracket_the_point_estimate() -> None:
    arb = L2LogisticArbiter(SCREENING)
    result = arb.score({"birads": None, "density": None})
    assert result.p_lower_bound <= result.p_positive <= result.p_upper_bound
    # All three missing coefficients are positive, so the lower bound is the
    # point estimate and the upper bound is driven by all-True.
    assert result.p_lower_bound == pytest.approx(SCREENING_BASE_RATE, abs=1e-6)
    assert result.p_upper_bound == pytest.approx(_sigmoid(arb.intercept + 2.3), abs=1e-6)
    assert result.p_upper_bound - result.p_lower_bound > 0.4


def test_fully_observed_case_has_degenerate_bounds() -> None:
    arb = L2LogisticArbiter(SCREENING)
    result = arb.score(
        {
            "birads": "BI_RADS_4",
            "density": "C_heterogeneously_dense",
            "prior_biopsy_history": False,
            "family_history_first_degree": False,
            "brca_status_known_pathogenic": False,
            "age_norm": 61.0,
            "years_since_last_mammo_norm": 2.0,
        }
    )
    assert result.missing_features == []
    assert result.p_lower_bound == result.p_positive == result.p_upper_bound


def test_artefact_declaring_nonzero_unknown_is_rejected_at_load(tmp_path: Path) -> None:
    """The artefact and the code cannot disagree about missingness."""
    bad = json.loads(SCREENING.read_text())
    bad["feature_encodings"]["family_history_first_degree"]["unknown"] = 0.5
    path = tmp_path / "bad_unknown.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown=0.5"):
        L2LogisticArbiter(path)


def test_shipped_templates_declare_no_nonzero_unknown_level() -> None:
    for artefact in sorted(MODELS.glob("*_template_v0.json")):
        model = json.loads(artefact.read_text())
        for name, spec in model["feature_encodings"].items():
            if isinstance(spec, dict) and set(spec) <= {"true", "false", "unknown"}:
                assert spec.get("unknown") in (None, 0.0), f"{artefact.name}:{name}"
        # And each carries an explicit, auditable policy block.
        assert model["missingness"]["unobserved_boolean_encoding"] == 0.0
        assert model["missingness"]["fitted_missingness_indicator_available"] is False


def test_declared_indicator_is_used_when_and_only_when_fitted(tmp_path: Path) -> None:
    """Option (a): an indicator applies only if the artefact declares a coefficient."""
    model = json.loads(SCREENING.read_text())
    feature = "brca_status_known_pathogenic"
    model["coefficients"][f"{feature}{MISSING_INDICATOR_SUFFIX}"] = -0.37
    path = tmp_path / "with_indicator.json"
    path.write_text(json.dumps(model))

    arb = L2LogisticArbiter(path)
    result = arb.score({"birads": None, "density": None, feature: None})
    assert result.missingness_policy == MISSINGNESS_DECLARED_INDICATOR
    assert result.term_contributions[f"{feature}{MISSING_INDICATOR_SUFFIX}"] == pytest.approx(-0.37)
    # The base feature still contributes zero; the indicator carries the effect.
    assert result.term_contributions[feature] == 0.0
    assert result.logit == pytest.approx(arb.intercept - 0.37, abs=1e-6)

    # Observed values must never trigger the indicator term.
    observed = arb.score({"birads": None, "density": None, feature: True})
    assert f"{feature}{MISSING_INDICATOR_SUFFIX}" not in observed.term_contributions
    assert observed.missingness_policy == MISSINGNESS_REFERENCE_LEVEL


def test_sum_of_terms_invariant_survives_missingness_terms() -> None:
    arb = L2LogisticArbiter(THERAPY)
    result = arb.score({"histology": None, "grade": None})
    assert sum(result.term_contributions.values()) == pytest.approx(result.logit, abs=1e-6)
