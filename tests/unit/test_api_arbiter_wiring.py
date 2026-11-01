"""Verify the L3 arbiters return real scores with exact inputs and outputs.

Exercises:
  - /v1/stage/screening/score (screening_arbiter_v1, n=2006)
  - /v1/stage/biopsy/score (biopsy_arbiter_v1, n=1977)
  - /v1/stage/therapy/score (therapy_arbiter_v1, n=1386)
  - sum(term_contributions) == logit invariant on all three trained arbiters
  - bucket -> recommendation mapping on real feature vectors
  - refusal to fabricate intercept-only arbiter_score when features={} on /v1/biopsy/analyze
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


SCREENING_FEATURES = {
    "age_norm": 62.0,
    "birads": "BI_RADS_4",
    "brca_status_known_pathogenic": False,
    "density": "C_heterogeneously_dense",
    "family_history_first_degree": True,
    "prior_biopsy_history": False,
    "years_since_last_mammo_norm": 1.5,
}

BIOPSY_FEATURES = {
    "brca_status_known_pathogenic": False,
    "family_history_first_degree": True,
    "growth_delta_norm": 4.0,
    "lesion_type": "mass_spiculated",
    "prior_biopsy_benign_at_site": False,
    "size_norm": 22.0,
    "us_correlate_hypoechoic_mass": True,
    "us_correlate_simple_cyst": False,
}

THERAPY_FEATURES = {
    "age_at_diagnosis_norm": 54.0,
    "brca_status_known_pathogenic": False,
    "er_status_positive": True,
    "pr_status_positive": True,
    "her2_status_positive": False,
    "grade": "3",
    "histology": "invasive_ductal",
    "ki67_norm": 38.0,
    "node_status_positive": True,
    "tumor_size_norm": 28.0,
}


def _arbiter_block_shape(body: dict) -> dict:
    """Assert the arbiter_score block has the required fields & invariants."""
    ab = body["arbiter_score"]
    assert ab is not None, "arbiter_score must be present in response body"
    for k in (
        "model_name",
        "p_positive",
        "logit",
        "risk_bucket",
        "recommendation",
        "term_contributions",
        "driving_feature",
        "driving_feature_contribution",
        "positive_class",
        "n_training",
        "model_state",
        "caveat",
    ):
        assert k in ab, f"arbiter_score missing field: {k}"
    assert ab["n_training"] > 0
    assert ab["model_state"] == "frozen"
    assert "Real-patient retrospective" in ab["caveat"]
    assert "not prospectively validated" in ab["caveat"]
    assert ab["risk_bucket"] in {"LOW", "MID", "HIGH"}
    assert 0.0 <= ab["p_positive"] <= 1.0
    return ab


def test_screening_endpoint_returns_real_arbiter_score(client: TestClient) -> None:
    resp = client.post("/v1/stage/screening/score", json={"features": SCREENING_FEATURES})
    assert resp.status_code == 200, resp.text
    ab = _arbiter_block_shape(resp.json())
    assert ab["model_name"] == "screening_arbiter_v1"
    assert ab["n_training"] == 2006
    assert math.isclose(ab["p_positive"], 0.47036, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(ab["logit"], -0.118699, rel_tol=0.0, abs_tol=1e-6)
    assert ab["risk_bucket"] == "MID"
    assert ab["recommendation"] == "SHORT_INTERVAL_6MO_FOLLOWUP"
    assert ab["driving_feature"] == "birads_BI_RADS_4"
    assert math.isclose(ab["driving_feature_contribution"], 0.231575, rel_tol=0.0, abs_tol=1e-6)


def test_biopsy_endpoint_returns_real_arbiter_score(client: TestClient) -> None:
    resp = client.post("/v1/stage/biopsy/score", json={"features": BIOPSY_FEATURES})
    assert resp.status_code == 200, resp.text
    ab = _arbiter_block_shape(resp.json())
    assert ab["model_name"] == "biopsy_arbiter_v1"
    assert ab["n_training"] == 1977
    assert math.isclose(ab["p_positive"], 0.840387, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(ab["logit"], 1.661108, rel_tol=0.0, abs_tol=1e-6)
    assert ab["risk_bucket"] == "HIGH"
    assert ab["recommendation"] == "PROCEED_TO_CORE_NEEDLE_BIOPSY"
    assert ab["driving_feature"] == "lesion_type_mass_spiculated"
    assert math.isclose(ab["driving_feature_contribution"], 2.193218, rel_tol=0.0, abs_tol=1e-6)


def test_therapy_endpoint_returns_real_arbiter_score(client: TestClient) -> None:
    resp = client.post("/v1/stage/therapy/score", json={"features": THERAPY_FEATURES})
    assert resp.status_code == 200, resp.text
    ab = _arbiter_block_shape(resp.json())
    assert ab["model_name"] == "therapy_arbiter_v1"
    assert ab["n_training"] == 1386
    assert math.isclose(ab["p_positive"], 0.400548, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(ab["logit"], -0.403183, rel_tol=0.0, abs_tol=1e-6)
    assert ab["risk_bucket"] == "MID"
    assert ab["recommendation"] == "INTERMEDIATE_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE"
    assert ab["driving_feature"] == "age_at_diagnosis_norm"
    assert math.isclose(ab["driving_feature_contribution"], -4.889608, rel_tol=0.0, abs_tol=1e-6)


def test_biopsy_arbiter_sum_of_terms_matches_logit(client: TestClient) -> None:
    resp = client.post("/v1/stage/biopsy/score", json={"features": BIOPSY_FEATURES})
    assert resp.status_code == 200
    ab = _arbiter_block_shape(resp.json())
    terms_sum = sum(ab["term_contributions"].values())
    assert abs(terms_sum - ab["logit"]) < 1e-4, (
        f"sum(term_contributions)={terms_sum} != logit={ab['logit']}"
    )


def test_therapy_arbiter_bucket_matches_recommendation(client: TestClient) -> None:
    resp = client.post("/v1/stage/therapy/score", json={"features": THERAPY_FEATURES})
    assert resp.status_code == 200
    ab = _arbiter_block_shape(resp.json())
    bucket_to_rec = {
        "LOW": "LOW_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "MID": "INTERMEDIATE_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "HIGH": "HIGH_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
    }
    assert ab["recommendation"] == bucket_to_rec[ab["risk_bucket"]]


def test_empty_feature_routes_refuse_intercept_only_arbiter_score(client: TestClient) -> None:
    """Routes without an explicit feature vector must not emit an intercept-only constant."""
    resp = client.post("/v1/biopsy/analyze", json={"report_text": "invasive ductal carcinoma"})
    assert resp.status_code == 200
    assert resp.json()["arbiter_score"] is None
