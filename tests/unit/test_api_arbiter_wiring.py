"""Verify the current explicit-feature L3 arbiter and L5 contracts.

PLAN.md §4a: 'Every response body carries: stage_output, arbiter_score,
term_contributions, driving_feature, evidence[]'.

The imaging/text endpoints do not fabricate an intercept-only arbiter score
from missing clinical features. Biopsy has a dedicated explicit feature route;
therapy accepts a complete ``therapy_features`` panel; and the current breast
L5 response exposes ``report_parse`` / ``therapy_triage`` at top level.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cbis_ddsm"
DICOM_FIXTURES = sorted(FIXTURE_DIR.glob("*.dcm"))


def _arbiter_block_shape(body: dict) -> None:
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
    # Trained v1 production contract.
    assert ab["n_training"] > 0
    assert ab["model_state"] == "frozen"
    assert "Real-patient retrospective" in ab["caveat"]
    assert "not prospectively validated" in ab["caveat"]
    # Bucket must be one of the three
    assert ab["risk_bucket"] in {"LOW", "MID", "HIGH"}
    # p_positive must be sigmoid-consistent
    assert 0.0 <= ab["p_positive"] <= 1.0


SPICULATED_MASS = {
    "lesion_type": "mass_spiculated",
    "size_norm": 22.0,
    "growth_delta_norm": 3.2,
    "prior_biopsy_benign_at_site": False,
    "family_history_first_degree": True,
    "brca_status_known_pathogenic": False,
    "us_correlate_hypoechoic_mass": True,
    "us_correlate_simple_cyst": False,
}

CIRCUMSCRIBED_MASS = {
    **SPICULATED_MASS,
    "lesion_type": "mass_circumscribed",
}

LOW_CHEMO_RECEIPT_PROFILE = {
    "histology": "invasive_ductal",
    "grade": 1,
    "er_positive": True,
    "pr_positive": True,
    "her2_positive": False,
    "ki67_pct": 10.0,
    "tumor_size_mm": 10.0,
    "lymph_nodes_pos": 0,
    "brca_pathogenic": False,
    "age_years": 54.0,
}

HIGH_CHEMO_RECEIPT_PROFILE = {
    "histology": "invasive_ductal",
    "grade": 3,
    "er_positive": False,
    "pr_positive": False,
    "her2_positive": False,
    "ki67_pct": 70.0,
    "tumor_size_mm": 70.0,
    "lymph_nodes_pos": 4,
    "brca_pathogenic": False,
    "age_years": 54.0,
}


# ── explicit biopsy + therapy feature contracts ─────────────────────


def test_spiculated_mass_scores_above_circumscribed_mass(
    client: TestClient,
) -> None:
    """The validated morphology contrast must survive the API boundary."""
    spiculated = client.post(
        "/v1/stage/biopsy/score",
        json={"features": SPICULATED_MASS},
    )
    circumscribed = client.post(
        "/v1/stage/biopsy/score",
        json={"features": CIRCUMSCRIBED_MASS},
    )
    assert spiculated.status_code == circumscribed.status_code == 200
    spiculated_score = spiculated.json()["arbiter_score"]
    circumscribed_score = circumscribed.json()["arbiter_score"]
    _arbiter_block_shape(spiculated.json())
    _arbiter_block_shape(circumscribed.json())
    assert spiculated_score["model_name"] == "biopsy_arbiter_v1"
    assert (
        spiculated_score["p_positive"]
        > circumscribed_score["p_positive"]
    )
    assert (
        spiculated_score["recommendation"]
        == "PROCEED_TO_CORE_NEEDLE_BIOPSY"
    )


def test_high_risk_breast_profile_has_higher_observed_chemo_receipt_score(
    client: TestClient,
) -> None:
    """The score is receipt propensity—not efficacy—and must preserve contrast."""
    low = client.post(
        "/v1/therapy/reason",
        json={
            "therapy_features": LOW_CHEMO_RECEIPT_PROFILE,
            "patient_context": {},
        },
    )
    high = client.post(
        "/v1/therapy/reason",
        json={
            "therapy_features": HIGH_CHEMO_RECEIPT_PROFILE,
            "patient_context": {},
        },
    )
    assert low.status_code == high.status_code == 200
    low_score = low.json()["arbiter_score"]
    high_score = high.json()["arbiter_score"]
    _arbiter_block_shape(low.json())
    _arbiter_block_shape(high.json())
    assert high_score["p_positive"] > low_score["p_positive"]
    assert high_score["positive_class"] == "observed_chemotherapy_receipt"
    assert "not treatment response" in high_score["caveat"]
    assert high_score["manski"]["identified"] is True
    assert high_score["manski"]["width"] == pytest.approx(0.0)


# ── screening requires DICOM bytes ────────────────────────────────────


@pytest.mark.skipif(not DICOM_FIXTURES, reason="No CBIS fixtures available for this test")
def test_screening_endpoint_refuses_intercept_only_arbiter(client: TestClient) -> None:
    dcm = DICOM_FIXTURES[0]
    b64 = base64.b64encode(dcm.read_bytes()).decode("ascii")
    resp = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": b64})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Whether MedSigLIP is enabled or fail-closed is environment-dependent;
    # neither branch may invent a clinical arbiter from imaging alone.
    assert body["arbiter_score"] is None


# ── invariant: sum(term_contributions) == logit ─────────────────────


def test_biopsy_arbiter_sum_of_terms_matches_logit(client: TestClient) -> None:
    """The wire-level ArbiterScore must preserve the sum-of-terms invariant.

    If serialization drops or rounds terms differently from the logit, our
    'why did the model say that?' UI would be dishonest.
    """
    resp = client.post(
        "/v1/stage/biopsy/score",
        json={"features": SPICULATED_MASS},
    )
    body = resp.json()
    ab = body["arbiter_score"]
    terms_sum = sum(ab["term_contributions"].values())
    assert abs(terms_sum - ab["logit"]) < 1e-4, (
        f"sum(term_contributions)={terms_sum} != logit={ab['logit']}"
    )


def test_therapy_arbiter_bucket_matches_recommendation(client: TestClient) -> None:
    """PLAN.md invariant: recommendation string must match the LOW/MID/HIGH bucket."""
    resp = client.post(
        "/v1/therapy/reason",
        json={
            "therapy_features": HIGH_CHEMO_RECEIPT_PROFILE,
            "patient_context": {},
        },
    )
    ab = resp.json()["arbiter_score"]
    bucket_to_rec = {
        "LOW": "LOW_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "MID": "INTERMEDIATE_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "HIGH": "HIGH_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
    }
    assert ab["recommendation"] == bucket_to_rec[ab["risk_bucket"]]


# ── /v1/case/full chains through all three arbiters ─────────────────


def test_case_full_returns_top_level_therapy_triage(client: TestClient) -> None:
    """Current breast L5 returns top-level specialist blocks, not legacy nesting."""
    resp = client.post(
        "/v1/case/full",
        json={
            "therapy_features": HIGH_CHEMO_RECEIPT_PROFILE,
            "run_co_scientist": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["therapy_triage"] is not None
    assert body["therapy_triage"]["model_name"] == "therapy_arbiter_v1"
    assert (
        body["therapy_triage"]["positive_class"]
        == "observed_chemotherapy_receipt"
    )
    assert "not treatment response" in body["therapy_triage"]["caveat"]
    assert body["biopsy"] is None
    assert body["therapy"] is None
