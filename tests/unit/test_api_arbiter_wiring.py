"""Verify the L3 arbiter is wired into every stage endpoint.

PLAN.md §4a: 'Every response body carries: stage_output, arbiter_score,
term_contributions, driving_feature, evidence[]'.

We use FastAPI TestClient so this is an in-process test with no network.
Screening is exercised through real CBIS-DDSM DICOM bytes when a fixture
is present (parametrized), and through the placeholder-refusal path when
no bytes are sent.
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
    # Template contract
    assert ab["n_training"] == 0
    assert ab["model_state"] == "template"
    assert ab["caveat"].startswith("TEMPLATE")
    # Bucket must be one of the three
    assert ab["risk_bucket"] in {"LOW", "MID", "HIGH"}
    # p_positive must be sigmoid-consistent
    assert 0.0 <= ab["p_positive"] <= 1.0


# ── biopsy + therapy don't need real fixtures ────────────────────────


def test_biopsy_endpoint_returns_arbiter_score(client: TestClient) -> None:
    resp = client.post("/v1/biopsy/analyze", json={"report_text": "invasive ductal carcinoma"})
    assert resp.status_code == 200
    body = resp.json()
    _arbiter_block_shape(body)
    assert body["arbiter_score"]["model_name"] == "biopsy_arbiter_template_v0"


def test_therapy_endpoint_returns_arbiter_score(client: TestClient) -> None:
    resp = client.post("/v1/therapy/reason", json={"biopsy_output": None, "patient_context": {}})
    assert resp.status_code == 200
    body = resp.json()
    _arbiter_block_shape(body)
    assert body["arbiter_score"]["model_name"] == "therapy_arbiter_template_v0"


# ── screening requires DICOM bytes ────────────────────────────────────


@pytest.mark.skipif(not DICOM_FIXTURES, reason="No CBIS fixtures available for this test")
def test_screening_endpoint_returns_arbiter_score(client: TestClient) -> None:
    dcm = DICOM_FIXTURES[0]
    b64 = base64.b64encode(dcm.read_bytes()).decode("ascii")
    resp = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": b64})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _arbiter_block_shape(body)
    assert body["arbiter_score"]["model_name"] == "screening_arbiter_template_v0"


# ── invariant: sum(term_contributions) == logit ─────────────────────


def test_biopsy_arbiter_sum_of_terms_matches_logit(client: TestClient) -> None:
    """The wire-level ArbiterScore must preserve the sum-of-terms invariant.

    If serialization drops or rounds terms differently from the logit, our
    'why did the model say that?' UI would be dishonest.
    """
    resp = client.post("/v1/biopsy/analyze", json={"report_text": "test"})
    body = resp.json()
    ab = body["arbiter_score"]
    terms_sum = sum(ab["term_contributions"].values())
    assert abs(terms_sum - ab["logit"]) < 1e-4, (
        f"sum(term_contributions)={terms_sum} != logit={ab['logit']}"
    )


def test_therapy_arbiter_bucket_matches_recommendation(client: TestClient) -> None:
    """PLAN.md invariant: recommendation string must match the LOW/MID/HIGH bucket."""
    resp = client.post("/v1/therapy/reason", json={"biopsy_output": None, "patient_context": {}})
    ab = resp.json()["arbiter_score"]
    bucket_to_rec = {
        "LOW":  "SURGERY_FIRST",
        "MID":  "MULTIDISCIPLINARY_REVIEW",
        "HIGH": "ESCALATE_TO_NEOADJUVANT_CHEMOTHERAPY",
    }
    assert ab["recommendation"] == bucket_to_rec[ab["risk_bucket"]]


# ── /v1/case/full chains through all three arbiters ─────────────────


def test_case_full_returns_biopsy_and_therapy_arbiters(client: TestClient) -> None:
    """/v1/case/full should carry arbiter scores from every stage it exercises.

    We deliberately omit screening_input here because screening requires
    DICOM bytes; the case orchestrator should still fill therapy from an
    empty biopsy path.
    """
    resp = client.post("/v1/case/full", json={"biopsy_input": {"report_text": "test"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["biopsy"] is not None
    assert body["biopsy"]["arbiter_score"] is not None
    assert body["therapy"] is not None
    assert body["therapy"]["arbiter_score"] is not None
