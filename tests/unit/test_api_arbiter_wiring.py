"""Verify how L3 template arbiters are (and are not) wired into the endpoints.

This file previously asserted the retired contract: that every stage endpoint
always returns a populated ``arbiter_score``, derived from a regex parse of
free-text ``report_text`` and from an empty ``patient_context``.  That is
exactly the behaviour the production-integrity strip removed, because it let a
hand-drafted ``n_training=0`` template emit a patient-level probability from
data nobody supplied.

The honest contract now under test:

* **biopsy** never emits a template ``arbiter_score``.  Prognosis comes from the
  METABRIC-fitted Breast DSS v3 arbiter and only when the complete explicit
  seven-feature vector is supplied -- no parser imputation.
* **therapy** emits an illustrative research triage decomposition *only* from a
  complete explicit feature vector.  Missing features suppress scoring; they are
  never encoded at the 0.5 midpoint.
* ``lymph_nodes_pos == 0`` is observed-negative and must contribute exactly
  ``0.0`` log-odds, distinct from "nodes unknown", which suppresses the score.
* the sum-of-terms invariant must survive serialisation.

In-process FastAPI TestClient; no network.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app

# The exact log-odds deltas encoded in therapy_arbiter_template_v0.json.
NODE_COEF = 2.1
GRADE_2_TO_3_DELTA = 1.2
TUMOR_SIZE_50MM_DELTA = 1.4
AGE_10Y_DELTA = -0.05

COMPLETE_TRIAGE: dict[str, Any] = {
    "histology": "invasive_ductal",
    "grade": 2,
    "er_positive": True,
    "pr_positive": True,
    "her2_positive": False,
    "ki67_pct": 20.0,
    "tumor_size_mm": 25.0,
    "lymph_nodes_pos": 0,
    "brca_pathogenic": False,
    "age_years": 55.0,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _triage(client: TestClient, **overrides: Any) -> dict | None:
    payload = dict(COMPLETE_TRIAGE)
    for k, v in overrides.items():
        if v is _OMIT:
            payload.pop(k, None)
        else:
            payload[k] = v
    resp = client.post(
        "/v1/therapy/reason",
        json={"biopsy_output": None, "patient_context": {}, "therapy_features": payload},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["therapy_triage"]


class _Omit:
    pass


_OMIT = _Omit()


def _arbiter_block_shape(ab: dict) -> None:
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
    assert ab["n_training"] == 0
    assert ab["model_state"] == "template"
    assert ab["caveat"].startswith("TEMPLATE")
    assert ab["risk_bucket"] in {"LOW", "MID", "HIGH"}
    assert 0.0 <= ab["p_positive"] <= 1.0


# --------------------------------------------------------------------------- #
# biopsy: the template arbiter is retired
# --------------------------------------------------------------------------- #


def test_biopsy_endpoint_emits_no_template_arbiter_score(client: TestClient) -> None:
    """A free-text report must not produce a template patient probability."""
    resp = client.post(
        "/v1/biopsy/analyze", json={"report_text": "invasive ductal carcinoma"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["arbiter_score"] is None, (
        "biopsy must not emit the n_training=0 template arbiter; prognosis is "
        "Breast DSS v3 on an explicit feature vector only"
    )
    assert "biopsy_arbiter_template_v0" not in resp.text


def test_biopsy_dss_suppressed_without_complete_explicit_vector(
    client: TestClient,
) -> None:
    """No seven-feature vector -> no prognosis, and a receipt says why."""
    resp = client.post("/v1/biopsy/analyze", json={"report_text": "invasive ductal carcinoma"})
    body = resp.json()
    assert body.get("dss_prognosis") is None
    stages = {r["stage"]: r for r in body.get("stage_receipts", [])}
    receipt = stages.get("breast_dss_prognosis")
    assert receipt is not None, "a breast_dss_prognosis receipt must be emitted"
    assert receipt["status"] == "skipped_not_applicable"
    assert receipt["required"] is False
    # Strictness check: the required ClinicalBERT parse must FAIL loudly when
    # its Modal URL is unset rather than silently degrading to a regex parser.
    parse = stages.get("clinicalbert_pathology_parse")
    assert parse is not None and parse["required"] is True
    assert parse["status"] == "failed_required"


# --------------------------------------------------------------------------- #
# therapy: explicit inputs only
# --------------------------------------------------------------------------- #


def test_therapy_arbiter_scores_only_from_explicit_inputs(client: TestClient) -> None:
    ab = _triage(client)
    assert ab is not None, "a complete explicit vector must produce a triage score"
    _arbiter_block_shape(ab)
    assert ab["model_name"] == "therapy_arbiter_template_v0"


def test_therapy_arbiter_gates_when_nodes_missing(client: TestClient) -> None:
    """Absent nodes are gated, not imputed and not silently dropped.

    MIGRATED from ``test_therapy_arbiter_suppressed_when_nodes_missing``.

    The original invariant -- an unobserved node status is never imputed to
    0.5 -- is preserved and strengthened. The old contract returned
    ``therapy_triage: None`` with a "skipped" receipt, which satisfied the
    letter of "never impute" while telling the caller nothing: they could not
    distinguish "we declined to score" from "we scored and it was low", and
    they never learned how much the missing covariate actually cost.

    The new contract refuses with HTTP 422 and hands back the identified set.
    On this panel an unobserved ``node_status_positive`` (coefficient +2.1,
    the largest in the artefact) spans [0.168682, 0.623634] -- width 0.454952,
    nearly half the probability scale from a single missing field. That is the
    number the old test threw away.
    """
    payload = {k: v for k, v in COMPLETE_TRIAGE.items() if k != "lymph_nodes_pos"}
    resp = client.post(
        "/v1/therapy/reason",
        json={"biopsy_output": None, "patient_context": {},
              "therapy_features": payload},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"] == "ManskiBoundsExceeded"
    assert body["missing_features"] == ["node_status_positive"]
    assert body["bounds"] == [0.168682, 0.623634]
    assert body["width"] == 0.454952
    assert body["max_width"] == 0.25
    assert body["public_bypass_available"] is False
    # The original invariant, checked directly: no imputed point estimate
    # anywhere in the refusal, and specifically nothing near 0.5.
    assert body["point_estimate_withheld"] is True
    assert "p_positive" not in body and "arbiter_score" not in body

    resp = client.post(
        "/v1/therapy/reason",
        json={"biopsy_output": None, "patient_context": {}, "therapy_features": None},
    )
    body = resp.json()
    assert body["therapy_triage"] is None
    assert body["arbiter_score"] is None
    stages = {r["stage"]: r for r in body["stage_receipts"]}
    triage_receipt = stages["therapy_research_triage"]
    assert triage_receipt["status"] == "skipped_not_applicable"
    assert any("nothing to identify" in w for w in triage_receipt["warnings"]), (
        f"the skip reason must say why nothing was scored: "
        f"{triage_receipt['warnings']}"
    )

    # MIGRATED assertion. The old test proved "unknown is never 0.5" by
    # grepping for the string "unknown=0.5" in a warning message -- a claim
    # about prose, which a rewording silently voids. The invariant is now
    # checked where it actually lives: in the frozen artefact, whose boolean
    # encodings must map `unknown` to null, and in the loader, which refuses
    # any template declaring anything else.
    import json as _json
    from pathlib import Path as _Path

    artefact = _json.loads(
        _Path(__file__).resolve().parents[2].joinpath(
            "src/oncology_arbiter/arbiter/models/therapy_arbiter_template_v0.json"
        ).read_text()
    )
    bool_encodings = {
        k: v for k, v in artefact["feature_encodings"].items()
        if isinstance(v, dict) and set(v) <= {"true", "false", "unknown"}
    }
    assert bool_encodings, "expected boolean-encoded features in the artefact"
    for name, enc in bool_encodings.items():
        assert enc["unknown"] is None, (
            f"{name} declares unknown={enc['unknown']!r}; an unobserved "
            f"boolean must stay unobserved, never be imputed"
        )
        assert enc["false"] == 0.0 and enc["true"] == 1.0


def test_zero_nodes_is_observed_negative_contributing_exactly_zero(
    client: TestClient,
) -> None:
    """lymph_nodes_pos=0 is data, not absence: contribution must be exactly 0.0."""
    ab = _triage(client, lymph_nodes_pos=0)
    assert ab is not None
    contrib = ab["term_contributions"]["node_status_positive"]
    assert contrib == 0.0, f"zero nodes must contribute exactly 0.0, got {contrib}"
    assert contrib != pytest.approx(NODE_COEF * 0.5), "0.5 midpoint encoding detected"


def test_one_positive_node_moves_logit_by_exactly_the_node_coefficient(
    client: TestClient,
) -> None:
    zero = _triage(client, lymph_nodes_pos=0)
    one = _triage(client, lymph_nodes_pos=1)
    assert zero is not None and one is not None
    assert one["logit"] - zero["logit"] == pytest.approx(NODE_COEF, abs=1e-9)
    assert one["term_contributions"]["node_status_positive"] == pytest.approx(
        NODE_COEF, abs=1e-9
    )


@pytest.mark.parametrize(
    "field,low,high,delta",
    [
        ("grade", 2, 3, GRADE_2_TO_3_DELTA),
        ("tumor_size_mm", 25.0, 75.0, TUMOR_SIZE_50MM_DELTA),
        ("age_years", 55.0, 65.0, AGE_10Y_DELTA),
    ],
)
def test_encoded_deltas_are_exact(
    client: TestClient, field: str, low: Any, high: Any, delta: float
) -> None:
    a = _triage(client, **{field: low})
    b = _triage(client, **{field: high})
    assert a is not None and b is not None
    assert b["logit"] - a["logit"] == pytest.approx(delta, abs=1e-9)


def test_therapy_arbiter_sum_of_terms_matches_logit(client: TestClient) -> None:
    """If serialisation drops or rounds terms, the 'why' UI would be dishonest."""
    ab = _triage(client)
    assert ab is not None
    terms_sum = sum(ab["term_contributions"].values())
    assert terms_sum == pytest.approx(ab["logit"], abs=1e-9), (
        f"sum(term_contributions)={terms_sum} != logit={ab['logit']}"
    )


def test_therapy_bucket_matches_recommendation(client: TestClient) -> None:
    ab = _triage(client)
    assert ab is not None
    bucket_to_rec = {
        "LOW": "SURGERY_FIRST",
        "MID": "MULTIDISCIPLINARY_REVIEW",
        "HIGH": "ESCALATE_TO_NEOADJUVANT_CHEMOTHERAPY",
    }
    assert ab["recommendation"] == bucket_to_rec[ab["risk_bucket"]]


def test_no_term_contribution_is_ever_a_half_coefficient(client: TestClient) -> None:
    """Regression guard for the retired unknown=0.5 midpoint encoding."""
    ab = _triage(client)
    assert ab is not None
    for name, value in ab["term_contributions"].items():
        assert value != pytest.approx(NODE_COEF * 0.5), f"{name} looks midpoint-encoded"


# --------------------------------------------------------------------------- #
# /v1/case/full
# --------------------------------------------------------------------------- #


def test_case_full_emits_no_template_arbiters_from_free_text(client: TestClient) -> None:
    """A report_text-only case must not manufacture stage probabilities."""
    resp = client.post("/v1/case/full", json={"biopsy_input": {"report_text": "test"}})
    assert resp.status_code == 200
    body = resp.json()
    if body.get("biopsy") is not None:
        assert body["biopsy"]["arbiter_score"] is None
    assert body.get("therapy") is None, (
        "therapy recommendations require the authenticated SL bridge; the "
        "retired rules-lite proxy used to fill this field"
    )
