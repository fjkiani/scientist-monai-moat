"""Frozen breast DSS v3 model and biopsy API integration tests."""
from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.models.breast_dss_arbiter import (
    ARTIFACT_NAME,
    BreastDssArbiterError,
    load_breast_dss_artifact,
    score_breast_dss,
)

EXPECTED_ARTIFACT_SHA256 = (
    "fc5cf139aa7912c3813bd80d889f5fabda1c210ef561aac0f4277e68879215f5"
)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    for flag in (
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY",
    ):
        monkeypatch.delenv(flag, raising=False)
    return TestClient(create_app())


def _features() -> dict:
    return {
        "age": 58.0,
        "tumor_size_mm": 22.0,
        "nodes_positive": 1,
        "grade": 2,
        "er_positive": True,
        "pr_positive": True,
        "her2_positive": False,
    }


def test_packaged_artifact_is_byte_identical_to_audit_copy() -> None:
    artifact, digest = load_breast_dss_artifact()
    assert digest == EXPECTED_ARTIFACT_SHA256
    assert artifact["artifact_name"] == ARTIFACT_NAME
    assert artifact["n"] == 1375
    assert artifact["events"] == 601
    assert artifact["incremental_rule"]["verdict"] == "PASSES vs both comparators"
    assert artifact["supersedes"].endswith("claim WITHDRAWN)")


def test_scorer_reconstructs_logit_and_never_imputes() -> None:
    f = _features()
    internal = {
        "age": f["age"],
        "tumor_size_mm": f["tumor_size_mm"],
        "nodes_positive": f["nodes_positive"],
        "grade": f["grade"],
        "er_pos": f["er_positive"],
        "pr_pos": f["pr_positive"],
        "her2_pos": f["her2_positive"],
    }
    result = score_breast_dss(internal)
    assert result.model_name == ARTIFACT_NAME
    assert math.isclose(
        result.score, 1.0 / (1.0 + math.exp(-result.logit)), rel_tol=0, abs_tol=1e-15
    )
    assert math.isclose(
        result.logit,
        -0.24702382790127209 + sum(result.term_contributions.values()),
        rel_tol=0,
        abs_tol=1e-12,
    )
    with pytest.raises(BreastDssArbiterError, match="missing"):
        score_breast_dss({k: v for k, v in internal.items() if k != "age"})


def test_biopsy_endpoint_matches_frozen_scorer_for_clinical_profile(
    client: TestClient,
) -> None:
    expected = score_breast_dss({
        "age": 58.0,
        "tumor_size_mm": 22.0,
        "nodes_positive": 1,
        "grade": 2,
        "er_pos": True,
        "pr_pos": True,
        "her2_pos": False,
    })
    response = client.post("/v1/biopsy/analyze", json={"dss_features": _features()})
    assert response.status_code == 200, response.text
    body = response.json()
    dss = body["dss_prognosis"]
    assert dss["model_name"] == "breast_dss_arbiter_v3_metabric"
    assert dss["model_state"] == "frozen"
    assert dss["n_training"] == 1375
    assert dss["events"] == 601
    assert dss["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert dss["disease_specific_mortality_score"] == pytest.approx(expected.score)
    assert dss["logit"] == pytest.approx(expected.logit)
    assert dss["endpoint_label"] == "disease-specific mortality"
    assert set(dss["term_contributions"]) == {
        "age", "tumor_size_mm", "nodes_positive", "grade",
        "er_pos", "pr_pos", "her2_pos",
    }
    assert any("explicit_features:no_imputation" in w for w in body["warnings"])


def test_more_advanced_profile_scores_above_small_node_negative_profile(
    client: TestClient,
) -> None:
    small_node_negative = {
        "age": 58.0,
        "tumor_size_mm": 10.0,
        "nodes_positive": 0,
        "grade": 1,
        "er_positive": True,
        "pr_positive": True,
        "her2_positive": False,
    }
    larger_node_positive = {
        **small_node_negative,
        "tumor_size_mm": 65.0,
        "nodes_positive": 8,
        "grade": 3,
    }
    low = client.post(
        "/v1/biopsy/analyze",
        json={"dss_features": small_node_negative},
    )
    high = client.post(
        "/v1/biopsy/analyze",
        json={"dss_features": larger_node_positive},
    )
    assert low.status_code == high.status_code == 200
    low_score = low.json()["dss_prognosis"]["disease_specific_mortality_score"]
    high_score = high.json()["dss_prognosis"]["disease_specific_mortality_score"]
    assert high_score > low_score


def test_biopsy_endpoint_rejects_incomplete_dss_vector(client: TestClient) -> None:
    bad = _features()
    del bad["nodes_positive"]
    response = client.post("/v1/biopsy/analyze", json={"dss_features": bad})
    assert response.status_code == 422


def test_therapy_schema_deprecates_alias_and_response_emits_only_prognosis_name(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["TherapyRequest"]
    legacy = request_schema["properties"]["legacy_therapy_benefit_model_name"]
    assert legacy["deprecated"] is True
    assert "therapy_prognosis_metabric_v1" in legacy["description"]

    # Deprecated alias alone is not a usable therapy input — fail closed (400).
    empty = client.post(
        "/v1/therapy/reason",
        json={"legacy_therapy_benefit_model_name": "therapy_benefit_metabric_v1"},
    )
    assert empty.status_code == 400, empty.text

    # With an explicit triage panel, the alias remaps and the response never
    # re-emits the retired benefit-model name.
    response = client.post(
        "/v1/therapy/reason",
        json={
            "legacy_therapy_benefit_model_name": "therapy_benefit_metabric_v1",
            "therapy_features": {
                "histology": "invasive_ductal",
                "grade": 2,
                "er_positive": True,
                "pr_positive": True,
                "her2_positive": False,
                "ki67_pct": 15.0,
                "tumor_size_mm": 20.0,
                "lymph_nodes_pos": 0,
                "brca_pathogenic": False,
                "age_years": 55.0,
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prognostic_model_name"] == "therapy_prognosis_metabric_v1"
    assert body["prognostic_model_executed"] is False
    assert body["prognostic_score"] is None
    assert "Single-arm METABRIC prognosis" in body["prognostic_uncertainty"]
    assert "cannot identify a causal treatment effect" in body["prognostic_uncertainty"]
    assert "therapy_benefit_metabric_v1" not in json.dumps(body)
    assert "deprecated_prognostic_model_alias_remapped" in body["warnings"]
