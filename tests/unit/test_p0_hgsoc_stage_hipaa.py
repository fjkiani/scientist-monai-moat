"""P0: HGSOC must 501, incomplete stage panels must 422, HIPAA must redact."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.hipaa_redact import redact_phi

_SCREENING = {
    "birads": "BI_RADS_4",
    "density": "C_heterogeneously_dense",
    "prior_biopsy_history": True,
    "family_history_first_degree": False,
    "brca_status_known_pathogenic": False,
    "age_norm": 55.0,
    "years_since_last_mammo_norm": 2.0,
}
_THERAPY = {
    "histology": "invasive_ductal",
    "grade": "3",
    "er_status_positive": True,
    "pr_status_positive": False,
    "her2_status_positive": False,
    "node_status_positive": True,
    "brca_status_known_pathogenic": False,
    "ki67_norm": 40.0,
    "tumor_size_norm": 35.0,
    "age_at_diagnosis_norm": 45.0,
}
_BIOPSY = {
    "lesion_type": "mass_spiculated",
    "size_norm": 15.0,
    "growth_delta_norm": 3.0,
    "prior_biopsy_benign_at_site": False,
    "family_history_first_degree": True,
    "brca_status_known_pathogenic": False,
    "us_correlate_hypoechoic_mass": True,
    "us_correlate_simple_cyst": False,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_hgsoc_case_full_is_501_and_skips_breast_therapy(client: TestClient) -> None:
    response = client.post(
        "/v1/case/full?cancer=hgsoc",
        json={
            "run_co_scientist": False,
            "therapy_features": {
                "histology": "invasive_ductal",
                "er_positive": True,
                "pr_positive": True,
                "her2_positive": False,
            },
        },
    )
    assert response.status_code == 501, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "hgsoc_case_full_not_implemented"
    assert detail["cancer"] == "hgsoc"
    assert "p_positive" not in response.text
    assert "dss_prognosis" not in response.text
    assert "therapy_triage" not in response.text


def test_stage_empty_panel_is_422(client: TestClient) -> None:
    for route in (
        "/v1/stage/screening/score",
        "/v1/stage/biopsy/score",
        "/v1/stage/therapy/score",
    ):
        response = client.post(route, json={"features": {}})
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["error"] == "empty_arbiter_features"
        assert "p_positive" not in response.text


def test_stage_therapy_missing_receptors_is_422_not_negative(client: TestClient) -> None:
    partial = dict(_THERAPY)
    partial.pop("er_status_positive")
    partial.pop("pr_status_positive")
    partial.pop("her2_status_positive")
    response = client.post("/v1/stage/therapy/score", json={"features": partial})
    assert response.status_code == 422, response.text
    missing = response.json()["detail"]["missing"]
    assert "er_status_positive" in missing
    assert "pr_status_positive" in missing
    assert "her2_status_positive" in missing
    assert "p_positive" not in response.text


def test_stage_screening_missing_birads_is_422(client: TestClient) -> None:
    partial = dict(_SCREENING)
    partial.pop("birads")
    response = client.post("/v1/stage/screening/score", json={"features": partial})
    assert response.status_code == 422, response.text
    assert "birads" in response.json()["detail"]["missing"]


def test_stage_complete_panels_score_posted_values(client: TestClient) -> None:
    screening = client.post("/v1/stage/screening/score", json={"features": _SCREENING})
    biopsy = client.post("/v1/stage/biopsy/score", json={"features": _BIOPSY})
    er_pos = client.post("/v1/stage/therapy/score", json={"features": _THERAPY})
    er_neg = dict(_THERAPY)
    er_neg["er_status_positive"] = False
    er_neg_resp = client.post("/v1/stage/therapy/score", json={"features": er_neg})
    assert screening.status_code == 200, screening.text
    assert biopsy.status_code == 200, biopsy.text
    assert er_pos.status_code == 200, er_pos.text
    assert er_neg_resp.status_code == 200, er_neg_resp.text
    assert screening.json()["arbiter_score"]["model_name"] == "screening_arbiter_v1"
    assert biopsy.json()["arbiter_score"]["model_name"] == "biopsy_arbiter_v1"
    assert er_pos.json()["arbiter_score"]["model_name"] == "therapy_arbiter_v1"
    assert (
        er_pos.json()["arbiter_score"]["p_positive"]
        != er_neg_resp.json()["arbiter_score"]["p_positive"]
    )


def test_hipaa_redactor_strips_patient_id_and_name() -> None:
    raw = {
        "patient_id": "MBD4-LOF-DEMO-01",
        "name": "Ada Patient",
        "synthetic_lethality": {
            "patient_id": "MBD4-LOF-DEMO-01",
            "note": "see MBD4-LOF-DEMO-01 in the chart for Ada Patient",
        },
    }
    redacted = redact_phi(raw)
    blob = json.dumps(redacted)
    assert redacted["patient_id"] == "[REDACTED]"
    assert redacted["name"] == "[REDACTED]"
    assert redacted["synthetic_lethality"]["patient_id"] == "[REDACTED]"
    assert "MBD4-LOF-DEMO-01" not in blob
    assert "Ada Patient" not in blob


def test_hipaa_mode_bundle_does_not_echo_phi(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "oncology_arbiter"
        / "api"
        / "static"
        / "demo_samples"
        / "ak_mbd4_lof_case.json"
    )
    posted = json.loads(bundle_path.read_text())
    posted = copy.deepcopy(posted)
    posted["name"] = "Ada Patient"
    monkeypatch.setenv("HIPAA_MODE", "true")
    response = client.post("/v1/tumor_board/bundle", json=posted)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provenance"]["model_state"] == "loaded_hipaa_redactor"
    assert "MBD4-LOF-DEMO-01" not in response.text
    assert "Ada Patient" not in response.text
