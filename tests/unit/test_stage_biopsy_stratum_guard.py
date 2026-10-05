"""stage-biopsy stratum guard: calc-only morphology must fail closed (422)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app
from oncology_arbiter.arbiter.stage_biopsy_wiring import (
    UNVALIDATED_LESION_STRATUM_CODE,
    UNVALIDATED_LESION_STRATUM_MESSAGE,
    UnvalidatedLesionStratumError,
    enforce_validated_lesion_stratum,
)

_ROUTE = "/v1/stage/biopsy/score"

_MASS_PANEL = {
    "lesion_type": "mass_spiculated",
    "size_norm": 15.0,
    "growth_delta_norm": 3.0,
    "prior_biopsy_benign_at_site": False,
    "family_history_first_degree": True,
    "brca_status_known_pathogenic": False,
    "us_correlate_hypoechoic_mass": True,
    "us_correlate_simple_cyst": False,
}


def test_enforce_rejects_calc_lesion_type_string() -> None:
    with pytest.raises(UnvalidatedLesionStratumError):
        enforce_validated_lesion_stratum({"lesion_type": "calcification_pleomorphic"})


def test_enforce_allows_mass_lesion_type_string() -> None:
    enforce_validated_lesion_stratum({"lesion_type": "mass_spiculated"})


def test_enforce_allows_mixed_calc_and_mass_one_hot() -> None:
    enforce_validated_lesion_stratum({
        "lesion_type_calcification_amorphous": 1.0,
        "lesion_type_mass_circumscribed": 1.0,
    })


def test_stage_biopsy_api_calc_only_returns_422(client: TestClient) -> None:
    resp = client.post(
        _ROUTE,
        json={"features": {"lesion_type": "calcification_amorphous"}},
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == UNVALIDATED_LESION_STRATUM_CODE
    assert detail["message"] == UNVALIDATED_LESION_STRATUM_MESSAGE


def test_stage_biopsy_api_mass_panel_allowed(client: TestClient) -> None:
    resp = client.post(_ROUTE, json={"features": _MASS_PANEL})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["arbiter_score"]["model_name"] == "biopsy_arbiter_v1"
    assert body["arbiter_score"]["n_training"] > 0


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())
