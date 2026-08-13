"""Regression gates for the retired ovarian prognostic arbiter."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_ovarian_retirement.py"
SPEC = importlib.util.spec_from_file_location("ovarian_retirement_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_corrected_cohort_excludes_unknown_horizon_outcomes() -> None:
    rows = [
        {"case_id": "a", "vital": "Dead", "days_to_birth": -20000, "days_to_death": 100, "days_to_follow_up": None, "figo": "III"},
        {"case_id": "b", "vital": "Dead", "days_to_birth": -21000, "days_to_death": 900, "days_to_follow_up": None, "figo": "IV"},
        {"case_id": "c", "vital": "Alive", "days_to_birth": -22000, "days_to_death": None, "days_to_follow_up": 1000, "figo": "I_II"},
        {"case_id": "d", "vital": "Alive", "days_to_birth": -22000, "days_to_death": None, "days_to_follow_up": 100, "figo": "III"},
    ]
    corrected, counts = MODULE.build_corrected_cohort(rows)
    assert [row["y"] for row in corrected] == [1, 0, 0]
    assert counts["secondary_drop_censored_before_horizon"] == 1


def test_retirement_source_cannot_emit_patient_probability() -> None:
    source = inspect.getsource(create_app)
    start = source.index("def _ovarian_retirement")
    end = source.index("def _run_medgemma_co_scientist", start)
    retirement_source = source[start:end]
    prohibited_keys = ('"p_positive":', '"risk_bucket":', '"driving_feature":', '"probability":')
    assert all(key not in retirement_source for key in prohibited_keys)


def test_hgsoc_dynamic_response_has_no_patient_score(monkeypatch) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    response = TestClient(create_app()).post(
        "/v1/tumor_board/dynamic",
        json={"cancer": "hgsoc", "run_co_scientist": False},
    )
    assert response.status_code == 200
    ovarian = response.json()["ovarian_arbiter"]
    assert ovarian["model_state"] == "retired"
    assert not ({"p_positive", "probability", "risk_bucket", "driving_feature"} & ovarian.keys())
