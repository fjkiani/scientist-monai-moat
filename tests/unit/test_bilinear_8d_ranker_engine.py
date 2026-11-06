"""Frozen bilinear_ml_loto engine vs PATH A default. Fixtures come from the JSON contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app
from oncology_arbiter.arbiter.ml_8d_trial_decoder import (
    score_bilinear_ml_loto,
)
from oncology_arbiter.arbiter.pathway_8d_ranker import AXES_8D

_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "oncology_arbiter"
    / "arbiter"
    / "models"
    / "crispro_8d_bilinear_decoder_v1.json"
)
_MODELS_COPY = (
    Path(__file__).resolve().parents[2] / "models" / "crispro_8d_bilinear_decoder_v1.json"
)

_SIGNED = [
    "d1_ddr_rs",
    "d2_hibernation_sen",
    "d3_rtk_egfr_addiction",
    "d4_ras_raf_oncogenic",
    "d5_io_effector",
    "d6_stroma_myeloid_barrier",
    "d7_surface_adc_payload",
    "d8_angio_metabolic_safety",
]


def _contract() -> dict:
    return json.loads(_CONTRACT.read_text())


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_contract_signed_axes_and_path_a_rss_axis() -> None:
    data = _contract()
    assert data["signed_axes_8d"] == _SIGNED
    assert data["W_MoA_shape"] == [8, 15]
    assert len(data["moa_classes_15"]) == 15
    assert data["moa_classes_15"][12] == "BCLxL_Dasatinib_Senolytic"
    assert AXES_8D == ("ddr", "mapk", "pi3k", "vegf", "her2", "io", "efflux", "rss")
    assert AXES_8D[-1] == "rss"
    assert tuple(data["signed_axes_8d"]) != AXES_8D
    assert _MODELS_COPY.read_bytes() == _CONTRACT.read_bytes()


def test_bilinear_scores_json_door2_vector_without_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _contract()
    cohort = next(
        c for c in data["loto_patient_cohorts"] if c["cohort_id"] == "GSE165897_Door2_n11"
    )
    p = cohort["learned_loto_vector_8d"]
    assert len(p) == 8

    def _refuse_refit(*_args, **_kwargs):
        raise AssertionError("query scoring must not refit W_MoA")

    monkeypatch.setattr(
        "oncology_arbiter.arbiter.ml_8d_trial_decoder.fit_bilinear_w_moa",
        _refuse_refit,
    )
    body = score_bilinear_ml_loto(
        {
            "p_signed_8d": p,
            "moa_weights": {"BCLxL_Dasatinib_Senolytic": 1.0},
        }
    )
    col = data["moa_classes_15"].index("BCLxL_Dasatinib_Senolytic")
    trial = np.asarray(data["W_MoA_matrix"], dtype=float)[:, col]
    expected = float(np.dot(np.asarray(p, dtype=float), trial) / np.linalg.norm(trial))
    assert body["engine"] == "bilinear_ml_loto"
    assert body["refit_on_query"] is False
    assert body["fallback_to_path_a"] is False
    assert body["coordinate_system"] == "signed_bilinear_8d"
    assert abs(body["score"] - expected) < 1e-9


def test_default_engine_stays_path_a_gated(client: TestClient) -> None:
    resp = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "crc",
            "mode": "iii_c",
            "clinical_features": {
                "tmb": 3.3,
                "hrd": "HRD-Intermediate",
                "fga": 0.2671,
                "kras_mut": True,
                "immune_z": 0.45,
                "gate2_hot": True,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "path_a_gated"
    assert body["engine"] != "bilinear_ml_loto"


def test_bilinear_missing_features_is_structured_error(client: TestClient) -> None:
    resp = client.post("/v1/ranker/8d/score", json={"engine": "bilinear_ml_loto"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["engine"] == "bilinear_ml_loto"
    assert detail["fallback"] is False
    assert "p_signed_8d" in detail["missing"]
    assert "top_trial" not in detail
