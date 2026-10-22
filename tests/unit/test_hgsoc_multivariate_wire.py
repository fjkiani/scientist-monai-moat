"""HGSOC multivariate platinum-resistance v2 — OOF honesty + case/full unlock."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("AUTH_MODE", "off")
os.environ.setdefault("ONCOLOGY_ARBITER_AUTH_MODE", "off")

from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.deps import compute_models_loaded
from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.arbiter import stage_resistance_multivariate_wiring as mvr

MOAT = Path(__file__).resolve().parents[2]
UNI = 0.6836


def test_artifact_sha_and_proof_honesty():
    proof = mvr.load_proof_receipt()
    assert proof["honest_n"] == 75
    assert proof["artifact_sha256"] == mvr.ARTIFACT_SHA256
    assert proof["beats_univariate_loocv"] is False
    uni = next(c for c in proof["comparison_oof"] if c["model"] == "univariate_mfap4")
    assert abs(uni["cv5"]["auroc"] - 0.6801) < 0.01
    assert uni["loocv"]["auroc"] < UNI + 0.01  # OOF uni ~0.65, in-sample proof 0.6836
    multi = [c for c in proof["comparison_oof"] if c["model"] != "univariate_mfap4"]
    assert all(c["loocv"]["auroc"] < uni["loocv"]["auroc"] + 1e-9 for c in multi) or True
    # Strong honesty: no multivariate LOOCV exceeds univariate LOOCV
    assert max(c["loocv"]["auroc"] for c in multi) < uni["loocv"]["auroc"]
    assert proof["top_features"][0]["feature"] == "EMT_MFAP4"
    bundle = mvr.load_multivariate_bundle()
    assert bundle["n_training"] == 75
    assert float(bundle["loocv_auroc"]) < 0.75


def test_models_loaded_badge():
    assert compute_models_loaded()["ovarian_resistance_multivariate_v2"] == ModelState.LOADED


def test_case_full_hgsoc_200_with_features():
    """Best-OOF: serve MFAP4 v1; multivariate joblib stays ablation-only."""
    from oncology_arbiter.arbiter.stage_mfap4_wiring import (
        ARTIFACT_SHA256 as MFAP4_SHA,
    )

    client = TestClient(create_app())
    features = {
        "EMT_MFAP4": 8.17,
        "EMT_EFEMP1": 8.0,
        "EMT_VIM": 9.0,
        "EMT_CDH1": 7.5,
        "EMT_SNAI1": 5.0,
        "DDR_BRCA1": 7.0,
        "DDR_BRCA2": 6.5,
        "DDR_RAD51": 7.2,
        "DDR_PARP1": 8.1,
        "DDR_ATM": 7.0,
        "DDR_ATR": 7.5,
        "ratio_mes_epi": (8.17 + 9.0 + 8.0) - 7.5,
        "grade_ord": 3.0,
        "figo_ord": 3.0,
        "residual_ord": 1.0,
        "brca1_mut": 0.0,
        "hist_serous": 1.0,
        "hist_undiff": 0.0,
    }
    resp = client.post(
        "/v1/case/full?cancer=hgsoc",
        json={"resistance_features": features},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cancer"] == "hgsoc"
    oa = body["ovarian_arbiter"]
    assert oa["model_name"] == "ovarian_mfap4_arbiter_v1"
    assert oa["production_scorer"] == "ovarian_mfap4_arbiter_v1"
    assert oa["model_state"] == "loaded"
    assert 0.0 < oa["p_positive"] < 1.0
    assert oa["beats_univariate_loocv"] is False
    assert abs(float(oa["mfap4_permutation_importance"]) - 0.1377) < 1e-3
    assert oa["artifact_sha256"] == MFAP4_SHA
    assert oa["ablation_artifact_sha256"] == mvr.ARTIFACT_SHA256
    assert oa["provenance"]["beats_univariate_loocv"] is False


def test_case_full_hgsoc_200_empty_body_identity():
    client = TestClient(create_app())
    resp = client.post("/v1/case/full?cancer=hgsoc", json={})
    assert resp.status_code == 200, resp.text
    oa = resp.json()["ovarian_arbiter"]
    assert oa["model_state"] == "loaded"
    assert oa["model_name"] == "ovarian_mfap4_arbiter_v1"
    assert oa["p_positive"] is None
    assert oa["beats_univariate_loocv"] is False


def test_dynamic_hgsoc_no_longer_501():
    client = TestClient(create_app())
    resp = client.post(
        "/v1/tumor_board/dynamic",
        json={"cancer": "hgsoc", "resistance_features": {"EMT_MFAP4": 8.17}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cancer"] == "hgsoc"
    assert body["ovarian_arbiter"]["model_name"] == "ovarian_mfap4_arbiter_v1"
