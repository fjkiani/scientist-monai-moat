"""HGSOC MFAP4 platinum-resistance wire — GSE63885 honest n=75."""
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
from oncology_arbiter.arbiter import load_arbiter, mfap4_arbiter
from oncology_arbiter.arbiter import stage_mfap4_wiring as mfap4

PROOF_AUROC = 0.6836
RETRACTED = 0.7625
MOAT = Path(__file__).resolve().parents[2]


def test_artifact_identity_and_auroc_matches_proof():
    model = load_arbiter("mfap4")
    assert model.model_name == "ovarian_mfap4_arbiter_v1"
    assert mfap4_arbiter().model_name == "ovarian_mfap4_arbiter_v1"
    assert model.n_training == 75
    held = float(model.performance["held_out_test_auroc"])
    assert abs(held - PROOF_AUROC) < 5e-4
    assert held < 0.75
    assert held != RETRACTED
    assert abs(held - RETRACTED) > 0.05
    assert model.positive_class == "platinum_resistant"
    assert "mfap4_expr" in model.feature_encodings


def test_proof_receipt_on_disk():
    proof = mfap4.load_proof_receipt()
    assert proof["status"] == "LOCKED"
    assert proof["probe"] == "212713_at"
    assert proof["honest"]["n"] == 75
    assert abs(float(proof["honest"]["auroc"]) - PROOF_AUROC) < 5e-4
    assert proof["retracted_na_as_sensitive"]["auroc"] == RETRACTED
    assert proof["artifact_sha256"] == mfap4.ARTIFACT_SHA256
    metrics = json.loads(
        (MOAT / "docs/proofs/gse63885_mfap4_metrics.json").read_text()
    )
    assert metrics["artifact_sha256"] == mfap4.ARTIFACT_SHA256


def test_score_monotone_higher_expr_higher_p_resistant():
    model = mfap4.load_stage_mfap4_arbiter()
    low = model.score({"mfap4_expr": 7.0})
    high = model.score({"mfap4_expr": 9.0})
    assert high.p_positive > low.p_positive
    # Resistant mean ~8.17 should lean higher than sensitive mean ~7.68
    r = model.score({"mfap4_expr": 8.1705})
    s = model.score({"mfap4_expr": 7.6807})
    assert r.p_positive > s.p_positive


def test_models_loaded_badge_mfap4():
    badges = compute_models_loaded()
    assert badges["ovarian_mfap4_v1"] == ModelState.LOADED


def test_hgsoc_mfap4_routes():
    client = TestClient(create_app())

    health = client.get("/v1/hgsoc/mfap4/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["probe"] == "212713_at"
    assert body["n_training"] == 75
    assert abs(float(body["auroc"]) - PROOF_AUROC) < 5e-4
    assert body["case_full"] is False

    metrics = client.get("/v1/hgsoc/mfap4/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["data"]["honest"]["n"] == 75

    scored = client.post(
        "/v1/hgsoc/mfap4/score",
        json={"mfap4_expr": 8.1705},
    )
    assert scored.status_code == 200
    arb = scored.json()["arbiter_score"]
    assert arb["model_name"] == "ovarian_mfap4_arbiter_v1"
    assert arb["n_training"] == 75
    assert arb["positive_class"] == "platinum_resistant"
    assert 0.0 < arb["p_positive"] < 1.0

    # Empty / missing feature must 422 — no toy default
    bad = client.post("/v1/hgsoc/mfap4/score", json={})
    assert bad.status_code == 422


def test_case_full_hgsoc_unlocked_200():
    """Best-OOF unlock: case/full?cancer=hgsoc returns HTTP 200 via MFAP4 v1."""
    client = TestClient(create_app())
    resp = client.post("/v1/case/full?cancer=hgsoc", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cancer"] == "hgsoc"
    assert body["ovarian_arbiter"]["model_name"] == "ovarian_mfap4_arbiter_v1"
    assert body["ovarian_arbiter"]["beats_univariate_loocv"] is False
