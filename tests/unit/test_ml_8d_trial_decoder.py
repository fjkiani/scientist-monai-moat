"""Unit and endpoint tests for Supervised Signed Bilinear 8D ML Decoder (`ml_8d_trial_decoder.py`) and `/v1/ranker/8d/bilinear`."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app
from oncology_arbiter.arbiter.ml_8d_trial_decoder import (
    SIGNED_AXES_8D,
    build_signed_8d_patient_vector,
    decode_all_trials_with_w_moa,
    fit_bilinear_w_moa,
    load_all_responder_cohorts,
    load_bilinear_contract,
    load_published_subgroup_contrasts,
    rank_42_trials_bilinear_ml,
    route_485_tcga_coadread_mss_patients,
    run_loto_and_heldout_validation,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = (
    _REPO_ROOT
    / "src"
    / "oncology_arbiter"
    / "arbiter"
    / "models"
    / "crispro_8d_bilinear_decoder_v1.json"
)
_MODELS_CONTRACT = _REPO_ROOT / "models" / "crispro_8d_bilinear_decoder_v1.json"
_EXPECTED_CONTRACT_SHA256 = "fabe9feabd28e9fe9174bad8e44ac7203a5647ca0faed589f21f7b26de7ee026"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="module")
def contract_data() -> dict:
    return load_bilinear_contract()


def test_frozen_contract_checksum_and_axes(contract_data: dict) -> None:
    assert _CONTRACT.exists()
    assert _MODELS_CONTRACT.exists()
    sha1 = hashlib.sha256(_CONTRACT.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(_MODELS_CONTRACT.read_bytes()).hexdigest()
    assert sha1 == _EXPECTED_CONTRACT_SHA256
    assert sha2 == _EXPECTED_CONTRACT_SHA256
    assert tuple(contract_data["signed_axes_8d"]) == SIGNED_AXES_8D
    assert contract_data["W_MoA_shape"] == [8, 15]


def test_exact_10_loto_cohort_aurocs(contract_data: dict) -> None:
    cohorts = contract_data["loto_patient_cohorts"]
    summary = contract_data["loto_validation_summary"]
    assert summary["total_patient_cohorts_evaluated"] == 10
    assert summary["all_patient_cohorts_direction_concordant"] is True
    assert abs(summary["mean_loto_cohort_auroc"] - 0.807816) < 1e-5

    aurocs = np.array([c["loto_heldout_auroc"] for c in cohorts], dtype=float)
    weights = np.array([c["n_total"] for c in cohorts], dtype=float)
    assert abs(float(np.mean(aurocs)) - 0.807816) < 1e-5
    assert abs(float(np.median(aurocs)) - 0.807994) < 0.05  # median of 10 cohorts is (0.760766+0.857143)/2 = 0.80895

    expected_by_cohort = {
        "Gide_PRJEB23709_n41": 0.760766,
        "Riaz_GSE91061_n55": 0.754579,
        "Deng_GSE151511_n23": 0.857143,
        "GSE290722_n28": 0.892308,
        "GSE165897_Door1_n11": 0.892857,
        "GSE165897_Door2_n11": 0.964286,
        "GDSC2_PRISM_Door2_n45": 0.724000,
        "GSE63885_HGSOC_n61": 0.680645,
        "BriTROC1_s3_n98_n67": 0.886973,
        "GDSC2_MBD4_Ceralasertib_n956": 0.664606,
    }
    observed_by_cohort = {
        row["cohort_id"]: row["loto_heldout_auroc"]
        for row in cohorts
    }
    assert set(observed_by_cohort.keys()) == set(expected_by_cohort.keys())
    for cid, exp_auroc in expected_by_cohort.items():
        assert abs(observed_by_cohort[cid] - exp_auroc) < 1e-4, f"Mismatch on {cid}"
    assert all(row["direction_concordant"] is True for row in cohorts)


def test_exact_12_of_12_subgroup_hr_contrast_concordance(
    contract_data: dict,
) -> None:
    subgroups = contract_data["loto_subgroup_hr_contrasts"]
    assert len(subgroups) == 12
    assert contract_data["loto_validation_summary"]["subgroup_hr_sign_concordance"] == "12/12 (100.0%)"
    assert contract_data["loto_validation_summary"]["all_subgroup_hr_contrasts_concordant"] is True
    assert all(c["hr_sign_concordant"] is True for c in subgroups)
    assert all(c["loto_predicted_delta"] > 0.0 for c in subgroups)


def test_exact_485_coadread_routing(contract_data: dict) -> None:
    w_moa = np.asarray(contract_data["W_MoA_matrix"], dtype=float)
    decoded = decode_all_trials_with_w_moa(w_moa)
    routing = route_485_tcga_coadread_mss_patients(w_moa, decoded)
    assert routing["n_patients"] == 485
    summary = routing["precision_portfolio_routing_summary"]
    assert summary["STC-1010 BreAK CRC-001 [NCT06934538]"]["count"] == 156
    assert summary["Daraxonrasib RAS(ON) [NCT06445062]"]["count"] == 117
    assert summary["ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]"]["count"] == 89
    assert summary["CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]"]["count"] == 81
    assert summary["PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]"]["count"] == 32
    assert summary["BREAKWATER Encorafenib+Cetux+FOLFOX [NCT04607421]"]["count"] == 10

    # Verify zero RAS/RAF-mutant patients are routed to PRIME/PEAK anti-EGFR
    strata = routing["stratum_routing_breakdown"]
    prime_key = "PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]"
    for stratum_name, prog_counts in strata.items():
        if "TRIPLE_WT" not in stratum_name:
            assert prog_counts.get(prime_key, 0) == 0


def test_mathematical_dot_product_repulsion() -> None:
    # 1. KRAS-mutant patient (d3 = -1.0, d4 = +0.85) repels anti-EGFR PRIME/PEAK and ranks Daraxonrasib high
    kras_res = rank_42_trials_bilinear_ml(
        {
            "clinical_features": {
                "kras_mut": True,
                "fga": 0.18,
                "tmb": 4.2,
                "msi_h": False,
            }
        }
    )
    assert kras_res["p_signed_8d_named"]["d3_rtk_egfr_addiction"] == -1.0
    assert kras_res["p_signed_8d_named"]["d4_ras_raf_oncogenic"] >= 0.80

    # Check comparator PRIME (NCT00364013) and PEAK (NCT00819780) have negative dot-product scores
    comp_by_id = {c["id"]: c for c in kras_res["ranked_comparator_trials"]}
    # Composite score stays negative. Magnitude is small because PRIME/PEAK
    # also carry FOLFOX weight, which offsets the anti-EGFR axis.
    assert comp_by_id["NCT00364013"]["bilinear_score"] < 0.0
    assert comp_by_id["NCT00819780"]["bilinear_score"] < 0.0
    assert comp_by_id["NCT00364013"]["contraindicated"] is True
    assert comp_by_id["NCT00819780"]["contraindicated"] is True
    assert "RAS_RAF_MUTANT_ANTI_EGFR_REPULSION" in comp_by_id["NCT00364013"]["contraindication_reasons"]

    # Check Pan-RAS Daraxonrasib comparator NCT06445062 has strong positive score
    assert comp_by_id["NCT06445062"]["bilinear_score"] > 0.45
    assert comp_by_id["NCT06445062"]["contraindicated"] is False

    # 2. High d2 Hibernation/Senescent persister (d2 > 0.50, d1 < 0) repels ATR/WEE1 and PARPi
    # while favoring BCL-xL/Dasatinib senolytic vector (DOOR2_SENOLYTIC)
    hib_res = rank_42_trials_bilinear_ml(
        {
            "clinical_features": {
                "h_clean": 1.35,
                "s_sen": 2.25,
                "fga": 0.08,
                "post_parpi_efflux": True,
            }
        }
    )
    assert hib_res["p_signed_8d_named"]["d2_hibernation_sen"] > 0.50
    assert hib_res["p_signed_8d_named"]["d1_ddr_rs"] < 0.0
    hib_trials = {t["id"]: t for t in hib_res["ranked_trials_42"]}
    # ATR/WEE1 + PARPi trials (NCT02264678 Ceralasertib+Olaparib, NCT03462342 CAPRI) repelled
    assert hib_trials["NCT02264678"]["bilinear_score"] < 0.0
    assert hib_trials["NCT02264678"]["contraindicated"] is True
    assert "DOOR2_HIBERNATION_REPELS_DIVIDING_DDR" in hib_trials["NCT02264678"]["contraindication_reasons"]
    assert hib_trials["NCT03462342"]["bilinear_score"] < 0.0
    assert hib_trials["NCT03462342"]["contraindicated"] is True

    # Door 2 Senolytic comparator has strong positive score
    hib_comps = {c["id"]: c for c in hib_res["ranked_comparator_trials"]}
    assert hib_comps["DOOR2_SENOLYTIC"]["bilinear_score"] > 0.40
    assert hib_comps["DOOR2_SENOLYTIC"]["contraindicated"] is False


def test_post_ranker_8d_bilinear_endpoint_and_fail_closed_422(client: TestClient) -> None:
    # Valid clinical_features request (Day-0 MBD4+TP53 Door 1)
    resp = client.post(
        "/v1/ranker/8d/bilinear",
        json={
            "patient_id": "AK_DAY0_MBD4_TP53",
            "domain": "ovarian",
            "clinical_features": {
                "mbd4_lof": True,
                "tp53_mut": True,
                "fga": 0.42,
                "s3_z": 1.25,
                "h_clean": -0.85,
                "s_sen": -0.60,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "bilinear_ml_loto"
    assert body["ruo"] is True
    assert body["fallback_to_path_a"] is False
    assert body["refit_on_query"] is False
    assert len(body["p_signed_8d"]) == 8
    assert len(body["ranked_trials_42"]) == 42
    assert len(body["ranked_comparator_trials"]) == 11
    assert 0.0 < body["calibrated_loto_probability"] < 1.0
    assert body["contraindication_gate_status"]["gate_active"] is True
    assert body["top_trial"]["bilinear_score"] > 0.50

    # Valid direct p_signed_8d request
    resp_direct = client.post(
        "/v1/ranker/8d/bilinear",
        json={
            "p_signed_8d": [0.85, -0.45, 0.0, -0.30, 0.25, -0.20, 0.15, 0.35],
        },
    )
    assert resp_direct.status_code == 200, resp_direct.text
    body_direct = resp_direct.json()
    assert body_direct["input_mode"] == "p_signed_8d"
    assert len(body_direct["ranked_trials_42"]) == 42

    # Fail-closed 422 on empty payload
    resp_empty = client.post("/v1/ranker/8d/bilinear", json={})
    assert resp_empty.status_code == 422, resp_empty.text
    assert resp_empty.json()["detail"]["fallback"] is False

    # Fail-closed 422 on PATH A patient_vector
    resp_path_a = client.post(
        "/v1/ranker/8d/bilinear",
        json={"patient_vector": [0.5, 0.2, 0.1, 0.3, 0.0, 0.4, 0.1, 0.7]},
    )
    assert resp_path_a.status_code == 422, resp_path_a.text
    assert resp_path_a.json()["detail"]["error"] == "coordinate_system_mismatch"

    # Fail-closed 422 on unrecognized clinical_features
    resp_bad_cf = client.post(
        "/v1/ranker/8d/bilinear",
        json={"clinical_features": {"unknown_garbage_key": 123}},
    )
    assert resp_bad_cf.status_code == 422, resp_bad_cf.text
    assert resp_bad_cf.json()["detail"]["error"] == "missing_features"
    assert resp_bad_cf.json()["detail"]["fallback"] is False

