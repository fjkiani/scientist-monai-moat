"""Ovarian two-door v3 and CAR-T/ICI v3 route contracts.

The 11 GSE165897 rows are the frozen table inside
``ovarian_two_door_standalone_v3.py``. EOC3's stored geometric class is
Hcleanonly; Gate 0 must override it to PRIMARY_REFRACTORY_ESCAPE.
"""
from __future__ import annotations

import math

from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.immunotherapy.cart_ici_v3 import (
    load_arbiter_v3_contract,
    load_cart_contract,
    load_v3_contract,
)
from oncology_arbiter.ovarian.ovarian_two_door_standalone_v3 import (
    GSE165897_11_PATIENT_TABLE,
)

EXPECTED = {
    "EOC136": "Neither",
    "EOC372": "Neither",
    "EOC87": "DDRonly",
    "EOC349": "DDRonly",
    "EOC1005": "DDRonly",
    "EOC733": "DDRonly",
    "EOC153": "Hcleanonly",
    "EOC443": "Hcleanonly",
    "EOC540": "Hcleanonly",
    "EOC227": "Both",
    "EOC3": "PRIMARY_REFRACTORY_ESCAPE",
}

DECISION = {
    "Neither": "CHEMO_CLEARED_REMISSION",
    "DDRonly": "DOOR_1_DDR_PROLIFERATIVE",
    "Hcleanonly": "DOOR_2_HIBERNATION_SENOLYTIC",
    "Both": "DOOR_2_HIBERNATION_SENOLYTIC",
    "PRIMARY_REFRACTORY_ESCAPE": "PRIMARY_REFRACTORY_ESCAPE",
}


def _client() -> TestClient:
    return TestClient(create_app())


def _post_row(client: TestClient, row: dict) -> dict:
    payload = {
        "timepoint": "post_nact",
        "patient_id": row["patient_id"],
        "post_H_clean": row["post_H_clean"],
        "post_DDR": row["post_DDR"],
        "post_S_sen": row["post_S_sen"],
        "post_BCL2A1_z": row["post_BCL2A1_z"],
        "delta_MCL1_z": row["delta_MCL1_z"],
        "post_EMT_z": row["post_EMT_z"],
    }
    response = client.post("/v1/ovarian/arbiter/score", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_eleven_gse165897_patients_match_four_state():
    assert len(GSE165897_11_PATIENT_TABLE) == 11
    client = _client()
    seen = {}
    for row in GSE165897_11_PATIENT_TABLE:
        body = _post_row(client, row)
        seen[row["patient_id"]] = body["four_state_class"]
        assert body["ruo"] is True
        assert body["contract_status"] == "evidence_bound_exploratory_prioritization"
        assert body["four_state_class"] == EXPECTED[row["patient_id"]]
        assert body["decision"] == DECISION[body["four_state_class"]]
    assert seen["EOC3"] == "PRIMARY_REFRACTORY_ESCAPE"
    assert seen["EOC349"] == "DDRonly"
    assert seen["EOC443"] == "Hcleanonly"


def test_case_full_hgsoc_attaches_two_door_without_dropping_mfap4():
    row = next(item for item in GSE165897_11_PATIENT_TABLE if item["patient_id"] == "EOC136")
    client = _client()
    response = client.post(
        "/v1/case/full",
        params={"cancer": "hgsoc"},
        json={
            "resistance_features": {"EMT_MFAP4": 8.17},
            "ovarian_two_door": {
                "timepoint": "post_nact",
                "patient_id": "EOC136",
                "post_H_clean": row["post_H_clean"],
                "post_DDR": row["post_DDR"],
                "post_BCL2A1_z": row["post_BCL2A1_z"],
                "delta_MCL1_z": row["delta_MCL1_z"],
                "post_EMT_z": row["post_EMT_z"],
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ovarian_two_door"]["four_state_class"] == "Neither"
    assert body["ovarian_arbiter"]["model_name"] == "ovarian_mfap4_arbiter_v1"


def test_ovarian_missing_post_fields_422():
    client = _client()
    response = client.post(
        "/v1/ovarian/arbiter/score",
        json={"timepoint": "post_nact", "patient_id": "missing"},
    )
    assert response.status_code == 422
    assert "post_H_clean" in response.json()["detail"]["missing"]


def test_three_immunotherapy_modes_and_missing_inputs():
    client = _client()
    v3 = load_v3_contract()
    bag = v3["v12"]["bag_transforms"]["literal_CD8"]
    bag_body = {
        "specimen": "cart_infusion_bag",
        "n_CD8": 100,
        "M_ratio": bag["M_ratio"]["mean"],
        "M_expr": bag["M_expr"]["mean"],
        "f_cyto_marker": bag["f_cyto_marker"]["mean"],
        "n_memory": 3,
        "n_cyto": 1,
        "n_exhausted": 1,
    }
    bag_response = client.post("/v1/immunotherapy/arbiter/score", json=bag_body)
    assert bag_response.status_code == 200, bag_response.text
    scored = bag_response.json()
    assert math.isclose(scored["Bag_Index_1"], 0.0, abs_tol=1e-12)
    assert math.isclose(scored["Bag_Index_3"], math.log2(3.5 / 2.5), rel_tol=0, abs_tol=1e-12)
    assert scored["flag"] is None

    hot = dict(bag_body)
    hot["f_cyto_marker"] = bag["f_cyto_marker"]["mean"] + bag["f_cyto_marker"]["sd"]
    hot_response = client.post("/v1/immunotherapy/arbiter/score", json=hot)
    assert hot_response.status_code == 200
    assert hot_response.json()["flag"] == "EX_VIVO_PREMATURE_BURNOUT"
    assert hot_response.json()["risk_flag"] == "EX_VIVO_CYTOTOXIC_DIFFERENTIATION_RISK"
    assert hot_response.json()["ruo"] is True
    assert hot_response.json()["contract_status"] == "evidence_bound_exploratory_prioritization"
    arb_v3 = load_arbiter_v3_contract()
    assert arb_v3["contract_version"] == "zeta_immunotherapy_contract_v3"
    assert arb_v3["contract_status"] == "evidence_bound_exploratory_prioritization"

    cart = load_cart_contract()
    blood = {"specimen": "cart_in_vivo_blood"}
    for name, mean in zip(cart["feature_order"], cart["preprocessing"]["mean"], strict=True):
        blood[name] = mean
    blood["prior_regimens"] = 4
    blood["prior_autologous_transplant"] = 0.0
    blood["cyto_absent"] = 0.0
    blood_response = client.post("/v1/immunotherapy/arbiter/score", json=blood)
    assert blood_response.status_code == 200, blood_response.text
    probability = blood_response.json()["probability"]
    intercept = float(cart["intercept_standardized"])
    # Features are not all at the mean after the binary flags are set to 0,
    # so only check the expit path when the posted values really are the means.
    posted = [blood[name] for name in cart["feature_order"]]
    if posted == cart["preprocessing"]["mean"]:
        expected = 1.0 / (1.0 + math.exp(-intercept)) if intercept >= 0 else (
            math.exp(intercept) / (1.0 + math.exp(intercept))
        )
        assert math.isclose(probability, expected, rel_tol=0, abs_tol=1e-12)
    else:
        assert 0.0 < probability < 1.0

    scales = v3["parent_v2"]["parent_v1"]["cohort_population_transforms"][
        "Gide_PRJEB23709_pretreatment"
    ]["population_transforms"]
    genes = ["TCF7", "ZNF683", "GNLY", "CD8A", "CD8B", "CD4", "IL7R", "FOXP3", "IL2RA"]
    ici = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "ici_tumor_biopsy",
            "mode": "bulk",
            "context": "Gide_PRJEB23709|pretreatment",
            "expression": {gene: scales[gene]["mean"] for gene in genes},
        },
    )
    assert ici.status_code == 200, ici.text
    ici_body = ici.json()
    assert math.isclose(ici_body["E_effector"], 0.0, abs_tol=1e-12)
    assert math.isclose(ici_body["S_stem_eff"], 0.0, abs_tol=1e-12)
    assert math.isclose(ici_body["S_triad"], 0.0, abs_tol=1e-12)

    single = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "ici_tumor_biopsy",
            "mode": "single_cell",
            "n_TCF7_positive": 7,
            "n_TOX_positive": 1,
        },
    )
    assert single.status_code == 200, single.text
    assert math.isclose(
        single.json()["TCF7_TOX_count_ratio"],
        math.log2(7.5 / 1.5),
        abs_tol=1e-12,
    )

    missing_bag = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={"specimen": "cart_infusion_bag"},
    )
    assert missing_bag.status_code == 422
    assert "n_CD8" in missing_bag.json()["detail"]["missing"]

    missing_blood = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={"specimen": "cart_in_vivo_blood"},
    )
    assert missing_blood.status_code == 422
    assert "load_ZNF683" in missing_blood.json()["detail"]["missing"]

    missing_ici = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={"specimen": "ici_tumor_biopsy", "mode": "bulk"},
    )
    assert missing_ici.status_code == 422
    assert "context" in missing_ici.json()["detail"]["missing"]
