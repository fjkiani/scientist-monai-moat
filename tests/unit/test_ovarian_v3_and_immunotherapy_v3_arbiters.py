"""Real-patient unit tests for Ovarian Two-Door v3 and Immunotherapy v12 (Contract v3).

Every test feeds real patient feature vectors from the frozen cohort contracts
and asserts exact numerical outputs (to 1e-12) and L5 Elo tournament rankings:
  - Ovarian Post-NACT (GSE165897, n=11): both pre-computed table inputs and raw
    component z-scores (stress_34_clean, proliferation_106, BCL2L1, PMAIP1, MCL1)
  - Ovarian Gate 1 (CCR 2026 B092 MBD4 c.1293delA + TP53 mut + BRCA WT + HRD<42)
  - Ovarian Gate 2 (GSE63885 MFAP4 probe 212713_at + AACR 2026 A065 zFAP/zCXCL10)
  - Immunotherapy cart_infusion_bag (Deng GSE151511, n=23 literal_CD8: ac01, ac12, ac18, ac23)
  - Immunotherapy cart_in_vivo_blood (GSE290722 v9, n=28: P1, P3, P26)
  - Immunotherapy ici_tumor_biopsy bulk (Gide PD1_10, PD1_35; Riaz Pt101) & single-cell (GSE120575 Mel112, Mel04.3, Mel60)
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

EXPECTED_CLASS = {
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

EXPECTED_DECISION = {
    "Neither": "CHEMO_CLEARED_REMISSION",
    "DDRonly": "DOOR_1_DDR_PROLIFERATIVE",
    "Hcleanonly": "DOOR_2_HIBERNATION_SENOLYTIC",
    "Both": "DOOR_2_HIBERNATION_SENOLYTIC",
    "PRIMARY_REFRACTORY_ESCAPE": "PRIMARY_REFRACTORY_ESCAPE",
}

EXPECTED_RECOMMEND = {
    "Neither": ([], []),
    "DDRonly": (["Ceralasertib", "Adavosertib"], []),
    "Hcleanonly": (["Navitoclax", "ABT-737"], ["Venetoclax"]),
    "Both": (["Navitoclax", "ABT-737"], ["Venetoclax"]),
    "PRIMARY_REFRACTORY_ESCAPE": ([], []),
}

# Frozen component z-scores from ovarian_hibernation_two_door_contract_v3.json
# (complete_primary_patient_table, SHA 253502d030f4e6b050257996420ff52c260d5f5ef75cd840bc7cc1f8b674fd65)
GSE165897_RAW_Z_COMPONENTS: dict[str, dict[str, float]] = {
    "EOC1005": {
        "z_stress_34_clean": -0.1120646176022793,
        "z_proliferation_106": -0.4757244454900954,
        "z_BCL2L1": 0.8152400789610472,
        "z_PMAIP1": 0.884550201215719,
        "z_MCL1": 0.40467391059149505,
        "post_DDR": 0.0608244465174617,
        "expected_H_clean": 0.3636598278878161,
        "expected_S_sen": 1.6587761974730872,
    },
    "EOC136": {
        "z_stress_34_clean": 0.35404320233153397,
        "z_proliferation_106": 0.05639947084031436,
        "z_BCL2L1": -0.09812894321672483,
        "z_PMAIP1": -0.745235129148274,
        "z_MCL1": -0.6587961165964555,
        "post_DDR": 0.042086486191433986,
        "expected_H_clean": 0.2976437314912196,
        "expected_S_sen": 0.11307577572267635,
    },
    "EOC153": {
        "z_stress_34_clean": 1.8758459121278785,
        "z_proliferation_106": -1.4843951070780101,
        "z_BCL2L1": 0.9589820809859554,
        "z_PMAIP1": 2.3307289818047603,
        "z_MCL1": 2.483581114460374,
        "post_DDR": 0.035789058341352986,
        "expected_H_clean": 3.3602410192058887,
        "expected_S_sen": 4.1663709675362295,
    },
    "EOC227": {
        "z_stress_34_clean": 0.776790925446493,
        "z_proliferation_106": -0.5658394769475096,
        "z_BCL2L1": 0.8303941605892418,
        "z_PMAIP1": -0.366976009040742,
        "z_MCL1": 0.14901661622124818,
        "post_DDR": 0.0603097752866016,
        "expected_H_clean": 1.3426304023940026,
        "expected_S_sen": 1.657031937721254,
    },
    "EOC3": {
        "z_stress_34_clean": 2.4207277517540744,
        "z_proliferation_106": -0.1539101086949194,
        "z_BCL2L1": -0.8983082629397073,
        "z_PMAIP1": 1.3654440405595578,
        "z_MCL1": 2.3769455076831547,
        "post_DDR": 0.047761554835575135,
        "expected_H_clean": 2.574637860448994,
        "expected_S_sen": 0.6648281303856893,
    },
    "EOC349": {
        "z_stress_34_clean": -0.7084127546285611,
        "z_proliferation_106": -1.724955697349814,
        "z_BCL2L1": 2.4348897568965517,
        "z_PMAIP1": -0.896602276614109,
        "z_MCL1": -0.4271389998441182,
        "post_DDR": 0.05547302403492261,
        "expected_H_clean": 1.016542942721253,
        "expected_S_sen": 2.981969422847814,
    },
    "EOC372": {
        "z_stress_34_clean": -0.7951535582125554,
        "z_proliferation_106": -1.4235548455298639,
        "z_BCL2L1": 0.09302840825334793,
        "z_PMAIP1": -0.8590437464641325,
        "z_MCL1": 0.06138116102000304,
        "post_DDR": 0.04644650745488275,
        "expected_H_clean": 0.6284012873173085,
        "expected_S_sen": -0.19899521191347916,
    },
    "EOC443": {
        "z_stress_34_clean": 0.12884678559986784,
        "z_proliferation_106": -1.6603741899471973,
        "z_BCL2L1": 0.9673672216199461,
        "z_PMAIP1": 0.6318759380579834,
        "z_MCL1": -0.23381428997478504,
        "post_DDR": 0.05212375139760441,
        "expected_H_clean": 1.7892209755470652,
        "expected_S_sen": 3.62227842519978,
    },
    "EOC540": {
        "z_stress_34_clean": -0.44187796138059704,
        "z_proliferation_106": -1.735943047083877,
        "z_BCL2L1": -1.4700088120629762,
        "z_PMAIP1": 0.6227615360974101,
        "z_MCL1": -0.9480094003821525,
        "post_DDR": 0.03578019329012831,
        "expected_H_clean": 1.29406508570328,
        "expected_S_sen": 1.3948272101198667,
    },
    "EOC733": {
        "z_stress_34_clean": -0.2392875595976043,
        "z_proliferation_106": 1.2945686609902025,
        "z_BCL2L1": -0.19554756965813605,
        "z_PMAIP1": -0.9242778709392094,
        "z_MCL1": 0.9891716536234315,
        "post_DDR": 0.10297662711873573,
        "expected_H_clean": -1.5338562205878068,
        "expected_S_sen": -3.642853314808584,
    },
    "EOC87": {
        "z_stress_34_clean": 0.9711516413871655,
        "z_proliferation_106": 1.0314311680446182,
        "z_BCL2L1": -1.0162863894430834,
        "z_PMAIP1": 0.7460549092649178,
        "z_MCL1": 0.5762238144946158,
        "post_DDR": 0.10592860657412567,
        "expected_H_clean": -0.06027952665745273,
        "expected_S_sen": -0.9067348213302341,
    },
}


def _client() -> TestClient:
    return TestClient(create_app())


def test_eleven_gse165897_patients_from_table_and_raw_z_components() -> None:
    """Verify all 11 GSE165897 patients from both pre-computed scores and raw z-components."""
    assert len(GSE165897_11_PATIENT_TABLE) == 11
    client = _client()
    for row in GSE165897_11_PATIENT_TABLE:
        pid = row["patient_id"]
        exp_cls = EXPECTED_CLASS[pid]
        exp_dec = EXPECTED_DECISION[exp_cls]
        exp_rec, exp_contra = EXPECTED_RECOMMEND[exp_cls]

        # 1. Direct table input -> exact class, decision, recommendations, and L5 Elo bridge
        resp = client.post(
            "/v1/ovarian/arbiter/score",
            json={
                "timepoint": "post_nact",
                "patient_id": pid,
                "post_H_clean": row["post_H_clean"],
                "post_DDR": row["post_DDR"],
                "post_S_sen": row["post_S_sen"],
                "post_BCL2A1_z": row["post_BCL2A1_z"],
                "delta_MCL1_z": row["delta_MCL1_z"],
                "post_EMT_z": row["post_EMT_z"],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ruo"] is True
        assert body["contract_status"] == "evidence_bound_exploratory_prioritization"
        assert body["four_state_class"] == exp_cls
        assert body["decision"] == exp_dec
        assert body["recommend"] == exp_rec
        assert body["contraindicated"] == exp_contra
        assert body["gate0_primary_refractory"] is (pid == "EOC3")
        argon = body["argon_interpretation"]
        assert argon["model"] == "gemini-4-argon"
        assert argon["temperature"] == 0.0
        assert argon["deterministic"] is True
        assert argon["argon_input"]["patient_id"] == pid
        assert exp_cls in argon["argon_output"]["biological_state_meaning"]
        assert body["elo_bridge"]["top_candidate"] in argon["argon_output"]["elo_ranking_meaning"]

        # 2. Raw component z-scores -> derive post_H_clean & post_S_sen to 1e-12
        comp = GSE165897_RAW_Z_COMPONENTS[pid]
        raw_resp = client.post(
            "/v1/ovarian/arbiter/score",
            json={
                "timepoint": "post_nact",
                "patient_id": pid,
                "z_stress_34_clean": comp["z_stress_34_clean"],
                "z_proliferation_106": comp["z_proliferation_106"],
                "z_BCL2L1": comp["z_BCL2L1"],
                "z_PMAIP1": comp["z_PMAIP1"],
                "z_MCL1": comp["z_MCL1"],
                "post_DDR": comp["post_DDR"],
                "post_BCL2A1_z": row["post_BCL2A1_z"],
                "delta_MCL1_z": row["delta_MCL1_z"],
                "post_EMT_z": row["post_EMT_z"],
            },
        )
        assert raw_resp.status_code == 200, raw_resp.text
        raw_body = raw_resp.json()
        assert math.isclose(
            raw_body["scores"]["post_H_clean"],
            comp["expected_H_clean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            raw_body["scores"]["post_S_sen"],
            comp["expected_S_sen"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert raw_body["four_state_class"] == exp_cls
        assert raw_body["decision"] == exp_dec


def test_ovarian_gate1_mbd4_lof_c1293delA_and_hrd_gate() -> None:
    """Verify Gate 1 CCR 2026 B092 MBD4-LOF rule with exact inputs and outputs."""
    client = _client()
    base_row = next(r for r in GSE165897_11_PATIENT_TABLE if r["patient_id"] == "EOC136")

    # MBD4 c.1293delA + TP53 mut + BRCA WT + HRD 28 (<42) -> Gate 1 fires
    fired_resp = client.post(
        "/v1/ovarian/arbiter/score",
        json={
            "timepoint": "post_nact",
            "patient_id": "AK_MBD4_c1293delA",
            "post_H_clean": base_row["post_H_clean"],
            "post_DDR": base_row["post_DDR"],
            "post_BCL2A1_z": base_row["post_BCL2A1_z"],
            "delta_MCL1_z": base_row["delta_MCL1_z"],
            "post_EMT_z": base_row["post_EMT_z"],
            "mbd4_lof": True,
            "mbd4_hgvs_c": "c.1293delA",
            "tp53_mut": True,
            "brca_wt": True,
            "hrd": 28.0,
        },
    )
    assert fired_resp.status_code == 200, fired_resp.text
    fired = fired_resp.json()
    g1 = fired["gate1_mbd4"]
    assert g1 == {
        "fired": True,
        "rule": "CCR_2026_B092_MBD4_LOF",
        "parp_monotherapy": "demoted",
        "parp_null_p": 0.605,
        "recommend": ["Ceralasertib", "Gemcitabine"],
        "ceralasertib_ln_ic50_delta": -0.74,
        "ceralasertib_p": 0.021,
    }
    assert fired["parp_monotherapy"] == "demoted"
    assert fired["recommend"] == ["Ceralasertib", "Gemcitabine"]
    ranks = {r["drug_id"]: r["rank"] for r in fired["elo_bridge"]["ranked_candidates"]}
    assert ranks["Ceralasertib"] == 1
    assert ranks["Gemcitabine"] == 2
    assert ranks["Olaparib"] == 8

    # HRD 55.0 (>=42) -> Gate 1 does not fire
    unfired_resp = client.post(
        "/v1/ovarian/arbiter/score",
        json={
            "timepoint": "post_nact",
            "patient_id": "MBD4_HRD_HIGH",
            "post_H_clean": base_row["post_H_clean"],
            "post_DDR": base_row["post_DDR"],
            "post_BCL2A1_z": base_row["post_BCL2A1_z"],
            "delta_MCL1_z": base_row["delta_MCL1_z"],
            "post_EMT_z": base_row["post_EMT_z"],
            "mbd4_lof": True,
            "tp53_mut": True,
            "brca_wt": True,
            "hrd": 55.0,
        },
    )
    assert unfired_resp.status_code == 200
    assert unfired_resp.json()["gate1_mbd4"]["fired"] is False


def test_ovarian_gate2_pretreatment_mfap4_and_aacr_a065() -> None:
    """Verify Gate 2 pre-treatment MFAP4 (GSE63885 212713_at) + AACR A065 stroma-immune."""
    client = _client()

    # GSE63885 platinum-resistant mean mfap4_expr=8.1705, FAP-high / CXCL10-low
    res_resp = client.post(
        "/v1/ovarian/arbiter/score",
        json={
            "timepoint": "pre_treatment",
            "patient_id": "GSE63885_resistant_mean",
            "mfap4_expr": 8.1705,
            "z_FAP": 0.85,
            "z_CXCL10": -0.62,
        },
    )
    assert res_resp.status_code == 200, res_resp.text
    res_body = res_resp.json()
    assert res_body["decision"] == "PRETREATMENT_STROMA_IMMUNE"
    g2_res = res_body["gate2"]
    assert g2_res["aacr_positive"] is False
    assert math.isclose(g2_res["mfap4"]["p_positive"], 0.500593, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(g2_res["mfap4"]["logit"], 0.002373, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(g2_res["ledger"]["tcga_os_hr"], 0.5933707512190046, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(g2_res["ledger"]["gse32062_rmst_months"], 8.859547458905148, rel_tol=0.0, abs_tol=1e-12)
    assert res_body["elo_bridge"]["top_candidate"] == "Ceralasertib"

    # GSE63885 platinum-sensitive mean mfap4_expr=7.6807, AACR A065 positive (zFAP<0 & zCXCL10>0)
    sen_resp = client.post(
        "/v1/ovarian/arbiter/score",
        json={
            "timepoint": "pre_treatment",
            "patient_id": "GSE63885_sensitive_aacr_pos",
            "mfap4_expr": 7.6807,
            "z_FAP": -0.74,
            "z_CXCL10": 0.91,
        },
    )
    assert sen_resp.status_code == 200, sen_resp.text
    sen_body = sen_resp.json()
    g2_sen = sen_body["gate2"]
    assert g2_sen["aacr_positive"] is True
    assert math.isclose(g2_sen["mfap4"]["p_positive"], 0.414039, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(g2_sen["mfap4"]["logit"], -0.347294, rel_tol=0.0, abs_tol=1e-6)
    assert sen_body["elo_bridge"]["top_candidate"] == "Carboplatin_Paclitaxel"


def test_case_full_hgsoc_attaches_two_door_and_elo_without_dropping_mfap4() -> None:
    row_153 = next(item for item in GSE165897_11_PATIENT_TABLE if item["patient_id"] == "EOC153")
    client = _client()
    response = client.post(
        "/v1/case/full",
        params={"cancer": "hgsoc"},
        json={
            "resistance_features": {"EMT_MFAP4": 8.1705},
            "ovarian_two_door": {
                "timepoint": "post_nact",
                "patient_id": "EOC153",
                "post_H_clean": row_153["post_H_clean"],
                "post_DDR": row_153["post_DDR"],
                "post_S_sen": row_153["post_S_sen"],
                "post_BCL2A1_z": row_153["post_BCL2A1_z"],
                "delta_MCL1_z": row_153["delta_MCL1_z"],
                "post_EMT_z": row_153["post_EMT_z"],
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ovarian_two_door"]["four_state_class"] == "Hcleanonly"
    assert body["ovarian_two_door"]["decision"] == "DOOR_2_HIBERNATION_SENOLYTIC"
    assert body["ovarian_arbiter"]["model_name"] == "ovarian_mfap4_arbiter_v1"
    assert math.isclose(body["ovarian_arbiter"]["p_positive"], 0.500593, rel_tol=0.0, abs_tol=1e-6)
    assert body["elo_ranked_hypotheses"][0]["drug_id"] == "Navitoclax"


def test_immunotherapy_bag_deng_gse151511_real_patients() -> None:
    """Verify all 23 real Deng GSE151511 literal_CD8 infusion bags and named patients ac01, ac12, ac18, ac23."""
    client = _client()
    v3 = load_v3_contract()
    arb_v3 = load_arbiter_v3_contract()
    assert arb_v3["contract_version"] == "zeta_immunotherapy_contract_v3"
    assert arb_v3["contract_status"] == "evidence_bound_exploratory_prioritization"

    # Named real patient 1: ac01 (GSM4579891, 3-mo CR=1, v12 actual_patient_inference_examples.bag)
    ac01_resp = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "cart_infusion_bag",
            "bag_gate": "literal_CD8",
            "n_CD8": 1520,
            "M_ratio": 1.14149467448654,
            "M_expr": 1.087685923313923,
            "f_cyto_marker": 0.78421052631579,
            "n_memory": 1375,
            "n_cyto": 1192,
            "n_exhausted": 623,
        },
    )
    assert ac01_resp.status_code == 200, ac01_resp.text
    ac01 = ac01_resp.json()
    assert math.isclose(ac01["Bag_Index_1"], 0.5516124652233043, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ac01["Bag_Index_2"], 0.6875185306880747, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ac01["Bag_Index_3"], -0.4004107904839966, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ac01["Bag_low_cyto"], 0.21578947368421053, rel_tol=0.0, abs_tol=1e-12)
    assert ac01["risk_flag"] is None
    assert ac01["flag"] is None
    assert ac01["elo_bridge"]["top_candidate"] == "Autologous_CD19_CART_Infusion"

    # Named real patient 2: ac18 (CR=0, high f_cyto_marker=0.915032679738562 -> differentiation risk)
    ac18_row = next(
        r for r in v3["v12"]["patient_inputs"]["bag"]
        if r["patient_id"] == "ac18" and r["gate"] == "literal_CD8"
    )
    ac18_resp = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "cart_infusion_bag",
            "bag_gate": "literal_CD8",
            "n_CD8": ac18_row["n_CD8"],
            "M_ratio": ac18_row["M_ratio"],
            "M_expr": ac18_row["M_expr"],
            "f_cyto_marker": ac18_row["f_cyto_marker"],
            "n_memory": ac18_row["n_memory"],
            "n_cyto": ac18_row["n_cyto"],
            "n_exhausted": ac18_row["n_exhausted"],
        },
    )
    assert ac18_resp.status_code == 200, ac18_resp.text
    ac18 = ac18_resp.json()
    assert math.isclose(ac18["Bag_Index_1"], ac18_row["Bag_Index_1"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ac18["Bag_Index_3"], ac18_row["Bag_Index_3"], rel_tol=0.0, abs_tol=1e-12)
    assert ac18["risk_flag"] == "EX_VIVO_CYTOTOXIC_DIFFERENTIATION_RISK"
    assert ac18["flag"] == "EX_VIVO_PREMATURE_BURNOUT"
    assert ac18["elo_bridge"]["top_candidate"] == "Memory_Enriched_Bag_Hold_or_Re_Expansion"

    # Full 23-patient Deng GSE151511 literal_CD8 cohort verification
    bag_rows = [
        r for r in v3["v12"]["patient_inputs"]["bag"]
        if r["gate"] == "literal_CD8" and r.get("primary_CR") is not None and r.get("n_CD8", 0) > 0
    ]
    assert len(bag_rows) == 23
    for r in bag_rows:
        resp = client.post(
            "/v1/immunotherapy/arbiter/score",
            json={
                "specimen": "cart_infusion_bag",
                "bag_gate": "literal_CD8",
                "n_CD8": r["n_CD8"],
                "M_ratio": r["M_ratio"],
                "M_expr": r["M_expr"],
                "f_cyto_marker": r["f_cyto_marker"],
                "n_memory": r["n_memory"],
                "n_cyto": r["n_cyto"],
                "n_exhausted": r["n_exhausted"],
            },
        )
        assert resp.status_code == 200
        b = resp.json()
        assert math.isclose(b["Bag_Index_1"], r["Bag_Index_1"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(b["Bag_Index_2"], r["Bag_Index_2"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(b["Bag_Index_3"], r["Bag_Index_3"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(b["Bag_low_cyto"], r["Bag_low_cyto"], rel_tol=0.0, abs_tol=1e-12)


def test_immunotherapy_blood_gse290722_real_patients_p1_p3_p26() -> None:
    """Verify real GSE290722 leukapheresis patients P1, P3, and P26 against exact Ridge probabilities."""
    client = _client()
    cart = load_cart_contract()
    expected_probs = {
        "P1": (0.021457469350329782, "Cytoreductive_SPD_Bridging_Before_CART"),
        "P3": (0.04682110546649535, "Cytoreductive_SPD_Bridging_Before_CART"),
        "P26": (0.6420115698079313, "Standard_Leukapheresis_CART_Proceed"),
    }
    for ex in cart["inference_examples"]:
        pid = ex["patient_id"]
        exp_prob, exp_top = expected_probs[pid]
        resp = client.post(
            "/v1/immunotherapy/arbiter/score",
            json={"specimen": "cart_in_vivo_blood", **ex["features"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ruo"] is True
        assert body["contract_status"] == "evidence_bound_exploratory_prioritization"
        assert math.isclose(body["probability"], exp_prob, rel_tol=0.0, abs_tol=1e-12)
        assert body["elo_bridge"]["top_candidate"] == exp_top

    # Verify alias keys (log1p_spd, prior_auto_sct) on real patient P26 produce identical probability
    p26_feats = next(x["features"] for x in cart["inference_examples"] if x["patient_id"] == "P26")
    alias_resp = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "cart_in_vivo_blood",
            "log1p_spd": p26_feats["log_baseline_spd"],
            "prior_regimens": p26_feats["prior_regimens"],
            "prior_auto_sct": p26_feats["prior_autologous_transplant"],
            "load_ZNF683": p26_feats["load_ZNF683"],
            "load_GNLY": p26_feats["load_GNLY"],
            "cyto_absent": p26_feats["cyto_absent"],
        },
    )
    assert alias_resp.status_code == 200
    assert math.isclose(alias_resp.json()["probability"], 0.6420115698079313, rel_tol=0.0, abs_tol=1e-12)


def test_immunotherapy_ici_biopsy_gide_riaz_and_gse120575_real_patients() -> None:
    """Verify real melanoma bulk (PD1_10, PD1_35, Pt101) and single-cell (Mel112, Mel04.3, Mel60) patients."""
    client = _client()
    v3 = load_v3_contract()
    mel_rows = v3["v12"]["patient_inputs"]["melanoma"]
    genes = ["TCF7", "ZNF683", "GNLY", "CD8A", "CD8B", "CD4", "IL7R", "FOXP3", "IL2RA"]

    real_bulk_cases = [
        (
            "PD1_10",
            "pretreatment",
            "Gide_PRJEB23709|pretreatment",
            -0.58824133622504,
            0.00599034319851,
            -0.3549585701883237,
            -0.23831718716996553,
            "Combination_IO_Stroma_or_Trial_Escalation",
        ),
        (
            "PD1_35",
            "pretreatment",
            "Gide_PRJEB23709|pretreatment",
            1.187228927951933,
            0.378646431347146,
            1.250957136413626,
            1.282821240644472,
            "Anti_PD1_Checkpoint_Blockade",
        ),
        (
            "Pt101",
            "on_treatment",
            "Riaz_GSE91061|on_treatment",
            1.229768129932729,
            -0.288048557357124,
            1.800752789572415,
            2.086245119392258,
            "Anti_PD1_Checkpoint_Blockade",
        ),
    ]
    for pid, tp, ctx, exp_eff, exp_triad, exp_stem, exp_half, exp_top in real_bulk_cases:
        row = next(r for r in mel_rows if r["patient_id"] == pid and r["timepoint"] == tp)
        resp = client.post(
            "/v1/immunotherapy/arbiter/score",
            json={
                "specimen": "ici_tumor_biopsy",
                "mode": "bulk",
                "context": ctx,
                "expression": {g: row[f"expr_{g}"] for g in genes},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert math.isclose(body["E_effector"], exp_eff, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(body["S_triad"], exp_triad, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(body["S_stem_eff"], exp_stem, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(body["S_stem_eff_TCF7_half"], exp_half, rel_tol=0.0, abs_tol=1e-12)
        assert body["elo_bridge"]["top_candidate"] == exp_top

    # Real GSE120575 single-cell patients: Mel112 (26, 2), Mel04.3 (19, 24), Mel60 (3, 21)
    real_sc_cases = [
        ("Mel112", 26, 2, 3.405992359675837, "Anti_PD1_Checkpoint_Blockade"),
        ("Mel04.3", 19, 24, -0.32930762525296, "Combination_IO_Stroma_or_Trial_Escalation"),
        ("Mel60", 3, 21, -2.618909832644494, "Combination_IO_Stroma_or_Trial_Escalation"),
    ]
    for _pid, n_tcf7, n_tox, exp_ratio, exp_top in real_sc_cases:
        sc_resp = client.post(
            "/v1/immunotherapy/arbiter/score",
            json={
                "specimen": "ici_tumor_biopsy",
                "mode": "single_cell",
                "n_TCF7_positive": n_tcf7,
                "n_TOX_positive": n_tox,
            },
        )
        assert sc_resp.status_code == 200, sc_resp.text
        sc_body = sc_resp.json()
        assert math.isclose(sc_body["TCF7_TOX_count_ratio"], exp_ratio, rel_tol=0.0, abs_tol=1e-12)
        assert sc_body["elo_bridge"]["top_candidate"] == exp_top


def test_ovarian_and_immunotherapy_missing_inputs_fail_closed_422() -> None:
    client = _client()
    ov_miss = client.post(
        "/v1/ovarian/arbiter/score",
        json={"timepoint": "post_nact", "patient_id": "missing"},
    )
    assert ov_miss.status_code == 422
    assert "post_H_clean" in ov_miss.json()["detail"]["missing"]

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

    # Contradictory cyto_absent=1 with non-zero load_ZNF683 must fail closed 422
    contradict_blood = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "cart_in_vivo_blood",
            "log_baseline_spd": 7.5632,
            "prior_regimens": 4,
            "prior_autologous_transplant": 0,
            "load_ZNF683": 0.45,
            "load_GNLY": 0.0,
            "cyto_absent": 1,
        },
    )
    assert contradict_blood.status_code == 422
    assert "load_ZNF683" in contradict_blood.json()["detail"]["missing"]

    missing_ici = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={"specimen": "ici_tumor_biopsy", "mode": "bulk"},
    )
    assert missing_ici.status_code == 422
    assert "context" in missing_ici.json()["detail"]["missing"]


def test_argon_patient_interpreter_is_deterministic_across_all_cohorts(monkeypatch) -> None:
    """Verify deterministic Argon input->output mapping and temperature=0.0 enforcement."""
    from oncology_arbiter.models.gemini_argon_client import GeminiArgonClient
    from oncology_arbiter.orchestrator.argon_patient_interpreter import (
        interpret_patient_with_argon,
    )

    client = _client()
    row_153 = next(r for r in GSE165897_11_PATIENT_TABLE if r["patient_id"] == "EOC153")
    payload = {
        "timepoint": "post_nact",
        "patient_id": "EOC153",
        "post_H_clean": row_153["post_H_clean"],
        "post_DDR": row_153["post_DDR"],
        "post_S_sen": row_153["post_S_sen"],
        "post_BCL2A1_z": row_153["post_BCL2A1_z"],
        "delta_MCL1_z": row_153["delta_MCL1_z"],
        "post_EMT_z": row_153["post_EMT_z"],
    }
    r1 = client.post("/v1/ovarian/arbiter/score", json=payload).json()["argon_interpretation"]
    r2 = client.post("/v1/ovarian/arbiter/score", json=payload).json()["argon_interpretation"]
    assert r1 == r2
    assert r1["temperature"] == 0.0
    assert r1["deterministic"] is True
    assert "3.3602" in r1["argon_output"]["biological_state_meaning"]
    assert "Navitoclax" in r1["argon_output"]["elo_ranking_meaning"]

    # Also verify GeminiArgonClient forces temperature=0.0 when deterministic=True
    captured_temps: list[float] = []

    def _fake_interaction_model(self, *, system, user, max_tokens, temperature):
        captured_temps.append(temperature)
        return (
            '{"biological_state_meaning": "EOC153 post_H_clean=3.3602 > 1.0165 -> Hcleanonly hibernation.", '
            '"gate_and_compartment_meaning": "Gate 3 Door 2 senolytic vulnerability.", '
            '"elo_ranking_meaning": "Navitoclax #1 over Venetoclax/Olaparib.", '
            '"actionable_summary_ruo": "Prioritize BCL-xL senolytic (RUO)."}',
            "req_argon_det_01",
        )

    monkeypatch.setattr(GeminiArgonClient, "_chat_interaction_model", _fake_interaction_model)
    argon_llm = GeminiArgonClient(api_key="AIzaSyDeterministicUnitTestKey000000000", deterministic=True)
    resp = argon_llm.chat([{"role": "user", "content": r1["argon_prompt"]}], temperature=0.9)
    assert captured_temps == [0.0]
    assert resp.request_id == "req_argon_det_01"

    body_153 = client.post("/v1/ovarian/arbiter/score", json=payload).json()
    live_interp = interpret_patient_with_argon(body_153, llm=argon_llm)
    assert live_interp["mode"] == "live_gemini_argon"
    assert live_interp["temperature"] == 0.0
    assert live_interp["request_id"] == "req_argon_det_01"
    assert "3.3602" in live_interp["argon_output"]["biological_state_meaning"]

