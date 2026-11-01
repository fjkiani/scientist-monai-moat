"""End-to-end integration tests for Ovarian Two-Door v3, Immunotherapy v12, and L5 Elo bridge."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("AUTH_MODE", "off")
os.environ.setdefault("ONCOLOGY_ARBITER_AUTH_MODE", "off")

from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.immunotherapy.cart_ici_v3 import load_cart_contract, load_v3_contract
from oncology_arbiter.ovarian.ovarian_two_door_standalone_v3 import (
    GSE165897_11_PATIENT_TABLE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROOFS_DIR = REPO_ROOT / "docs" / "proofs"

EXPECTED_CLASSES = {
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


def _client() -> TestClient:
    return TestClient(create_app())


def _ovarian_payload(row: dict) -> dict:
    return {
        "timepoint": "post_nact",
        "patient_id": row["patient_id"],
        "post_H_clean": row["post_H_clean"],
        "post_DDR": row["post_DDR"],
        "post_S_sen": row["post_S_sen"],
        "post_BCL2A1_z": row["post_BCL2A1_z"],
        "delta_MCL1_z": row["delta_MCL1_z"],
        "post_EMT_z": row["post_EMT_z"],
    }


def test_1_gse165897_all_11_patients_and_l5_elo_rankings() -> None:
    client = _client()
    by_id = {}
    for row in GSE165897_11_PATIENT_TABLE:
        resp = client.post("/v1/ovarian/arbiter/score", json=_ovarian_payload(row))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        pid = row["patient_id"]
        by_id[pid] = body
        assert body["ruo"] is True
        assert body["contract_status"] == "evidence_bound_exploratory_prioritization"
        assert body["four_state_class"] == EXPECTED_CLASSES[pid]
        assert body["elo_bridge"]["ruo"] is True
        assert body["elo_bridge"]["contract_status"] == "evidence_bound_exploratory_prioritization"

    eoc153_ranks = {
        r["drug_id"]: r["rank"]
        for r in by_id["EOC153"]["elo_bridge"]["ranked_candidates"]
    }
    assert by_id["EOC153"]["elo_bridge"]["top_mechanism_class"] == "BCL-xL_senolytic"
    assert eoc153_ranks["Navitoclax"] == 1
    assert eoc153_ranks["Navitoclax"] < eoc153_ranks["Venetoclax"]
    assert eoc153_ranks["Navitoclax"] < eoc153_ranks["Olaparib"]

    eoc1005_bridge = by_id["EOC1005"]["elo_bridge"]
    assert eoc1005_bridge["top_candidate"] == "Ceralasertib"
    assert eoc1005_bridge["ranked_candidates"][0]["drug_id"] == "Ceralasertib"

    # Also verify /v1/case/full?cancer=hgsoc wires the bridge into elo_ranked_hypotheses
    row_153 = next(r for r in GSE165897_11_PATIENT_TABLE if r["patient_id"] == "EOC153")
    cf = client.post(
        "/v1/case/full?cancer=hgsoc",
        json={
            "resistance_features": {"EMT_MFAP4": 8.1705},
            "run_co_scientist": False,
            "ovarian_two_door": _ovarian_payload(row_153),
        },
    )
    assert cf.status_code == 200, cf.text
    cf_body = cf.json()
    assert cf_body["ovarian_two_door"]["four_state_class"] == "Hcleanonly"
    assert math.isclose(cf_body["ovarian_arbiter"]["p_positive"], 0.500593, rel_tol=0.0, abs_tol=1e-6)
    assert cf_body["elo_ranked_hypotheses"][0]["drug_id"] == "Navitoclax"


def test_2_mbd4_lof_brca_wt_promotes_ceralasertib_gemcitabine_demotes_olaparib() -> None:
    client = _client()
    row = next(r for r in GSE165897_11_PATIENT_TABLE if r["patient_id"] == "EOC136")
    payload = {
        **_ovarian_payload(row),
        "mbd4_lof": True,
        "mbd4_hgvs_c": "c.1293delA",
        "tp53_mut": True,
        "brca_wt": True,
        "hrd": 28.0,
    }
    resp = client.post("/v1/ovarian/arbiter/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["gate1_mbd4"]["fired"] is True
    assert body["parp_monotherapy"] == "demoted"
    bridge = body["elo_bridge"]
    assert "Ceralasertib" in bridge["promoted"]
    assert "Gemcitabine" in bridge["promoted"]
    assert "Olaparib" in bridge["demoted"]
    ranks = {r["drug_id"]: r["rank"] for r in bridge["ranked_candidates"]}
    assert ranks["Ceralasertib"] == 1
    assert ranks["Gemcitabine"] == 2
    assert ranks["Olaparib"] > ranks["Gemcitabine"]


def test_3_immunotherapy_three_compartments_score_and_route_to_l5() -> None:
    client = _client()
    v3 = load_v3_contract()
    cart = load_cart_contract()

    # 1. Real Deng GSE151511 patient ac01 (GSM4579891, CR=1) & ac18 (CR=0 burnout bag)
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
    bag_body = ac01_resp.json()
    assert bag_body["ruo"] is True
    assert bag_body["contract_status"] == "evidence_bound_exploratory_prioritization"
    assert math.isclose(bag_body["Bag_Index_1"], 0.5516124652233043, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(bag_body["Bag_Index_2"], 0.6875185306880747, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(bag_body["Bag_Index_3"], -0.4004107904839966, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(bag_body["Bag_low_cyto"], 0.21578947368421053, rel_tol=0.0, abs_tol=1e-12)
    bag_metrics = bag_body["elo_bridge"]["receipts"]["metrics_literal_CD8"]["Bag_Index_3_primary_CR"]
    assert round(float(bag_metrics["auroc"]), 3) == 0.881
    assert round(float(bag_metrics["q"]), 5) == 0.00831
    assert round(float(bag_metrics["q_global_new_primary"]), 4) == 0.0191
    assert bag_body["elo_bridge"]["top_candidate"] == "Autologous_CD19_CART_Infusion"

    # 2. Real GSE290722 patient P26 (responder, prob=0.6420115698079313) & P1 (non-responder, prob=0.021457469350329782)
    p26_feats = next(x["features"] for x in cart["inference_examples"] if x["patient_id"] == "P26")
    blood_resp = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={"specimen": "cart_in_vivo_blood", **p26_feats},
    )
    assert blood_resp.status_code == 200, blood_resp.text
    blood_body = blood_resp.json()
    assert blood_body["ruo"] is True
    assert math.isclose(blood_body["probability"], 0.6420115698079313, rel_tol=0.0, abs_tol=1e-12)
    blood_eval = blood_body["elo_bridge"]["receipts"]["evaluation_receipt"]
    assert round(float(blood_eval["auroc"]), 3) == 0.782
    assert round(float(blood_eval["within_fold_mean"]), 3) == 0.823
    assert round(float(blood_eval["brier"]), 3) == 0.199
    assert blood_body["elo_bridge"]["top_candidate"] == "Standard_Leukapheresis_CART_Proceed"

    # 3. Real Gide_PRJEB23709|pretreatment patient PD1_35 (ERR2208952, responder y=1)
    mel_rows = v3["v12"]["patient_inputs"]["melanoma"]
    pd1_35 = next(r for r in mel_rows if r["patient_id"] == "PD1_35" and r["timepoint"] == "pretreatment")
    genes = ["TCF7", "ZNF683", "GNLY", "CD8A", "CD8B", "CD4", "IL7R", "FOXP3", "IL2RA"]
    ici_resp = client.post(
        "/v1/immunotherapy/arbiter/score",
        json={
            "specimen": "ici_tumor_biopsy",
            "mode": "bulk",
            "context": "Gide_PRJEB23709|pretreatment",
            "expression": {g: pd1_35[f"expr_{g}"] for g in genes},
        },
    )
    assert ici_resp.status_code == 200, ici_resp.text
    ici_body = ici_resp.json()
    assert math.isclose(ici_body["E_effector"], 1.187228927951933, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ici_body["S_triad"], 0.378646431347146, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ici_body["S_stem_eff"], 1.250957136413626, rel_tol=0.0, abs_tol=1e-12)
    gide_metrics = ici_body["elo_bridge"]["receipts"]["gide_pretreatment_metrics"]
    assert round(float(gide_metrics["ZNF683_GNLY_effector_response"]["auroc"]), 3) == 0.804
    assert round(float(gide_metrics["ZNF683_GNLY_effector_OS"]["hr"]), 3) == 0.370
    assert ici_body["elo_bridge"]["top_candidate"] == "Anti_PD1_Checkpoint_Blockade"


def test_4_empty_or_malformed_payloads_fail_closed_422_on_all_routes() -> None:
    client = _client()
    assert client.post("/v1/ovarian/arbiter/score", json={}).status_code == 422
    assert (
        client.post(
            "/v1/ovarian/arbiter/score",
            json={"timepoint": "pre_treatment"},
        ).status_code
        == 422
    )
    assert client.post("/v1/immunotherapy/arbiter/score", json={}).status_code == 422
    assert (
        client.post(
            "/v1/immunotherapy/arbiter/score",
            json={"specimen": "cart_infusion_bag"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/immunotherapy/arbiter/score",
            json={"specimen": "cart_in_vivo_blood"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/immunotherapy/arbiter/score",
            json={"specimen": "ici_tumor_biopsy", "mode": "bulk"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/case/full?cancer=hgsoc",
            json={"ovarian_two_door": {"timepoint": "post_nact"}},
        ).status_code
        == 422
    )


def test_5_open_archive_receipts_exist_and_verify_by_sha256() -> None:
    gdsc2_path = PROOFS_DIR / "gdsc2_ddr_bh3_tertile_shift_receipt.json"
    pfi_path = PROOFS_DIR / "gse165897_two_door_pfi_separation_receipt.json"
    assert gdsc2_path.is_file()
    assert pfi_path.is_file()
    assert len(hashlib.sha256(gdsc2_path.read_bytes()).hexdigest()) == 64
    assert len(hashlib.sha256(pfi_path.read_bytes()).hexdigest()) == 64

    gdsc2 = json.loads(gdsc2_path.read_text())
    pfi = json.loads(pfi_path.read_text())
    assert gdsc2["ruo"] is True
    assert gdsc2["contract_status"] == "evidence_bound_exploratory_prioritization"
    assert (
        gdsc2["archive"]["sha256"]
        == "f980a2ff7f1c8a680cc0e3dafbbce0ec07a16711a5483bd620513fa7bb90a011"
    )
    assert (
        gdsc2["source_members"]["tertile_profiles"]["sha256"]
        == "5a36cd3ad622ef6c370f5d7e44b0ee21ca1426bc461f3f8f168840db8824d79d"
    )
    assert (
        gdsc2["source_members"]["class_sensitivity"]["sha256"]
        == "1ddd1d0cdb6caceed8980ac882c8ba73b8dd66362556f5e6f8f02860e0642787"
    )
    assert pfi["ruo"] is True
    assert pfi["contract_status"] == "evidence_bound_exploratory_prioritization"
    assert (
        pfi["source_sha256"]
        == "8e98492229f25fb83d4b195f73a68908eb7b39932d88796f58bf2f5d75f6f87e"
    )
    assert (
        pfi["contract_v3_sha256"]
        == "253502d030f4e6b050257996420ff52c260d5f5ef75cd840bc7cc1f8b674fd65"
    )
    omnibus = pfi["omnibus_4state_kruskal_wallis"]
    assert round(float(omnibus["primary_all11"]["exact_omnibus_label_permutation_p"]), 4) == 0.0514
    assert (
        round(
            float(
                omnibus["post_remission_n10_excluding_gate0_eoc3"][
                    "exact_omnibus_label_permutation_p"
                ]
            ),
            5,
        )
        == 0.00333
    )
