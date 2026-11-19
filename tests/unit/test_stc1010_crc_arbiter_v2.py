"""Comprehensive unit & FastAPI integration tests for STC-1010 MSS CRC Biomarker Lock v2.

Verifies:
  1. Exact 6-domain gene-level equations (``D1..D6``, ``D1_D2_Vaccine_Effector_Index``,
     ``D1_D2_D3_D4_Immune_Competence_Index``, ``STC1010_Readiness_Score``) and
     all-tumor ``ddof=0`` population references across ``TCGA`` (``n=592`` all-tumor,
     ``n=490/492`` MSS) and ``GSE39582`` (``n=566`` all-tumor, ``n=444/444`` MSS).
  2. Embedded 12 prevalence rows, 126 Cox rows (102 valid fits in BH family),
     63 60-month RMST rows, nested 5-fold L2 classifier AUROCs (``0.9894`` TCGA OOF,
     ``0.9376`` GSE39582 CV), and ``KRAS G12D`` ``CD47`` / ``BRAF V600E`` ``D6`` Cliff's deltas.
  3. All 4 ``BreAK CRC-001`` Regimen Engineering Strata:
     - ``STRATUM_1_DIRECT_VACCINE_PERMISSIVE`` (``D1 > 0`` and ``APM_Intact``)
     - ``STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN`` (``D1 > 0`` and ``TAP1/TAP2/PSMB8 < MSS Q20``
       + ``B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED`` when ``b2m_truncating_lof == True``)
     - ``STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE`` (``VEGF_TOX_Exhaustion > Q75`` or
       ``Treg_Load > Q75`` or ``SPP1_Core_Barrier > Q75`` +
       ``KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47`` when ``kras_g12d == True``)
     - ``STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION`` (``D1 <= 0`` or
       ``TGFB_CAF_Barrier > Q75`` or ``liver_metastasis == True``)
  4. 8D ``G(p, t) * PATH A`` integration with ``D1 + APM_Intact`` -> ``io`` and ``G(p, t)``.
  5. FastAPI ``POST /v1/crc/stc1010/score`` HTTP 200 responses and ``HTTP 422``
     fail-closed behavior on empty or incomplete payloads.
"""
from __future__ import annotations

import math
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app
from oncology_arbiter.crc import (
    CLASS_MODULE_MEAN,
    CLASS_MODULE_SD,
    CLASSIFIER_NESTED_CV_METRICS,
    COEF,
    CONTRACT_STATUS_V2,
    DOMAINS,
    FEATURES,
    GENES,
    GENOTYPE_CONTRASTS,
    INTERCEPT,
    MODULES,
    REF,
    STRATUM_1_DIRECT_VACCINE_PERMISSIVE,
    STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN,
    STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE,
    STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION,
    cox_results,
    prevalence_results,
    rmst_results,
    score as standalone_score,
    score_stc1010_crc_case,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _expr_from_z(z_overrides: Dict[str, float], reference: str = "TCGA") -> Dict[str, float]:
    """Construct a 42-gene log2 expression dict from target gene z-scores (default z=0.0)."""
    gm, gs, *_ = REF[reference]
    out: Dict[str, float] = {}
    for g, m, s in zip(GENES, gm, gs):
        z_val = float(z_overrides.get(g, 0.0))
        out[g] = m + z_val * s
    return out


def test_exact_6_domain_gene_equations_and_embedded_934_cohort_ledger() -> None:
    """Verify exact 6-domain equations, 42-gene panel, and 934-patient ledger rows."""
    assert len(GENES) == 42
    assert "ARG1" not in GENES
    assert "CCL15" not in GENES
    assert DOMAINS == (
        "D1_trafficking",
        "D2_presentation",
        "D3_effector",
        "D4_triad",
        "D5_myeloid",
        "D6_CMS1like",
    )

    # Verify nested CV AUROCs
    assert math.isclose(
        CLASSIFIER_NESTED_CV_METRICS["TCGA_OOF"]["AUROC"], 0.9894002343666772, rel_tol=1e-12
    )
    assert CLASSIFIER_NESTED_CV_METRICS["TCGA_OOF"]["n_labeled"] == 361
    assert math.isclose(
        CLASSIFIER_NESTED_CV_METRICS["GSE39582_CV"]["AUROC"], 0.937567084078712, rel_tol=1e-12
    )
    assert CLASSIFIER_NESTED_CV_METRICS["GSE39582_CV"]["n_labeled"] == 267

    # Verify 12 prevalence rows, 126 Cox rows (102 valid fits), and 63 RMST rows
    prev = prevalence_results()
    cox = cox_results()
    rmst = rmst_results()
    assert len(prev) == 12
    assert len(cox) == 126
    assert sum(1 for r in cox if r["status"] == "OK") == 102
    assert len(rmst) == 63

    prev_map = {(r["cohort"], r["definition"]): r for r in prev}
    assert prev_map[("TCGA", "D1_Plus_D2_Permissive_MSS")]["n_positive_complete"] == 139
    assert prev_map[("TCGA", "D1_Plus_D2_Permissive_MSS")]["n_complete"] == 490
    assert math.isclose(prev_map[("TCGA", "D1_Plus_D2_Permissive_MSS")]["percent_complete"], 28.367, abs_tol=1e-3)
    assert prev_map[("GSE39582", "D1_Plus_D2_Permissive_MSS")]["n_positive_complete"] == 140
    assert prev_map[("GSE39582", "D1_Plus_D2_Permissive_MSS")]["n_complete"] == 444
    assert math.isclose(prev_map[("GSE39582", "D1_Plus_D2_Permissive_MSS")]["percent_complete"], 31.532, abs_tol=1e-3)

    assert prev_map[("TCGA", "AACR_A065_Positive_MSS")]["n_positive_complete"] == 72
    assert prev_map[("TCGA", "AACR_A065_APM_Intact_MSS")]["n_positive_complete"] == 62
    assert prev_map[("GSE39582", "AACR_A065_Positive_MSS")]["n_positive_complete"] == 77
    assert prev_map[("GSE39582", "AACR_A065_APM_Intact_MSS")]["n_positive_complete"] == 71

    assert prev_map[("TCGA", "CMS1_Like_MSS")]["n_positive_complete"] == 71
    assert prev_map[("GSE39582", "CMS1_Like_MSS")]["n_positive_complete"] == 54
    assert prev_map[("TCGA", "Trafficking_Permissive_Primed_MSS")]["n_positive_complete"] == 24
    assert prev_map[("GSE39582", "Trafficking_Permissive_Primed_MSS")]["n_positive_complete"] == 22

    # Verify KRAS G12D CD47 and BRAF V600E D6 Cliff's deltas
    kras_cd47 = GENOTYPE_CONTRASTS["KRAS_G12D_vs_WT_z_CD47"]
    assert math.isclose(kras_cd47["TCGA_raw"]["cliffs_delta"], -0.294771, abs_tol=1e-6)
    assert math.isclose(kras_cd47["TCGA_raw"]["q"], 0.030412, abs_tol=1e-6)
    assert math.isclose(kras_cd47["TCGA_CD8_PTPRC"]["cliffs_delta"], -0.264706, abs_tol=1e-6)
    assert math.isclose(kras_cd47["TCGA_CD8_PTPRC"]["q"], 0.046026, abs_tol=1e-6)
    braf_d6 = GENOTYPE_CONTRASTS["BRAF_V600E_vs_WT_D6_CMS1like"]
    assert math.isclose(braf_d6["TCGA_raw"]["cliffs_delta"], 0.770948, abs_tol=1e-6)
    assert math.isclose(braf_d6["GSE39582_raw"]["cliffs_delta"], 0.306122, abs_tol=1e-6)

    # Verify exact manual formula parity on non-trivial z-score vector
    z_test = {
        "CXCL9": 1.2, "CXCL10": 1.6, "CXCL11": 0.8, "CXCR3": 0.4,       # CXCR3_Trafficking = 1.0
        "TGFB1": -0.6, "ACTA2": -0.3, "FAP": -0.9, "COL1A1": -0.3, "FN1": -0.3, "THBS1": 0.0,  # TGFB = -0.4
        "TAP1": 0.8, "TAP2": 0.8, "PSMB8": 0.8, "PSMB9": 0.8,
        "B2M": 0.8, "HLA-A": 0.8, "HLA-B": 0.8, "HLA-C": 0.8,           # APM_TAP_Score = 0.8
        "VEGFA": -0.4, "TOX": -0.6, "ZNF683": 0.8, "GNLY": 0.4,         # VEGF_TOX = -0.5 - 0.6 = -1.1 -> D3 = +1.1
        "CD8A": 1.0, "CD8B": 1.0, "CD4": 0.5, "IL7R": 0.5, "TCF7": 0.5,
        "FOXP3": -0.4, "IL2RA": -0.2,                                   # S_triad = 0.7 - (-0.3) = 1.0 -> D4 = 1.0
        "CXCL3": -0.5, "CXCL8": -0.5, "CD47": -0.4, "SLC25A22": -0.2, "IRF2": 0.4,  # KRAS_Axis = -2.0
        "SPP1": -0.8, "CXCL12": -0.4, "ITGAM": -0.4, "CD68": -0.4,      # SPP1_Core = -0.5
    }
    expr = _expr_from_z(z_test, "TCGA")
    s = standalone_score(expr, "TCGA", mss=True)

    assert math.isclose(s["CXCR3_Trafficking"], 1.0, abs_tol=1e-12)
    assert math.isclose(s["TGFB_CAF_Barrier"], -0.4, abs_tol=1e-12)
    assert math.isclose(s["D1_trafficking"], 1.4, abs_tol=1e-12)
    assert math.isclose(s["D2_presentation"], 0.8, abs_tol=1e-12)
    assert math.isclose(s["VEGF_TOX_Exhaustion"], -1.1, abs_tol=1e-12)
    assert math.isclose(s["D3_effector"], 1.1, abs_tol=1e-12)
    assert math.isclose(s["D4_triad"], 1.0, abs_tol=1e-12)
    assert math.isclose(s["KRAS_IRF2_CXCL3_Axis"], -2.0, abs_tol=1e-12)
    assert math.isclose(s["SPP1_Core_Barrier"], -0.5, abs_tol=1e-12)

    _, _, mm, ms, dm, ds, _, _ = REF["TCGA"]
    mm_map = dict(zip(FEATURES, mm))
    ms_map = dict(zip(FEATURES, ms))
    expected_d5 = -0.5 * (
        ((-2.0 - mm_map["KRAS_IRF2_CXCL3_Axis"]) / ms_map["KRAS_IRF2_CXCL3_Axis"])
        + ((-0.5 - mm_map["SPP1_Core_Barrier"]) / ms_map["SPP1_Core_Barrier"])
    )
    assert math.isclose(s["D5_myeloid"], expected_d5, abs_tol=1e-12)

    fv = [(s[k] - a) / b for k, a, b in zip(FEATURES, CLASS_MODULE_MEAN, CLASS_MODULE_SD)]
    expected_d6 = INTERCEPT + sum(w * v for w, v in zip(COEF, fv))
    assert math.isclose(s["D6_CMS1like"], expected_d6, abs_tol=1e-12)

    dz = [
        (s[dom] - m) / sd
        for dom, m, sd in zip(DOMAINS, dm, ds)
    ]
    assert math.isclose(s["D1_D2_Vaccine_Effector_Index"], 0.5 * (dz[0] + dz[1]), abs_tol=1e-12)
    assert math.isclose(s["D1_D2_D3_D4_Immune_Competence_Index"], sum(dz[:4]) / 4.0, abs_tol=1e-12)
    assert s["APM_intact"] is True
    assert s["AACR_A065_Positive_MSS"] is True
    assert s["D1_Plus_D2_Permissive_MSS"] is True


def test_all_four_break_crc001_regimen_strata_and_pathway_8d_integration() -> None:
    """Verify Stratum 1, Stratum 2 (+ B2M LOF flag), Stratum 3 (+ KRAS G12D CD47 rule), and Stratum 4."""
    # --- 1. STRATUM_1_DIRECT_VACCINE_PERMISSIVE ---
    z_s1 = {
        "CXCL9": 1.2, "CXCL10": 1.4, "CXCL11": 1.0, "CXCR3": 0.8,
        "FAP": -0.8, "TGFB1": -0.5, "ACTA2": -0.4, "COL1A1": -0.4, "FN1": -0.4, "THBS1": -0.3,
        "TAP1": 0.9, "TAP2": 0.9, "PSMB8": 0.8, "PSMB9": 0.7,
        "B2M": 0.8, "HLA-A": 0.7, "HLA-B": 0.7, "HLA-C": 0.7,
    }
    res_s1 = score_stc1010_crc_case(
        {
            "patient_id": "CRC-S1",
            "reference": "TCGA",
            "log_expression": _expr_from_z(z_s1, "TCGA"),
            "kras_mut": True,
        }
    )
    assert res_s1["ruo"] is True
    assert res_s1["contract_status"] == CONTRACT_STATUS_V2
    assert res_s1["gates"]["APM_Intact"] is True
    assert res_s1["gates"]["AACR_A065_Positive"] is True
    assert res_s1["break_crc001_regimen_plan"]["primary_stratum"] == STRATUM_1_DIRECT_VACCINE_PERMISSIVE
    # In Stratum 1 (D1 > 0 and APM_Intact), STC_1010 ranks #1 across all 8 CRC trials
    assert res_s1["pathway_8d"]["top_trial"] == "STC_1010"
    assert res_s1["pathway_8d"]["stc1010_rank"] == 1
    assert res_s1["pathway_8d"]["patient_vector_8d"][5] == 0.70  # io axis elevated to 0.70

    # --- 2. STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN (+ B2M LOF flag) ---
    z_s2 = dict(z_s1)
    z_s2["TAP1"] = -1.2  # Below TCGA MSS Q20 (-0.8460)
    z_s2["TAP2"] = -1.1  # Below TCGA MSS Q20 (-0.8426)
    z_s2["B2M"] = 0.5    # B2M intact
    res_s2 = score_stc1010_crc_case(
        {
            "patient_id": "CRC-S2",
            "reference": "TCGA",
            "log_expression": _expr_from_z(z_s2, "TCGA"),
            "b2m_truncating_lof": False,
        }
    )
    assert res_s2["domains"]["D1_trafficking"] > 0.0
    assert res_s2["gates"]["APM_Intact"] is False
    assert res_s2["break_crc001_regimen_plan"]["primary_stratum"] == STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN
    assert res_s2["break_crc001_regimen_plan"]["barrier_flags"]["b2m_intact"] is True
    assert "B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED" not in res_s2["break_crc001_regimen_plan"]["flags"]

    # Now test with b2m_truncating_lof == True -> emits B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED and gates STC_1010 G=0.0
    res_s2_b2m = score_stc1010_crc_case(
        {
            "patient_id": "CRC-S2-B2M-LOF",
            "reference": "TCGA",
            "log_expression": _expr_from_z(z_s2, "TCGA"),
            "b2m_truncating_lof": True,
        }
    )
    assert "B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED" in res_s2_b2m["break_crc001_regimen_plan"]["flags"]
    assert res_s2_b2m["pathway_8d"]["stc1010_gated_fit"] == 0.0

    # --- 3. STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE (+ KRAS G12D CD47 rule) ---
    z_s3 = dict(z_s1)
    z_s3["VEGFA"] = 2.0
    z_s3["TOX"] = 2.0
    z_s3["FOXP3"] = 1.8
    z_s3["IL2RA"] = 1.8
    z_s3["SPP1"] = 1.5
    res_s3 = score_stc1010_crc_case(
        {
            "patient_id": "CRC-S3-G12D",
            "reference": "TCGA",
            "log_expression": _expr_from_z(z_s3, "TCGA"),
            "kras_g12d": True,
        }
    )
    plan_s3 = res_s3["break_crc001_regimen_plan"]
    assert plan_s3["primary_stratum"] == STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE
    assert plan_s3["barrier_flags"]["vegf_tox_exhaustion_gt_q75"] is True
    assert plan_s3["barrier_flags"]["treg_load_gt_q75"] is True
    assert "KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47" in plan_s3["flags"]
    assert any("KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47" in m for m in plan_s3["regimen_mandates"])

    # --- 4. STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION ---
    z_s4 = {
        "CXCL9": -1.0, "CXCL10": -1.2, "CXCL11": -1.0, "CXCR3": -0.8,
        "TGFB1": 1.5, "ACTA2": 1.4, "FAP": 1.6, "COL1A1": 1.5, "FN1": 1.4, "THBS1": 1.2,
    }
    res_s4 = score_stc1010_crc_case(
        {
            "patient_id": "CRC-S4-LIVER",
            "reference": "TCGA",
            "log_expression": _expr_from_z(z_s4, "TCGA"),
            "liver_metastasis": True,
            "kras_mut": True,
        }
    )
    plan_s4 = res_s4["break_crc001_regimen_plan"]
    assert res_s4["domains"]["D1_trafficking"] <= 0.0
    assert plan_s4["primary_stratum"] == STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION
    assert plan_s4["barrier_flags"]["tgfb_caf_barrier_gt_q75"] is True
    assert plan_s4["barrier_flags"]["liver_metastasis"] is True
    # In Stratum 4 (D1 <= 0), io is suppressed to 0.10 so anti-VEGF (HORIZON_III / VELOUR) outranks STC_1010
    assert res_s4["pathway_8d"]["patient_vector_8d"][5] == 0.10
    assert res_s4["pathway_8d"]["top_trial"] != "STC_1010"


def test_fastapi_crc_stc1010_score_endpoint_and_422_fail_closed(client: TestClient) -> None:
    """Verify POST /v1/crc/stc1010/score returns 200 on valid inputs and 422 on empty/partial inputs."""
    # 1. Valid 42-gene GSE39582 payload -> HTTP 200
    z_gse = {
        "CXCL9": 1.0, "CXCL10": 1.2, "CXCL11": 0.9, "CXCR3": 0.7,
        "FAP": -0.6, "TGFB1": -0.4, "ACTA2": -0.3, "COL1A1": -0.3, "FN1": -0.3, "THBS1": -0.3,
        "TAP1": 0.6, "TAP2": 0.6, "PSMB8": 0.6, "PSMB9": 0.6,
        "B2M": 0.6, "HLA-A": 0.6, "HLA-B": 0.6, "HLA-C": 0.6,
    }
    resp_ok = client.post(
        "/v1/crc/stc1010/score",
        json={
            "patient_id": "GSE39582-TEST-01",
            "reference": "GSE39582",
            "log_expression": _expr_from_z(z_gse, "GSE39582"),
            "kras_mut": True,
            "kras_g12d": True,
        },
    )
    assert resp_ok.status_code == 200, resp_ok.text
    body = resp_ok.json()
    assert body["ruo"] is True
    assert body["contract_status"] == "exploratory_research_rationale"
    assert body["reference"] == "GSE39582"
    assert body["gates"]["APM_Intact"] is True
    assert body["gates"]["AACR_A065_Positive"] is True
    assert body["break_crc001_regimen_plan"]["primary_stratum"] == STRATUM_1_DIRECT_VACCINE_PERMISSIVE
    assert "KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47" in body["break_crc001_regimen_plan"]["flags"]
    assert body["pathway_8d"]["top_trial"] == "STC_1010"
    assert len(body["pathway_8d"]["ranked_trials"]) == 8

    # 2. Valid precomputed domain payload -> HTTP 200
    resp_dom = client.post(
        "/v1/crc/stc1010/score",
        json={
            "patient_id": "PRECOMP-DOM-01",
            "reference": "TCGA",
            "D1_trafficking": 0.85,
            "D2_presentation": 0.65,
            "D3_effector": 0.40,
            "D4_triad": 0.35,
            "D5_myeloid": 0.20,
            "D6_CMS1like": -1.5,
            "APM_intact": True,
            "z_FAP": -0.5,
            "z_CXCL10": 0.9,
        },
    )
    assert resp_dom.status_code == 200, resp_dom.text
    body_dom = resp_dom.json()
    assert body_dom["input_mode"] == "precomputed_domains"
    assert body_dom["gates"]["APM_Intact"] is True
    assert body_dom["gates"]["AACR_A065_Positive"] is True
    assert body_dom["break_crc001_regimen_plan"]["primary_stratum"] == STRATUM_1_DIRECT_VACCINE_PERMISSIVE

    # 3. Empty payload {} -> HTTP 422 fail-closed
    resp_empty = client.post("/v1/crc/stc1010/score", json={})
    assert resp_empty.status_code == 422
    detail_empty = resp_empty.json()["detail"]
    assert detail_empty["error"] == "missing_required_inputs"

    # 4. Incomplete 42-gene log_expression -> HTTP 422 listing missing genes
    partial_expr = {"CXCL9": 8.5, "CXCL10": 7.9}
    resp_partial = client.post(
        "/v1/crc/stc1010/score",
        json={"reference": "TCGA", "log_expression": partial_expr},
    )
    assert resp_partial.status_code == 422
    detail_partial = resp_partial.json()["detail"]
    assert detail_partial["error"] == "missing_required_inputs"
    assert "TAP1" in detail_partial["missing"]
    assert "SPP1" in detail_partial["missing"]

    # 5. Incomplete precomputed domains -> HTTP 422 listing missing domains
    resp_missing_dom = client.post(
        "/v1/crc/stc1010/score",
        json={"reference": "TCGA", "D1_trafficking": 0.5, "D2_presentation": 0.4},
    )
    assert resp_missing_dom.status_code == 422
    detail_dom = resp_missing_dom.json()["detail"]
    assert detail_dom["error"] == "missing_required_inputs"
    assert "D3_effector" in detail_dom["missing"]
    assert "D6_CMS1like" in detail_dom["missing"]
