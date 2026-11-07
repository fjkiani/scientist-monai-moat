"""Unit tests for the deterministic 8D/7D PATH A + Contraindication Mask Ranker.

Verifies:
  1. All 18/18 raw PATH A fits across the 6 GBM trial classes and 3 BrM patient
     profiles match ``8d_vector_run_results.json`` to 4 decimal places.
  2. EGFR TKI (``egfr_tkis_erlotinib``) scores ``0.9625`` (#1) on ``EGFR_mut_BrM``
     (``egfr_mut=True``), ``0.0000`` (blocked, ``raw_fit=0.6325``,
     ``axis_masked_fit=0.1833``) on ``ZEB1_high_KRAS_mut`` (``kras_mut=True,
     egfr_mut=False``), and ``0.0000`` (``raw_fit=0.7334``,
     ``axis_masked_fit=0.0000``) on ``PTEN_null_ZEB1_high`` (``pten_null=True,
     egfr_mut=False``).
  3. ``IMblaze370`` scores raw ``0.8262`` on ``P_A`` and ``0.6751`` on ``P_B``,
     and is gated to ``0.0000`` (``axis_masked_fit=0.2267`` on ``P_A`` and
     ``0.0252`` on ``P_B``).
  4. On the 485-patient TCGA-COADREAD MSS cohort:
     - Legacy rules: ``STC_1010`` is #1 in ``0 / 473`` non-POLE (``TMB < 10``)
       patients and ``9 / 485`` overall (``TMB >= 25`` POLE only).
     - Mode ``iii-C`` (Two-Gate Precision Stratum, ``gate2_z >= -0.0257``):
       ``STC_1010`` is #1 in ``146 / 146`` (``100.00%``) of Gate-2 Immune-Hot
       non-POLE (``TMB < 10``) MSS CRC patients (``84/84`` Hot Triple-WT and
       ``62/62`` Hot RAS/RAF-mut) and ``0 / 327`` (``0.00%``) of Immune-Cold
       non-POLE MSS CRC patients.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app
from oncology_arbiter.arbiter.pathway_8d_ranker import (
    AXES_7D,
    AXES_8D,
    CONTRACT_STATUS,
    CRC_TRIAL_VECTORS_8D,
    ContraindicationContext,
    GATE2_CUTPOINT_Z,
    GBM_PROFILE_VECTORS_8D,
    GBM_REPORTED_FITS_8D,
    GBM_TRIAL_VECTORS_8D,
    IO_CORE_PATIENT_P_A_7D,
    IO_CORE_PATIENT_P_B_7D,
    IO_CORE_TRIAL_VECTORS_7D,
    build_crc_patient_vector_8d,
    evaluate_brm_gbm_two_door_arbiter,
    evaluate_contraindication_gate,
    evaluate_trial_path_a,
    fit_axis_masked,
    fit_path_a,
    l2_norm,
    load_tcga_coadread_mss_485_cohort,
    rank_trials_path_a,
    U6_CLOSURE_VERDICT,
)


def test_gbm_18_of_18_raw_path_a_fits_match_run_results_json() -> None:
    """Test 1: All 18/18 raw PATH A fits across 6 GBM classes x 3 profiles match to 4 decimals."""
    assert AXES_8D == ("ddr", "mapk", "pi3k", "vegf", "her2", "io", "efflux", "rss")
    assert AXES_7D == ("ddr", "mapk", "pi3k", "vegf", "her2", "io", "efflux")

    # Verify exact trial L2 norms
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["egfr_tkis_erlotinib"]), math.sqrt(1.19), rel_tol=1e-12)
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["checkpoint_CheckMate"]), math.sqrt(0.90), rel_tol=1e-12)
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["cilengitide_CENTRIC"]), math.sqrt(0.62), rel_tol=1e-12)
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["mtor_pi3k_inhibitors"]), math.sqrt(0.82), rel_tol=1e-12)
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["rindopepimut_ACTIV"]), 0.70, rel_tol=1e-12)
    assert math.isclose(l2_norm(GBM_TRIAL_VECTORS_8D["bevacizumab_AVAglio"]), 0.90, rel_tol=1e-12)

    count = 0
    for profile_id, p_vec in GBM_PROFILE_VECTORS_8D.items():
        expected_map = GBM_REPORTED_FITS_8D[profile_id]
        for trial_id, expected_fit in expected_map.items():
            t_vec = GBM_TRIAL_VECTORS_8D[trial_id]
            computed_fit = fit_path_a(p_vec, t_vec)
            assert round(computed_fit, 4) == expected_fit, (
                f"Mismatch on {profile_id} / {trial_id}: got {computed_fit:.6f}, expected {expected_fit:.4f}"
            )
            count += 1

    assert count == 18


def test_egfr_tki_contraindication_gate_across_three_gbm_brm_profiles() -> None:
    """Test 2: EGFR TKI scores 0.9625 (#1) on EGFR_mut_BrM and 0.0000 (blocked) on KRAS-mut & PTEN-null."""
    # 1. Positive control: EGFR_mut_BrM (egfr_mut=True)
    ctx_egfr = ContraindicationContext(egfr_mut=True, mss=True)
    ranked_egfr = rank_trials_path_a(
        GBM_PROFILE_VECTORS_8D["EGFR_mut_BrM"],
        GBM_TRIAL_VECTORS_8D,
        ctx=ctx_egfr,
    )
    top_egfr = ranked_egfr[0]
    assert top_egfr.trial_id == "egfr_tkis_erlotinib"
    assert top_egfr.rank == 1
    assert round(top_egfr.raw_fit, 4) == 0.9625
    assert round(top_egfr.axis_masked_fit, 4) == 0.9625
    assert round(top_egfr.gated_fit, 4) == 0.9625
    assert round(top_egfr.combined_score, 4) == 0.8138
    assert top_egfr.contraindicated is False
    assert top_egfr.ruo is True
    assert top_egfr.contract_status == CONTRACT_STATUS

    # 2. ZEB1_high_KRAS_mut (kras_mut=True, egfr_mut=False) -> blocked (G=0.0, axis_masked=0.1833)
    ctx_kras = ContraindicationContext(kras_mut=True, egfr_mut=False, mss=True)
    ranked_kras = rank_trials_path_a(
        GBM_PROFILE_VECTORS_8D["ZEB1_high_KRAS_mut"],
        GBM_TRIAL_VECTORS_8D,
        ctx=ctx_kras,
    )
    egfr_on_kras = next(r for r in ranked_kras if r.trial_id == "egfr_tkis_erlotinib")
    assert round(egfr_on_kras.raw_fit, 4) == 0.6325
    assert round(egfr_on_kras.axis_masked_fit, 4) == 0.1833
    assert egfr_on_kras.gated_fit == 0.0000
    assert egfr_on_kras.contraindicated is True
    assert egfr_on_kras.rank == 6
    assert "UPSTREAM_EGFR_TKI_BYPASSED_BY_DOWNSTREAM_KRAS_OR_PTEN" in egfr_on_kras.reasons
    # Checkpoint CheckMate (0.5692) and Cilengitide CENTRIC (0.5207) now lead
    assert ranked_kras[0].trial_id == "checkpoint_CheckMate"
    assert round(ranked_kras[0].gated_fit, 4) == 0.5692

    # 3. PTEN_null_ZEB1_high (pten_null=True, egfr_mut=False) -> blocked (G=0.0, raw=0.7334, axis_masked=0.0000)
    ctx_pten = ContraindicationContext(pten_null=True, egfr_mut=False, mss=True)
    ranked_pten = rank_trials_path_a(
        GBM_PROFILE_VECTORS_8D["PTEN_null_ZEB1_high"],
        GBM_TRIAL_VECTORS_8D,
        ctx=ctx_pten,
    )
    egfr_on_pten = next(r for r in ranked_pten if r.trial_id == "egfr_tkis_erlotinib")
    assert round(egfr_on_pten.raw_fit, 4) == 0.7334
    assert round(egfr_on_pten.axis_masked_fit, 4) == 0.0000
    assert egfr_on_pten.gated_fit == 0.0000
    assert egfr_on_pten.contraindicated is True
    assert egfr_on_pten.rank == 6
    assert ranked_pten[0].trial_id == "mtor_pi3k_inhibitors"
    assert round(ranked_pten[0].gated_fit, 4) == 0.9497


def test_imblaze370_raw_and_gated_scores_on_p_a_and_p_b() -> None:
    """Test 3: IMblaze370 scores raw 0.8262 on P_A and 0.6751 on P_B, and is gated to 0.0000."""
    t_imb = IO_CORE_TRIAL_VECTORS_7D["IMblaze370"]
    p_a_prompt = (0.10, 0.70, 0.15, 0.25, 0.05, 0.40, 0.15)
    p_b_prompt = (0.10, 0.70, 0.15, 0.25, 0.05, 0.10, 0.15)

    for p_a, p_b in ((p_a_prompt, p_b_prompt), (IO_CORE_PATIENT_P_A_7D, IO_CORE_PATIENT_P_B_7D)):
        # P_A (liver_met=False, mss=True)
        ctx_a = ContraindicationContext(kras_mut=True, mss=True, liver_met=False)
        res_a = evaluate_trial_path_a("IMblaze370", p_a, t_imb, ctx=ctx_a)
        assert round(res_a.raw_fit, 4) == 0.8262
        assert round(res_a.axis_masked_fit, 4) == 0.2267
        assert res_a.gated_fit == 0.0000
        assert res_a.contraindicated is True
        assert "FALSIFIED_MEKI_PDL1_IN_MSS_CRC_OS_HR_1_00" in res_a.reasons

        # P_B (liver_met=True / tgfb_barrier_high=True, mss=True)
        ctx_b = ContraindicationContext(
            kras_mut=True,
            mss=True,
            liver_met=True,
            tgfb_barrier_high=True,
        )
        res_b = evaluate_trial_path_a("IMblaze370", p_b, t_imb, ctx=ctx_b)
        assert round(res_b.raw_fit, 4) == 0.6751
        assert round(res_b.axis_masked_fit, 4) == 0.0252
        assert res_b.gated_fit == 0.0000
        assert res_b.contraindicated is True


def test_tcga_coadread_485_mss_cohort_legacy_vs_mode_iiic_twogate_precision() -> None:
    """Test 4: Verify 485-patient TCGA-COADREAD MSS cohort rankings under Legacy and Mode iii-C."""
    cohort = load_tcga_coadread_mss_485_cohort()
    assert len(cohort) == 485

    non_pole = [row for row in cohort if row["is_non_pole_tmb_lt_10"]]
    assert len(non_pole) == 473

    # --- Part A: Legacy rules (mapk=0.70 flat, TMB-only io, no contraindication gate) ---
    legacy_stc_top1_all = 0
    legacy_stc_top1_non_pole = 0
    legacy_stc_top1_pole_ge25 = 0

    for row in cohort:
        p_leg = build_crc_patient_vector_8d(
            hrd_proxy=row["hrd_proxy"],
            brca_somatic=row["brca_somatic"],
            tmb=row["tmb"],
            aneuploidy_score=row["aneuploidy_score"],
            fraction_genome_altered=row["fraction_genome_altered"],
            mode="legacy",
        )
        ranked_leg = rank_trials_path_a(
            p_leg,
            CRC_TRIAL_VECTORS_8D,
            ctx=None,
            apply_gate=False,
        )
        # STC_1010 strict win against best comparator
        stc_fit = next(r.gated_fit for r in ranked_leg if r.trial_id == "STC_1010")
        best_comp = max(r.gated_fit for r in ranked_leg if r.trial_id != "STC_1010")
        if stc_fit > best_comp:
            legacy_stc_top1_all += 1
            if row["is_non_pole_tmb_lt_10"]:
                legacy_stc_top1_non_pole += 1
            if row["tmb"] >= 25.0:
                legacy_stc_top1_pole_ge25 += 1

    assert legacy_stc_top1_non_pole == 0, f"Expected 0/473 in non-POLE legacy, got {legacy_stc_top1_non_pole}"
    assert legacy_stc_top1_all == 9, f"Expected 9/485 overall in legacy, got {legacy_stc_top1_all}"
    assert legacy_stc_top1_pole_ge25 == 9

    # --- Part B: Mode iii-C (Two-Gate Precision Stratum, gate2_z >= -0.0257) ---
    for conv in ("f07", "ws5v2"):
        hot_non_pole_total = 0
        hot_non_pole_stc_wins = 0
        hot_wt_total = 0
        hot_wt_stc_wins = 0
        hot_mut_total = 0
        hot_mut_stc_wins = 0
        cold_non_pole_total = 0
        cold_non_pole_stc_wins = 0

        for row in non_pole:
            ctx = ContraindicationContext(
                kras_mut=row["kras_mut"],
                nras_mut=row["nras_mut"],
                braf_mut=row["braf_mut"],
                mss=True,
                gate2_immune_hot=row["gate2_immune_hot"],
            )
            p_iiic = build_crc_patient_vector_8d(
                hrd_proxy=row["hrd_proxy"],
                brca_somatic=row["brca_somatic"],
                tmb=row["tmb"],
                aneuploidy_score=row["aneuploidy_score"],
                fraction_genome_altered=row["fraction_genome_altered"],
                kras_mut=row["kras_mut"],
                nras_mut=row["nras_mut"],
                braf_mut=row["braf_mut"],
                gate2_z=row["gate2_z"],
                gate2_immune_hot=row["gate2_immune_hot"],
                mode="iiic_twogate_precision",
                mapk_convention=conv,
            )
            ranked = rank_trials_path_a(
                p_iiic,
                CRC_TRIAL_VECTORS_8D,
                ctx=ctx,
                apply_gate=True,
            )
            stc_fit = next(r.gated_fit for r in ranked if r.trial_id == "STC_1010")
            best_comp = max(r.gated_fit for r in ranked if r.trial_id != "STC_1010")
            stc_wins = stc_fit > best_comp

            # Verify PACCE is always hard-blocked and PRIME/PEAK_FOLFIRI are blocked in RAS/RAF-mut
            pacce_res = next(r for r in ranked if r.trial_id == "PACCE")
            assert pacce_res.gated_fit == 0.0
            if row["ras_raf_mut"]:
                prime_res = next(r for r in ranked if r.trial_id == "PRIME")
                assert prime_res.gated_fit == 0.0

            is_hot = row["gate2_z"] >= GATE2_CUTPOINT_Z
            if is_hot:
                hot_non_pole_total += 1
                if stc_wins:
                    hot_non_pole_stc_wins += 1
                if row["triple_wt"]:
                    hot_wt_total += 1
                    if stc_wins:
                        hot_wt_stc_wins += 1
                else:
                    hot_mut_total += 1
                    if stc_wins:
                        hot_mut_stc_wins += 1
            else:
                cold_non_pole_total += 1
                if stc_wins:
                    cold_non_pole_stc_wins += 1

        assert hot_non_pole_total == 146
        assert hot_non_pole_stc_wins == 146, (
            f"Expected 146/146 Hot wins under {conv}, got {hot_non_pole_stc_wins}/{hot_non_pole_total}"
        )
        assert hot_wt_total == 84
        assert hot_wt_stc_wins == 84
        assert hot_mut_total == 62
        assert hot_mut_stc_wins == 62
        assert cold_non_pole_total == 327
        assert cold_non_pole_stc_wins == 0, (
            f"Expected 0/327 Cold wins under {conv}, got {cold_non_pole_stc_wins}/{cold_non_pole_total}"
        )


def test_modes_iiia_iiib_and_io_core_7d_and_n0147_axis_mask() -> None:
    """Test 5: Verify Modes iii-A (449/473), iii-B (273/473), N0147/PRIME/PACCE masks, and 7 IO-CORE trials."""
    # 1. Verify all 7 IO-CORE 7D trials on P_A and P_B against path_a_scores.yaml
    expected_io_core = {
        "QUILT_2_004": (0.5815, 0.3553),
        "GRANITE": (0.5411, 0.2832),
        "GVAX": (0.4486, 0.1495),
        "IMblaze370": (0.8262, 0.6751),
        "AtezoTRIBE": (0.5843, 0.3671),
        "MEDITREME": (0.5037, 0.2089),
        "CO_26": (0.4000, 0.1000),
    }
    for trial_id, (exp_a, exp_b) in expected_io_core.items():
        t_vec = IO_CORE_TRIAL_VECTORS_7D[trial_id]
        assert round(fit_path_a(IO_CORE_PATIENT_P_A_7D, t_vec), 4) == exp_a
        assert round(fit_path_a(IO_CORE_PATIENT_P_B_7D, t_vec), 4) == exp_b

    # 2. Verify N0147 axis mask (g_mapk = 0.0 in RAS/RAF-mut)
    ctx_ras = ContraindicationContext(kras_mut=True, mss=True)
    gate_G_n0147, mask_g_n0147, reasons_n0147 = evaluate_contraindication_gate("N0147", ctx_ras, dim=8)
    assert gate_G_n0147 == 1.0
    assert mask_g_n0147 == (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert "CETUXIMAB_COMPONENT_MASKED_IN_RAS_RAF_MUT" in reasons_n0147

    # 3. Verify Modes iii-A and iii-B across the 485 cohort (and raw fallback loader parity)
    cohort_v2 = load_tcga_coadread_mss_485_cohort()
    cohort_fallback = load_tcga_coadread_mss_485_cohort(v2_gated_csv_path= None if False else __import__("pathlib").Path("/nonexistent/v2.csv"))
    assert len(cohort_v2) == len(cohort_fallback) == 485

    iiia_non_pole_wins = 0
    iiia_all_wins = 0
    iiib_non_pole_wins = 0
    iiib_hot_non_pole_wins = 0
    iiib_all_wins = 0

    for row_v2, row_fb in zip(cohort_v2, cohort_fallback):
        assert row_v2["patient_id"] == row_fb["patient_id"]
        assert row_v2["ras_raf_mut"] == row_fb["ras_raf_mut"]
        assert row_v2["gate2_immune_hot"] == row_fb["gate2_immune_hot"]

        ctx = ContraindicationContext(
            kras_mut=row_v2["kras_mut"],
            nras_mut=row_v2["nras_mut"],
            braf_mut=row_v2["braf_mut"],
            mss=True,
            gate2_immune_hot=row_v2["gate2_immune_hot"],
        )

        # Mode iii-A
        p_iiia = build_crc_patient_vector_8d(
            hrd_proxy=row_v2["hrd_proxy"],
            brca_somatic=row_v2["brca_somatic"],
            tmb=row_v2["tmb"],
            aneuploidy_score=row_v2["aneuploidy_score"],
            fraction_genome_altered=row_v2["fraction_genome_altered"],
            kras_mut=row_v2["kras_mut"],
            nras_mut=row_v2["nras_mut"],
            braf_mut=row_v2["braf_mut"],
            gate2_z=row_v2["gate2_z"],
            mode="iiia_f07_continuous",
        )
        ranked_iiia = rank_trials_path_a(p_iiia, CRC_TRIAL_VECTORS_8D, ctx=ctx)
        stc_iiia = next(r.gated_fit for r in ranked_iiia if r.trial_id == "STC_1010")
        comp_iiia = max(r.gated_fit for r in ranked_iiia if r.trial_id != "STC_1010")
        if stc_iiia > comp_iiia:
            iiia_all_wins += 1
            if row_v2["is_non_pole_tmb_lt_10"]:
                iiia_non_pole_wins += 1

        # Mode iii-B
        p_iiib = build_crc_patient_vector_8d(
            hrd_proxy=row_v2["hrd_proxy"],
            brca_somatic=row_v2["brca_somatic"],
            tmb=row_v2["tmb"],
            aneuploidy_score=row_v2["aneuploidy_score"],
            fraction_genome_altered=row_v2["fraction_genome_altered"],
            kras_mut=row_v2["kras_mut"],
            nras_mut=row_v2["nras_mut"],
            braf_mut=row_v2["braf_mut"],
            gate2_z=row_v2["gate2_z"],
            mode="iiib_ws5v2_continuous",
        )
        ranked_iiib = rank_trials_path_a(p_iiib, CRC_TRIAL_VECTORS_8D, ctx=ctx)
        stc_iiib = next(r.gated_fit for r in ranked_iiib if r.trial_id == "STC_1010")
        comp_iiib = max(r.gated_fit for r in ranked_iiib if r.trial_id != "STC_1010")
        if stc_iiib > comp_iiib:
            iiib_all_wins += 1
            if row_v2["is_non_pole_tmb_lt_10"]:
                iiib_non_pole_wins += 1
                if row_v2["gate2_immune_hot"]:
                    iiib_hot_non_pole_wins += 1

    assert iiia_non_pole_wins == 449
    assert iiia_all_wins == 461
    assert iiib_non_pole_wins == 273
    assert iiib_hot_non_pole_wins == 144
    assert iiib_all_wins == 285


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_fastapi_ranker_8d_crc_hot_vs_cold_and_modes(client: TestClient) -> None:
    """Test 6: POST /v1/ranker/8d/score for domain='crc' (Hot vs Cold under mode='iii_c' and direct vector)."""
    # 1. Immune-Hot KRAS-mut MSS CRC patient via clinical_features -> STC_1010 #1, PRIME/PEAK/PACCE gated 0.0
    resp_hot = client.post(
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
    assert resp_hot.status_code == 200, resp_hot.text
    body_hot = resp_hot.json()
    assert body_hot["ruo"] is True
    assert "RESEARCH USE ONLY" in body_hot["ruo_disclaimer"]
    assert body_hot["domain"] == "crc"
    assert body_hot["mode"] == "iii_c"
    assert body_hot["axis_order"] == list(AXES_8D)
    assert len(body_hot["patient_vector"]) == 8
    assert body_hot["patient_vector"][5] == 0.70  # Hot tertile z >= 0.3026 -> io = 0.70
    assert body_hot["top_trial"] == "STC_1010"
    assert body_hot["recommended_trial_id"] == "STC_1010"
    assert body_hot["top_gated_trial_id"] == "STC_1010"
    assert body_hot["stc_1010_rank_1"] is True
    assert len(body_hot["rankings"]) == 8
    assert body_hot["rankings"][0]["trial_id"] == "STC_1010"
    assert body_hot["rankings"][0]["rank"] == 1
    assert body_hot["rankings"][0]["fit_gated"] > 0.70

    by_id_hot = {r["trial_id"]: r for r in body_hot["rankings"]}
    assert by_id_hot["PRIME"]["fit_gated"] == 0.0
    assert by_id_hot["PRIME"]["contraindicated"] is True
    assert by_id_hot["PEAK_FOLFIRI"]["fit_gated"] == 0.0
    assert by_id_hot["PACCE"]["fit_gated"] == 0.0

    # 2. Immune-Cold KRAS-mut MSS CRC patient via clinical_features -> STC_1010 NOT #1, PRIME/PEAK/PACCE gated 0.0
    resp_cold = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "CRC",
            "mode": "iii-c",
            "clinical_features": {
                "tmb": 3.3,
                "hrd": "HRD-Intermediate",
                "fga": 0.2671,
                "kras_mut": True,
                "immune_z": -0.60,
                "gate2_hot": False,
            },
        },
    )
    assert resp_cold.status_code == 200, resp_cold.text
    body_cold = resp_cold.json()
    assert body_cold["ruo"] is True
    assert body_cold["domain"] == "crc"
    assert body_cold["mode"] == "iii_c"
    assert body_cold["patient_vector"][5] == 0.10  # Cold suppression -> io = 0.10
    assert body_cold["stc_1010_rank_1"] is False
    assert body_cold["top_trial"] != "STC_1010"
    assert body_cold["rankings"][0]["trial_id"] != "STC_1010"

    by_id_cold = {r["trial_id"]: r for r in body_cold["rankings"]}
    assert by_id_cold["PRIME"]["fit_gated"] == 0.0
    assert by_id_cold["PRIME"]["trial_gate_G"] == 0.0
    assert "DOWNSTREAM_RAS_RAF_MUTATION_BLOCKS_ANTI_EGFR" in by_id_cold["PRIME"]["contraindication_reasons"]
    assert by_id_cold["PEAK_FOLFIRI"]["fit_gated"] == 0.0
    assert by_id_cold["PACCE"]["fit_gated"] == 0.0
    assert "HARD_BLOCK_ANTI_EGFR_PLUS_BEV_TOXICITY (PACCE OS HR=1.43)" in by_id_cold["PACCE"]["contraindication_reasons"]

    # 3. Direct 8D patient_vector input for CRC
    resp_vec = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "crc",
            "mode": "ii_a",
            "patient_vector": [0.15, 0.55, 0.15, 0.25, 0.05, 0.50, 0.1258, 0.1068],
            "contraindications": {"kras_mut": True},
        },
    )
    assert resp_vec.status_code == 200, resp_vec.text
    body_vec = resp_vec.json()
    assert body_vec["mode"] == "ii_a"
    assert body_vec["top_trial"] == "STC_1010"
    assert body_vec["stc_1010_rank_1"] is True


def test_fastapi_ranker_8d_gbm_profiles_and_direct_vectors(client: TestClient) -> None:
    """Test 7: POST /v1/ranker/8d/score for domain='gbm' across EGFR_mut_BrM, ZEB1_high_KRAS_mut, and PTEN_null."""
    # 1. EGFR_mut_BrM -> egfr_tkis_erlotinib #1 with 0.9625
    resp_egfr = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "gbm",
            "profile_id": "EGFR_mut_BrM",
        },
    )
    assert resp_egfr.status_code == 200, resp_egfr.text
    body_egfr = resp_egfr.json()
    assert body_egfr["ruo"] is True
    assert body_egfr["domain"] == "gbm"
    assert body_egfr["axis_order"] == list(AXES_8D)
    assert body_egfr["top_trial"] == "egfr_tkis_erlotinib"
    assert len(body_egfr["rankings"]) == 6
    assert body_egfr["rankings"][0]["trial_id"] == "egfr_tkis_erlotinib"
    assert round(body_egfr["rankings"][0]["fit_raw"], 4) == 0.9625
    assert round(body_egfr["rankings"][0]["fit_gated"], 4) == 0.9625
    assert body_egfr["rankings"][0]["contraindicated"] is False

    # 2. ZEB1_high_KRAS_mut -> egfr_tkis_erlotinib gated to 0.0000 (raw 0.6325, axis_masked 0.1833), checkpoint_CheckMate #1 (0.5692)
    resp_kras = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "gbm",
            "profile_id": "ZEB1_high_KRAS_mut",
        },
    )
    assert resp_kras.status_code == 200, resp_kras.text
    body_kras = resp_kras.json()
    assert body_kras["top_trial"] == "checkpoint_CheckMate"
    assert round(body_kras["rankings"][0]["fit_gated"], 4) == 0.5692
    egfr_kras = next(r for r in body_kras["rankings"] if r["trial_id"] == "egfr_tkis_erlotinib")
    assert round(egfr_kras["fit_raw"], 4) == 0.6325
    assert round(egfr_kras["fit_axis_masked"], 4) == 0.1833
    assert egfr_kras["fit_gated"] == 0.0
    assert egfr_kras["contraindicated"] is True
    assert egfr_kras["rank"] == 6

    # 3. Direct 8D vector for PTEN_null_ZEB1_high -> mtor_pi3k_inhibitors #1 (0.9497), egfr_tkis_erlotinib gated to 0.0000
    resp_pten = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "gbm",
            "patient_vector": list(GBM_PROFILE_VECTORS_8D["PTEN_null_ZEB1_high"]),
            "contraindications": {"pten_null": True, "egfr_mut": False},
        },
    )
    assert resp_pten.status_code == 200, resp_pten.text
    body_pten = resp_pten.json()
    assert body_pten["top_trial"] == "mtor_pi3k_inhibitors"
    assert round(body_pten["rankings"][0]["fit_gated"], 4) == 0.9497
    egfr_pten = next(r for r in body_pten["rankings"] if r["trial_id"] == "egfr_tkis_erlotinib")
    assert round(egfr_pten["fit_raw"], 4) == 0.7334
    assert round(egfr_pten["fit_axis_masked"], 4) == 0.0000
    assert egfr_pten["fit_gated"] == 0.0


def test_fastapi_ranker_8d_io_core_p_a_and_p_b(client: TestClient) -> None:
    """Test 8: POST /v1/ranker/8d/score for domain='io_core' on P_A and P_B -> IMblaze370 gated to 0.0000."""
    # 1. P_A via profile_id
    resp_pa = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "io_core",
            "profile_id": "P_A",
        },
    )
    assert resp_pa.status_code == 200, resp_pa.text
    body_pa = resp_pa.json()
    assert body_pa["ruo"] is True
    assert body_pa["domain"] == "io_core"
    assert body_pa["axis_order"] == list(AXES_7D)
    assert len(body_pa["patient_vector"]) == 7
    assert len(body_pa["rankings"]) == 7
    assert body_pa["top_trial"] == "AtezoTRIBE"
    assert round(body_pa["rankings"][0]["fit_gated"], 4) == 0.5843

    imb_pa = next(r for r in body_pa["rankings"] if r["trial_id"] == "IMblaze370")
    assert round(imb_pa["fit_raw"], 4) == 0.8262
    assert round(imb_pa["fit_axis_masked"], 4) == 0.2267
    assert imb_pa["fit_gated"] == 0.0
    assert imb_pa["contraindicated"] is True
    assert "FALSIFIED_MEKI_PDL1_IN_MSS_CRC_OS_HR_1_00" in imb_pa["contraindication_reasons"]

    # 2. P_B via direct 8D vector (automatically sliced to 7D for io_core) + liver_met/tgfb_high
    resp_pb = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "io_core",
            "patient_vector": [*IO_CORE_PATIENT_P_B_7D, 0.10],
            "contraindications": {"kras_mut": True, "mss": True, "liver_met": True, "tgfb_high": True},
        },
    )
    assert resp_pb.status_code == 200, resp_pb.text
    body_pb = resp_pb.json()
    assert len(body_pb["patient_vector"]) == 7
    imb_pb = next(r for r in body_pb["rankings"] if r["trial_id"] == "IMblaze370")
    assert round(imb_pb["fit_raw"], 4) == 0.6751
    assert round(imb_pb["fit_axis_masked"], 4) == 0.0252
    assert imb_pb["fit_gated"] == 0.0
    assert imb_pb["contraindicated"] is True


def test_fastapi_ranker_8d_fail_closed_422_validation(client: TestClient) -> None:
    """Test 9: Fail-closed HTTP 422 on empty body {}, wrong vector length, out-of-bounds values, and invalid domain/mode."""
    # 1. Empty body {}
    r_empty = client.post("/v1/ranker/8d/score", json={})
    assert r_empty.status_code == 422

    # 2. Only domain/mode without patient_vector, clinical_features, or profile_id
    r_no_inputs = client.post("/v1/ranker/8d/score", json={"domain": "crc", "mode": "iii_c"})
    assert r_no_inputs.status_code == 422

    # 3. Empty clinical_features dict {}
    r_empty_cf = client.post("/v1/ranker/8d/score", json={"domain": "crc", "clinical_features": {}})
    assert r_empty_cf.status_code == 422

    # 4. Wrong vector length (5D vector)
    r_5d = client.post(
        "/v1/ranker/8d/score",
        json={"domain": "crc", "patient_vector": [0.1, 0.2, 0.3, 0.4, 0.5]},
    )
    assert r_5d.status_code == 422

    # 5. Out-of-bounds vector component > 1.0 (1.5)
    r_oob_high = client.post(
        "/v1/ranker/8d/score",
        json={"domain": "crc", "patient_vector": [0.1, 1.5, 0.2, 0.3, 0.0, 0.5, 0.1, 0.1]},
    )
    assert r_oob_high.status_code == 422

    # 6. Out-of-bounds vector component < 0.0 (-0.1)
    r_oob_low = client.post(
        "/v1/ranker/8d/score",
        json={"domain": "gbm", "patient_vector": [0.1, -0.1, 0.2, 0.3, 0.0, 0.5, 0.1, 0.1]},
    )
    assert r_oob_low.status_code == 422

    # 7. Invalid domain
    r_bad_domain = client.post(
        "/v1/ranker/8d/score",
        json={"domain": "invalid_domain", "patient_vector": [0.1, 0.2, 0.2, 0.3, 0.0, 0.5, 0.1, 0.1]},
    )
    assert r_bad_domain.status_code == 422

    # 8. Invalid mode
    r_bad_mode = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "crc",
            "mode": "unknown_mode_xyz",
            "patient_vector": [0.1, 0.2, 0.2, 0.3, 0.0, 0.5, 0.1, 0.1],
        },
    )
    assert r_bad_mode.status_code == 422


def test_evaluate_brm_gbm_two_door_arbiter_door1_door2_myeloid_and_nanocarrier_lock(
    client: TestClient,
) -> None:
    """Test 10: Verify Two-Door BrM/GBM drug, resistance, myeloid firewall, and 3-Stage nanocarrier lock."""
    # 1. Door 1 EGFR-mutant BrM (low ZEB1/EMT, S_sen=0.0)
    res_d1 = evaluate_brm_gbm_two_door_arbiter(
        profile_id="EGFR_mut_BrM",
        zeb1_or_mesenchymal_high=False,
        h_clean_z=-0.5,
        s_sen_z=-0.5,
    )
    assert res_d1["ruo"] is True
    assert res_d1["primary_door"] == "DOOR_1_PROLIFERATIVE_DDR_EGFR"
    assert res_d1["is_door2"] is False
    assert res_d1["emt_resistant_egfr_tkis"] == []
    assert res_d1["myeloid_firewall"]["active"] is True
    assert res_d1["nanocarrier_3stage_lock"]["and_gate_triggered"] is True
    assert res_d1["nanocarrier_3stage_lock"]["unshielded_lrp1_monoligand_allowed"] is False
    assert res_d1["nanocarrier_3stage_lock"]["stage3_tumor_core_lock_antigens"] == [
        "MUC1",
        "CLDN7",
        "EPCAM",
        "DSG2",
    ]
    assert res_d1["u6_closure_verdict"]["claim_id"] == "U6"
    assert res_d1["u6_closure_verdict"]["nsclc_reproduction"]["spearman_rho"] == -0.6238
    assert res_d1["u6_closure_verdict"]["gbm_single_gene_zeb1_falsification"]["spearman_rho"] == 0.0846
    assert (
        res_d1["u6_closure_verdict"]["gbm_cns_two_door_rescue"]["GBM_S_sen_to_ITGAV"]["rho"]
        == -0.3415
    )

    # 2. Door 2 ZEB1-high / S_sen-high BrM/GBM -> AZD2014, AZD8055, GNE-317, GSK2256098C, Dasatinib,
    #    BCL-xL senolytics, Verubecestat+Anti-VSIG4 prioritized; Osimertinib/Zorifertinib EMT-resistant;
    #    Venetoclax & AZD5991 contraindicated!
    res_d2 = evaluate_brm_gbm_two_door_arbiter(
        profile_id="ZEB1_high_KRAS_mut",
        zeb1_or_mesenchymal_high=True,
        h_clean_z=0.80,
        s_sen_z=1.50,
    )
    assert res_d2["is_door2"] is True
    assert res_d2["is_s_sen_high"] is True
    assert res_d2["zeb1_or_mesenchymal_high"] is True
    assert res_d2["primary_door"] == "DOOR_2_HIBERNATING_COLONIZATION_SENOLYTIC_FAK_MTOR"
    assert "Osimertinib" in res_d2["emt_resistant_egfr_tkis"]
    assert "Zorifertinib (AZD3759)" in res_d2["emt_resistant_egfr_tkis"]

    drug_ids = {d["drug_id"] for d in res_d2["prioritized_drugs"]}
    assert {
        "AZD2014",
        "AZD8055",
        "GNE-317",
        "GSK2256098C",
        "Dasatinib",
        "WEHI-539_Navitoclax_ABT-737",
        "Verubecestat_Anti_VSIG4_Myeloid_Reprogramming",
    }.issubset(drug_ids)

    contra_names = {c["drug_name"] for c in res_d2["contraindicated_drugs"]}
    assert "Venetoclax" in contra_names
    assert "AZD5991" in contra_names
    assert "Osimertinib" in contra_names
    assert "Zorifertinib (AZD3759)" in contra_names
    assert "Temozolomide" in contra_names

    # 3. Verify POST /v1/ranker/8d/score for domain='gbm' surfaces brm_gbm_two_door_arbiter
    resp_gbm = client.post(
        "/v1/ranker/8d/score",
        json={
            "domain": "gbm",
            "profile_id": "PTEN_null_ZEB1_high",
            "clinical_features": {
                "zeb1_or_mesenchymal_high": True,
                "h_clean_z": 1.1,
                "s_sen_z": 1.3,
            },
        },
    )
    assert resp_gbm.status_code == 200, resp_gbm.text
    body = resp_gbm.json()
    arb = body["brm_gbm_two_door_arbiter"]
    assert arb is not None
    assert arb["is_door2"] is True
    assert arb["is_s_sen_high"] is True
    assert "Osimertinib" in arb["emt_resistant_egfr_tkis"]
    assert any(c["drug_name"] == "Venetoclax" for c in arb["contraindicated_drugs"])
    assert any(c["drug_name"] == "AZD5991" for c in arb["contraindicated_drugs"])
    assert U6_CLOSURE_VERDICT["claim_id"] == "U6"



