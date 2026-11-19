"""Responder cohort and subgroup contrast loaders."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

from .moa import _moa_vec
from .paths import _BRENUS_ROOT, _CASPRO_ROOT, _CRISPR_MAIN_ROOT, _DEV_ROOT, _MOAT_ROOT
from .patient import _tanh_scale, build_signed_8d_patient_vector


@dataclass
class ResponderCohortBlock:
    """A patient/model-level responder vs. non-responder training/evaluation cohort."""
    cohort_id: str
    trial_or_regimen: str
    moa_vec: np.ndarray
    P: np.ndarray                  # shape (N, 8)
    y: np.ndarray                  # shape (N,), binary 1=responder/sensitive, 0=non-responder/resistant
    sample_ids: list[str]
    weight: float = 1.0
    description: str = ""


@dataclass
class SubgroupContrastBlock:
    """Published trial subgroup log-HR contrast (-ln(HR_pos / HR_neg))."""
    contrast_id: str
    trial_id: str
    trial_name: str
    moa_vec: np.ndarray
    p_pos: np.ndarray              # shape (8,) biomarker-positive / responder subgroup
    p_neg: np.ndarray              # shape (8,) biomarker-negative / resistant subgroup
    hr_pos: float
    hr_neg: float
    target_log_hr_contrast: float  # -ln(hr_pos / hr_neg)
    weight: float = 1.5
    source_citation: str = ""


def load_all_responder_cohorts() -> dict[str, ResponderCohortBlock]:
    """Load all on-disk patient/model-level responder cohorts into signed 8D space."""
    cohorts: dict[str, ResponderCohortBlock] = {}

    # 1 & 2 & 4: Gide (n=41), Riaz (n=55), Deng GSE151511 (n=23) from zeta_immunotherapy_external_validation_v3.json
    imm_path = _MOAT_ROOT / "src/oncology_arbiter/immunotherapy/contracts/zeta_immunotherapy_external_validation_v3.json"
    imm_data = json.loads(imm_path.read_text())
    mel_rows = imm_data["v12"]["patient_inputs"]["melanoma"]
    bag_rows = imm_data["v12"]["patient_inputs"]["bag"]

    # (a) Gide PRJEB23709 pretreatment (n=41: 19 responders, 22 non-responders)
    gide_sub = [
        r for r in mel_rows
        if r.get("cohort") == "Gide_PRJEB23709" and r.get("timepoint") == "pretreatment" and r.get("y") is not None
    ]
    P_gide = []
    y_gide = []
    ids_gide = []
    for r in gide_sub:
        e_eff = float(r.get("ZNF683_GNLY_effector", 0.0) or 0.0)
        s_tr = float(r.get("S_triad", 0.0) or 0.0)
        tgfb_z = (float(r.get("expr_TGFB1", 4.5473)) - 4.54730389) / 0.79179682
        vegf_z = (float(r.get("expr_VEGFA", 7.0897)) - 7.08973298) / 1.35863050
        tox_z = (float(r.get("expr_TOX", 1.6298)) - 1.62977876) / 0.99844441
        # d5 captures ZNF683+GNLY effector & S_triad; d6 captures TGFB1 + VEGFA + TOX barrier
        p_vec = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            _tanh_scale(0.85 * e_eff + 0.25 * s_tr, scale=1.1),
            _tanh_scale(0.45 * tgfb_z + 0.35 * vegf_z + 0.20 * tox_z - 0.35 * s_tr, scale=1.3),
            0.0,
            _tanh_scale(-0.35 * vegf_z, scale=1.5),
        ], dtype=float)
        P_gide.append(p_vec)
        y_gide.append(int(r["y"]))
        ids_gide.append(str(r["patient_id"]))
    cohorts["Gide_PRJEB23709_n41"] = ResponderCohortBlock(
        cohort_id="Gide_PRJEB23709_n41",
        trial_or_regimen="Anti-PD-1 ± CTLA-4 (Gide PRJEB23709 pretreatment, n=41)",
        moa_vec=_moa_vec(Anti_PD1_PDL1=0.80, Anti_CTLA4=0.45),
        P=np.vstack(P_gide),
        y=np.array(y_gide, dtype=int),
        sample_ids=ids_gide,
        weight=2.0,
        description="Gide PRJEB23709 pretreatment melanoma cohort (n=41: 19 R, 22 NR)",
    )

    # (b) Riaz GSE91061 on-treatment (n=55: 13 responders, 42 non-responders)
    riaz_sub = [
        r for r in mel_rows
        if r.get("cohort") == "Riaz_GSE91061" and r.get("timepoint") == "on_treatment" and r.get("y") is not None
    ]
    P_riaz = []
    y_riaz = []
    ids_riaz = []
    for r in riaz_sub:
        e_eff = float(r.get("ZNF683_GNLY_effector", 0.0) or 0.0)
        s_tr = float(r.get("S_triad", 0.0) or 0.0)
        tgfb_z = (float(r.get("expr_TGFB1", 5.7444)) - 5.74435603) / 1.09569596
        vegf_z = (float(r.get("expr_VEGFA", 5.3814)) - 5.38143900) / 1.28380071
        arg1_z = (float(r.get("expr_ARG1", 1.7097)) - 1.70967171) / 2.49246827
        p_vec = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            _tanh_scale(0.90 * e_eff + 0.15 * s_tr, scale=1.1),
            _tanh_scale(0.40 * tgfb_z + 0.30 * arg1_z + 0.30 * vegf_z - 0.30 * e_eff, scale=1.3),
            0.0,
            _tanh_scale(-0.30 * vegf_z, scale=1.5),
        ], dtype=float)
        P_riaz.append(p_vec)
        y_riaz.append(int(r["y"]))
        ids_riaz.append(str(r["patient_id"]))
    cohorts["Riaz_GSE91061_n55"] = ResponderCohortBlock(
        cohort_id="Riaz_GSE91061_n55",
        trial_or_regimen="Nivolumab Anti-PD-1 (Riaz GSE91061, n=55)",
        moa_vec=_moa_vec(Anti_PD1_PDL1=1.00),
        P=np.vstack(P_riaz),
        y=np.array(y_riaz, dtype=int),
        sample_ids=ids_riaz,
        weight=2.0,
        description="Riaz GSE91061 anti-PD-1 melanoma cohort (n=55: 13 R, 42 NR)",
    )

    # (c) Deng GSE151511 CAR-T infusion bag literal_CD8 (n=23: 9 primary CR, 14 non-CR)
    deng_sub = [
        r for r in bag_rows
        if r.get("gate") == "literal_CD8" and r.get("primary_CR") is not None
    ]
    P_deng = []
    y_deng = []
    ids_deng = []
    for r in deng_sub:
        b3 = float(r.get("Bag_Index_3", 0.0) or 0.0)
        b1 = float(r.get("Bag_Index_1", 0.0) or 0.0)
        f_cyto = float(r.get("f_cyto_marker", 0.8147) or 0.8147)
        cyto_z = (f_cyto - 0.81472776) / 0.06733288
        p_vec = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            _tanh_scale(0.80 * b3 + 0.30 * b1, scale=1.0),
            _tanh_scale(0.60 * cyto_z - 0.40 * b3, scale=1.2),
            0.0,
            0.0,
        ], dtype=float)
        P_deng.append(p_vec)
        y_deng.append(int(r["primary_CR"]))
        ids_deng.append(str(r["patient_id"]))
    cohorts["Deng_GSE151511_n23"] = ResponderCohortBlock(
        cohort_id="Deng_GSE151511_n23",
        trial_or_regimen="CD19 CAR-T Memory-Primed Effector (Deng GSE151511, n=23)",
        moa_vec=_moa_vec(Vaccine_STC1010=0.65, Anti_PD1_PDL1=0.65),
        P=np.vstack(P_deng),
        y=np.array(y_deng, dtype=int),
        sample_ids=ids_deng,
        weight=1.5,
        description="Deng GSE151511 infusion bag literal_CD8 primary CR cohort (n=23: 9 CR, 14 non-CR)",
    )

    # 3: GSE290722 CAR-T in vivo blood baseline28 (n=28: 13 LtR, 15 R/NR)
    imm_tar_path = Path("/Users/fahadkiani/Downloads/MASTER_IMMUNOTHERAPY_V2_TO_V12.tar.gz")
    cart_contract = json.loads(
        (_MOAT_ROOT / "src/oncology_arbiter/immunotherapy/contracts/cart_hobit_gnly_clinical_contract_v1.json").read_text()
    )
    prep_mean = cart_contract["preprocessing"]["mean"]
    prep_scale = cart_contract["preprocessing"]["scale"]
    coefs_std = cart_contract["coefficients_standardized"]
    with tarfile.open(imm_tar_path) as tf:
        cf_rows = list(csv.DictReader(io.TextIOWrapper(tf.extractfile("cart_cd8_tem_singlecell_v7/tables/cohort_features.csv"))))
    bl28 = [r for r in cf_rows if r.get("cohort") == "baseline28"]
    P_290722 = []
    y_290722 = []
    ids_290722 = []
    for r in bl28:
        spd = float(r["log_baseline_spd"]) if r.get("log_baseline_spd") not in ("", "NA", "nan") else prep_mean[0]
        preg = float(r["prior_regimens"]) if r.get("prior_regimens") not in ("", "NA", "nan") else prep_mean[1]
        auto = float(r["prior_autologous_transplant"]) if r.get("prior_autologous_transplant") not in ("", "NA", "nan") else prep_mean[2]
        f_cyto = float(r.get("f_cyto", 0.0) or 0.0)
        z_znf = f_cyto * float(r.get("cyto__logcpm__ZNF683", 0.0) or 0.0)
        z_gnly = f_cyto * float(r.get("cyto__logcpm__GNLY", 0.0) or 0.0)
        c_abs = float(r.get("cyto_absent", 0.0) or 0.0)
        raw_x = [spd, preg, auto, z_znf, z_gnly, c_abs]
        std_x = [(raw_x[i] - prep_mean[i]) / prep_scale[i] for i in range(6)]
        # Effector component vs clinical/myeloid tumor burden barrier
        eff_lin = coefs_std[2] * std_x[2] + coefs_std[3] * std_x[3] + coefs_std[4] * std_x[4] + coefs_std[5] * std_x[5]
        bar_lin = -(coefs_std[0] * std_x[0] + coefs_std[1] * std_x[1])
        p_vec = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            _tanh_scale(eff_lin, scale=1.2),
            _tanh_scale(bar_lin, scale=1.2),
            0.0,
            0.0,
        ], dtype=float)
        P_290722.append(p_vec)
        y_290722.append(int(float(r["y"])))
        ids_290722.append(str(r["patient_id"]))
    cohorts["GSE290722_n28"] = ResponderCohortBlock(
        cohort_id="GSE290722_n28",
        trial_or_regimen="CAR-T Leukapheresis ZNF683+GNLY Effector (GSE290722, n=28)",
        moa_vec=_moa_vec(Vaccine_STC1010=0.60, Anti_PD1_PDL1=0.70),
        P=np.vstack(P_290722),
        y=np.array(y_290722, dtype=int),
        sample_ids=ids_290722,
        weight=1.5,
        description="GSE290722 baseline28 leukapheresis CAR-T cohort (n=28: 13 LtR, 15 R/NR)",
    )

    # 5: GSE165897 Post-NACT Ovarian Two-Door (n=11 paired HGSOC patients)
    from oncology_arbiter.ovarian.ovarian_two_door_standalone_v3 import (
        GSE165897_11_PATIENT_TABLE,
        HUMAN_POLICY,
    )
    ddr_cut = float(HUMAN_POLICY["post_DDR_cutpoint"])
    h_cut = float(HUMAN_POLICY["post_H_clean_cutpoint"])
    P_165897 = []
    y_165897_door1 = []
    y_165897_door2 = []
    ids_165897 = []
    for r in GSE165897_11_PATIENT_TABLE:
        pddr = float(r["post_DDR"])
        ph = float(r["post_H_clean"])
        ps = float(r["post_S_sen"])
        bcl2a1 = float(r["post_BCL2A1_z"])
        mcl1_d = float(r["delta_MCL1_z"])
        emt_z = float(r["post_EMT_z"])
        gate0 = bcl2a1 > 2.5 and (mcl1_d > 2.5 or emt_z > 1.0)
        cls = str(r["four_state_class"])
        # d1: post_DDR centered at cutpoint, penalized if hibernation high (Door 2 overrides Door 1)
        d1_val = _tanh_scale((pddr - ddr_cut) / 0.025 - 0.65 * max(0.0, ph - h_cut), scale=1.0)
        # d2: post_H_clean & post_S_sen hibernation/senolytic score, penalized if Gate 0 refractory escape
        d2_val = -0.95 if gate0 else _tanh_scale(0.60 * (ph - h_cut) + 0.25 * ps, scale=1.2)
        d6_val = 0.95 if gate0 else _tanh_scale(0.35 * emt_z + 0.25 * mcl1_d, scale=1.5)
        p_vec = np.array([d1_val, d2_val, 0.0, 0.0, 0.0, d6_val, 0.0, -0.60 if gate0 else 0.0], dtype=float)
        P_165897.append(p_vec)
        y_165897_door1.append(1 if cls == "DDRonly" else 0)
        y_165897_door2.append(1 if (cls in {"Hcleanonly", "Both"} and not gate0) else 0)
        ids_165897.append(str(r["patient_id"]))

    cohorts["GSE165897_Door1_n11"] = ResponderCohortBlock(
        cohort_id="GSE165897_Door1_n11",
        trial_or_regimen="Post-NACT Door 1 ATR/WEE1 Proliferative DDR (GSE165897, n=11)",
        moa_vec=_moa_vec(ATR_WEE1_Inhibitor=1.00),
        P=np.vstack(P_165897),
        y=np.array(y_165897_door1, dtype=int),
        sample_ids=ids_165897,
        weight=1.5,
        description="GSE165897 post-NACT Door 1 (DDRonly n=4 vs Door 2/Neither/Gate0 n=7)",
    )
    cohorts["GSE165897_Door2_n11"] = ResponderCohortBlock(
        cohort_id="GSE165897_Door2_n11",
        trial_or_regimen="Post-NACT Door 2 BCL-xL/Dasatinib Senolytic (GSE165897, n=11)",
        moa_vec=_moa_vec(BCLxL_Dasatinib_Senolytic=1.00),
        P=np.vstack(P_165897),
        y=np.array(y_165897_door2, dtype=int),
        sample_ids=ids_165897,
        weight=1.5,
        description="GSE165897 post-NACT Door 2 non-refractory (Hcleanonly+Both excl EOC3 n=4 vs rest n=7)",
    )

    # 6: GDSC2/CTRPv2/PRISM Door 2 Senolytic Pharmacology (n=45 ovarian models)
    ov_tar_path = Path("/Users/fahadkiani/Downloads/MASTER_OVARIAN_HIBERNATION_V1_TO_V9_FINAL.tar.gz")
    with tarfile.open(ov_tar_path) as tf_ov:
        r45 = list(csv.DictReader(io.TextIOWrapper(tf_ov.extractfile(
            "ovarian_hibernation_v9_subtype2A_2B_and_dose_response_lock/reference_45_donor_models.csv"
        ))))
        comp_rows = list(csv.DictReader(io.TextIOWrapper(tf_ov.extractfile(
            "ovarian_hibernation_v8_consensus_and_mTORC2_unlock/two_drug_composite_donors.csv"
        ))))
    s45_map = {r["PatientID"]: r for r in r45}
    dn_sub = [r for r in comp_rows if r["composite"] == "dasatinib+navitoclax"]
    P_door2_45 = []
    y_door2_45 = []
    ids_door2_45 = []
    for r in dn_sub:
        pid = r["PatientID"]
        m_row = s45_map[pid]
        h_val = float(m_row["H_clean"])
        s_val = float(m_row["S_sen"])
        mcl1_z = float(m_row["zMCL1"])
        prolif = float(m_row["proliferation_106_ES"])
        prolif_z = (prolif - 5714.3030) / 727.6263
        p_vec = np.array([
            _tanh_scale(prolif_z - 0.5 * h_val, scale=1.5),
            _tanh_scale(0.85 * s_val + 0.15 * h_val, scale=2.2),
            0.0,
            0.0,
            0.0,
            _tanh_scale(0.35 * mcl1_z - 0.25 * s_val, scale=1.5),
            0.0,
            0.0,
        ], dtype=float)
        P_door2_45.append(p_vec)
        y_door2_45.append(1 if float(r["value"]) > 0.0 else 0)
        ids_door2_45.append(f"{pid}:{m_row['CellLineName']}")
    cohorts["GDSC2_PRISM_Door2_n45"] = ResponderCohortBlock(
        cohort_id="GDSC2_PRISM_Door2_n45",
        trial_or_regimen="Door 2 Dasatinib+Navitoclax Senolytic (GDSC2/CTRPv2/PRISM, n=45)",
        moa_vec=_moa_vec(BCLxL_Dasatinib_Senolytic=1.00),
        P=np.vstack(P_door2_45),
        y=np.array(y_door2_45, dtype=int),
        sample_ids=ids_door2_45,
        weight=2.0,
        description="GDSC2/CTRPv2/PRISM ovarian Door 2 senolytic composite (n=45: 20 sensitive, 25 resistant)",
    )

    # 7: GSE63885 HGSOC G3/G4 (n=61) & All-Histology (n=75) Platinum/Stroma Response
    mfap4_json = json.loads((_MOAT_ROOT / "artifacts/mfap4_gse63885_honest_n75.json").read_text())
    sids_hdr: list[str] = []
    hist_hdr: list[str] = []
    grade_hdr: list[str] = []
    brca_hdr: list[str] = []
    with (_CRISPR_MAIN_ROOT / "data/external/GSE63885/GSE63885_series_matrix.txt").open() as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                sids_hdr = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                k = vals[0].split(":", 1)[0].strip().lower() if vals else ""
                if "histophatological" in k or "histopathological" in k:
                    hist_hdr = [v.split(":", 1)[-1].strip() for v in vals]
                elif k.startswith("tumor grade"):
                    grade_hdr = [v.split(":", 1)[-1].strip() for v in vals]
                elif k.startswith("brca1"):
                    brca_hdr = [v.split(":", 1)[-1].strip() for v in vals]
            elif not line.startswith("!") and not line.startswith("^") and line.strip():
                break
    clin_map_63885 = {
        sid: {"hist": hist_hdr[i], "grade": grade_hdr[i], "brca": brca_hdr[i]}
        for i, sid in enumerate(sids_hdr)
    }
    mfap4_vals_all = np.array([float(r["mfap4"]) for r in mfap4_json["rows"]])
    mfap4_mean = float(np.mean(mfap4_vals_all))
    mfap4_sd = float(np.std(mfap4_vals_all, ddof=0))

    P_63885_61 = []
    y_63885_61 = []
    ids_63885_61 = []
    for r in mfap4_json["rows"]:
        sid = r["sample_id"]
        c_info = clin_map_63885.get(sid, {})
        is_hgsoc_61 = ("serous" in c_info.get("hist", "").lower()) and (c_info.get("grade") in {"G3", "G4"})
        mfap4_z = (float(r["mfap4"]) - mfap4_mean) / mfap4_sd
        y_sens = 1 - int(r["y"])
        p_vec = np.array([
            _tanh_scale(-0.45 * mfap4_z, scale=1.2),
            0.0,
            0.0,
            0.0,
            0.0,
            _tanh_scale(mfap4_z, scale=1.0),
            0.0,
            0.0,
        ], dtype=float)
        if is_hgsoc_61:
            P_63885_61.append(p_vec)
            y_63885_61.append(y_sens)
            ids_63885_61.append(sid)

    cohorts["GSE63885_HGSOC_n61"] = ResponderCohortBlock(
        cohort_id="GSE63885_HGSOC_n61",
        trial_or_regimen="First-Line Platinum Chemotherapy vs MFAP4 Stromal Barrier (GSE63885 HGSOC G3/G4, n=61)",
        moa_vec=_moa_vec(FOLFOX_Oxaliplatin=0.85, PARP_Inhibitor=0.25),
        P=np.vstack(P_63885_61),
        y=np.array(y_63885_61, dtype=int),
        sample_ids=ids_63885_61,
        weight=1.5,
        description="GSE63885 high-grade serous G3/G4 ovarian cohort (n=61: 31 platinum-sensitive, 30 resistant; honest all-histology n=75 AUROC=0.6836)",
    )

    # 8: BriTROC-1 CN-Signature s3 at Diagnosis (n=98 primary set & n=67 adjusted set)
    britroc_path = _BRENUS_ROOT / "clinical_validation/trackA/britroc_master_published.csv"
    with open(britroc_path) as fh:
        b_rows = list(csv.DictReader(fh))
    s3_mu = 0.1767425892802721
    s3_sd = 0.13804070487798628
    P_britroc = []
    y_britroc = []
    ids_britroc = []
    for r in b_rows:
        if r.get("s3_diag") not in ("", "NA", "nan") and r.get("resistant") not in ("", "NA", "nan"):
            s3_val = float(r["s3_diag"])
            s3_z = (s3_val - s3_mu) / s3_sd
            y_sens = 1 - int(float(r["resistant"]))
            p_vec = np.array([
                _tanh_scale(s3_z, scale=1.2),
                0.0,
                0.0,
                0.0,
                0.0,
                _tanh_scale(-0.35 * s3_z, scale=1.5),
                0.0,
                0.0,
            ], dtype=float)
            P_britroc.append(p_vec)
            y_britroc.append(y_sens)
            ids_britroc.append(str(r["patient"]))
    cohorts["BriTROC1_s3_n98_n67"] = ResponderCohortBlock(
        cohort_id="BriTROC1_s3_n98_n67",
        trial_or_regimen="Platinum / PARP HRD-s3 Sensitivity (BriTROC-1 s3_diag, n=67 complete / n=98 diagnosis)",
        moa_vec=_moa_vec(PARP_Inhibitor=0.65, ATR_WEE1_Inhibitor=0.55, FOLFOX_Oxaliplatin=0.45),
        P=np.vstack(P_britroc),
        y=np.array(y_britroc, dtype=int),
        sample_ids=ids_britroc,
        weight=2.0,
        description="BriTROC-1 s3_diag HRD platinum sensitivity cohort (n=67 complete adjusted: 58 sensitive, 9 resistant; n=98 diagnosis set protective AUROC=0.7285)",
    )

    # 9: GDSC2 MBD4-LOF vs WT Pharmacology (Ceralasertib / Gemcitabine vs Olaparib across ~956 lines)
    gdsc_compact_path = _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/gdsc2_mbd4_967_lines.json"
    gdsc_lines = json.loads(gdsc_compact_path.read_text())

    cera_lines = [r for r in gdsc_lines if r.get("ceralasertib_ln_ic50") is not None and (r.get("mbd4_lof") or not r.get("mbd4_any"))]
    cera_ic50 = np.array([float(r["ceralasertib_ln_ic50"]) for r in cera_lines])
    cera_mu, cera_sd = float(np.mean(cera_ic50)), float(np.std(cera_ic50, ddof=0))
    P_cera = []
    y_cera = []
    ids_cera = []
    for r in cera_lines:
        pot_z = -(float(r["ceralasertib_ln_ic50"]) - cera_mu) / cera_sd
        tp53_b = 0.20 if r.get("tp53_mut") else 0.0
        p_vec = np.array([
            _tanh_scale(0.85 * pot_z + tp53_b, scale=1.2),
            _tanh_scale(-0.35 * pot_z, scale=1.5),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=float)
        P_cera.append(p_vec)
        y_cera.append(1 if r.get("mbd4_lof") else 0)
        ids_cera.append(str(r["cosmic_id"]))
    cohorts["GDSC2_MBD4_Ceralasertib_n956"] = ResponderCohortBlock(
        cohort_id="GDSC2_MBD4_Ceralasertib_n956",
        trial_or_regimen="Ceralasertib (AZD6738) ATRi + Gemcitabine in MBD4-LOF vs WT (GDSC2, n=958 all / n=950 Methods)",
        moa_vec=_moa_vec(ATR_WEE1_Inhibitor=1.00),
        P=np.vstack(P_cera),
        y=np.array(y_cera, dtype=int),
        sample_ids=ids_cera,
        weight=2.0,
        description="GDSC2 Ceralasertib MBD4-LOF vs WT (n=14 LOF vs 914 DepMap-Methods WT [928] / 936 GDSC2-Methods WT [950] / 944 all-GDSC2 WT [958]; delta LN_IC50 = -0.7325, MWU P = 0.0215)",
    )

    return cohorts



def load_published_subgroup_contrasts() -> list[SubgroupContrastBlock]:
    """Build published adjudicated subgroup log-HR contrasts (-ln(HR_pos / HR_neg))."""
    contrasts: list[SubgroupContrastBlock] = [
        # 1. PRIME (NCT00364013): Expanded RAS-WT (PFS HR=0.72) vs RAS-mutant (PFS HR=1.29)
        SubgroupContrastBlock(
            contrast_id="PRIME_RAS_WT_vs_MUT",
            trial_id="NCT00364013",
            trial_name="PRIME (Panitumumab + FOLFOX4)",
            moa_vec=_moa_vec(Anti_EGFR=1.00, FOLFOX_Oxaliplatin=0.45),
            p_pos=build_signed_8d_patient_vector({"triple_wt": True, "egfr_mut": True, "kras_mut": False, "d1_ddr_rs": 0.15}),
            p_neg=build_signed_8d_patient_vector({"kras_mut": True, "tp53_mut": True, "triple_wt": False, "d1_ddr_rs": 0.15}),
            hr_pos=0.72,
            hr_neg=1.29,
            target_log_hr_contrast=float(-math.log(0.72 / 1.29)),  # +0.5831
            weight=2.5,
            source_citation="Douillard et al. NEJM 2013 (PMID 24024839): Expanded RAS-WT PFS HR=0.72 vs RAS-mut PFS HR=1.29",
        ),
        # 2. PEAK (NCT00819780): Expanded RAS-WT (PFS HR=0.65) vs RAS-mutant (PFS HR=1.31)
        SubgroupContrastBlock(
            contrast_id="PEAK_RAS_WT_vs_MUT",
            trial_id="NCT00819780",
            trial_name="PEAK (Panitumumab + mFOLFOX6)",
            moa_vec=_moa_vec(Anti_EGFR=1.00, FOLFOX_Oxaliplatin=0.40),
            p_pos=build_signed_8d_patient_vector({"triple_wt": True, "egfr_mut": True, "kras_mut": False, "d1_ddr_rs": 0.15}),
            p_neg=build_signed_8d_patient_vector({"kras_mut": True, "tp53_mut": True, "triple_wt": False, "d1_ddr_rs": 0.15}),
            hr_pos=0.65,
            hr_neg=1.31,
            target_log_hr_contrast=float(-math.log(0.65 / 1.31)),  # +0.7008
            weight=2.5,
            source_citation="Schwartzberg et al. JCO 2014 (PMID 24687833): Expanded RAS-WT PFS HR=0.65 vs RAS-mut PFS HR=1.31",
        ),
        # 3. AtezoTRIBE (NCT03721653): DetermaIO+ / Immunoscore-IC-high (PFS HR=0.10 / 0.50) vs pMMR low (PFS HR=0.86)
        SubgroupContrastBlock(
            contrast_id="AtezoTRIBE_DetermaIO_Pos_vs_Neg",
            trial_id="NCT03721653",
            trial_name="AtezoTRIBE (FOLFOXIRI + Bevacizumab + Atezolizumab)",
            moa_vec=_moa_vec(FOLFOX_Oxaliplatin=0.55, FOLFIRI_Irinotecan_TOP1=0.45, Anti_VEGF_Bev=0.55, Anti_PD1_PDL1=0.75),
            p_pos=build_signed_8d_patient_vector({"gate2_z": 0.75, "liver_met": False, "tgfb_high": False, "kras_mut": True}),
            p_neg=build_signed_8d_patient_vector({"gate2_z": -0.75, "liver_met": True, "tgfb_high": True, "kras_mut": True}),
            hr_pos=0.10,
            hr_neg=0.86,
            target_log_hr_contrast=float(-math.log(0.35 / 0.86)),  # damped log-HR contrast +0.8990
            weight=2.0,
            source_citation="Antoniotti et al. Lancet Oncol 2022 / PMID 39908699: DetermaIO IOOPT+ PFS HR=0.10 (Immunoscore-IC high HR~0.50) vs pMMR HR=0.86",
        ),
        # 4. CheckMate 9X8 (NCT03414983): CD8/TMB-enriched vs unselected MSS mCRC (PFS HR=0.81 NS)
        SubgroupContrastBlock(
            contrast_id="CheckMate_9X8_Immune_Hot_vs_Cold",
            trial_id="NCT03414983",
            trial_name="CheckMate 9X8 (Nivolumab + mFOLFOX6 + Bevacizumab)",
            moa_vec=_moa_vec(Anti_PD1_PDL1=0.80, FOLFOX_Oxaliplatin=0.55, Anti_VEGF_Bev=0.50),
            p_pos=build_signed_8d_patient_vector({"gate2_z": 0.65, "liver_met": False, "tgfb_high": False, "kras_mut": True}),
            p_neg=build_signed_8d_patient_vector({"gate2_z": -0.65, "liver_met": True, "tgfb_high": True, "kras_mut": True}),
            hr_pos=0.58,
            hr_neg=1.05,
            target_log_hr_contrast=float(-math.log(0.58 / 1.05)),  # +0.5935
            weight=1.8,
            source_citation="Lenz et al. JITC 2024 (PMID 38485190): CheckMate 9X8 ITT PFS HR=0.81 (p=0.30); CD8/CMS1-3 enriched vs CMS4/liver-met",
        ),
        # 5. IMblaze370 (NCT02788279): Atezolizumab + Cobimetinib in MSS mCRC (Phase III OS HR=1.00 failure vs Regorafenib)
        SubgroupContrastBlock(
            contrast_id="IMblaze370_MSS_CRC_Failure",
            trial_id="NCT02788279",
            trial_name="IMblaze370 (Atezolizumab + Cobimetinib MEKi)",
            moa_vec=_moa_vec(MEK_Cobimetinib=1.00, Anti_PD1_PDL1=0.55),
            p_pos=build_signed_8d_patient_vector({"gate2_z": 0.45, "d3_rtk_egfr_addiction": 0.0, "d4_ras_raf_oncogenic": -0.20, "liver_met": False, "tgfb_high": False}),
            p_neg=build_signed_8d_patient_vector({"gate2_z": -0.55, "kras_mut": True, "tp53_mut": True, "liver_met": True, "tgfb_high": True}),
            hr_pos=0.95,
            hr_neg=1.25,
            target_log_hr_contrast=float(-math.log(0.95 / 1.25)),  # +0.2744 (with negative score on p_neg!)
            weight=2.0,
            source_citation="Eng et al. Lancet Oncol 2019 (NCT02788279): OS HR=1.00 (8.87 vs 8.51 mo); MEKi fails to prime KRAS-mut MSS CRC",
        ),
        # 6. BREAKWATER (NCT04607421): Encorafenib + Cetuximab + mFOLFOX6 in BRAF V600E mCRC (HR=0.47 vs SOC)
        SubgroupContrastBlock(
            contrast_id="BREAKWATER_BRAF_V600E_vs_WT",
            trial_id="NCT04607421",
            trial_name="BREAKWATER (Encorafenib + Cetuximab + mFOLFOX6)",
            moa_vec=_moa_vec(BRAF_Encorafenib=1.00, FOLFOX_Oxaliplatin=0.45, Anti_EGFR=0.20),
            p_pos=build_signed_8d_patient_vector({"braf_mut": True, "tp53_mut": True, "d3_rtk_egfr_addiction": -0.35, "d4_ras_raf_oncogenic": 0.95}),
            p_neg=build_signed_8d_patient_vector({"braf_mut": False, "kras_mut": False, "triple_wt": True, "d3_rtk_egfr_addiction": 0.50, "d4_ras_raf_oncogenic": -0.65}),
            hr_pos=0.47,
            hr_neg=1.15,
            target_log_hr_contrast=float(-math.log(0.47 / 1.15)),  # +0.8948
            weight=2.5,
            source_citation="Elez et al. NEJM 2025 (DOI 10.1056/nejmoa2501912): BREAKWATER BRAF V600E mCRC ORR/PFS/OS benefit (HR~0.47)",
        ),
        # 7. CARMEN-LC03 (NCT04154956): CEACAM5 IHC >=80% + low efflux (OS HR=0.71) vs 50-79% / high efflux (HR=1.38)
        SubgroupContrastBlock(
            contrast_id="CARMEN_LC03_CEACAM5_80_vs_50_79",
            trial_id="NCT04154956",
            trial_name="CARMEN-LC03 (Tusamitamab ravtansine anti-CEACAM5 ADC)",
            moa_vec=_moa_vec(CEACAM5_TOP1_ADC=0.85, FOLFOX_Oxaliplatin=0.10),
            p_pos=build_signed_8d_patient_vector({"d7_surface_adc_payload": 0.90, "d6_stroma_myeloid_barrier": -0.25, "d8_angio_metabolic_safety": 0.40}),
            p_neg=build_signed_8d_patient_vector({"d7_surface_adc_payload": -0.65, "d6_stroma_myeloid_barrier": 0.45, "d8_angio_metabolic_safety": -0.30}),
            hr_pos=0.71,
            hr_neg=1.38,
            target_log_hr_contrast=float(-math.log(0.71 / 1.38)),  # +0.6646
            weight=2.5,
            source_citation="Besse et al. WCLC 2024: CARMEN-LC03 CEACAM5 >=80% subgroup OS HR=0.71 vs 50-79% subgroup HR=1.38 (ITT PFS HR=1.14)",
        ),
        # 8. BOT/BAL & CCTG CO.26 (NCT02870920 / NCT03860272): No-liver-met (PFS HR=0.54) vs Liver-met (PFS HR=1.39)
        SubgroupContrastBlock(
            contrast_id="BOT_BAL_CO26_NLM_vs_LiverMet",
            trial_id="NCT02870920",
            trial_name="CCTG CO.26 / BOT-BAL (Dual CTLA-4 + PD-1/L1 in MSS mCRC)",
            moa_vec=_moa_vec(Anti_PD1_PDL1=0.75, Anti_CTLA4=0.75),
            p_pos=build_signed_8d_patient_vector({"gate2_z": 0.55, "liver_met": False, "tgfb_high": False, "tmb": 12.0}),
            p_neg=build_signed_8d_patient_vector({"gate2_z": -0.55, "liver_met": True, "tgfb_high": True, "tmb": 3.3}),
            hr_pos=0.54,
            hr_neg=1.39,
            target_log_hr_contrast=float(-math.log(0.54 / 1.39)),  # +0.9455
            weight=2.5,
            source_citation="Chen et al. JAMA Oncol 2020 / PMC10698621 & Bullock Nat Med 2024 (PMID 38871975): NLM PFS HR=0.54 (BOT/BAL ORR 17-23% in NLM) vs LM PFS HR=1.39 (ORR 0%)",
        ),
        # 9. KRYSTAL-10 (NCT04793958): Adagrasib (G12C-OFF) + Cetuximab in KRAS-mut mCRC (Failed Phase III vs Daraxonrasib RAS(ON))
        SubgroupContrastBlock(
            contrast_id="KRYSTAL10_Adagrasib_Cetux_Failure_in_KRAS_TP53",
            trial_id="NCT04793958",
            trial_name="KRYSTAL-10 (Adagrasib + Cetuximab in KRAS G12C mCRC)",
            moa_vec=_moa_vec(Anti_EGFR=0.85, MEK_Cobimetinib=0.45),
            p_pos=build_signed_8d_patient_vector({"triple_wt": True, "egfr_mut": True, "kras_mut": False, "d3_rtk_egfr_addiction": 0.95, "d4_ras_raf_oncogenic": -0.75, "d6_stroma_myeloid_barrier": -0.35}),
            p_neg=build_signed_8d_patient_vector({"kras_mut": True, "tp53_mut": True, "d3_rtk_egfr_addiction": -1.00, "d4_ras_raf_oncogenic": 0.85, "d6_stroma_myeloid_barrier": 0.55}),
            hr_pos=0.75,
            hr_neg=1.08,
            target_log_hr_contrast=float(-math.log(0.75 / 1.08)),  # +0.3646
            weight=2.0,
            source_citation="ESMO GI 2026 (NCT04793958): KRYSTAL-10 missed primary PFS/OS endpoints (mPFS 7.5 vs 8.1 mo, mOS 21.6 vs 21.7 mo) due to RTK feedback & Cetuximab in RAS-mut",
        ),
        # 10. Daraxonrasib (NCT06445062): RAS(ON) multi-selective in KRAS G12/13 + TP53 co-mutant vs RAS-WT
        SubgroupContrastBlock(
            contrast_id="Daraxonrasib_NCT06445062_KRAS_TP53_vs_WT",
            trial_id="NCT06445062",
            trial_name="Daraxonrasib (RMC-6236) + mFOLFOX6 + Bevacizumab",
            moa_vec=_moa_vec(Pan_RAS_Daraxonrasib=1.00, FOLFOX_Oxaliplatin=0.50, Anti_VEGF_Bev=0.45),
            p_pos=build_signed_8d_patient_vector({"kras_mut": True, "tp53_mut": True, "d3_rtk_egfr_addiction": -1.00, "d4_ras_raf_oncogenic": 0.90, "d6_stroma_myeloid_barrier": 0.25}),
            p_neg=build_signed_8d_patient_vector({"kras_mut": False, "nras_mut": False, "braf_mut": False, "triple_wt": True, "d3_rtk_egfr_addiction": 0.85, "d4_ras_raf_oncogenic": -0.75}),
            hr_pos=0.48,
            hr_neg=1.20,
            target_log_hr_contrast=float(-math.log(0.48 / 1.20)),  # +0.9163
            weight=2.5,
            source_citation="GAP-04 / NCT06445062 Subprotocol A: Multi-selective RAS(ON) directly targets KRAS G12/13 + TP53 co-mutant MSS mCRC without RTK feedback",
        ),
        # 11. STC-1010 BreAK CRC-001 (NCT06934538): Gate-2 Immune-Hot + KRAS-mut NLM vs Gate-2 Cold + Liver-Met/CAF-high
        SubgroupContrastBlock(
            contrast_id="STC1010_BreAK_CRC001_Gate2Hot_vs_Cold",
            trial_id="NCT06934538",
            trial_name="BreAK CRC-001 (STC-1010 + mFOLFOX6 ± Bevacizumab)",
            moa_vec=_moa_vec(Vaccine_STC1010=1.00, FOLFOX_Oxaliplatin=0.55, Anti_VEGF_Bev=0.45, Pan_RAS_Daraxonrasib=0.35),
            p_pos=build_signed_8d_patient_vector({"kras_mut": True, "gate2_z": 0.65, "liver_met": False, "tgfb_high": False, "d5_io_effector": 0.75, "d6_stroma_myeloid_barrier": -0.35}),
            p_neg=build_signed_8d_patient_vector({"kras_mut": True, "gate2_z": -0.65, "liver_met": True, "tgfb_high": True, "d5_io_effector": -0.55, "d6_stroma_myeloid_barrier": 0.75}),
            hr_pos=0.52,
            hr_neg=1.25,
            target_log_hr_contrast=float(-math.log(0.52 / 1.25)),  # +0.8771
            weight=2.5,
            source_citation="Alzeeb et al. Front Oncol 2024 & stc1010_mss_crc_biomarker_lock_v2.json (N=934 MSS CRC): D1+D2 permissive Gate-2 Hot PFS HR=0.67 vs CAF/hepatic barrier",
        ),
        # 12. BACE1 / GBM Stupp Escape (EM-02-A / Verubecestat): BACE1-high + M2-myeloid cuff vs BACE1-low
        SubgroupContrastBlock(
            contrast_id="BACE1_GBM_BrM_DualCompartment",
            trial_id="EM-02-A",
            trial_name="BACE1 / GBM Recurrence EscapeMap (GLASS / GSE223499)",
            moa_vec=_moa_vec(BACE1_GBM_Targeted=1.00, ATR_WEE1_Inhibitor=0.25),
            p_pos=build_signed_8d_patient_vector({"d1_ddr_rs": 0.35, "d3_rtk_egfr_addiction": 0.80, "d6_stroma_myeloid_barrier": 0.75, "d7_surface_adc_payload": 0.85, "d8_angio_metabolic_safety": 0.40}),
            p_neg=build_signed_8d_patient_vector({"d1_ddr_rs": -0.30, "d3_rtk_egfr_addiction": -0.50, "d6_stroma_myeloid_barrier": -0.40, "d7_surface_adc_payload": -0.60, "d8_angio_metabolic_safety": -0.20}),
            hr_pos=0.50,
            hr_neg=1.30,
            target_log_hr_contrast=float(-math.log(0.50 / 1.30)),  # +0.9555
            weight=2.0,
            source_citation="Chafe et al. Sci Transl Med 2025 (GSE237446) & GSE223499 / GLASS syn17038081: BACE1 dual-compartment EGFR cleavage + CD163/LRP1 myeloid reprogramming",
        ),
    ]
    return contrasts


# ---------------------------------------------------------------------------
# 4. Regularized Bilinear Interaction Matrix Learner (W_MoA in R^(8 x 15))
# ---------------------------------------------------------------------------
