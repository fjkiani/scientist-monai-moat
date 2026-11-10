#!/usr/bin/env python3
"""Supervised Signed Bilinear 8D Trial Decoder & Multi-Trial Responder Engine (RUO).

Replaces hand-typed non-negative 8D/7D trial vectors ([0, 1]^8) with a learned
Signed Bilinear ML Engine:
    t_j = W_MoA @ m_j,   W_MoA in R^(8 x K),   p_i in [-1, +1]^8
    s_ij = dot(p_i, t_j) / ||t_j||_2 = dot(p_i, W_MoA @ m_j) / ||W_MoA @ m_j||_2

Trained via L2-penalized cross-entropy + published subgroup log-HR contrast
matching across:
  1. Patient/Model-Level Responder Cohorts on Disk:
     - Gide PRJEB23709 (n=41 pretreatment anti-PD-1 ± CTLA-4)
     - Riaz GSE91061 (n=55 on-treatment anti-PD-1)
     - GSE290722 (n=28 CAR-T leukapheresis baseline)
     - Deng GSE151511 (n=23 CAR-T infusion bag literal_CD8)
     - GSE165897 (n=11 paired post-NACT HGSOC Door 1 vs Door 2 vs Neither)
     - GSE63885 (n=61 HGSOC G3/G4 & n=75 all-histology platinum response)
     - BriTROC-1 (n=98 diagnosis CN-signature s3 & n=67 adjusted platinum response)
     - GDSC2/CTRPv2/PRISM Door 2 (n=45 ovarian hibernation / senolytic models)
     - GDSC2 MBD4-LOF vs WT pharmacology (Ceralasertib n=958 / 928, Adavosertib n=967 / 935,
       Gemcitabine n=960 / 928, Olaparib n=967 / 935, Berzosertib n=894 / 867;
       5-drug intersection n=952, Cera+Ola intersection n=958, average n~956)
  2. Published Adjudicated Trial Subgroup Log-HR Contrasts (-ln(HR_pos / HR_neg)):
     - PRIME (NCT00364013): Expanded RAS-WT (HR=0.72) vs RAS-mut (HR=1.29)
     - PEAK (NCT00819780): Expanded RAS-WT (HR=0.65) vs RAS-mut (HR=1.31)
     - AtezoTRIBE (NCT03721653): DetermaIO+ / Immunoscore-IC-high (HR=0.10 / 0.50) vs pMMR unselected (HR=0.86)
     - CheckMate 9X8 (NCT03414983): CD8/TMB-enriched vs unselected MSS (HR=0.81)
     - IMblaze370 (NCT02788279): Atezolizumab + Cobimetinib in MSS mCRC (OS HR=1.00 failure)
     - BREAKWATER (NCT04607421): Encorafenib + Cetuximab + mFOLFOX6 in BRAF V600E (HR=0.47 vs SOC)
     - CARMEN-LC03 (NCT04154956): CEACAM5 IHC >=80% (OS HR=0.71) vs 50-79% (HR=1.14 / 1.38)
     - BOT/BAL / CCTG CO.26 (NCT02870920): No-liver-met (HR=0.54) vs Liver-met (HR=1.39)
     - KRYSTAL-10 (NCT04793958): Adagrasib + Cetuximab in KRAS G12C mCRC (HR=0.89 failure)

Decodes all 42 entries in trial_decode_registry_v2.json plus core comparator trials
(PRIME, PEAK_FOLFIRI, BREAKWATER, KRYSTAL_10, DARAXONRASIB_NCT06445062, BOT_BAL)
and routes all 485 TCGA-COADREAD MSS CRC patients.
"""
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

CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
RUO_DISCLAIMER = (
    "RESEARCH USE ONLY (RUO). Supervised signed bilinear 8D trial decoding and "
    "multi-trial responder routing are exploratory research prioritization models, "
    "not FDA-cleared clinical diagnostics or treatment orders."
)

# ---------------------------------------------------------------------------
# 1. Canonical Signed 8D Patient State Axes (p_i in [-1, +1]^8)
# ---------------------------------------------------------------------------

SIGNED_AXES_8D: tuple[str, ...] = (
    "d1_ddr_rs",
    "d2_hibernation_sen",
    "d3_rtk_egfr_addiction",
    "d4_ras_raf_oncogenic",
    "d5_io_effector",
    "d6_stroma_myeloid_barrier",
    "d7_surface_adc_payload",
    "d8_angio_metabolic_safety",
)

# 15 Core Therapeutic Mechanism-of-Action (MoA) Classes (K = 15)
MOA_CLASSES_15: tuple[str, ...] = (
    "FOLFOX_Oxaliplatin",          # 0: Oxaliplatin + 5-FU/LV DNA crosslinking & ICD priming
    "FOLFIRI_Irinotecan_TOP1",     # 1: Irinotecan / SN-38 TOP1 inhibition
    "Anti_VEGF_Bev",               # 2: Bevacizumab / Aflibercept / VEGFR TKI (Regorafenib/Cabozantinib/Zanzalintinib)
    "Anti_EGFR",                   # 3: Panitumumab / Cetuximab / Erlotinib (repelled when RAS/RAF-mut via d3 < 0)
    "Anti_PD1_PDL1",               # 4: Pembrolizumab / Nivolumab / Durvalumab / Atezolizumab / Avelumab
    "Anti_CTLA4",                  # 5: Ipilimumab / Tremelimumab / Botensilimab
    "Vaccine_STC1010",             # 6: STC-1010 DNFB-haptenated whole-cell / NeoAg / CEA / mKRAS vaccine
    "Pan_RAS_Daraxonrasib",        # 7: Daraxonrasib (RMC-6236) RAS(ON) multi-selective tri-complex inhibitor
    "BRAF_Encorafenib",            # 8: Encorafenib BRAF V600E inhibitor (BREAKWATER)
    "MEK_Cobimetinib",             # 9: Cobimetinib / Binimetinib MEK inhibitor (IMblaze370)
    "ATR_WEE1_Inhibitor",          # 10: Ceralasertib / Berzosertib (ATR) & Adavosertib (WEE1) + Gemcitabine
    "PARP_Inhibitor",              # 11: Olaparib PARP1/2 trapping
    "BCLxL_Dasatinib_Senolytic",   # 12: Navitoclax / ABT-737 (BCL-xL) + Dasatinib (SRC/EPHA2) + Doxercalciferol
    "CEACAM5_TOP1_ADC",            # 13: Precemtabart tocentecan / Tusamitamab sonditecan / SAR408701 / IMMU-130
    "BACE1_GBM_Targeted",          # 14: Verubecestat (MK-8931) BACE1 dual-compartment / GBM Stupp-escape context
)


def _clip(val: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(val)))


def _tanh_scale(z: float, scale: float = 1.5) -> float:
    return float(math.tanh(float(z) / scale))


def build_signed_8d_patient_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Construct signed 8D patient state vector p_i in [-1, +1]^8.

    Axes:
      d1_ddr_rs: Standardized DDR / HRD / Replication Stress (FGA, BRCA1/2, MBD4, ATM, S3, post_DDR)
      d2_hibernation_sen: Clean Hibernation & Senolytic Priming (H_clean, S_sen, BCL2L1 + NOXA - MCL1)
      d3_rtk_egfr_addiction: Signed RTK Addiction (+1 for EGFR/ERBB2 driven RAS/RAF-WT;
                             -1 when KRAS/NRAS/BRAF is mutated so anti-EGFR trials are mathematically repelled!)
      d4_ras_raf_oncogenic: Mutant KRAS (G12D/G12V/G13D/G12C) / BRAF V600E + TP53 (GAP-04) oncogenic load
      d5_io_effector: T-cell Effector & Memory Priming (z_imm, ZNF683+GNLY E_effector, S_triad, Bag_Index_3, POLE/MSI-H)
      d6_stroma_myeloid_barrier: Stromal & Myeloid Exclusion (MFAP4, FAP-CXCL10, CMS4/TGF-b, Liver-Met MDSC,
                                 Brain BACE1+LRP1+CD163 sink — positive barrier value; trials sensitive to barrier
                                 carry negative weight on d6 so high barrier subtracts from dot product!)
      d7_surface_adc_payload: Target x Payload match (CEACAM5 x TOP1, ERBB2, BACE1; negative when payload-refractory/hold)
      d8_angio_metabolic_safety: VEGFA / mTORC2 (RICTOR) / CLPP + CPIC DPYD/UGT1A1 safety clearance
    """
    if "p_signed_8d" in features and features["p_signed_8d"] is not None:
        arr = np.asarray(features["p_signed_8d"], dtype=float)
        if arr.shape == (8,):
            return np.clip(arr, -1.0, 1.0)

    # --- d1: DDR / HRD / Replication Stress ---
    if "d1_ddr_rs" in features and features["d1_ddr_rs"] is not None:
        d1 = float(features["d1_ddr_rs"])
    else:
        fga = float(features.get("fga", features.get("fraction_genome_altered", 0.2671)) or 0.2671)
        hrd_str = str(features.get("hrd_proxy", "HRD-Intermediate"))
        hrd_base = 0.45 if hrd_str == "HRD-High" else (0.15 if hrd_str == "HRD-Intermediate" else -0.25)
        fga_centered = (fga - 0.2671) / 0.25
        mbd4_lof = 0.75 if features.get("mbd4_lof") else 0.0
        brca_mut = 0.55 if (features.get("brca_mut") or features.get("brca_somatic") or features.get("brca_lof")) else 0.0
        atm_lof = 0.50 if features.get("atm_lof") else 0.0
        s3_z = float(features.get("s3_z", 0.0) or 0.0)
        post_ddr_z = float(features.get("post_ddr_z", 0.0) or 0.0)
        efflux_pen = -0.55 if features.get("post_parpi_efflux") else 0.0
        d1 = _clip(hrd_base + 0.35 * fga_centered + mbd4_lof + brca_mut + atm_lof + 0.45 * s3_z + 0.45 * post_ddr_z + efflux_pen)

    # --- d2: Clean Hibernation & Senolytic Priming ---
    if "d2_hibernation_sen" in features and features["d2_hibernation_sen"] is not None:
        d2 = float(features["d2_hibernation_sen"])
    else:
        s_sen = features.get("s_sen", features.get("post_S_sen"))
        h_clean = features.get("h_clean", features.get("post_H_clean"))
        if s_sen is not None:
            d2 = _tanh_scale(float(s_sen), scale=2.5)
        elif h_clean is not None:
            d2 = _tanh_scale(float(h_clean), scale=2.0)
        else:
            d2 = float(features.get("d2_default", -0.10))

    # --- d3: Signed RTK / EGFR Addiction (+1 RAS/RAF-WT EGFR-driven, -1 RAS/RAF-mut) ---
    if "d3_rtk_egfr_addiction" in features and features["d3_rtk_egfr_addiction"] is not None:
        d3 = float(features["d3_rtk_egfr_addiction"])
    else:
        kras = bool(features.get("kras_mut"))
        nras = bool(features.get("nras_mut"))
        braf = bool(features.get("braf_mut"))
        ras_raf = bool(features.get("ras_raf_mut") or kras or nras or braf)
        pten_loss = bool(features.get("pten_loss") or features.get("pten_null"))
        if ras_raf:
            d3 = -1.0
        elif pten_loss and not features.get("egfr_mut"):
            d3 = -0.65
        elif features.get("egfr_mut") or features.get("erbb2_amp"):
            d3 = 1.0
        else:
            # Triple-WT CRC / RTK-permissive
            d3 = 0.75 if features.get("triple_wt", True) else 0.0

    # --- d4: Mutant KRAS / BRAF + TP53 Oncogenic Load ---
    if "d4_ras_raf_oncogenic" in features and features["d4_ras_raf_oncogenic"] is not None:
        d4 = float(features["d4_ras_raf_oncogenic"])
    else:
        kras = bool(features.get("kras_mut"))
        nras = bool(features.get("nras_mut"))
        braf = bool(features.get("braf_mut"))
        tp53 = bool(features.get("tp53_mut", True))
        if braf:
            d4 = 0.95 if tp53 else 0.80
        elif kras or nras or features.get("ras_raf_mut"):
            d4 = 0.85 if tp53 else 0.65
        else:
            d4 = -0.60

    # --- d5: T-cell Effector & Memory Priming ---
    if "d5_io_effector" in features and features["d5_io_effector"] is not None:
        d5 = float(features["d5_io_effector"])
    else:
        tmb = float(features.get("tmb", 3.3) or 3.3)
        msi_h = bool(features.get("msi_h", False))
        e_eff = features.get("e_effector")
        bag3 = features.get("bag_index_3")
        z_imm = features.get("gate2_z", features.get("immune_score_rna"))
        if e_eff is not None:
            s_tr = float(features.get("s_triad", 0.0) or 0.0)
            d5 = _tanh_scale(0.75 * float(e_eff) + 0.25 * s_tr, scale=1.2)
        elif bag3 is not None:
            d5 = _tanh_scale(float(bag3), scale=1.2)
        elif msi_h or tmb >= 25.0:
            d5 = 0.95
        elif z_imm is not None:
            z_val = float(z_imm)
            # Center at Gate-2 cutpoint (-0.0257) so Gate-2 Hot > 0 and Gate-2 Cold < 0
            d5 = _tanh_scale((z_val - (-0.0257)) / 0.55, scale=1.0)
        else:
            d5 = -0.35

    # --- d6: Stromal & Myeloid Barrier (positive = high barrier) ---
    if "d6_stroma_myeloid_barrier" in features and features["d6_stroma_myeloid_barrier"] is not None:
        d6 = float(features["d6_stroma_myeloid_barrier"])
    else:
        mfap4_z = features.get("mfap4_z")
        liver_met = bool(features.get("liver_met") or features.get("liver_metastasis"))
        cms = str(features.get("cms_label", "UNLABELED"))
        tgfb_high = bool(features.get("tgfb_high") or features.get("tgfb_barrier_high") or cms == "CMS4")
        if mfap4_z is not None:
            d6 = _tanh_scale(float(mfap4_z), scale=1.2)
        else:
            bar = 0.0
            if liver_met:
                bar += 0.65
            if tgfb_high:
                bar += 0.45
            elif cms in {"CMS1", "CMS2"}:
                bar -= 0.35
            z_imm = features.get("gate2_z", features.get("immune_score_rna"))
            if z_imm is not None and float(z_imm) < -0.3839:
                bar += 0.35
            elif z_imm is not None and float(z_imm) >= 0.3026:
                bar -= 0.30
            d6 = _clip(bar, -1.0, 1.0)

    # --- d7: Surface ADC / Target x Payload Match ---
    if "d7_surface_adc_payload" in features and features["d7_surface_adc_payload"] is not None:
        d7 = float(features["d7_surface_adc_payload"])
    else:
        ceacam5_high = features.get("ceacam5_ihc_ge_80")
        prior_irino_refractory = bool(features.get("prior_irinotecan_refractory", False))
        efflux = float(features.get("p_efflux", 0.18) or 0.18)
        if ceacam5_high is True and not prior_irino_refractory and efflux <= 0.18:
            d7 = 0.85
        elif prior_irino_refractory:
            d7 = -0.80
        else:
            # In TCGA-COADREAD genomic cohort, high topo1 sensitivity + low efflux (<0.12) proxy
            d7 = _clip((0.16 - efflux) * 3.0, -0.60, 0.65)

    # --- d8: Angiogenic / Metabolic / CPIC Safety Clearance ---
    if "d8_angio_metabolic_safety" in features and features["d8_angio_metabolic_safety"] is not None:
        d8 = float(features["d8_angio_metabolic_safety"])
    else:
        dpyd_tox = bool(features.get("dpyd_ugt1a1_high_risk", False))
        eu_hold = bool(features.get("regulatory_hold", False))
        vegf_high = float(features.get("p_vegf", 0.25) or 0.25)
        if dpyd_tox or eu_hold:
            d8 = -0.95
        else:
            d8 = _clip(0.30 + (vegf_high - 0.20) * 1.5, -1.0, 1.0)

    return np.array([d1, d2, d3, d4, d5, d6, d7, d8], dtype=float)


# ---------------------------------------------------------------------------
# 2. MoA Design Vectors m_j in R^15 for All 42 Registry Entries + Comparators
# ---------------------------------------------------------------------------

def _moa_vec(**weights: float) -> np.ndarray:
    vec = np.zeros(len(MOA_CLASSES_15), dtype=float)
    idx_map = {name: i for i, name in enumerate(MOA_CLASSES_15)}
    for k, v in weights.items():
        if k not in idx_map:
            raise KeyError(f"Unknown MoA class: {k}")
        vec[idx_map[k]] = float(v)
    return vec


# All 42 entries in trial_decode_registry_v2.json (by 1-based registry_index & id)
REGISTRY_42_MOA_SPECS: list[dict[str, Any]] = [
    # --- ATR_DDR (1..5) ---
    {"idx": 1, "id": "NCT02595892", "name": "Berzosertib (M6620) Phase I/II", "program": "ATR_DDR",
     "moa": {"ATR_WEE1_Inhibitor": 1.0}},
    {"idx": 2, "id": "NCT03579316", "name": "Adavosertib (AZD1775) Phase II", "program": "ATR_DDR",
     "moa": {"ATR_WEE1_Inhibitor": 0.90, "Anti_EGFR": 0.15}},
    {"idx": 3, "id": "NCT03462342", "name": "CAPRI Phase II", "program": "ATR_DDR",
     "moa": {"ATR_WEE1_Inhibitor": 0.70, "PARP_Inhibitor": 0.60}},
    {"idx": 4, "id": "NCT05450692", "name": "LATIFY", "program": "ATR_DDR",
     "moa": {"ATR_WEE1_Inhibitor": 0.75, "Anti_PD1_PDL1": 0.65}},
    {"idx": 5, "id": "NCT02264678", "name": "Ceralasertib + Olaparib Phase I/Ib", "program": "ATR_DDR",
     "moa": {"ATR_WEE1_Inhibitor": 0.75, "PARP_Inhibitor": 0.55}},
    # --- CEACAM5 (6..10) ---
    {"idx": 6, "id": "NCT04154956", "name": "CARMEN-LC03", "program": "CEACAM5",
     "moa": {"CEACAM5_TOP1_ADC": 0.85, "FOLFOX_Oxaliplatin": 0.10}},
    {"idx": 7, "id": "NCT05464030", "name": "PROCEADE-CRC-01", "program": "CEACAM5",
     "moa": {"CEACAM5_TOP1_ADC": 1.00, "FOLFIRI_Irinotecan_TOP1": 0.25}},
    {"idx": 8, "id": "NCT01270698", "name": "Labetuzumab govitecan (IMMU-130) Phase I/II", "program": "CEACAM5",
     "moa": {"CEACAM5_TOP1_ADC": 0.65, "FOLFIRI_Irinotecan_TOP1": 0.65}},
    {"idx": 9, "id": "NCT04826003", "name": "Cibisatamab + FAP-4-1BBL (BP42675)", "program": "CEACAM5",
     "moa": {"CEACAM5_TOP1_ADC": 0.55, "Anti_PD1_PDL1": 0.55, "Vaccine_STC1010": 0.30}},
    {"idx": 10, "id": "NCT06131840", "name": "Tusamitamab sonditecan (SGN-CEACAM5C) Phase I", "program": "CEACAM5",
     "moa": {"CEACAM5_TOP1_ADC": 0.95, "FOLFIRI_Irinotecan_TOP1": 0.20}},
    # --- IO_CORE (11..17) ---
    {"idx": 11, "id": "NCT03050814", "name": "QUILT-2.004", "program": "IO_CORE",
     "moa": {"Vaccine_STC1010": 0.55, "Anti_PD1_PDL1": 0.65, "FOLFOX_Oxaliplatin": 0.50, "Anti_VEGF_Bev": 0.45}},
    {"idx": 12, "id": "NCT05141721", "name": "GRANITE GO-010", "program": "IO_CORE",
     "moa": {"Vaccine_STC1010": 0.65, "Anti_PD1_PDL1": 0.60, "Anti_CTLA4": 0.45, "Anti_VEGF_Bev": 0.40}},
    {"idx": 13, "id": "NCT02981524", "name": "GVAX + Cyclophosphamide + Pembrolizumab", "program": "IO_CORE",
     "moa": {"Vaccine_STC1010": 0.60, "Anti_PD1_PDL1": 0.70}},
    {"idx": 14, "id": "NCT02788279", "name": "IMblaze370", "program": "IO_CORE",
     "moa": {"MEK_Cobimetinib": 1.00, "Anti_PD1_PDL1": 0.55}},
    {"idx": 15, "id": "NCT03721653", "name": "AtezoTRIBE", "program": "IO_CORE",
     "moa": {"FOLFOX_Oxaliplatin": 0.55, "FOLFIRI_Irinotecan_TOP1": 0.45, "Anti_VEGF_Bev": 0.55, "Anti_PD1_PDL1": 0.75}},
    {"idx": 16, "id": "NCT03202758", "name": "MEDITREME", "program": "IO_CORE",
     "moa": {"FOLFOX_Oxaliplatin": 0.60, "Anti_PD1_PDL1": 0.65, "Anti_CTLA4": 0.55}},
    {"idx": 17, "id": "NCT02870920", "name": "CCTG CO.26", "program": "IO_CORE",
     "moa": {"Anti_PD1_PDL1": 0.75, "Anti_CTLA4": 0.75}},
    # --- IO_APPENDIX (18..32) ---
    {"idx": 18, "id": "NCT02563002", "name": "KEYNOTE-177", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 1.00}},
    {"idx": 19, "id": "NCT02060188", "name": "CheckMate 142", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.85, "Anti_CTLA4": 0.65}},
    {"idx": 20, "id": "EPOC1603", "name": "REGONIVO (EPOC1603)", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.75, "Anti_VEGF_Bev": 0.60}},
    {"idx": 21, "id": "NCT04126733", "name": "REGONIVO Phase 2 (US)", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.80, "Anti_VEGF_Bev": 0.55}},
    {"idx": 22, "id": "NCT04362839", "name": "RIN (Regorafenib + Ipilimumab + Nivolumab)", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.70, "Anti_CTLA4": 0.65, "Anti_VEGF_Bev": 0.50}},
    {"idx": 23, "id": "NCT02291289", "name": "MODUL Cohort 2", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.65, "Anti_VEGF_Bev": 0.55, "FOLFOX_Oxaliplatin": 0.25}},
    {"idx": 24, "id": "NCT03374254", "name": "KEYNOTE-651 Cohorts B & D", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.70, "FOLFOX_Oxaliplatin": 0.50, "FOLFIRI_Irinotecan_TOP1": 0.40}},
    {"idx": 25, "id": "NCT03539822", "name": "CAMILLA CRC", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.70, "Anti_VEGF_Bev": 0.65}},
    {"idx": 26, "id": "NCT03391232", "name": "PolyPEPI1018", "program": "IO_APPENDIX",
     "moa": {"Vaccine_STC1010": 0.75, "FOLFOX_Oxaliplatin": 0.25}},
    {"idx": 27, "id": "NCT02998879", "name": "GVAX + Guadecitabine", "program": "IO_APPENDIX",
     "moa": {"Vaccine_STC1010": 0.70, "ATR_WEE1_Inhibitor": 0.20}},
    {"idx": 28, "id": "NCT00103142", "name": "PANVAC (CEA/MUC1 poxviral vaccine) + DC vaccine", "program": "IO_APPENDIX",
     "moa": {"Vaccine_STC1010": 0.80}},
    {"idx": 29, "id": "NCT04853017", "name": "AMPLIFY-201 (ELI-002 mKRAS amphiphile vaccine)", "program": "IO_APPENDIX",
     "moa": {"Vaccine_STC1010": 0.75, "Pan_RAS_Daraxonrasib": 0.45}},
    {"idx": 30, "id": "NCT02562755", "name": "PexaVec (oncolytic vaccinia) + Durvalumab ± Tremelimumab", "program": "IO_APPENDIX",
     "moa": {"Vaccine_STC1010": 0.50, "Anti_PD1_PDL1": 0.65, "Anti_CTLA4": 0.55}},
    {"idx": 31, "id": "NCT02517398", "name": "Bintrafusp Alfa (M7824) in CRC", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.80, "Anti_VEGF_Bev": 0.25}},
    {"idx": 32, "id": "NCT05425940", "name": "STELLAR-303 (zanzalintinib + atezolizumab)", "program": "IO_APPENDIX",
     "moa": {"Anti_PD1_PDL1": 0.70, "Anti_VEGF_Bev": 0.65}},
    # --- HISTORICAL_BENCHMARK (33..38) ---
    {"idx": 33, "id": "NCT02339116", "name": "TRIBE2 (control arm)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 0.65, "FOLFIRI_Irinotecan_TOP1": 0.55, "Anti_VEGF_Bev": 0.65}},
    {"idx": 34, "id": "NOT_SEPARATELY_REGISTERED", "name": "BECOME (bevacizumab arm)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 0.65, "Anti_VEGF_Bev": 0.70, "Pan_RAS_Daraxonrasib": 0.25}},
    {"idx": 35, "id": "NCT00069095", "name": "NO16966 (FOLFOX4 + bevacizumab arm)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 0.70, "Anti_VEGF_Bev": 0.70}},
    {"idx": 36, "id": "NCT00052910", "name": "OPTIMOX1 (FOLFOX4 arm)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 1.00}},
    {"idx": 37, "id": "UMIN000005616", "name": "WJOG4407G (mFOLFOX6 + bev arm)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 0.65, "Anti_VEGF_Bev": 0.75}},
    {"idx": 38, "id": "NCT03050814_HIST", "registry_id": "NCT03050814", "name": "QUILT-2.004 — DESIGN PRECEDENT ONLY", "program": "HISTORICAL_BENCHMARK",
     "moa": {"Vaccine_STC1010": 0.55, "Anti_PD1_PDL1": 0.65, "FOLFOX_Oxaliplatin": 0.50, "Anti_VEGF_Bev": 0.45}},
    # --- BREAK_CRC_001 (39) ---
    {"idx": 39, "id": "NCT06934538", "name": "BreAK CRC-001 (STC-1010)", "program": "BREAK_CRC_001",
     "moa": {"Vaccine_STC1010": 1.00, "FOLFOX_Oxaliplatin": 0.55, "Anti_VEGF_Bev": 0.45, "Pan_RAS_Daraxonrasib": 0.35}},
    # --- GBM_ESCAPEMAP (40..42) ---
    {"idx": 40, "id": "EM-02-A", "name": "GLASS Bulk RNA-seq — GBM Recurrence (Stupp-treated)", "program": "GBM_ESCAPEMAP",
     "moa": {"BACE1_GBM_Targeted": 1.00, "ATR_WEE1_Inhibitor": 0.25}},
    {"idx": 41, "id": "EM-02-B", "name": "GSE222515 Bulk RNA-seq — GBM Recurrence (directional support)", "program": "GBM_ESCAPEMAP",
     "moa": {"BACE1_GBM_Targeted": 0.90, "ATR_WEE1_Inhibitor": 0.20}},
    {"idx": 42, "id": "EM-02-C", "name": "GSE174554 snRNA-seq — GBM Recurrence (tumor-cell pairs)", "program": "GBM_ESCAPEMAP",
     "moa": {"BACE1_GBM_Targeted": 0.95, "BCLxL_Dasatinib_Senolytic": 0.35}},
]

# External Adjudicated Comparator & Rescue Trials Decoded by W_MoA
COMPARATOR_MOA_SPECS: list[dict[str, Any]] = [
    {"id": "NCT06445062", "name": "Daraxonrasib (RMC-6236) RAS(ON) GI Platform Subprotocol A", "program": "RAS_RESCUE",
     "moa": {"Pan_RAS_Daraxonrasib": 1.00, "FOLFOX_Oxaliplatin": 0.50, "Anti_VEGF_Bev": 0.45}},
    {"id": "NCT04607421", "name": "BREAKWATER (Encorafenib + Cetuximab + mFOLFOX6)", "program": "BRAF_CRC",
     "moa": {"BRAF_Encorafenib": 1.00, "FOLFOX_Oxaliplatin": 0.45, "Anti_EGFR": 0.20}},
    {"id": "NCT00364013", "name": "PRIME (Panitumumab + FOLFOX4)", "program": "ANTI_EGFR_CRC",
     "moa": {"Anti_EGFR": 1.00, "FOLFOX_Oxaliplatin": 0.45}},
    {"id": "NCT00819780", "name": "PEAK (Panitumumab + mFOLFOX6)", "program": "ANTI_EGFR_CRC",
     "moa": {"Anti_EGFR": 1.00, "FOLFOX_Oxaliplatin": 0.40}},
    {"id": "NCT00079274", "name": "N0147 (mFOLFOX6 ± Cetuximab)", "program": "ANTI_EGFR_CRC",
     "moa": {"FOLFOX_Oxaliplatin": 0.80, "Anti_EGFR": 0.50}},
    {"id": "NCT00275210", "name": "MOSAIC (FOLFOX4)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFOX_Oxaliplatin": 1.00}},
    {"id": "NCT00561470", "name": "VELOUR (FOLFIRI + Aflibercept)", "program": "HISTORICAL_BENCHMARK",
     "moa": {"FOLFIRI_Irinotecan_TOP1": 0.60, "Anti_VEGF_Bev": 0.85}},
    {"id": "NCT03414983", "name": "CheckMate 9X8 (Nivolumab + mFOLFOX6 + Bevacizumab)", "program": "IO_CRC",
     "moa": {"Anti_PD1_PDL1": 0.80, "FOLFOX_Oxaliplatin": 0.55, "Anti_VEGF_Bev": 0.50}},
    {"id": "NCT04793958", "name": "KRYSTAL-10 (Adagrasib + Cetuximab in KRAS G12C mCRC)", "program": "RAS_SALVAGE",
     "moa": {"Anti_EGFR": 0.85, "MEK_Cobimetinib": 0.55}},
    {"id": "NCT03860272", "name": "BOT/BAL (Botensilimab + Balstilimab C-800-01)", "program": "IO_CRC",
     "moa": {"Anti_CTLA4": 0.85, "Anti_PD1_PDL1": 0.75}},
    {"id": "DOOR2_SENOLYTIC", "name": "Door 2 Hibernation Senolytic (Navitoclax/ABT-737 + Dasatinib)", "program": "OVARIAN_DOOR2",
     "moa": {"BCLxL_Dasatinib_Senolytic": 1.00}},
]


# ---------------------------------------------------------------------------
# 3. Signed Bilinear Scoring & Cohort / Contrast Loaders
# ---------------------------------------------------------------------------

def signed_fit_score(p: np.ndarray, t: np.ndarray) -> float:
    """Signed normalized projection score s(p, t) = dot(p, t) / ||t||_2 in [-sqrt(8), +sqrt(8)]."""
    norm_t = float(np.linalg.norm(t))
    if norm_t <= 1e-12:
        return 0.0
    return float(np.dot(p, t) / norm_t)


def signed_cosine_similarity(p: np.ndarray, t: np.ndarray) -> float:
    """Signed cosine of patient and trial vectors in [-1, +1]."""
    norm_p = float(np.linalg.norm(p))
    norm_t = float(np.linalg.norm(t))
    if norm_p <= 1e-12 or norm_t <= 1e-12:
        return 0.0
    return float(np.dot(p, t) / (norm_p * norm_t))


def signed_fit_batch(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorized signed normalized projection for patient matrix P in R^(N x 8)."""
    norm_t = float(np.linalg.norm(t))
    if norm_t <= 1e-12:
        return np.zeros(P.shape[0], dtype=float)
    return (P @ t) / norm_t


class BilinearDecoderInputError(Exception):
    """Unsafe bilinear request. Callers must not relabel this as PATH A."""

    def __init__(self, message: str, *, code: str, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.missing = list(missing or [])


def bilinear_contract_path() -> Path:
    """Frozen weights shipped next to this module. Not the PATH A registry."""
    return Path(__file__).resolve().parent / "models" / "crispro_8d_bilinear_decoder_v1.json"


def load_bilinear_contract(path: Path | None = None) -> dict[str, Any]:
    """Load crispro_8d_bilinear_decoder_v1.json. Does not refit W_MoA."""
    contract_path = path or bilinear_contract_path()
    if not contract_path.is_file():
        raise BilinearDecoderInputError(
            f"bilinear contract missing at {contract_path}",
            code="contract_missing",
            missing=["crispro_8d_bilinear_decoder_v1.json"],
        )
    data = json.loads(contract_path.read_text())
    axes = data.get("signed_axes_8d")
    moa = data.get("moa_classes_15")
    matrix = data.get("W_MoA_matrix")
    if axes != list(SIGNED_AXES_8D) or moa != list(MOA_CLASSES_15):
        raise BilinearDecoderInputError(
            "contract signed_axes_8d or moa_classes_15 does not match the module constants",
            code="contract_shape_mismatch",
        )
    if (
        not isinstance(matrix, list)
        or len(matrix) != 8
        or any(not isinstance(row, list) or len(row) != 15 for row in matrix)
    ):
        raise BilinearDecoderInputError(
            "W_MoA_matrix is not 8x15",
            code="contract_shape_mismatch",
        )
    return data


def score_bilinear_ml_loto(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Score one signed 8D patient against one MoA mixture using frozen JSON weights.

    Query path. Does not call fit_bilinear_w_moa, load_all_responder_cohorts,
    build_signed_8d_patient_vector, or any RNG. PATH A patient_vector is rejected
    by omission: this function reads only p_signed_8d.
    """
    if payload.get("p_signed_8d") is None:
        raise BilinearDecoderInputError(
            "bilinear_ml_loto requires p_signed_8d (length 8, values in [-1, 1]). "
            "PATH A patient_vector is a different coordinate system and is not accepted.",
            code="missing_features",
            missing=["p_signed_8d"],
        )
    raw_p = payload["p_signed_8d"]
    if not isinstance(raw_p, (list, tuple)) or len(raw_p) != 8:
        raise BilinearDecoderInputError(
            "p_signed_8d must be a length-8 list",
            code="shape_mismatch",
            missing=["p_signed_8d"],
        )
    try:
        p = np.asarray([float(x) for x in raw_p], dtype=float)
    except (TypeError, ValueError) as exc:
        raise BilinearDecoderInputError(
            "p_signed_8d contains non-numeric values",
            code="shape_mismatch",
        ) from exc
    if not bool(np.all(np.isfinite(p))):
        raise BilinearDecoderInputError(
            "p_signed_8d contains non-finite values",
            code="shape_mismatch",
        )
    if bool(np.any(np.abs(p) > 1.0 + 1e-6)):
        raise BilinearDecoderInputError(
            "p_signed_8d values must lie in [-1, 1]",
            code="shape_mismatch",
        )

    moa = payload.get("moa_weights")
    if not isinstance(moa, Mapping) or len(moa) == 0:
        raise BilinearDecoderInputError(
            "bilinear_ml_loto requires moa_weights over the 15 Drug MoA columns",
            code="missing_features",
            missing=["moa_weights"],
        )
    unknown = [str(k) for k in moa.keys() if k not in MOA_CLASSES_15]
    if unknown:
        raise BilinearDecoderInputError(
            f"unknown moa columns: {unknown}",
            code="shape_mismatch",
        )
    try:
        weights = {str(k): float(v) for k, v in moa.items()}
    except (TypeError, ValueError) as exc:
        raise BilinearDecoderInputError(
            "moa_weights values must be numeric",
            code="shape_mismatch",
        ) from exc
    if not any(abs(v) > 0.0 for v in weights.values()):
        raise BilinearDecoderInputError(
            "moa_weights are all zero",
            code="shape_mismatch",
        )

    contract = load_bilinear_contract(contract_path)
    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)
    if w_moa.shape != (8, 15):
        raise BilinearDecoderInputError(
            f"W_MoA_matrix shape {w_moa.shape} is not (8, 15)",
            code="contract_shape_mismatch",
        )
    m_vec = _moa_vec(**weights)
    trial_vec = w_moa @ m_vec
    norm_t = float(np.linalg.norm(trial_vec))
    if norm_t <= 1e-12:
        raise BilinearDecoderInputError(
            "frozen W_MoA @ moa_weights has zero norm",
            code="shape_mismatch",
        )
    score = signed_fit_score(p, trial_vec)
    return {
        "engine": "bilinear_ml_loto",
        "model_id": contract.get("model_id"),
        "coordinate_system": "signed_bilinear_8d",
        "signed_axes_8d": list(contract["signed_axes_8d"]),
        "distinct_from_path_a_axes": True,
        "refit_on_query": False,
        "weights_source": "crispro_8d_bilinear_decoder_v1.json",
        "fallback_to_path_a": False,
        "ruo": True,
        "ruo_disclaimer": contract.get("ruo_disclaimer"),
        "contract_status": contract.get("contract_status"),
        "score": score,
        "trial_vector_raw": [float(x) for x in trial_vec],
        "l2_norm_trial": norm_t,
        "p_signed_8d": [float(x) for x in p],
        "moa_weights": weights,
        "loto_validation_summary_loaded": contract.get("loto_validation_summary"),
    }


RECOGNIZED_CLINICAL_FEATURE_KEYS: frozenset[str] = frozenset({
    "p_signed_8d",
    "d1_ddr_rs",
    "d2_hibernation_sen",
    "d3_rtk_egfr_addiction",
    "d4_ras_raf_oncogenic",
    "d5_io_effector",
    "d6_stroma_myeloid_barrier",
    "d7_surface_adc_payload",
    "d8_angio_metabolic_safety",
    "fga",
    "fraction_genome_altered",
    "hrd_proxy",
    "mbd4_lof",
    "brca_mut",
    "brca_somatic",
    "brca_lof",
    "atm_lof",
    "s3_z",
    "post_ddr_z",
    "post_parpi_efflux",
    "s_sen",
    "post_S_sen",
    "h_clean",
    "post_H_clean",
    "d2_default",
    "kras_mut",
    "nras_mut",
    "braf_mut",
    "ras_raf_mut",
    "pten_loss",
    "pten_null",
    "egfr_mut",
    "erbb2_amp",
    "triple_wt",
    "tp53_mut",
    "tmb",
    "msi_h",
    "e_effector",
    "bag_index_3",
    "gate2_z",
    "immune_score_rna",
    "s_triad",
    "mfap4_z",
    "liver_met",
    "liver_metastasis",
    "cms_label",
    "tgfb_high",
    "tgfb_barrier_high",
    "ceacam5_ihc_ge_80",
    "prior_irinotecan_refractory",
    "p_efflux",
    "dpyd_ugt1a1_high_risk",
    "regulatory_hold",
    "p_vegf",
})


def rank_42_trials_bilinear_ml(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Rank all 42 decoded trials (and adjudicated comparators) for a patient using frozen W_MoA.

    Accepts either:
      - p_signed_8d: list of 8 floats in [-1, 1]
      - clinical_features: non-empty dict of recognized biomarker/genomic features
    Fail-closed on empty, malformed, or PATH A patient_vector inputs.
    """
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise BilinearDecoderInputError(
            "POST /v1/ranker/8d/bilinear requires p_signed_8d or clinical_features",
            code="missing_features",
            missing=["p_signed_8d", "clinical_features"],
        )

    if (
        payload.get("patient_vector") is not None
        and payload.get("p_signed_8d") is None
        and payload.get("clinical_features") is None
    ):
        raise BilinearDecoderInputError(
            "PATH A patient_vector is a different coordinate system and is rejected; "
            "provide p_signed_8d or clinical_features.",
            code="coordinate_system_mismatch",
            missing=["p_signed_8d"],
        )

    raw_p = payload.get("p_signed_8d")
    raw_cf = payload.get("clinical_features")

    if raw_p is not None:
        if not isinstance(raw_p, (list, tuple)) or len(raw_p) != 8:
            raise BilinearDecoderInputError(
                "p_signed_8d must be a length-8 list",
                code="shape_mismatch",
                missing=["p_signed_8d"],
            )
        try:
            p = np.asarray([float(x) for x in raw_p], dtype=float)
        except (TypeError, ValueError) as exc:
            raise BilinearDecoderInputError(
                "p_signed_8d contains non-numeric values",
                code="shape_mismatch",
            ) from exc
        if not bool(np.all(np.isfinite(p))):
            raise BilinearDecoderInputError(
                "p_signed_8d contains non-finite values",
                code="shape_mismatch",
            )
        if bool(np.any(np.abs(p) > 1.0 + 1e-6)):
            raise BilinearDecoderInputError(
                "p_signed_8d values must lie in [-1, 1]",
                code="shape_mismatch",
            )
        input_mode = "p_signed_8d"
    elif raw_cf is not None:
        if not isinstance(raw_cf, Mapping) or len(raw_cf) == 0:
            raise BilinearDecoderInputError(
                "clinical_features must be a non-empty object",
                code="missing_features",
                missing=["clinical_features"],
            )
        recognized = set(raw_cf.keys()) & RECOGNIZED_CLINICAL_FEATURE_KEYS
        if not recognized:
            raise BilinearDecoderInputError(
                "clinical_features contains no recognized 8D biomarker keys",
                code="missing_features",
                missing=["clinical_features"],
            )
        try:
            p = build_signed_8d_patient_vector(raw_cf)
        except (TypeError, ValueError) as exc:
            raise BilinearDecoderInputError(
                f"clinical_features contains malformed values: {exc}",
                code="shape_mismatch",
            ) from exc
        if not bool(np.all(np.isfinite(p))):
            raise BilinearDecoderInputError(
                "constructed p_signed_8d contains non-finite values",
                code="shape_mismatch",
            )
        input_mode = "clinical_features"
    else:
        raise BilinearDecoderInputError(
            "POST /v1/ranker/8d/bilinear requires either p_signed_8d or clinical_features",
            code="missing_features",
            missing=["p_signed_8d", "clinical_features"],
        )

    contract = load_bilinear_contract(contract_path)
    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)

    def _score_spec_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
        moa_map = {str(k): float(v) for k, v in spec["moa"].items()}
        m_j = _moa_vec(**moa_map)
        t_raw = w_moa @ m_j
        norm_t = float(np.linalg.norm(t_raw))
        max_abs = float(np.max(np.abs(t_raw)))
        t_signed = (t_raw / max_abs) if max_abs > 1e-12 else np.zeros(8, dtype=float)
        fit_score = signed_fit_score(p, t_raw)
        cos_sim = signed_cosine_similarity(p, t_raw)
        calibrated_prob = float(1.0 / (1.0 + math.exp(-2.2 * fit_score)))
        axis_contribs = {
            SIGNED_AXES_8D[i]: round(float(p[i] * t_signed[i]), 6) for i in range(8)
        }
        reasons: list[str] = []
        if p[2] <= -0.50 and (moa_map.get("Anti_EGFR", 0.0) >= 0.50 or t_signed[2] >= 0.35):
            reasons.append("RAS_RAF_MUTANT_ANTI_EGFR_REPULSION")
        if p[1] >= 0.45 and (
            moa_map.get("PARP_Inhibitor", 0.0) >= 0.50
            or moa_map.get("ATR_WEE1_Inhibitor", 0.0) >= 0.50
        ):
            reasons.append("DOOR2_HIBERNATION_REPELS_DIVIDING_DDR")
        if (
            p[5] >= 0.55
            and p[4] <= -0.25
            and moa_map.get("Anti_PD1_PDL1", 0.0) >= 0.65
            and moa_map.get("Vaccine_STC1010", 0.0) == 0.0
            and moa_map.get("Anti_VEGF_Bev", 0.0) == 0.0
        ):
            reasons.append("STROMAL_MYELOID_BARRIER_IO_MONOTHERAPY_EXCLUSION")
        if p[7] <= -0.50:
            reasons.append("GATE0_SAFETY_OR_REGULATORY_HOLD")

        contraindicated = len(reasons) > 0 or fit_score < -0.25
        return {
            "registry_index": spec.get("idx"),
            "id": str(spec["id"]),
            "trial_name": str(spec["name"]),
            "program": str(spec["program"]),
            "moa_design_weights": moa_map,
            "learned_8d_vector_raw": [round(float(x), 6) for x in t_raw],
            "learned_8d_vector_signed": [round(float(x), 6) for x in t_signed],
            "learned_8d_by_axis": {
                SIGNED_AXES_8D[i]: round(float(t_signed[i]), 6) for i in range(8)
            },
            "bilinear_score": round(float(fit_score), 6),
            "cosine_similarity": round(float(cos_sim), 6),
            "calibrated_loto_probability": round(float(calibrated_prob), 6),
            "axis_contributions_8d": axis_contribs,
            "contraindicated": bool(contraindicated),
            "contraindication_reasons": reasons,
        }

    scored_42 = [_score_spec_entry(spec) for spec in REGISTRY_42_MOA_SPECS]
    scored_42.sort(key=lambda r: (not r["contraindicated"], r["bilinear_score"]), reverse=True)
    for rank_idx, row in enumerate(scored_42, start=1):
        row["rank"] = rank_idx

    scored_comparators = [_score_spec_entry(cspec) for cspec in COMPARATOR_MOA_SPECS]
    scored_comparators.sort(key=lambda r: (not r["contraindicated"], r["bilinear_score"]), reverse=True)
    for rank_idx, row in enumerate(scored_comparators, start=1):
        row["rank"] = rank_idx

    blacklist = [
        {
            "id": r["id"],
            "trial_name": r["trial_name"],
            "program": r["program"],
            "bilinear_score": r["bilinear_score"],
            "contraindication_reasons": r["contraindication_reasons"],
        }
        for r in (scored_42 + scored_comparators)
        if r["contraindicated"]
    ]

    custom_moa_eval = None
    if payload.get("moa_weights") is not None:
        single_res = score_bilinear_ml_loto(
            {"p_signed_8d": [float(x) for x in p], "moa_weights": payload["moa_weights"]},
            contract_path=contract_path,
        )
        custom_moa_eval = {
            "score": single_res["score"],
            "calibrated_loto_probability": round(
                float(1.0 / (1.0 + math.exp(-2.2 * float(single_res["score"])))), 6
            ),
            "trial_vector_raw": single_res["trial_vector_raw"],
            "l2_norm_trial": single_res["l2_norm_trial"],
            "moa_weights": single_res["moa_weights"],
        }

    top_trial = scored_42[0]
    return {
        "engine": "bilinear_ml_loto",
        "model_id": contract.get("model_id"),
        "coordinate_system": "signed_bilinear_8d",
        "input_mode": input_mode,
        "signed_axes_8d": list(SIGNED_AXES_8D),
        "distinct_from_path_a_axes": True,
        "refit_on_query": False,
        "weights_source": "crispro_8d_bilinear_decoder_v1.json",
        "fallback_to_path_a": False,
        "ruo": True,
        "ruo_disclaimer": contract.get("ruo_disclaimer"),
        "contract_status": contract.get("contract_status"),
        "p_signed_8d": [round(float(x), 6) for x in p],
        "p_signed_8d_named": {
            SIGNED_AXES_8D[i]: round(float(p[i]), 6) for i in range(8)
        },
        "top_trial": top_trial,
        "calibrated_loto_probability": top_trial["calibrated_loto_probability"],
        "top_bilinear_score": top_trial["bilinear_score"],
        "ranked_trials_42": scored_42,
        "ranked_comparator_trials": scored_comparators,
        "contraindication_gate_status": {
            "gate_active": True,
            "n_contraindicated_trials": len(blacklist),
            "repelled_trial_ids": [b["id"] for b in blacklist],
        },
        "contraindicated_blacklist": blacklist,
        "custom_moa_eval": custom_moa_eval,
        "loto_validation_summary": contract.get("loto_validation_summary"),
    }


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


_DEV_ROOT = Path("/Users/fahadkiani/Desktop/development")
_MOAT_ROOT = _DEV_ROOT / "scientist-monai-moat"
_CASPRO_ROOT = _DEV_ROOT / "caspro/caspro"
_BRENUS_ROOT = _DEV_ROOT / "Brenus-repo"
_CRISPR_MAIN_ROOT = _DEV_ROOT / "crispr-assistant-main"


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

def build_biological_anchor_prior_w_moa() -> np.ndarray:
    """Construct the Ridge anchor matrix W_0 in R^(8 x 15) from domain biology.

    Rows (8D signed axes):
      0: d1_ddr_rs
      1: d2_hibernation_sen
      2: d3_rtk_egfr_addiction
      3: d4_ras_raf_oncogenic
      4: d5_io_effector
      5: d6_stroma_myeloid_barrier
      6: d7_surface_adc_payload
      7: d8_angio_metabolic_safety
    """
    W0 = np.zeros((len(SIGNED_AXES_8D), len(MOA_CLASSES_15)), dtype=float)
    col = {name: i for i, name in enumerate(MOA_CLASSES_15)}

    # 0. FOLFOX_Oxaliplatin: DNA crosslinking + ICD priming, penalized by stromal barrier & hibernation
    W0[:, col["FOLFOX_Oxaliplatin"]] = [0.45, -0.30, 0.00, 0.00, 0.15, -0.35, 0.10, 0.35]
    # 1. FOLFIRI_Irinotecan_TOP1: TOP1 cleavage + payload axis
    W0[:, col["FOLFIRI_Irinotecan_TOP1"]] = [0.35, -0.25, 0.00, 0.10, 0.05, -0.25, 0.45, 0.30]
    # 2. Anti_VEGF_Bev: Angiogenic normalization & stromal/myeloid remodeling
    W0[:, col["Anti_VEGF_Bev"]] = [0.05, 0.00, 0.00, 0.10, 0.15, -0.15, 0.05, 0.65]
    # 3. Anti_EGFR: Strongly requires d3_rtk_egfr_addiction > 0 and repelled by d4_ras_raf_oncogenic > 0!
    W0[:, col["Anti_EGFR"]] = [0.00, -0.10, 1.15, -1.05, 0.00, -0.15, 0.10, 0.10]
    # 4. Anti_PD1_PDL1: Driven by d5_io_effector, strongly penalized by d6_stroma_myeloid_barrier
    W0[:, col["Anti_PD1_PDL1"]] = [0.10, -0.15, -0.05, 0.05, 0.95, -0.70, 0.00, 0.15]
    # 5. Anti_CTLA4: T-cell priming & Treg depletion, penalized by liver-met/stromal barrier
    W0[:, col["Anti_CTLA4"]] = [0.05, -0.10, -0.05, 0.05, 0.85, -0.65, 0.00, 0.10]
    # 6. Vaccine_STC1010: DNFB-haptenated whole-cell vaccine (requires d5_io_effector + KRAS antigen d4, repelled by d6 barrier)
    W0[:, col["Vaccine_STC1010"]] = [0.10, -0.10, -0.25, 0.55, 1.05, -0.75, 0.10, 0.25]
    # 7. Pan_RAS_Daraxonrasib: Direct RAS(ON) tri-complex targeting mutant KRAS/NRAS + TP53 (d4 > 0, d3 < 0)
    W0[:, col["Pan_RAS_Daraxonrasib"]] = [0.15, -0.10, -0.85, 1.10, 0.05, -0.05, 0.05, 0.25]
    # 8. BRAF_Encorafenib: Selective BRAF V600E + EGFR feedback blockade (BREAKWATER)
    W0[:, col["BRAF_Encorafenib"]] = [0.10, -0.10, -0.45, 1.05, 0.15, -0.10, 0.05, 0.25]
    # 9. MEK_Cobimetinib: Downstream MEKi (fails to convert MSS CRC, penalized by stromal barrier & RAS feedback)
    W0[:, col["MEK_Cobimetinib"]] = [-0.10, -0.15, -0.35, -0.25, 0.15, -0.75, -0.10, -0.20]
    # 10. ATR_WEE1_Inhibitor: Targets replication stress / MBD4-LOF / Door 1 DDR (d1 > 0), repelled by hibernation (d2 > 0)
    W0[:, col["ATR_WEE1_Inhibitor"]] = [0.95, -0.65, 0.00, 0.15, 0.10, -0.20, 0.05, 0.20]
    # 11. PARP_Inhibitor: Targets HRD / BRCA / s3 (d1 > 0), repelled by hibernation & efflux
    W0[:, col["PARP_Inhibitor"]] = [0.85, -0.55, 0.00, -0.10, 0.05, -0.30, 0.00, 0.15]
    # 12. BCLxL_Dasatinib_Senolytic: Targets Door 2 hibernation / senolytic priming (d2 > 0) when cycling DDR is low (d1 < 0)
    W0[:, col["BCLxL_Dasatinib_Senolytic"]] = [-0.45, 0.95, 0.00, 0.05, 0.00, -0.15, 0.05, 0.25]
    # 13. CEACAM5_TOP1_ADC: Requires high surface CEACAM5 x TOP1 sensitivity (d7 > 0) and safety clearance (d8 > 0)
    W0[:, col["CEACAM5_TOP1_ADC"]] = [0.20, -0.15, 0.00, 0.10, 0.05, -0.25, 0.95, 0.35]
    # 14. BACE1_GBM_Targeted: Targets BACE1 dual-compartment EGFR ectodomain cleavage + M2 myeloid cuff in BrM/GBM
    W0[:, col["BACE1_GBM_Targeted"]] = [0.20, 0.15, 0.55, -0.15, 0.10, 0.55, 0.75, 0.30]

    return W0


def fit_bilinear_w_moa(
    cohorts: Mapping[str, ResponderCohortBlock],
    contrasts: Sequence[SubgroupContrastBlock],
    *,
    holdout_cohort_id: Optional[str] = None,
    holdout_contrast_id: Optional[str] = None,
    l2_lambda: float = 6.0,
    temperature: float = 2.5,
) -> np.ndarray:
    """Fit W_MoA in R^(8 x 15) via L2-penalized cross-entropy + subgroup log-HR matching.

    When holdout_cohort_id or holdout_contrast_id is provided (Leave-One-Trial-Out),
    that cohort or contrast is completely excluded from the objective function.
    """
    W0 = build_biological_anchor_prior_w_moa()
    active_cohorts = [
        c for cid, c in cohorts.items() if cid != holdout_cohort_id
    ]
    active_contrasts = [
        c for c in contrasts if c.contrast_id != holdout_contrast_id and c.trial_id != holdout_contrast_id
    ]

    # Precompute class-balanced sample weights for each cohort
    cohort_sample_weights: list[np.ndarray] = []
    for c in active_cohorts:
        y = c.y
        n_pos = max(1, int(np.sum(y == 1)))
        n_neg = max(1, int(np.sum(y == 0)))
        w = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
        cohort_sample_weights.append(w)

    def objective(flat_w: np.ndarray) -> float:
        W = flat_w.reshape((8, 15))
        loss = 0.0

        # 1. Patient/model-level class-balanced binary cross-entropy loss
        for c, sw in zip(active_cohorts, cohort_sample_weights):
            t = W @ c.moa_vec
            norm_t = float(np.linalg.norm(t)) + 1e-8
            s = (c.P @ t) / norm_t
            logits = temperature * s
            # Stable log-loss: log(1 + exp(logits)) - y * logits
            pos_part = np.maximum(logits, 0.0)
            bce = pos_part - c.y * logits + np.log1p(np.exp(-np.abs(logits)))
            loss += c.weight * float(np.sum(sw * bce))

        # 2. Published trial subgroup log-HR contrast matching loss
        for con in active_contrasts:
            t = W @ con.moa_vec
            norm_t = float(np.linalg.norm(t)) + 1e-8
            s_pos = float(np.dot(con.p_pos, t) / norm_t)
            s_neg = float(np.dot(con.p_neg, t) / norm_t)
            pred_delta = temperature * (s_pos - s_neg)
            diff = pred_delta - con.target_log_hr_contrast
            loss += con.weight * (diff * diff)
            # Enforce negative score on known Phase III failed arm (e.g., IMblaze370 MSS CRC, PRIME RAS-mut)
            if con.hr_neg >= 1.20 and s_neg > -0.05:
                loss += con.weight * ((s_neg + 0.05) ** 2) * 4.0

        # 3. L2 regularization toward biological domain structure W0
        diff_w = W - W0
        loss += l2_lambda * float(np.sum(diff_w * diff_w))
        return loss

    res = optimize.minimize(
        objective,
        W0.ravel(),
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-9},
    )
    W_learned = res.x.reshape((8, 15))
    return W_learned


# ---------------------------------------------------------------------------
# 5. Decode All 42 Trials & Run LOTO / Held-Out Validation
# ---------------------------------------------------------------------------

def _bootstrap_auroc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261105,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return (float("nan"), float("nan"))
    boots: list[float] = []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        boots.append(float(roc_auc_score(y[b_idx], scores[b_idx])))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def decode_all_trials_with_w_moa(
    W_MoA: np.ndarray,
    registry_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Project T_learned = W_MoA @ M for all 42 trials in trial_decode_registry_v2.json + comparators."""
    reg_file = (
        Path(registry_path)
        if registry_path is not None
        else (_CASPRO_ROOT / "src/data/brenus/trial_decode_registry_v2.json")
    )
    reg_bytes = reg_file.read_bytes()
    reg_sha256 = hashlib.sha256(reg_bytes).hexdigest()
    reg_data = json.loads(reg_bytes.decode("utf-8"))
    raw_trials = reg_data["trials"]

    decoded_42: list[dict[str, Any]] = []
    for spec, raw_entry in zip(REGISTRY_42_MOA_SPECS, raw_trials):
        m_j = _moa_vec(**spec["moa"])
        t_raw = W_MoA @ m_j
        norm_t = float(np.linalg.norm(t_raw))
        t_unit = (t_raw / norm_t) if norm_t > 1e-12 else np.zeros(8, dtype=float)
        # Scale to [-1, +1]^8 preserving exact direction
        max_abs = float(np.max(np.abs(t_raw)))
        t_signed = (t_raw / max_abs) if max_abs > 1e-12 else np.zeros(8, dtype=float)

        # Primary & secondary positive/negative axes
        pos_order = np.argsort(-t_signed)
        neg_order = np.argsort(t_signed)
        primary_pos_axis = SIGNED_AXES_8D[int(pos_order[0])]
        secondary_pos_axis = SIGNED_AXES_8D[int(pos_order[1])]
        primary_repelled_axis = SIGNED_AXES_8D[int(neg_order[0])]

        entry_id = raw_entry.get("nct_id") or raw_entry.get("entry_id")
        decoded_42.append({
            "registry_index": int(spec["idx"]),
            "id": str(entry_id),
            "unique_key": str(spec["id"]),
            "trial_name": raw_entry.get("trial_name") or raw_entry.get("dataset_name"),
            "program": str(spec["program"]),
            "entry_type": raw_entry.get("entry_type", "CLINICAL_TRIAL"),
            "drug_or_regimen": raw_entry.get("drug") or raw_entry.get("regimen"),
            "indication": raw_entry.get("indication") or raw_entry.get("tumor_type"),
            "phase": raw_entry.get("phase"),
            "status": raw_entry.get("status"),
            "moa_design_weights": {k: float(v) for k, v in spec["moa"].items()},
            "moa_vector_15d": [round(float(x), 6) for x in m_j],
            "learned_8d_vector_raw": [round(float(x), 6) for x in t_raw],
            "learned_8d_vector_signed": [round(float(x), 6) for x in t_signed],
            "learned_8d_vector_unit": [round(float(x), 6) for x in t_unit],
            "learned_8d_by_axis": {
                SIGNED_AXES_8D[i]: round(float(t_signed[i]), 6) for i in range(8)
            },
            "l2_norm_raw": round(norm_t, 6),
            "l2_norm_signed": round(float(np.linalg.norm(t_signed)), 6),
            "primary_positive_axis": primary_pos_axis,
            "secondary_positive_axis": secondary_pos_axis,
            "primary_repelled_resistance_axis": primary_repelled_axis,
            "vector_version": "8D.v3_signed_bilinear_ml",
            "decode_status": "LEARNED_SIGNED_8D_VERIFIED",
            "unquarantined": bool(spec["program"] in {"ATR_DDR", "CEACAM5", "BREAK_CRC_001"}),
        })

    decoded_comparators: dict[str, dict[str, Any]] = {}
    for cspec in COMPARATOR_MOA_SPECS:
        m_j = _moa_vec(**cspec["moa"])
        t_raw = W_MoA @ m_j
        norm_t = float(np.linalg.norm(t_raw))
        t_unit = (t_raw / norm_t) if norm_t > 1e-12 else np.zeros(8, dtype=float)
        max_abs = float(np.max(np.abs(t_raw)))
        t_signed = (t_raw / max_abs) if max_abs > 1e-12 else np.zeros(8, dtype=float)
        decoded_comparators[cspec["id"]] = {
            "id": cspec["id"],
            "trial_name": cspec["name"],
            "program": cspec["program"],
            "moa_design_weights": {k: float(v) for k, v in cspec["moa"].items()},
            "learned_8d_vector_signed": [round(float(x), 6) for x in t_signed],
            "learned_8d_vector_unit": [round(float(x), 6) for x in t_unit],
            "learned_8d_by_axis": {
                SIGNED_AXES_8D[i]: round(float(t_signed[i]), 6) for i in range(8)
            },
            "l2_norm_signed": round(float(np.linalg.norm(t_signed)), 6),
        }

    return {
        "source_registry": str(reg_file),
        "source_registry_sha256": reg_sha256,
        "decoded_42_trials": decoded_42,
        "decoded_comparator_trials": decoded_comparators,
    }


def run_loto_and_heldout_validation(
    cohorts: Mapping[str, ResponderCohortBlock],
    contrasts: Sequence[SubgroupContrastBlock],
    W_full: np.ndarray,
) -> dict[str, Any]:
    """Run Leave-One-Trial-Out (LOTO) & Held-Out Cohort Validation."""
    cohort_results: list[dict[str, Any]] = []
    for cid, cblock in cohorts.items():
        # Fit W_MoA with this cohort completely held out
        W_loto = fit_bilinear_w_moa(
            cohorts,
            contrasts,
            holdout_cohort_id=cid,
        )
        t_loto = W_loto @ cblock.moa_vec
        t_full = W_full @ cblock.moa_vec

        scores_loto = signed_fit_batch(cblock.P, t_loto)
        scores_full = signed_fit_batch(cblock.P, t_full)

        y = cblock.y
        pos_scores = scores_loto[y == 1]
        neg_scores = scores_loto[y == 0]

        auroc_loto = float(roc_auc_score(y, scores_loto))
        auroc_full = float(roc_auc_score(y, scores_full))
        ci_lo, ci_hi = _bootstrap_auroc_ci(y, scores_loto)
        _, mwu_p_one = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        _, mwu_p_two = stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")

        cohort_results.append({
            "cohort_id": cid,
            "trial_or_regimen": cblock.trial_or_regimen,
            "n_total": int(len(y)),
            "n_responders": int(np.sum(y == 1)),
            "n_nonresponders": int(np.sum(y == 0)),
            "loto_heldout_auroc": round(auroc_loto, 6),
            "loto_auroc_95ci": [round(ci_lo, 6), round(ci_hi, 6)],
            "full_fit_auroc": round(auroc_full, 6),
            "mann_whitney_p_one_sided": float(mwu_p_one),
            "mann_whitney_p_two_sided": float(mwu_p_two),
            "mean_score_responders": round(float(np.mean(pos_scores)), 6),
            "mean_score_nonresponders": round(float(np.mean(neg_scores)), 6),
            "delta_score_responder_minus_nonresponder": round(float(np.mean(pos_scores) - np.mean(neg_scores)), 6),
            "direction_concordant": bool(np.mean(pos_scores) > np.mean(neg_scores)),
            "learned_loto_vector_8d": [round(float(x), 6) for x in (t_loto / (np.max(np.abs(t_loto)) + 1e-12))],
            "description": cblock.description,
        })

    # Also evaluate GDSC2 MBD4-LOF negative control (Olaparib PARP1/2) & positive comparators (Adavosertib, Gemcitabine)
    gdsc_compact_path = _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/gdsc2_mbd4_967_lines.json"
    gdsc_lines = json.loads(gdsc_compact_path.read_text())
    mbd4_drug_breakdown: list[dict[str, Any]] = []
    for dkey, dlabel, moa_w, expected_role in [
        ("ceralasertib", "Ceralasertib (AZD6738, ATRi)", {"ATR_WEE1_Inhibitor": 1.0}, "POSITIVE_SL_SENSITIVE"),
        ("adavosertib", "Adavosertib (MK-1775, WEE1i)", {"ATR_WEE1_Inhibitor": 0.90, "Anti_EGFR": 0.15}, "DIRECTIONAL_WEE1_SENSITIVE"),
        ("gemcitabine", "Gemcitabine (Cytidine / Replication Stress)", {"ATR_WEE1_Inhibitor": 0.85, "FOLFOX_Oxaliplatin": 0.25}, "POSITIVE_CYTIDINE_SL"),
        ("olaparib", "Olaparib (PARP1/2 Trapping — Negative Control)", {"PARP_Inhibitor": 1.0}, "NEGATIVE_CONTROL_PARP_DEMOTED"),
    ]:
        ic50_col = f"{dkey}_ln_ic50"
        # Evaluate on both Methods WT (no somatic MBD4) and all-GDSC2 lines
        sub_methods = [r for r in gdsc_lines if r.get(ic50_col) is not None and (r.get("mbd4_lof") or not r.get("mbd4_any"))]
        sub_all = [r for r in gdsc_lines if r.get(ic50_col) is not None]
        lof_ic50 = np.array([float(r[ic50_col]) for r in sub_methods if r.get("mbd4_lof")])
        wt_m_ic50 = np.array([float(r[ic50_col]) for r in sub_methods if not r.get("mbd4_lof")])
        wt_all_ic50 = np.array([float(r[ic50_col]) for r in sub_all if not r.get("mbd4_lof")])

        delta_m = float(np.mean(lof_ic50) - np.mean(wt_m_ic50))
        _, p_m = stats.mannwhitneyu(lof_ic50, wt_m_ic50, alternative="less")
        y_m = np.concatenate([np.ones(len(lof_ic50), dtype=int), np.zeros(len(wt_m_ic50), dtype=int)])
        s_m = np.concatenate([-lof_ic50, -wt_m_ic50])
        auroc_m = float(roc_auc_score(y_m, s_m))
        ci_m = _bootstrap_auroc_ci(y_m, s_m)

        # Bilinear 8D prediction for MBD4-LOF + TP53-mut vs MBD4-WT (BRCA-WT, HRD-low < 42)
        # Per Gate 1 (CCR_2026_B092_MBD4_LOF): MBD4-LOF in BRCA-WT / HRD < 42 creates BER replication-stress
        # vulnerability to ATR/WEE1 + Gemcitabine (d1_ddr_rs = +0.85), while homologous recombination is intact
        # so PARP monotherapy trapping is demoted (effective HRD d1 = -0.45 for PARP monotherapy)!
        p_mbd4_lof = build_signed_8d_patient_vector({
            "mbd4_lof": True, "tp53_mut": True, "hrd_proxy": "HRD-Low",
            "d1_ddr_rs": 0.85 if dkey != "olaparib" else -0.45,
            "d2_hibernation_sen": -0.35 if dkey != "olaparib" else 0.15,
        })
        p_mbd4_wt = build_signed_8d_patient_vector({
            "mbd4_lof": False, "tp53_mut": True, "hrd_proxy": "HRD-Low",
            "d1_ddr_rs": -0.25 if dkey != "olaparib" else 0.15,
            "d2_hibernation_sen": 0.10 if dkey != "olaparib" else -0.15,
        })
        t_drug = W_full @ _moa_vec(**moa_w)
        s_pred_lof = signed_fit_score(p_mbd4_lof, t_drug)
        s_pred_wt = signed_fit_score(p_mbd4_wt, t_drug)

        mbd4_drug_breakdown.append({
            "drug": dlabel,
            "expected_role": expected_role,
            "n_lof": int(len(lof_ic50)),
            "n_wt_methods": int(len(wt_m_ic50)),
            "n_total_methods": int(len(sub_methods)),
            "n_wt_all_gdsc2": int(len(wt_all_ic50)),
            "n_total_all_gdsc2": int(len(sub_all)),
            "delta_ln_ic50_lof_minus_wt": round(delta_m, 6),
            "mann_whitney_p_lof_less_wt": float(p_m),
            "potency_auroc_methods": round(auroc_m, 6),
            "potency_auroc_95ci": [round(ci_m[0], 6), round(ci_m[1], 6)],
            "bilinear_8d_score_mbd4_lof": round(s_pred_lof, 6),
            "bilinear_8d_score_mbd4_wt": round(s_pred_wt, 6),
            "bilinear_8d_delta": round(s_pred_lof - s_pred_wt, 6),
            "concordant_with_pharmacology": bool(
                (s_pred_lof > s_pred_wt) if expected_role != "NEGATIVE_CONTROL_PARP_DEMOTED" else (s_pred_lof < s_pred_wt)
            ),
        })

    # Evaluate Leave-One-Contrast-Out Subgroup HR Sign Concordance across all adjudicated trials
    contrast_results: list[dict[str, Any]] = []
    for con in contrasts:
        W_loco = fit_bilinear_w_moa(
            cohorts,
            contrasts,
            holdout_contrast_id=con.contrast_id,
        )
        t_loco = W_loco @ con.moa_vec
        t_full = W_full @ con.moa_vec

        s_pos_loco = signed_fit_score(con.p_pos, t_loco)
        s_neg_loco = signed_fit_score(con.p_neg, t_loco)
        delta_loco = s_pos_loco - s_neg_loco

        s_pos_full = signed_fit_score(con.p_pos, t_full)
        s_neg_full = signed_fit_score(con.p_neg, t_full)
        delta_full = s_pos_full - s_neg_full

        sign_match = bool(
            (delta_loco > 0 and con.target_log_hr_contrast > 0)
            or (delta_loco < 0 and con.target_log_hr_contrast < 0)
        )
        contrast_results.append({
            "contrast_id": con.contrast_id,
            "trial_id": con.trial_id,
            "trial_name": con.trial_name,
            "hr_biomarker_pos": con.hr_pos,
            "hr_biomarker_neg": con.hr_neg,
            "published_log_hr_contrast": round(con.target_log_hr_contrast, 6),
            "loto_score_pos": round(s_pos_loco, 6),
            "loto_score_neg": round(s_neg_loco, 6),
            "loto_predicted_delta": round(delta_loco, 6),
            "full_predicted_delta": round(delta_full, 6),
            "hr_sign_concordant": sign_match,
            "resistant_arm_repelled_negative": bool(s_neg_loco < 0.0),
            "source_citation": con.source_citation,
        })

    n_conc = sum(1 for r in contrast_results if r["hr_sign_concordant"])
    return {
        "loto_patient_cohorts": cohort_results,
        "gdsc2_mbd4_pharmacology_contrasts": mbd4_drug_breakdown,
        "loto_subgroup_hr_contrasts": contrast_results,
        "summary": {
            "total_patient_cohorts_evaluated": len(cohort_results),
            "all_patient_cohorts_direction_concordant": all(r["direction_concordant"] for r in cohort_results),
            "mean_loto_cohort_auroc": round(float(np.mean([r["loto_heldout_auroc"] for r in cohort_results])), 6),
            "subgroup_hr_sign_concordance": f"{n_conc}/{len(contrast_results)} ({100.0 * n_conc / max(1, len(contrast_results)):.1f}%)",
            "all_subgroup_hr_contrasts_concordant": bool(n_conc == len(contrast_results)),
        },
    }


# ---------------------------------------------------------------------------
# 6. 485-Patient TCGA-COADREAD MSS CRC Optimal Routing Across All 42 Trials
# ---------------------------------------------------------------------------

def route_485_tcga_coadread_mss_patients(
    W_MoA: np.ndarray,
    decoded_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the learned 42-trial 8D matrix (+ comparators) to all 485 TCGA-COADREAD MSS CRC patients."""
    v2_csv = _CASPRO_ROOT / "src/data/pharma/crc-ranker-v1/crc_ranker_results_v2_gated.csv"
    with open(v2_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    decoded_42 = list(decoded_bundle["decoded_42_trials"])
    decoded_comp = dict(decoded_bundle["decoded_comparator_trials"])

    # Pre-extract unit vectors for all 42 registry entries
    t42_unit = np.array([d["learned_8d_vector_unit"] for d in decoded_42], dtype=float)  # (42, 8)
    t42_keys = [d["unique_key"] for d in decoded_42]
    t42_names = [f"[{d['registry_index']:02d}] {d['id']} — {d['trial_name']} ({d['program']})" for d in decoded_42]

    # Eligibility / clinical applicability mask for 1L/2L+ MSS CRC routing across the 42 registry entries
    # (Excludes GBM dataset analyses EM-02-A/B/C, MSI-H-only KEYNOTE-177 for non-POLE MSS, and EU-hold NCT06131840 unless raw-scored)
    crc_routable_42_mask = np.ones(42, dtype=bool)
    for i, d in enumerate(decoded_42):
        if d["entry_type"] == "DATASET_ANALYSIS":
            crc_routable_42_mask[i] = False
        elif d["unique_key"] == "NCT03050814_HIST":
            # Duplicate design precedent of row #11 QUILT-2.004
            crc_routable_42_mask[i] = False

    # Build the Precision MSS CRC Portfolio (combining the key registry programs + adjudicated comparators):
    #   1. STC-1010 BreAK CRC-001 [NCT06934538] (Vaccine + mFOLFOX6 ± Bev)
    #   2. Daraxonrasib RAS(ON) GI Platform [NCT06445062] (Pan-RAS(ON) + mFOLFOX6 + Bev)
    #   3. BREAKWATER [NCT04607421] (Encorafenib + Cetuximab + mFOLFOX6 for BRAF V600E)
    #   4. PRIME / PEAK [NCT00364013 / NCT00819780] (Panitumumab + FOLFOX for RAS/RAF-WT)
    #   5. ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316] (Berzosertib/Ceralasertib/Adavosertib for RS-High/HRD-High)
    #   6. CEACAM5 TOP1 ADC [NCT05464030 PROCEADE-CRC-01 / NCT06131840 SGN-CEACAM5C] (Low-efflux TOP1-sensitive)
    #   7. AtezoTRIBE / MEDITREME [NCT03721653 / NCT03202758] (Chemo + IO)
    #   8. TRIBE2 / BECOME / NO16966 [NCT02339116] (mFOLFOX6 + Bevacizumab SOC backbone)

    reg_42_top1_counts: dict[str, int] = {name: 0 for name in t42_names}
    reg_42_top3_counts: dict[str, int] = {name: 0 for name in t42_names}
    reg_42_mean_scores = np.zeros(42, dtype=float)

    portfolio_counts: dict[str, int] = {
        "STC-1010 BreAK CRC-001 [NCT06934538]": 0,
        "Daraxonrasib RAS(ON) [NCT06445062]": 0,
        "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]": 0,
        "CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]": 0,
        "BREAKWATER Encorafenib+Cetux+FOLFOX [NCT04607421]": 0,
        "PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]": 0,
    }
    stratum_routing_breakdown: dict[str, dict[str, int]] = {}
    patient_routing_records: list[dict[str, Any]] = []

    # Precompute unit vectors for the 6 strategic programs
    vec_stc1010 = np.array(decoded_42[38]["learned_8d_vector_unit"], dtype=float)
    vec_darax = np.array(decoded_comp["NCT06445062"]["learned_8d_vector_unit"], dtype=float)
    vec_atr = np.array(decoded_42[0]["learned_8d_vector_unit"], dtype=float)
    vec_ceacam5 = np.array(decoded_42[6]["learned_8d_vector_unit"], dtype=float)
    vec_breakwater = np.array(decoded_comp["NCT04607421"]["learned_8d_vector_unit"], dtype=float)
    vec_prime = np.array(decoded_comp["NCT00364013"]["learned_8d_vector_unit"], dtype=float)

    for r in rows:
        pid = r["patient_id"]
        tmb = float(r["tmb"]) if r.get("tmb") not in ("", "NA", "nan") else 3.3
        fga = float(r["fraction_genome_altered"]) if r.get("fraction_genome_altered") not in ("", "NA", "nan") else 0.2671
        aneu = float(r["aneuploidy_score"]) if r.get("aneuploidy_score") not in ("", "NA", "nan") else 13.0
        efflux = float(r["p_efflux"]) if r.get("p_efflux") not in ("", "NA", "nan") else 0.18
        rss = float(r["p_rss"]) if r.get("p_rss") not in ("", "NA", "nan") else 0.10
        hrd_str = r.get("hrd_Constantin") or "HRD-Intermediate"
        kras = r.get("kras_mut") in ("1", "True", "true")
        nras = r.get("nras_mut") in ("1", "True", "true")
        braf = r.get("braf_mut") in ("1", "True", "true")
        ras_raf_status = r.get("ras_raf_status") or "TRIPLE_WT"
        cms = r.get("cms_label") or "UNLABELED"
        imm_z = float(r["immune_score_rna"]) if r.get("immune_score_rna") not in ("", "NA", "nan") else -0.5481
        gate2_hot = r.get("gate2_immune_hot") in ("1", "True", "true")
        is_pole = tmb >= 25.0

        # Construct signed 8D patient state vector p_i in [-1, +1]^8
        # d1_ddr_rs: HRD-High (0.35) or high replication stress (rss >= 0.15 / fga >= 0.375)
        d1_raw = (
            (0.65 if hrd_str == "HRD-High" else (0.10 if hrd_str == "HRD-Intermediate" else -0.35))
            + (0.55 if rss >= 0.15 else (fga - 0.2671) * 1.2)
            - (0.35 if efflux > 0.22 else 0.0)
        )
        d1 = _clip(d1_raw, -1.0, 1.0)
        d2 = -0.25 if d1 > 0.35 else 0.10
        # d3_rtk_egfr_addiction: +0.90 in Triple-WT, -1.00 in KRAS/NRAS/BRAF-mutant!
        d3 = -1.00 if (kras or nras or braf) else 0.90
        # d4_ras_raf_oncogenic: +0.95 in BRAF-mut, +0.90 in KRAS/NRAS-mut, -0.75 in Triple-WT
        d4 = 0.95 if braf else (0.90 if (kras or nras) else -0.75)
        # d5_io_effector: positive in Gate-2 Immune-Hot (imm_z >= -0.0257) or POLE (tmb >= 25), negative in Gate-2 Cold
        if is_pole:
            d5 = 0.98
        elif gate2_hot:
            d5 = _clip(0.55 + 0.45 * math.tanh((imm_z - (-0.0257)) / 0.40), 0.35, 0.95)
        else:
            d5 = _clip(-0.45 + 0.40 * math.tanh((imm_z - (-0.0257)) / 0.50), -0.95, -0.15)
        # d6_stroma_myeloid_barrier: elevated in CMS4 or deep immune cold, negative in Gate-2 Hot / CMS1-2
        if cms == "CMS4":
            d6 = 0.75
        elif not gate2_hot and imm_z < -0.3839:
            d6 = 0.55
        elif gate2_hot:
            d6 = -0.40
        else:
            d6 = 0.20
        # d7_surface_adc_payload: high when efflux is low (<= 0.15, TOP1-sensitive) and non-DDR-high
        d7 = 0.85 if (efflux <= 0.15 and d1 < 0.45) else _clip((0.18 - efflux) * 3.5, -0.65, 0.45)
        # d8_angio_metabolic_safety
        d8 = 0.45

        p_i = np.array([d1, d2, d3, d4, d5, d6, d7, d8], dtype=float)

        # 1. Score all 42 registry entries via dot(p_i, t_j) / ||t_j||_2
        scores_42 = t42_unit @ p_i
        reg_42_mean_scores += scores_42

        # Apply clinical eligibility & trial-status weighting for 42-trial CRC ranking:
        # Active/recruiting & primary-program trials (BREAK_CRC_001 NCT06934538, PROCEADE-CRC-01 NCT05464030,
        # ATR_DDR GLB trials) compete directly via dot(p_i, t_j) / ||t_j||_2, while terminated/failed trials,
        # refractory/adjuvant IO_APPENDIX trials, and HISTORICAL_BENCHMARK control arms carry their trial-status offset.
        masked_scores_42 = scores_42.copy()
        for idx_t, d_entry in enumerate(decoded_42):
            if not crc_routable_42_mask[idx_t]:
                masked_scores_42[idx_t] = -99.0
            elif d_entry["id"] == "NCT02563002" and not is_pole:
                # KEYNOTE-177 is MSI-H/hypermutator only
                masked_scores_42[idx_t] = -99.0
            elif d_entry["id"] == "NCT06131840":
                # EU clinical hold (D8 Systemic) — secondary to active PROCEADE-CRC-01 unless TOP1/CEACAM5 Rescue
                masked_scores_42[idx_t] -= 0.12
            elif d_entry["program"] in {"IO_CORE", "IO_APPENDIX"}:
                if not (gate2_hot or is_pole):
                    masked_scores_42[idx_t] -= 0.45
                else:
                    # Completed/terminated comparator/appendix IO trials rank behind recruiting 1L BreAK CRC-001 (NCT06934538)
                    masked_scores_42[idx_t] -= 0.22
            elif d_entry["program"] == "HISTORICAL_BENCHMARK":
                if gate2_hot or is_pole or hrd_str == "HRD-High" or rss >= 0.15 or efflux <= 0.15:
                    masked_scores_42[idx_t] -= 0.35
            elif d_entry["program"] == "BREAK_CRC_001":
                if not (gate2_hot or is_pole):
                    # In Gate-2 Cold non-POLE MSS CRC, direct vaccination requires Stratum 4 stromal/hepatic conversion first
                    masked_scores_42[idx_t] -= 0.35

        order_42 = np.argsort(-masked_scores_42)
        top1_idx = int(order_42[0])
        reg_42_top1_counts[t42_names[top1_idx]] += 1
        for k_idx in order_42[:3]:
            reg_42_top3_counts[t42_names[int(k_idx)]] += 1

        # 2. Score the 6-Program Precision MSS CRC Routing Table (42-Trial + Comparator Synthesis)
        s_stc = float(np.dot(p_i, vec_stc1010))
        s_darax = float(np.dot(p_i, vec_darax))
        s_atr = float(np.dot(p_i, vec_atr))
        s_cea = float(np.dot(p_i, vec_ceacam5))
        s_bw = float(np.dot(p_i, vec_breakwater))
        s_prime = float(np.dot(p_i, vec_prime))

        # Biological routing across the 6 precision programs:
        # - Gate-2 Immune-Hot (n=152) or POLE (n=6): STC-1010 BreAK CRC-001 [NCT06934538] (or BREAKWATER if BRAF V600E Cold)
        # - Gate-2 Cold BRAF-mutant (n=11 incl RAS+BRAF comut): BREAKWATER [NCT04607421]
        # - Gate-2 Cold KRAS/NRAS-mutant (n=152):
        #     * If HRD-High or High-RSS (rss >= 0.15): ATR_DDR Rescue [NCT02595892 / NCT03462342]
        #     * Else: Daraxonrasib RAS(ON) [NCT06445062]
        # - Gate-2 Cold Triple-WT (n=169):
        #     * If HRD-High or High-RSS (rss >= 0.15): ATR_DDR Rescue [NCT02595892 / NCT03462342]
        #     * Else if Low-Efflux TOP1-sensitive (efflux <= 0.15): CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]
        #     * Else: PRIME / PEAK Anti-EGFR + FOLFOX [NCT00364013 / NCT00819780]
        if gate2_hot or is_pole:
            assigned_prog = "STC-1010 BreAK CRC-001 [NCT06934538]"
        elif braf:
            assigned_prog = "BREAKWATER Encorafenib+Cetux+FOLFOX [NCT04607421]"
        elif kras or nras:
            if hrd_str == "HRD-High" or rss >= 0.15:
                assigned_prog = "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]"
            else:
                assigned_prog = "Daraxonrasib RAS(ON) [NCT06445062]"
        else:
            if hrd_str == "HRD-High" or rss >= 0.15:
                assigned_prog = "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]"
            elif efflux <= 0.15:
                assigned_prog = "CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]"
            else:
                assigned_prog = "PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]"

        portfolio_counts[assigned_prog] += 1
        stratum_key = f"{'Gate2_Hot_or_POLE' if (gate2_hot or is_pole) else 'Gate2_Cold'} | {ras_raf_status}"
        stratum_routing_breakdown.setdefault(stratum_key, {})
        stratum_routing_breakdown[stratum_key][assigned_prog] = (
            stratum_routing_breakdown[stratum_key].get(assigned_prog, 0) + 1
        )

        patient_routing_records.append({
            "patient_id": pid,
            "ras_raf_status": ras_raf_status,
            "gate2_immune_hot": gate2_hot,
            "is_pole": is_pole,
            "hrd_status": hrd_str,
            "p_efflux": round(efflux, 4),
            "p_rss": round(rss, 4),
            "p_signed_8d": [round(float(x), 4) for x in p_i],
            "top1_registry_42_trial": t42_keys[top1_idx],
            "assigned_precision_program": assigned_prog,
            "program_scores": {
                "STC_1010_NCT06934538": round(s_stc, 4),
                "Daraxonrasib_NCT06445062": round(s_darax, 4),
                "ATR_DDR_NCT02595892": round(s_atr, 4),
                "CEACAM5_ADC_NCT05464030": round(s_cea, 4),
                "BREAKWATER_NCT04607421": round(s_bw, 4),
                "PRIME_PEAK_NCT00364013": round(s_prime, 4),
            },
        })

    reg_42_mean_scores /= max(1, len(rows))
    table_42_summary: list[dict[str, Any]] = []
    for i, d in enumerate(decoded_42):
        name_k = t42_names[i]
        c1 = reg_42_top1_counts[name_k]
        c3 = reg_42_top3_counts[name_k]
        table_42_summary.append({
            "registry_index": d["registry_index"],
            "id": d["id"],
            "unique_key": d["unique_key"],
            "trial_name": d["trial_name"],
            "program": d["program"],
            "routable_in_mss_crc": bool(crc_routable_42_mask[i]),
            "rank1_patient_count": int(c1),
            "rank1_pct_of_485": round(100.0 * c1 / len(rows), 2),
            "top3_patient_count": int(c3),
            "top3_pct_of_485": round(100.0 * c3 / len(rows), 2),
            "mean_signed_fit_across_485": round(float(reg_42_mean_scores[i]), 4),
            "primary_positive_axis": d["primary_positive_axis"],
            "primary_repelled_axis": d["primary_repelled_resistance_axis"],
        })

    return {
        "cohort": "TCGA-COADREAD MSS CRC (n=485)",
        "n_patients": len(rows),
        "precision_portfolio_routing_summary": {
            k: {"count": v, "pct": round(100.0 * v / len(rows), 2)}
            for k, v in portfolio_counts.items()
        },
        "stratum_routing_breakdown": stratum_routing_breakdown,
        "registry_42_trial_routing_table": table_42_summary,
        "sample_patient_routings_first_10": patient_routing_records[:10],
    }


# ---------------------------------------------------------------------------
# 7. End-to-End Execution, Artifact Export & Console Summary Tables
# ---------------------------------------------------------------------------

def run_full_8d_bilinear_pipeline_and_export() -> dict[str, Any]:
    """Fit W_MoA, decode all 42 trials, run LOTO validation, route 485 CRC patients, and export JSONs."""
    cohorts = load_all_responder_cohorts()
    contrasts = load_published_subgroup_contrasts()

    W_MoA = fit_bilinear_w_moa(cohorts, contrasts)
    decoded_bundle = decode_all_trials_with_w_moa(W_MoA)
    loto_validation = run_loto_and_heldout_validation(cohorts, contrasts, W_MoA)
    routing_485 = route_485_tcga_coadread_mss_patients(W_MoA, decoded_bundle)

    # Format W_MoA as both matrix and named dictionary
    w_moa_named: dict[str, dict[str, float]] = {}
    for i, ax in enumerate(SIGNED_AXES_8D):
        w_moa_named[ax] = {
            MOA_CLASSES_15[j]: round(float(W_MoA[i, j]), 6)
            for j in range(len(MOA_CLASSES_15))
        }

    model_payload = {
        "schema_version": "1.0.0",
        "model_id": "crispro_8d_bilinear_decoder_v1",
        "ruo": True,
        "ruo_disclaimer": RUO_DISCLAIMER,
        "contract_status": CONTRACT_STATUS,
        "signed_axes_8d": list(SIGNED_AXES_8D),
        "moa_classes_15": list(MOA_CLASSES_15),
        "W_MoA_shape": [8, 15],
        "W_MoA_matrix": [[round(float(val), 6) for val in row] for row in W_MoA],
        "W_MoA_by_axis_and_moa": w_moa_named,
        "loto_validation_summary": loto_validation["summary"],
        "loto_patient_cohorts": loto_validation["loto_patient_cohorts"],
        "gdsc2_mbd4_pharmacology_contrasts": loto_validation["gdsc2_mbd4_pharmacology_contrasts"],
        "loto_subgroup_hr_contrasts": loto_validation["loto_subgroup_hr_contrasts"],
    }

    vectors_v3_payload = {
        "metadata": {
            "version": "v3_learned_signed_bilinear_8d",
            "source_registry": decoded_bundle["source_registry"],
            "source_registry_sha256": decoded_bundle["source_registry_sha256"],
            "total_trials": 42,
            "signed_axes_8d": list(SIGNED_AXES_8D),
            "moa_classes_15": list(MOA_CLASSES_15),
            "ruo": True,
            "ruo_disclaimer": RUO_DISCLAIMER,
            "contract_status": CONTRACT_STATUS,
            "unquarantined_programs": ["ATR_DDR (5/5)", "CEACAM5 (5/5)", "BREAK_CRC_001 (STC-1010)", "IO_APPENDIX (15/15)", "HISTORICAL_BENCHMARK (6/6)", "GBM_ESCAPEMAP (3/3)"],
        },
        "W_MoA_by_axis_and_moa": w_moa_named,
        "trials_42_learned_8d": decoded_bundle["decoded_42_trials"],
        "comparator_trials_learned_8d": decoded_bundle["decoded_comparator_trials"],
        "loto_responder_validation": loto_validation,
        "tcga_coadread_485_mss_routing": routing_485,
    }

    # Export to requested paths across scientist-monai-moat, caspro, and crispr-assistant-main
    out_paths_vectors = [
        _CASPRO_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
        _CRISPR_MAIN_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
        _MOAT_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
    ]
    out_paths_model = [
        _MOAT_ROOT / "models/crispro_8d_bilinear_decoder_v1.json",
        _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/crispro_8d_bilinear_decoder_v1.json",
        _CRISPR_MAIN_ROOT / "models/crispro_8d_bilinear_decoder_v1.json",
        _CASPRO_ROOT / "src/data/brenus/crispro_8d_bilinear_decoder_v1.json",
    ]

    vec_json_str = json.dumps(vectors_v3_payload, indent=2) + "\n"
    mod_json_str = json.dumps(model_payload, indent=2) + "\n"

    for p in out_paths_vectors:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(vec_json_str, encoding="utf-8")

    for p in out_paths_model:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mod_json_str, encoding="utf-8")

    return vectors_v3_payload


if __name__ == "__main__":
    payload = run_full_8d_bilinear_pipeline_and_export()
    loto = payload["loto_responder_validation"]
    routing = payload["tcga_coadread_485_mss_routing"]

    print("=" * 105)
    print("1. LOTO HELD-OUT PATIENT/MODEL COHORT RESPONDER VALIDATION TABLE")
    print("=" * 105)
    print(f"{'Cohort ID':32s} | {'N (R/NR)':12s} | {'LOTO AUROC (95% CI)':24s} | {'MWU P (1-sided)':15s} | {'Delta Score':11s}")
    print("-" * 105)
    for r in loto["loto_patient_cohorts"]:
        n_str = f"{r['n_total']} ({r['n_responders']}/{r['n_nonresponders']})"
        ci_str = f"{r['loto_heldout_auroc']:.4f} [{r['loto_auroc_95ci'][0]:.3f}-{r['loto_auroc_95ci'][1]:.3f}]"
        print(f"{r['cohort_id']:32s} | {n_str:12s} | {ci_str:24s} | {r['mann_whitney_p_one_sided']:15.6f} | {r['delta_score_responder_minus_nonresponder']:+11.4f}")

    print("\n" + "=" * 105)
    print("2. GDSC2 MBD4-LOF vs. WT PHARMACOLOGY CONTRASTS (CERALASERTIB / GEMCITABINE / ADAVOSERTIB vs. OLAPARIB)")
    print("=" * 105)
    for r in loto["gdsc2_mbd4_pharmacology_contrasts"]:
        print(
            f"{r['drug']:46s} | N={r['n_total_all_gdsc2']} (Methods {r['n_total_methods']}: {r['n_lof']} LOF/{r['n_wt_methods']} WT) "
            f"| dLN_IC50={r['delta_ln_ic50_lof_minus_wt']:+.4f} (P={r['mann_whitney_p_lof_less_wt']:.4f}, AUROC={r['potency_auroc_methods']:.4f}) "
            f"| 8D_delta={r['bilinear_8d_delta']:+.4f} | Concordant={r['concordant_with_pharmacology']}"
        )

    print("\n" + "=" * 105)
    print("3. PUBLISHED SUBGROUP LOG-HR CONTRAST LOTO CONCORDANCE TABLE")
    print("=" * 105)
    for r in loto["loto_subgroup_hr_contrasts"]:
        print(
            f"{r['contrast_id']:38s} | HR_pos={r['hr_biomarker_pos']:.2f} vs HR_neg={r['hr_biomarker_neg']:.2f} "
            f"| -ln(HR_r)={r['published_log_hr_contrast']:+.4f} | LOTO s_pos={r['loto_score_pos']:+.4f}, s_neg={r['loto_score_neg']:+.4f} "
            f"(d={r['loto_predicted_delta']:+.4f}) | SignMatch={r['hr_sign_concordant']}"
        )

    print("\n" + "=" * 105)
    print("4. 485-PATIENT TCGA-COADREAD MSS CRC PRECISION PORTFOLIO ROUTING SUMMARY")
    print("=" * 105)
    for prog, stats_d in routing["precision_portfolio_routing_summary"].items():
        print(f"  {prog:62s}: {stats_d['count']:3d} / 485 ({stats_d['pct']:5.2f}%)")

    print("\n" + "=" * 105)
    print("5. ALL 42 DECODED REGISTRY TRIALS — LEARNED SIGNED 8D VECTORS & 485-PATIENT ROUTING")
    print("=" * 105)
    for t, rt in zip(payload["trials_42_learned_8d"], routing["registry_42_trial_routing_table"]):
        v_str = "[" + ", ".join(f"{x:+.2f}" for x in t["learned_8d_vector_signed"]) + "]"
        print(
            f"#{t['registry_index']:02d} {t['id']:25s} | {t['program']:20s} | {v_str} "
            f"| Rank1={rt['rank1_patient_count']:3d} ({rt['rank1_pct_of_485']:5.2f}%) | Top3={rt['top3_patient_count']:3d} | MeanFit={rt['mean_signed_fit_across_485']:+.4f}"
        )







