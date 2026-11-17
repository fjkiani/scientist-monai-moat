"""Signed 8D patient vector construction."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .constants import SIGNED_AXES_8D

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
