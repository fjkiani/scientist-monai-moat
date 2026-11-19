"""Fit W_MoA from cohorts (offline; not used on query path)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

from .constants import MOA_CLASSES_15, MOA_CLASSES_24, MOA_CLASSES_9_EXT, SIGNED_AXES_8D
from .cohorts import ResponderCohortBlock, SubgroupContrastBlock, load_all_responder_cohorts, load_published_subgroup_contrasts
from .moa import _moa_vec, pad_moa_vec_15_to_24
from .scoring import signed_fit_batch, signed_fit_score

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


def _coerce_moa_vec(moa: np.ndarray, moa_width: int) -> np.ndarray:
    m = np.asarray(moa, dtype=float).reshape(-1)
    if m.shape[0] == moa_width:
        return m
    if moa_width == 24 and m.shape[0] == 15:
        return pad_moa_vec_15_to_24(m)
    if moa_width == 15 and m.shape[0] == 24:
        return m[:15]
    raise ValueError(f"moa_vec length {m.shape[0]} incompatible with moa_width={moa_width}")


def fit_bilinear_w_moa(
    cohorts: Mapping[str, ResponderCohortBlock],
    contrasts: Sequence[SubgroupContrastBlock],
    *,
    holdout_cohort_id: Optional[str] = None,
    holdout_contrast_id: Optional[str] = None,
    l2_lambda: float = 6.0,
    temperature: float = 2.5,
    moa_width: int = 15,
    w0_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit W_MoA in R^(8 x K) via L2-penalized cross-entropy + subgroup log-HR matching.

    K defaults to 15 (v1). Pass moa_width=24 to train the full v2 matrix (pads 15D
    cohort/contrast MoA vectors). When holdout_* is set (LOTO), that block is excluded.
    """
    if moa_width not in (15, 24):
        raise ValueError(f"moa_width must be 15 or 24, got {moa_width}")
    if w0_override is not None:
        W0 = np.asarray(w0_override, dtype=float)
        if W0.shape != (8, moa_width):
            raise ValueError(f"w0_override shape {W0.shape} != (8, {moa_width})")
    elif moa_width == 15:
        W0 = build_biological_anchor_prior_w_moa()
    else:
        W0 = compose_w_moa_v2_from_v1_matrix(build_biological_anchor_prior_w_moa())

    active_cohorts = [
        c for cid, c in cohorts.items() if cid != holdout_cohort_id
    ]
    active_contrasts = [
        c for c in contrasts if c.contrast_id != holdout_contrast_id and c.trial_id != holdout_contrast_id
    ]

    # Precompute class-balanced sample weights + width-coerced MoA
    cohort_sample_weights: list[np.ndarray] = []
    cohort_moa: list[np.ndarray] = []
    for c in active_cohorts:
        y = c.y
        n_pos = max(1, int(np.sum(y == 1)))
        n_neg = max(1, int(np.sum(y == 0)))
        w = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
        cohort_sample_weights.append(w)
        cohort_moa.append(_coerce_moa_vec(c.moa_vec, moa_width))

    contrast_moa = [_coerce_moa_vec(con.moa_vec, moa_width) for con in active_contrasts]

    def objective(flat_w: np.ndarray) -> float:
        W = flat_w.reshape((8, moa_width))
        loss = 0.0

        # 1. Patient/model-level class-balanced binary cross-entropy loss
        for c, sw, mvec in zip(active_cohorts, cohort_sample_weights, cohort_moa):
            t = W @ mvec
            norm_t = float(np.linalg.norm(t)) + 1e-8
            s = (c.P @ t) / norm_t
            logits = temperature * s
            # Stable log-loss: log(1 + exp(logits)) - y * logits
            pos_part = np.maximum(logits, 0.0)
            bce = pos_part - c.y * logits + np.log1p(np.exp(-np.abs(logits)))
            loss += c.weight * float(np.sum(sw * bce))

        # 2. Published trial subgroup log-HR contrast matching loss
        for con, mvec in zip(active_contrasts, contrast_moa):
            t = W @ mvec
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
    W_learned = res.x.reshape((8, moa_width))
    return W_learned


def build_biological_anchor_prior_w_moa_extension() -> np.ndarray:
    """Receipt-backed 8×9 prior for v2 MoA extension columns (RUO)."""
    W_ext = np.zeros((len(SIGNED_AXES_8D), len(MOA_CLASSES_9_EXT)), dtype=float)
    col = {name: i for i, name in enumerate(MOA_CLASSES_9_EXT)}

    W_ext[:, col["VDR_Paricalcitol"]] = [0.05, 0.35, 0.00, 0.05, 0.10, -0.20, 0.05, 0.75]
    W_ext[:, col["FAK_ROCK_Inhibitor"]] = [-0.10, 0.25, 0.45, 0.05, 0.05, 0.85, 0.10, 0.15]
    W_ext[:, col["mTORC2_Inhibitor"]] = [0.10, 0.55, 0.00, 0.10, 0.05, -0.15, 0.00, 0.45]
    W_ext[:, col["PRMT5_MAT2A_MTAP"]] = [0.55, -0.25, 0.00, 0.65, 0.05, -0.10, 0.05, 0.20]
    W_ext[:, col["EP300_CREBBP"]] = [0.35, -0.15, 0.10, 0.05, 0.75, -0.55, 0.05, 0.25]
    W_ext[:, col["CDK2_PKMYT1"]] = [0.75, 0.40, 0.00, 0.25, 0.05, -0.10, 0.05, 0.15]
    W_ext[:, col["MCL1_BCLxL_TW37"]] = [-0.35, 0.90, 0.00, 0.05, 0.00, -0.10, 0.00, 0.20]
    W_ext[:, col["FOLR1_ADC"]] = [0.15, -0.10, 0.00, 0.05, 0.05, -0.20, 0.98, 0.40]
    W_ext[:, col["CDH6_ADC"]] = [0.15, -0.10, 0.00, 0.05, 0.05, -0.15, 0.92, 0.38]
    return W_ext


def compose_w_moa_v2_from_v1_matrix(W_v1: np.ndarray) -> np.ndarray:
    """Freeze v1 left block (8×15) and append receipt-backed W_ext (8×9)."""
    W_v1 = np.asarray(W_v1, dtype=float)
    if W_v1.shape != (8, 15):
        raise ValueError(f"W_v1 must be (8, 15), got {W_v1.shape}")
    return np.hstack([W_v1, build_biological_anchor_prior_w_moa_extension()])


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

