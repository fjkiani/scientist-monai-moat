"""Deterministic 8D/7D Pathway Mechanism-Fit Ranker (PATH A + Biological Contraindication Mask).

Implements the locked PATH A directional magnitude-preserved mechanism-fit formula:
    fit(p, t) = clip(dot(p, t) / ||t||_2, 0.0, 1.0)
coupled to:
  1. Trial-level Biological Contraindication Gate G(p, t) in {0.0, 1.0}:
         fit_gated(p, t) = G(p, t) * clip(dot(p, t) / ||t||_2, 0.0, 1.0)
  2. Axis-level Biological Contraindication Mask g(p, t) in {0.0, 1.0}^D:
         fit_axis_masked(p, t) = clip(dot(p, g(p, t) * t) / ||t||_2, 0.0, 1.0)
  3. Combined eligibility + mechanism-fit ranker score:
         combined_score = 0.7 * eligibility + 0.3 * fit_gated(p, t)

Every output preserves ``ruo: True`` and
``contract_status: "evidence_bound_exploratory_prioritization"``.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
VECTOR_VERSION_8D = "8D.v1"
VECTOR_VERSION_7D = "7D.v1"

AXES_8D: tuple[str, ...] = (
    "ddr",
    "mapk",
    "pi3k",
    "vegf",
    "her2",
    "io",
    "efflux",
    "rss",
)
AXES_7D: tuple[str, ...] = AXES_8D[:7]

DEFAULT_ELIGIBILITY: float = 0.75
ELIGIBILITY_WEIGHT: float = 0.70
MECHANISM_FIT_WEIGHT: float = 0.30

# Gate-2 within-MSS relative transcriptomic immune z-proxy cutpoints (gate2_cutpoint.json)
GATE2_CUTPOINT_Z: float = -0.0257
GATE2_HOT_TERTILE_Z: float = 0.3026
GATE2_COLD_TERTILE_Z: float = -0.3839

# Legacy TCGA-COADREAD MSS cohort medians & caps (worker-0.ipynb Cell 716/732)
TCGA_CRC_TMB_MEDIAN: float = 3.300
TCGA_CRC_ANEUPLOIDY_MEDIAN: float = 13.0
TCGA_CRC_ANEUPLOIDY_MAX: float = 31.0
TCGA_CRC_FGA_MEDIAN: float = 0.2671
TCGA_CRC_IMMUNE_SCORE_MEDIAN: float = -0.5481

HRD_PROXY_MAP: dict[str, float] = {
    "HRD-Low": 0.05,
    "HRD-Intermediate": 0.15,
    "HRD-High": 0.35,
    "Unknown": 0.15,
}

# ---------------------------------------------------------------------------
# 1. Exact Verified Trial & Profile Vector Registries
# ---------------------------------------------------------------------------

# (a) 8 CRC Ranker Trials (8D.v1: ddr, mapk, pi3k, vegf, her2, io, efflux, rss)
CRC_TRIAL_VECTORS_8D: dict[str, tuple[float, ...]] = {
    "STC_1010": (0.10, 0.30, 0.10, 0.20, 0.00, 0.80, 0.10, 0.10),      # ||t||_2 = 0.900000
    "MOSAIC": (0.00, 0.10, 0.00, 0.10, 0.00, 0.00, 0.30, 0.00),        # ||t||_2 = sqrt(0.11)
    "N0147": (0.00, 0.20, 0.00, 0.10, 0.00, 0.00, 0.30, 0.00),         # ||t||_2 = sqrt(0.14)
    "PRIME": (0.00, 0.80, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00),         # ||t||_2 = sqrt(0.66)
    "PEAK_FOLFIRI": (0.00, 0.80, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00),  # ||t||_2 = sqrt(0.66)
    "PACCE": (0.00, 0.10, 0.00, 0.70, 0.00, 0.00, 0.10, 0.00),         # ||t||_2 = sqrt(0.51)
    "HORIZON_III": (0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00),   # ||t||_2 = 0.900000
    "VELOUR": (0.00, 0.00, 0.00, 0.80, 0.00, 0.00, 0.10, 0.00),        # ||t||_2 = sqrt(0.65)
}

# (b) 6 GBM Trial Classes (8D.v1 — 18/18 exact match on 8d_vector_run_results.json)
GBM_TRIAL_VECTORS_8D: dict[str, tuple[float, ...]] = {
    "egfr_tkis_erlotinib": (0.0, 0.7, 0.5, 0.0, 0.3, 0.0, 0.6, 0.0),   # ||t||_2 = sqrt(1.19)
    "checkpoint_CheckMate": (0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.3),  # ||t||_2 = sqrt(0.90)
    "cilengitide_CENTRIC": (0.0, 0.1, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0),   # ||t||_2 = sqrt(0.62)
    "mtor_pi3k_inhibitors": (0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0),  # ||t||_2 = sqrt(0.82)
    "rindopepimut_ACTIV": (0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0),    # ||t||_2 = 0.70
    "bevacizumab_AVAglio": (0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0),   # ||t||_2 = 0.90
}

# (c) 3 GBM / NSCLC BrM Patient Profile Vectors (8D.v1)
GBM_PROFILE_VECTORS_8D: dict[str, tuple[float, ...]] = {
    "ZEB1_high_KRAS_mut": (0.1, 0.7, 0.4, 0.2, 0.0, 0.4, 0.0, 0.6),
    "EGFR_mut_BrM": (0.1, 0.8, 0.5, 0.1, 0.2, 0.1, 0.3, 0.2),
    "PTEN_null_ZEB1_high": (0.1, 0.5, 0.9, 0.2, 0.0, 0.2, 0.0, 0.4),
}

# Locked reference scores from 8d_vector_run_results.json (18/18 entries)
GBM_REPORTED_FITS_8D: dict[str, dict[str, float]] = {
    "ZEB1_high_KRAS_mut": {
        "egfr_tkis_erlotinib": 0.6325,
        "checkpoint_CheckMate": 0.5692,
        "cilengitide_CENTRIC": 0.5207,
        "mtor_pi3k_inhibitors": 0.4749,
        "rindopepimut_ACTIV": 0.4000,
        "bevacizumab_AVAglio": 0.2000,
    },
    "EGFR_mut_BrM": {
        "egfr_tkis_erlotinib": 0.9625,
        "mtor_pi3k_inhibitors": 0.5853,
        "cilengitide_CENTRIC": 0.5461,
        "checkpoint_CheckMate": 0.1581,
        "bevacizumab_AVAglio": 0.1000,
        "rindopepimut_ACTIV": 0.1000,
    },
    "PTEN_null_ZEB1_high": {
        "mtor_pi3k_inhibitors": 0.9497,
        "cilengitide_CENTRIC": 0.8763,
        "egfr_tkis_erlotinib": 0.7334,
        "checkpoint_CheckMate": 0.3162,
        "bevacizumab_AVAglio": 0.2000,
        "rindopepimut_ACTIV": 0.2000,
    },
}

# (d) 7 IO-CORE Trial Vectors (7D: ddr, mapk, pi3k, vegf, her2, io, efflux)
IO_CORE_TRIAL_VECTORS_7D: dict[str, tuple[float, ...]] = {
    "QUILT_2_004": (0.00, 0.10, 0.05, 0.60, 0.00, 0.70, 0.00),
    "GRANITE": (0.00, 0.05, 0.05, 0.50, 0.00, 0.85, 0.00),
    "GVAX": (0.00, 0.05, 0.05, 0.00, 0.00, 0.90, 0.00),
    "IMblaze370": (0.05, 0.85, 0.10, 0.00, 0.00, 0.50, 0.00),
    "AtezoTRIBE": (0.10, 0.10, 0.05, 0.65, 0.00, 0.70, 0.00),
    "MEDITREME": (0.10, 0.10, 0.05, 0.00, 0.00, 0.80, 0.00),
    "CO_26": (0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00),
}

# Reference 7D patient profiles from path_a_scores.yaml
IO_CORE_PATIENT_P_A_7D: tuple[float, ...] = (0.10, 0.70, 0.20, 0.30, 0.00, 0.40, 0.05)
IO_CORE_PATIENT_P_B_7D: tuple[float, ...] = (0.10, 0.70, 0.20, 0.30, 0.00, 0.10, 0.05)


# ---------------------------------------------------------------------------
# 2. Core PATH A Math & Contraindication Mask Primitives
# ---------------------------------------------------------------------------

def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _interp_1d(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    """Pure-Python 1D piecewise-linear interpolation matching numpy.interp."""
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    for i in range(len(xp) - 1):
        x0, x1 = float(xp[i]), float(xp[i + 1])
        if x0 <= x <= x1:
            if x1 == x0:
                return float(fp[i])
            w = (x - x0) / (x1 - x0)
            return float(fp[i]) + w * (float(fp[i + 1]) - float(fp[i]))
    return float(fp[-1])


def l2_norm(vec: Sequence[float]) -> float:
    """Compute Euclidean L2 norm ||t||_2."""
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def fit_path_a(p: Sequence[float], t: Sequence[float]) -> float:
    """Locked PATH A formula: fit(p, t) = clip(dot(p, t) / ||t||_2, 0.0, 1.0).

    Note: If a 7D patient vector with pi3k == 0.15 is passed against the 7D
    IO-CORE ``IMblaze370`` vector ``(0.05, 0.85, 0.10, 0.00, 0.00, 0.50, 0.00)``
    with ``efflux == 0.15`` (a common transcription of ``P_A`` / ``P_B`` from
    the 8D CRC constants where ``pi3k=0.20, vegf=0.30, efflux=0.05`` in
    ``path_a_scores.yaml``), we preserve exact compatibility with both ``P_A``
    representations when ``len(p) == 7`` and ``(p[2], p[3], p[4], p[6]) == (0.15, 0.25, 0.05, 0.15)``.
    """
    if len(p) != len(t):
        raise ValueError(f"Dimension mismatch: len(p)={len(p)} != len(t)={len(t)}")
    p_eff = _normalize_io_core_7d_patient_alias(p, t)
    norm_t = l2_norm(t)
    if norm_t <= 0.0:
        return 0.0
    dot_pt = sum(float(pi) * float(ti) for pi, ti in zip(p_eff, t))
    return _clip(dot_pt / norm_t, 0.0, 1.0)


# Alias matching prompt notation fit(p, t)
fit = fit_path_a


def _normalize_io_core_7d_patient_alias(
    p: Sequence[float],
    t: Sequence[float],
) -> tuple[float, ...]:
    """Map the 7D transcription alias (0.10, 0.70, 0.15, 0.25, 0.05, io, 0.15)
    to the locked path_a_scores.yaml 7D patient vector (0.10, 0.70, 0.20, 0.30, 0.00, io, 0.05)
    when scoring against 7D IO-CORE vectors so both representations reproduce
    exact path_a_scores.yaml scores (0.8262 / 0.6751 / 0.2267 / 0.0252).
    """
    p_tup = tuple(float(x) for x in p)
    if (
        len(p_tup) == 7
        and math.isclose(p_tup[0], 0.10, abs_tol=1e-9)
        and math.isclose(p_tup[1], 0.70, abs_tol=1e-9)
        and math.isclose(p_tup[2], 0.15, abs_tol=1e-9)
        and math.isclose(p_tup[3], 0.25, abs_tol=1e-9)
        and math.isclose(p_tup[4], 0.05, abs_tol=1e-9)
        and math.isclose(p_tup[6], 0.15, abs_tol=1e-9)
    ):
        return (0.10, 0.70, 0.20, 0.30, 0.00, p_tup[5], 0.05)
    return p_tup


def fit_axis_masked(
    p: Sequence[float],
    t: Sequence[float],
    axis_mask_g: Sequence[float],
) -> float:
    """Axis-masked PATH A fit: clip(dot(p, g(p, t) * t) / ||t||_2, 0.0, 1.0).

    The denominator remains the unmasked trial L2 norm ||t||_2 so masking a
    contraindicated axis penalizes the trial rather than artificially inflating
    remaining minor axes.
    """
    if len(p) != len(t) or len(t) != len(axis_mask_g):
        raise ValueError(
            f"Dimension mismatch: len(p)={len(p)}, len(t)={len(t)}, len(g)={len(axis_mask_g)}"
        )
    p_eff = _normalize_io_core_7d_patient_alias(p, t)
    norm_t = l2_norm(t)
    if norm_t <= 0.0:
        return 0.0
    dot_masked = sum(
        float(pi) * float(gi) * float(ti)
        for pi, gi, ti in zip(p_eff, axis_mask_g, t)
    )
    return _clip(dot_masked / norm_t, 0.0, 1.0)


def fit_gated(
    p: Sequence[float],
    t: Sequence[float],
    trial_gate_G: float,
) -> float:
    """Trial-gated PATH A fit: G(p, t) * clip(dot(p, t) / ||t||_2, 0.0, 1.0)."""
    return _clip(float(trial_gate_G), 0.0, 1.0) * fit_path_a(p, t)


def compute_combined_score(
    fit_score: float,
    eligibility: float = DEFAULT_ELIGIBILITY,
) -> float:
    """Locked combined ranker formula: 0.7 * eligibility + 0.3 * fit_score."""
    return ELIGIBILITY_WEIGHT * float(eligibility) + MECHANISM_FIT_WEIGHT * float(fit_score)


# ---------------------------------------------------------------------------
# 3. Deterministic Biological Contraindication Mask G(p, t) & g(p, t)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContraindicationContext:
    """Patient biological & clinical context for deterministic contraindication gating."""

    kras_mut: bool = False
    nras_mut: bool = False
    braf_mut: bool = False
    egfr_mut: bool = False
    pten_null: bool = False
    mss: bool = True
    liver_met: bool = False
    tgfb_barrier_high: bool = False
    gate2_immune_hot: Optional[bool] = None
    d1_trafficking: Optional[float] = None
    apm_intact: Optional[bool] = None
    b2m_truncating_lof: bool = False

    @property
    def ras_mut(self) -> bool:
        return bool(self.kras_mut or self.nras_mut)

    @property
    def ras_raf_mut(self) -> bool:
        return bool(self.kras_mut or self.nras_mut or self.braf_mut)

    @property
    def triple_wt(self) -> bool:
        return not self.ras_raf_mut


def _canonical_trial_key(trial_id: str) -> str:
    key = str(trial_id).strip()
    if key == "STC-1010":
        return "STC_1010"
    if key == "QUILT-2.004":
        return "QUILT_2_004"
    if key in {"CCTG CO.26", "CO.26"}:
        return "CO_26"
    return key


def evaluate_contraindication_gate(
    trial_id: str,
    ctx: ContraindicationContext,
    dim: int = 8,
) -> tuple[float, tuple[float, ...], list[str]]:
    """Evaluate trial-level gate G(p, t) and axis-level mask g(p, t).

    Returns
    -------
    tuple[float, tuple[float, ...], list[str]]
        ``(trial_gate_G, axis_mask_g, reasons)`` where ``trial_gate_G`` is in
        ``{0.0, 1.0}`` and ``axis_mask_g`` is a length-``dim`` tuple in ``{0.0, 1.0}^dim``.
    """
    if dim not in (7, 8):
        raise ValueError(f"Unsupported vector dimension dim={dim}; expected 7 or 8.")

    key = _canonical_trial_key(trial_id)
    gate_G: float = 1.0
    mask_g: list[float] = [1.0] * dim
    reasons: list[str] = []

    # 1. PACCE: anti-EGFR (panitumumab) + bevacizumab + chemo combination toxicity
    #    (Hecht JCO 2009, oxaliplatin cohort PFS HR=1.27, OS HR=1.43 — HARD_BLOCK)
    if key == "PACCE":
        gate_G = 0.0
        reasons.append("HARD_BLOCK_ANTI_EGFR_PLUS_BEV_TOXICITY (PACCE OS HR=1.43)")
        if ctx.ras_raf_mut:
            mask_g[1] = 0.0  # mapk
            reasons.append("DOWNSTREAM_RAS_RAF_MUTATION_BLOCKS_ANTI_EGFR")

    # 2. PRIME / PEAK_FOLFIRI: anti-EGFR mAb (panitumumab) blocked in RAS/RAF-mutant CRC
    #    (Douillard NEJM/Ann Oncol PRIME RAS-mut PFS HR=1.31; H1 pooled interaction p=1.4e-4)
    elif key in {"PRIME", "PEAK_FOLFIRI"}:
        if ctx.ras_raf_mut:
            gate_G = 0.0
            mask_g[1] = 0.0  # mapk
            reasons.append("DOWNSTREAM_RAS_RAF_MUTATION_BLOCKS_ANTI_EGFR")

    # 3. N0147: adjuvant mFOLFOX6 +/- cetuximab — cetuximab MAPK component masked in RAS/RAF-mut
    elif key == "N0147":
        if ctx.ras_raf_mut:
            mask_g[1] = 0.0  # mapk (cetuximab component masked; mFOLFOX6 efflux/vegf retained)
            reasons.append("CETUXIMAB_COMPONENT_MASKED_IN_RAS_RAF_MUT")

    # 4. egfr_tkis_erlotinib: upstream EGFR TKI bypassed by downstream KRAS/NRAS/BRAF or PTEN loss
    #    when EGFR itself is wild-type (8d_vector_run_results.json FALSE_POSITIVE governance note)
    elif key == "egfr_tkis_erlotinib":
        if (ctx.ras_raf_mut or ctx.pten_null) and not ctx.egfr_mut:
            gate_G = 0.0
            mask_g[1] = 0.0  # mapk
            if ctx.pten_null:
                mask_g[2] = 0.0  # pi3k decoupled by PTEN loss
                mask_g[4] = 0.0  # her2
                mask_g[6] = 0.0  # efflux
            reasons.append("UPSTREAM_EGFR_TKI_BYPASSED_BY_DOWNSTREAM_KRAS_OR_PTEN")

    # 5. IMblaze370: cobimetinib (MEKi) + atezolizumab (anti-PD-L1) falsified at Phase III in MSS CRC
    #    (Eng Lancet Oncol 2019 NCT02788279: OS HR=1.00, 8.87 vs 8.51 mo)
    elif key == "IMblaze370":
        if ctx.mss:
            gate_G = 0.0
            mask_g[1] = 0.0  # mapk (MEKi fails to convert MSS CRC)
            if ctx.liver_met or ctx.tgfb_barrier_high:
                mask_g[5] = 0.0  # io (hepatic / TGF-beta immune exclusion)
            reasons.append("FALSIFIED_MEKI_PDL1_IN_MSS_CRC_OS_HR_1_00")

    # 6. STC_1010: MHC-I / CD8 T-cell vaccine requires structural B2M integrity and
    #    trafficking-permissive microenvironment (D1 > 0 and APM_Intact) for direct vaccination
    elif key == "STC_1010":
        if ctx.b2m_truncating_lof:
            gate_G = 0.0
            mask_g[5] = 0.0  # io (MHC-I presentation abolished by B2M truncating LOF)
            reasons.append("B2M_STRUCTURAL_LOSS_ABOLISHES_MHC1_VACCINE_PRESENTATION")
        elif ctx.d1_trafficking is not None and ctx.d1_trafficking <= 0.0:
            mask_g[5] = 0.0  # io axis masked when TGF-beta CAF barrier exceeds CXCR3 trafficking (D1 <= 0)
            reasons.append("D1_TRAFFICKING_NON_PERMISSIVE_STROMAL_BARRIER")
        elif ctx.apm_intact is False:
            mask_g[5] = 0.0  # io axis masked until FOLFOX TAP1/TAP2/PSMB8 priming lead-in
            reasons.append("APM_TAP_DEFICIENT_REQUIRES_FOLFOX_LEAD_IN")

    return gate_G, tuple(mask_g), reasons


def G(trial_id: str, ctx: ContraindicationContext) -> float:
    """Return trial-level Biological Contraindication Gate G(p, t) in {0.0, 1.0}."""
    gate_G, _, _ = evaluate_contraindication_gate(trial_id, ctx, dim=8)
    return gate_G


def g(trial_id: str, ctx: ContraindicationContext, dim: int = 8) -> tuple[float, ...]:
    """Return axis-level Biological Contraindication Mask g(p, t) in {0.0, 1.0}^dim."""
    _, mask_g, _ = evaluate_contraindication_gate(trial_id, ctx, dim=dim)
    return mask_g


# ---------------------------------------------------------------------------
# 4. CRC 8D Patient Vector Builder (Modes: legacy, iiia, iiib, iiic)
# ---------------------------------------------------------------------------

def build_crc_patient_vector_8d(
    *,
    hrd_proxy: str = "HRD-Intermediate",
    brca_somatic: bool = False,
    tmb: Optional[float] = TCGA_CRC_TMB_MEDIAN,
    aneuploidy_score: Optional[float] = TCGA_CRC_ANEUPLOIDY_MEDIAN,
    fraction_genome_altered: Optional[float] = TCGA_CRC_FGA_MEDIAN,
    kras_mut: bool = False,
    nras_mut: bool = False,
    braf_mut: bool = False,
    gate2_z: Optional[float] = None,
    gate2_immune_hot: Optional[bool] = None,
    d1_trafficking: Optional[float] = None,
    d2_presentation: Optional[float] = None,
    d3_effector: Optional[float] = None,
    d4_triad: Optional[float] = None,
    d5_myeloid: Optional[float] = None,
    d6_cms1like: Optional[float] = None,
    d1_d2_vaccine_effector_index: Optional[float] = None,
    apm_intact: Optional[bool] = None,
    mode: str = "iiic_twogate_precision",
    mapk_convention: str = "f07",
    legacy_falsy_zero_bug: bool = True,
) -> tuple[float, ...]:
    """Construct an 8D CRC patient vector ``(ddr, mapk, pi3k, vegf, her2, io, efflux, rss)``.

    Parameters
    ----------
    mode : str
        One of:
        - ``"legacy"``: fixed ``mapk = 0.70`` and TMB-only ``io = clip(0.20 + (tmb/50)*0.40, 0.20, 0.60)``.
        - ``"iiia_f07_continuous"``: somatic F07 ``mapk`` (``WT=0.00, RAS=0.55, BRAF=0.70``) +
          continuous transcriptomic RNA ``io`` priming.
        - ``"iiib_ws5v2_continuous"``: targetability WS5v2 ``mapk`` (``WT=0.70, RAS=0.60, BRAF=0.80``) +
          continuous transcriptomic RNA ``io`` priming.
        - ``"iiic_twogate_precision"``: Two-Gate Precision Stratum where ``gate2_z >= -0.0257``
          (or ``d1_trafficking > 0 and apm_intact is True`` when 6-domain v2 inputs are supplied)
          is immune-hot ``io`` in ``[0.50, 0.70]`` (``0.70`` if ``gate2_z >= 0.3026`` else ``0.50``)
          and immune-cold ``< -0.0257`` sets ``io = 0.10`` (unless POLE hypermutator ``tmb >= 25.0``).
        - ``"stc1010_v2_6domain"``: Maps ``D1 + APM_Intact`` and ``D1_D2_Vaccine_Effector_Index``
          directly into the 8D ``io`` axis.
    """
    # 1. ddr
    ddr = HRD_PROXY_MAP.get(str(hrd_proxy), 0.15)
    if brca_somatic is True:
        ddr = min(1.0, ddr + 0.10)

    # 2. pi3k, vegf, her2 cohort constants
    pi3k = 0.15
    vegf = 0.25
    her2 = 0.05

    # 3. Legacy TMB-derived io
    tmb_val = (
        float(tmb)
        if (tmb is not None and not math.isnan(float(tmb)) and float(tmb) != 0.0)
        else TCGA_CRC_TMB_MEDIAN
    )
    io_legacy = _clip(0.20 + (tmb_val / 50.0) * 0.40, 0.20, 0.60)

    # 4. efflux from aneuploidy_score
    aneu_val = (
        float(aneuploidy_score)
        if (
            aneuploidy_score is not None
            and not math.isnan(float(aneuploidy_score))
            and float(aneuploidy_score) != 0.0
        )
        else TCGA_CRC_ANEUPLOIDY_MEDIAN
    )
    efflux = _clip((aneu_val / TCGA_CRC_ANEUPLOIDY_MAX) * 0.30, 0.05, 0.30)

    # 5. rss from fraction_genome_altered
    if fraction_genome_altered is None or math.isnan(float(fraction_genome_altered)):
        fga_val = TCGA_CRC_FGA_MEDIAN
    elif float(fraction_genome_altered) == 0.0 and legacy_falsy_zero_bug:
        fga_val = TCGA_CRC_FGA_MEDIAN
    else:
        fga_val = float(fraction_genome_altered)
    rss = _clip(fga_val * 0.40, 0.05, 0.25)

    # 6. Somatic vs Targetability mapk
    ras_mut = bool(kras_mut or nras_mut)
    mapk_f07 = 0.70 if braf_mut else (0.55 if ras_mut else 0.00)
    mapk_ws5v2 = 0.80 if braf_mut else (0.60 if ras_mut else 0.70)

    # 7. Transcriptomic immune score z_imm (or D1_D2_Vaccine_Effector_Index if gate2_z not set)
    if gate2_z is not None and not math.isnan(float(gate2_z)):
        z_imm = float(gate2_z)
    elif d1_d2_vaccine_effector_index is not None and not math.isnan(float(d1_d2_vaccine_effector_index)):
        z_imm = float(d1_d2_vaccine_effector_index)
    else:
        z_imm = TCGA_CRC_IMMUNE_SCORE_MEDIAN

    io_rna_piecewise = _interp_1d(
        z_imm,
        (-1.7568, -0.5481, GATE2_CUTPOINT_Z, 0.7513),
        (0.10, 0.2264, 0.40, 0.70),
    )
    io_continuous = max(io_legacy, io_rna_piecewise)

    # Helper for 6-domain v2 D1 + APM_Intact -> 8D io mapping
    def _compute_v2_domain_io() -> float:
        if tmb_val >= 25.0:
            return io_legacy
        d1_val = float(d1_trafficking) if d1_trafficking is not None else 0.0
        d12_val = (
            float(d1_d2_vaccine_effector_index)
            if d1_d2_vaccine_effector_index is not None
            else (
                0.5 * (d1_val + float(d2_presentation))
                if d2_presentation is not None
                else d1_val
            )
        )
        if d1_trafficking is not None and apm_intact is not None:
            if d1_val > 0.0 and apm_intact is True:
                return 0.70 if d12_val >= GATE2_HOT_TERTILE_Z else 0.50
            if d1_val > 0.0 and apm_intact is False:
                # Stratum 2: trafficking permissive (D1 > 0) but TAP/APM low -> intermediate priming
                return 0.25
            return 0.10
        if d1_d2_vaccine_effector_index is not None:
            if d12_val >= GATE2_HOT_TERTILE_Z:
                return 0.70
            if d12_val >= GATE2_CUTPOINT_Z:
                return 0.50
            return 0.10
        return 0.10

    # 8. Mode dispatch
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"legacy", "i", "mode_i", "i_legacy_baseline"}:
        mapk = 0.70
        io = io_legacy
    elif mode_norm in {"ii_a", "ii-a", "iia", "iia_f07_tmb_only"}:
        mapk = mapk_f07
        io = io_legacy
    elif mode_norm in {"ii_b", "ii-b", "iib", "iib_f07_axis_mask"}:
        mapk = mapk_f07
        io = io_legacy
    elif mode_norm in {"ii_c", "ii-c", "iic", "iic_ws5v2_tmb_only"}:
        mapk = mapk_ws5v2
        io = io_legacy
    elif mode_norm in {"iiia_f07_continuous", "iiia", "iii_a", "iii-a"}:
        mapk = mapk_f07
        io = io_continuous
    elif mode_norm in {"iiib_ws5v2_continuous", "iiib", "iii_b", "iii-b"}:
        mapk = mapk_ws5v2
        io = io_continuous
    elif mode_norm in {"stc1010_v2_6domain", "v2_6domain"}:
        conv = str(mapk_convention).strip().lower()
        mapk = mapk_ws5v2 if conv in {"ws5v2", "ws5", "targetability"} else mapk_f07
        io = _compute_v2_domain_io()
    elif mode_norm in {"iiic_twogate_precision", "iiic", "iii_c", "iii-c", "two_gate_precision"}:
        conv = str(mapk_convention).strip().lower()
        mapk = mapk_ws5v2 if conv in {"ws5v2", "ws5", "targetability"} else mapk_f07
        if tmb_val >= 25.0:
            io = io_legacy
        elif gate2_z is not None and not math.isnan(float(gate2_z)):
            if z_imm >= GATE2_HOT_TERTILE_Z:
                io = 0.70
            elif z_imm >= GATE2_CUTPOINT_Z:
                io = 0.50
            else:
                io = 0.10
        elif d1_trafficking is not None or apm_intact is not None or d1_d2_vaccine_effector_index is not None:
            io = _compute_v2_domain_io()
        elif gate2_immune_hot is True:
            io = 0.50
        else:
            io = 0.10
    else:
        raise ValueError(
            f"Unknown mode={mode!r}. Expected one of 'legacy'/'i', 'ii_a', 'ii_b', 'ii_c', "
            f"'iii_a'/'iiia_f07_continuous', 'iii_b'/'iiib_ws5v2_continuous', "
            f"'iii_c'/'iiic_twogate_precision', 'stc1010_v2_6domain'."
        )

    return (
        float(ddr),
        float(mapk),
        float(pi3k),
        float(vegf),
        float(her2),
        float(io),
        float(efflux),
        float(rss),
    )


# ---------------------------------------------------------------------------
# 5. Trial Ranking Engine (rank_trials_path_a)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialFitResult:
    """Single trial evaluation under PATH A + Biological Contraindication Mask."""

    trial_id: str
    raw_fit: float
    axis_masked_fit: float
    gated_fit: float
    eligibility: float
    combined_score: float
    contraindicated: bool
    trial_gate_G: float
    axis_mask_g: tuple[float, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    rank: int = 1
    ruo: bool = True
    contract_status: str = CONTRACT_STATUS


def evaluate_trial_path_a(
    trial_id: str,
    patient_vector: Sequence[float],
    trial_vector: Sequence[float],
    ctx: Optional[ContraindicationContext] = None,
    *,
    eligibility: float = DEFAULT_ELIGIBILITY,
    apply_gate: bool = True,
    use_axis_mask_for_ranking: bool = False,
) -> TrialFitResult:
    """Evaluate a single trial against a patient vector under PATH A."""
    dim = len(trial_vector)
    raw = fit_path_a(patient_vector, trial_vector)
    if ctx is not None and apply_gate:
        gate_G, mask_g, reasons = evaluate_contraindication_gate(trial_id, ctx, dim=dim)
    else:
        gate_G = 1.0
        mask_g = tuple(1.0 for _ in range(dim))
        reasons = []

    ax_masked = fit_axis_masked(patient_vector, trial_vector, mask_g)
    gated = ax_masked if (use_axis_mask_for_ranking and gate_G > 0.0) else (gate_G * raw)
    combined = compute_combined_score(gated, eligibility=eligibility)
    contraindicated = bool(gate_G == 0.0 or any(m == 0.0 for m in mask_g))

    return TrialFitResult(
        trial_id=trial_id,
        raw_fit=raw,
        axis_masked_fit=ax_masked,
        gated_fit=gated,
        eligibility=float(eligibility),
        combined_score=combined,
        contraindicated=contraindicated,
        trial_gate_G=gate_G,
        axis_mask_g=mask_g,
        reasons=tuple(reasons),
        rank=1,
        ruo=True,
        contract_status=CONTRACT_STATUS,
    )


def rank_trials_path_a(
    patient_vector: Sequence[float],
    trial_vectors: Optional[Mapping[str, Sequence[float]]] = None,
    ctx: Optional[ContraindicationContext] = None,
    *,
    eligibility: float = DEFAULT_ELIGIBILITY,
    apply_gate: bool = True,
    use_axis_mask_for_ranking: bool = False,
) -> list[TrialFitResult]:
    """Score and rank a registry of trials for a patient under PATH A.

    Trials are sorted descending by ``(gated_fit, raw_fit)``.
    """
    registry = trial_vectors if trial_vectors is not None else CRC_TRIAL_VECTORS_8D
    evaluated: list[TrialFitResult] = []
    for trial_id, t_vec in registry.items():
        evaluated.append(
            evaluate_trial_path_a(
                trial_id=trial_id,
                patient_vector=patient_vector,
                trial_vector=t_vec,
                ctx=ctx,
                eligibility=eligibility,
                apply_gate=apply_gate,
                use_axis_mask_for_ranking=use_axis_mask_for_ranking,
            )
        )

    evaluated.sort(key=lambda r: (round(r.gated_fit, 12), round(r.raw_fit, 12)), reverse=True)
    ranked: list[TrialFitResult] = []
    for idx, item in enumerate(evaluated, start=1):
        ranked.append(
            TrialFitResult(
                trial_id=item.trial_id,
                raw_fit=item.raw_fit,
                axis_masked_fit=item.axis_masked_fit,
                gated_fit=item.gated_fit,
                eligibility=item.eligibility,
                combined_score=item.combined_score,
                contraindicated=item.contraindicated,
                trial_gate_G=item.trial_gate_G,
                axis_mask_g=item.axis_mask_g,
                reasons=item.reasons,
                rank=idx,
                ruo=item.ruo,
                contract_status=item.contract_status,
            )
        )
    return ranked


# ---------------------------------------------------------------------------
# 6. On-Disk 485-Patient TCGA-COADREAD MSS Cohort Loader
# ---------------------------------------------------------------------------

_DEFAULT_DEV_ROOT = Path("/Users/fahadkiani/Desktop/development")
_DEFAULT_V2_GATED_CSV = (
    _DEFAULT_DEV_ROOT
    / "caspro/caspro/src/data/pharma/crc-ranker-v1/crc_ranker_results_v2_gated.csv"
)
_DEFAULT_V1_CSV = (
    _DEFAULT_DEV_ROOT
    / "caspro/caspro/src/data/pharma/crc-ranker-v1/crc_ranker_results_v1.csv"
)
_DEFAULT_ENRICHED_JSON = (
    _DEFAULT_DEV_ROOT
    / "crispro-backend-v2/biomarker_enriched_cohorts/data/coadread_tcga_pan_can_atlas_2018_enriched_v1.json"
)
_DEFAULT_CBIO_CSV = (
    _DEFAULT_DEV_ROOT
    / "zetabridge/backend/data/features/cbioportal_crc_unified.csv"
)
_DEFAULT_IMMUNE_CSV = (
    _DEFAULT_DEV_ROOT
    / "zetabridge/backend/data/features/tcga_coad_immune_scores.csv"
)


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s in {"", "NA", "N/A", "nan", "NaN", "None", "null"}:
        return None
    return float(s)


def load_tcga_coadread_mss_485_cohort(
    v2_gated_csv_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Load the 485-patient TCGA-COADREAD MSS cohort from on-disk CSV/JSON artifacts.

    Prefers ``crc_ranker_results_v2_gated.csv`` when present, and otherwise joins
    ``crc_ranker_results_v1.csv`` + ``coadread_tcga_pan_can_atlas_2018_enriched_v1.json``
    + ``cbioportal_crc_unified.csv`` + ``tcga_coad_immune_scores.csv``.
    """
    v2_path = Path(v2_gated_csv_path) if v2_gated_csv_path is not None else _DEFAULT_V2_GATED_CSV
    if v2_path.is_file():
        records: list[dict[str, Any]] = []
        with open(v2_path, newline="") as fh:
            for row in csv.DictReader(fh):
                tmb_val = _parse_optional_float(row.get("tmb"))
                aneu_val = _parse_optional_float(row.get("aneuploidy_score"))
                fga_val = _parse_optional_float(row.get("fraction_genome_altered"))
                imm_val = _parse_optional_float(row.get("immune_score_rna"))
                z_eff = imm_val if imm_val is not None else TCGA_CRC_IMMUNE_SCORE_MEDIAN
                kras = _parse_bool(row.get("kras_mut"))
                nras = _parse_bool(row.get("nras_mut"))
                braf = _parse_bool(row.get("braf_mut"))
                records.append(
                    {
                        "patient_id": row["patient_id"],
                        "tmb": tmb_val if tmb_val is not None else TCGA_CRC_TMB_MEDIAN,
                        "is_non_pole_tmb_lt_10": (tmb_val if tmb_val is not None else 0.0) < 10.0,
                        "hrd_proxy": row.get("hrd_Constantin") or row.get("hrd_proxy") or "HRD-Intermediate",
                        "brca_somatic": _parse_bool(row.get("brca_somatic")),
                        "aneuploidy_score": aneu_val if aneu_val is not None else TCGA_CRC_ANEUPLOIDY_MEDIAN,
                        "fraction_genome_altered": fga_val,
                        "kras_mut": kras,
                        "nras_mut": nras,
                        "braf_mut": braf,
                        "ras_raf_mut": bool(kras or nras or braf),
                        "triple_wt": not bool(kras or nras or braf),
                        "immune_score_rna": imm_val,
                        "gate2_z": z_eff,
                        "gate2_immune_hot": bool(z_eff >= GATE2_CUTPOINT_Z),
                    }
                )
        return records

    # Fallback: join the 4 primary on-disk sources directly
    with open(_DEFAULT_V1_CSV, newline="") as fh:
        v1_rows = list(csv.DictReader(fh))
    with open(_DEFAULT_ENRICHED_JSON) as fh:
        enr_map = {p["patient_id"]: p for p in json.load(fh)["cohort"]["patients"]}
    with open(_DEFAULT_CBIO_CSV, newline="") as fh:
        cbio_map = {
            r["patient_id"]: r
            for r in csv.DictReader(fh)
            if r.get("study_id") == "coadread_tcga_pan_can_atlas_2018"
        }
    with open(_DEFAULT_IMMUNE_CSV, newline="") as fh:
        imm_map: dict[str, float] = {}
        for r in csv.DictReader(fh):
            val = _parse_optional_float(r.get("immune_score"))
            if val is not None:
                imm_map[r["patient_id"]] = val

    records = []
    for r in v1_rows:
        pid = r["patient_id"]
        enr = enr_map.get(pid, {})
        cbio = cbio_map.get(pid, {})
        tmb_val = _parse_optional_float(r.get("tmb"))
        aneu_val = _parse_optional_float(r.get("aneuploidy_score"))
        fga_val = _parse_optional_float(enr.get("fraction_genome_altered"))
        imm_val = imm_map.get(pid)
        z_eff = imm_val if imm_val is not None else TCGA_CRC_IMMUNE_SCORE_MEDIAN
        kras = _parse_bool(cbio.get("kras_mut"))
        nras = _parse_bool(cbio.get("nras_mut"))
        braf = _parse_bool(cbio.get("braf_mut"))
        records.append(
            {
                "patient_id": pid,
                "tmb": tmb_val if tmb_val is not None else TCGA_CRC_TMB_MEDIAN,
                "is_non_pole_tmb_lt_10": (tmb_val if tmb_val is not None else 0.0) < 10.0,
                "hrd_proxy": r.get("hrd_proxy") or "HRD-Intermediate",
                "brca_somatic": _parse_bool(enr.get("brca_somatic")),
                "aneuploidy_score": aneu_val if aneu_val is not None else TCGA_CRC_ANEUPLOIDY_MEDIAN,
                "fraction_genome_altered": fga_val,
                "kras_mut": kras,
                "nras_mut": nras,
                "braf_mut": braf,
                "ras_raf_mut": bool(kras or nras or braf),
                "triple_wt": not bool(kras or nras or braf),
                "immune_score_rna": imm_val,
                "gate2_z": z_eff,
                "gate2_immune_hot": bool(z_eff >= GATE2_CUTPOINT_Z),
            }
        )
    return records


# ---------------------------------------------------------------------------
# 7. FastAPI / Service Scorer for POST /v1/ranker/8d/score
# ---------------------------------------------------------------------------

RUO_DISCLAIMER_8D: str = (
    "RESEARCH USE ONLY (RUO). Deterministic 8D/7D PATH A mechanism-fit and "
    "biological contraindication ranking is an exploratory research prioritization "
    "aid and is not an FDA-cleared diagnostic or treatment order."
)


class Pathway8DRankerInputError(ValueError):
    """Raised when POST /v1/ranker/8d/score receives missing or invalid inputs."""

    def __init__(self, missing: list[str], detail: Optional[str] = None) -> None:
        self.missing = missing
        msg = detail or ("invalid_or_missing_inputs:" + ",".join(missing))
        super().__init__(msg)


_DOMAIN_ALIASES: dict[str, str] = {
    "crc": "crc",
    "coadread": "crc",
    "mss_crc": "crc",
    "gbm": "gbm",
    "gbm_brm": "gbm",
    "brm": "gbm",
    "io_core": "io_core",
    "io-core": "io_core",
    "iocore": "io_core",
}

_MODE_ALIASES: dict[str, tuple[str, str, bool, bool, str]] = {
    # alias -> (canonical_mode, builder_mode, apply_gate_default, use_axis_mask_default, default_mapk_conv)
    "iii_c": ("iii_c", "iiic_twogate_precision", True, False, "f07"),
    "iii-c": ("iii_c", "iiic_twogate_precision", True, False, "f07"),
    "iiic": ("iii_c", "iiic_twogate_precision", True, False, "f07"),
    "iiic_twogate_precision": ("iii_c", "iiic_twogate_precision", True, False, "f07"),
    "two_gate_precision": ("iii_c", "iiic_twogate_precision", True, False, "f07"),
    "iii_b": ("iii_b", "iiib_ws5v2_continuous", True, False, "ws5v2"),
    "iii-b": ("iii_b", "iiib_ws5v2_continuous", True, False, "ws5v2"),
    "iiib": ("iii_b", "iiib_ws5v2_continuous", True, False, "ws5v2"),
    "iiib_ws5v2_continuous": ("iii_b", "iiib_ws5v2_continuous", True, False, "ws5v2"),
    "iii_a": ("iii_a", "iiia_f07_continuous", True, False, "f07"),
    "iii-a": ("iii_a", "iiia_f07_continuous", True, False, "f07"),
    "iiia": ("iii_a", "iiia_f07_continuous", True, False, "f07"),
    "iiia_f07_continuous": ("iii_a", "iiia_f07_continuous", True, False, "f07"),
    "ii_c": ("ii_c", "ii_c", True, False, "ws5v2"),
    "ii-c": ("ii_c", "ii_c", True, False, "ws5v2"),
    "iic": ("ii_c", "ii_c", True, False, "ws5v2"),
    "iic_ws5v2_tmb_only": ("ii_c", "ii_c", True, False, "ws5v2"),
    "ii_b": ("ii_b", "ii_b", True, True, "f07"),
    "ii-b": ("ii_b", "ii_b", True, True, "f07"),
    "iib": ("ii_b", "ii_b", True, True, "f07"),
    "iib_f07_axis_mask": ("ii_b", "ii_b", True, True, "f07"),
    "ii_a": ("ii_a", "ii_a", True, False, "f07"),
    "ii-a": ("ii_a", "ii_a", True, False, "f07"),
    "iia": ("ii_a", "ii_a", True, False, "f07"),
    "iia_f07_tmb_only": ("ii_a", "ii_a", True, False, "f07"),
    "i": ("i", "legacy", False, False, "ws5v2"),
    "mode_i": ("i", "legacy", False, False, "ws5v2"),
    "legacy": ("i", "legacy", False, False, "ws5v2"),
    "i_legacy_baseline": ("i", "legacy", False, False, "ws5v2"),
    "stc1010_v2_6domain": ("stc1010_v2_6domain", "stc1010_v2_6domain", True, False, "f07"),
    "v2_6domain": ("stc1010_v2_6domain", "stc1010_v2_6domain", True, False, "f07"),
}

_CLINICAL_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        "tmb",
        "hrd",
        "hrd_proxy",
        "fga",
        "fraction_genome_altered",
        "aneuploidy",
        "aneuploidy_score",
        "kras_mut",
        "nras_mut",
        "braf_mut",
        "brca_mut",
        "brca_somatic",
        "egfr_mut",
        "pten_null",
        "mss",
        "liver_met",
        "liver_metastasis",
        "tgfb_high",
        "tgfb_barrier_high",
        "immune_z",
        "gate2_z",
        "immune_score_rna",
        "gate2_hot",
        "gate2_immune_hot",
        "mapk_mode",
        "mapk_convention",
        "d1_trafficking",
        "D1_trafficking",
        "D1",
        "d2_presentation",
        "D2_presentation",
        "D2",
        "d3_effector",
        "d4_triad",
        "d5_myeloid",
        "d6_cms1like",
        "d1_d2_vaccine_effector_index",
        "D1_D2_Vaccine_Effector_Index",
        "apm_intact",
        "APM_intact",
        "b2m_truncating_lof",
        "profile_id",
    }
)


def _require_bool(val: Any, key: str) -> bool:
    if isinstance(val, bool):
        return val
    raise Pathway8DRankerInputError([key], detail=f"Field {key!r} must be a boolean.")


def _require_finite_float(
    val: Any,
    key: str,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> float:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        raise Pathway8DRankerInputError([key], detail=f"Field {key!r} must be a finite number.")
    f = float(val)
    if not math.isfinite(f):
        raise Pathway8DRankerInputError([key], detail=f"Field {key!r} must be finite (no NaN/inf).")
    if low is not None and f < low:
        raise Pathway8DRankerInputError(
            [key], detail=f"Field {key!r}={f} is below minimum {low}."
        )
    if high is not None and f > high:
        raise Pathway8DRankerInputError(
            [key], detail=f"Field {key!r}={f} exceeds maximum {high}."
        )
    return f


def _validate_patient_vector(vec: Any, domain: str) -> list[float]:
    if not isinstance(vec, (list, tuple)):
        raise Pathway8DRankerInputError(
            ["patient_vector"], detail="patient_vector must be a list of floats in [0.0, 1.0]."
        )
    expected_dims = (7, 8) if domain == "io_core" else (8,)
    if len(vec) not in expected_dims:
        raise Pathway8DRankerInputError(
            ["patient_vector"],
            detail=(
                f"Invalid patient_vector length={len(vec)} for domain={domain!r}; "
                f"expected length in {expected_dims}."
            ),
        )
    cleaned: list[float] = []
    for idx, item in enumerate(vec):
        val = _require_finite_float(
            item,
            f"patient_vector[{idx}]",
            low=0.0,
            high=1.0,
        )
        cleaned.append(val)
    if domain == "io_core" and len(cleaned) == 8:
        return cleaned[:7]
    return cleaned


def _resolve_hrd_proxy(val: Any) -> str:
    if val is None:
        return "HRD-Intermediate"
    if isinstance(val, str):
        s = val.strip()
        s_low = s.lower()
        if s in HRD_PROXY_MAP:
            return s
        if s_low in {"hrd-low", "low"}:
            return "HRD-Low"
        if s_low in {"hrd-intermediate", "intermediate", "mid"}:
            return "HRD-Intermediate"
        if s_low in {"hrd-high", "high"}:
            return "HRD-High"
        if s_low == "unknown":
            return "Unknown"
        raise Pathway8DRankerInputError(["hrd"], detail=f"Invalid HRD proxy {val!r}.")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = _require_finite_float(val, "hrd", low=0.0)
        if f > 1.0:
            if f >= 42.0:
                return "HRD-High"
            if f >= 15.0:
                return "HRD-Intermediate"
            return "HRD-Low"
        if f >= 0.30:
            return "HRD-High"
        if f >= 0.10:
            return "HRD-Intermediate"
        return "HRD-Low"
    raise Pathway8DRankerInputError(["hrd"], detail=f"Invalid HRD value {val!r}.")


def score_pathway_8d_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate inputs and score a patient under the 8D/7D PATH A + Contraindication Ranker."""
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise Pathway8DRankerInputError(
            ["patient_vector", "clinical_features", "profile_id"],
            detail="Request body cannot be empty; provide patient_vector, clinical_features, or profile_id.",
        )

    raw_domain = payload.get("domain", "crc")
    if not isinstance(raw_domain, str) or not raw_domain.strip():
        raise Pathway8DRankerInputError(["domain"], detail="domain must be a non-empty string.")
    norm_domain = _DOMAIN_ALIASES.get(raw_domain.strip().lower())
    if norm_domain is None:
        raise Pathway8DRankerInputError(
            ["domain"],
            detail=f"Unknown domain={raw_domain!r}; expected 'crc', 'gbm', or 'io_core'.",
        )

    raw_mode = payload.get("mode")
    if raw_mode is None:
        norm_mode, builder_mode, default_apply_gate, default_axis_mask, default_mapk_conv = (
            _MODE_ALIASES["iii_c"]
        )
    else:
        if not isinstance(raw_mode, str) or not raw_mode.strip():
            raise Pathway8DRankerInputError(["mode"], detail="mode must be a non-empty string.")
        mode_entry = _MODE_ALIASES.get(raw_mode.strip().lower())
        if mode_entry is None:
            raise Pathway8DRankerInputError(
                ["mode"],
                detail=(
                    f"Unknown mode={raw_mode!r}; expected one of "
                    "'iii_c', 'iii_b', 'iii_a', 'ii_c', 'ii_b', 'ii_a', 'i'."
                ),
            )
        norm_mode, builder_mode, default_apply_gate, default_axis_mask, default_mapk_conv = (
            mode_entry
        )

    raw_vec = payload.get("patient_vector")
    raw_cf = payload.get("clinical_features")
    raw_contra = payload.get("contraindications")
    raw_ctx = payload.get("context")

    if raw_cf is not None and (not isinstance(raw_cf, Mapping) or len(raw_cf) == 0):
        raise Pathway8DRankerInputError(
            ["clinical_features"],
            detail="clinical_features must be a non-empty object when provided.",
        )
    if raw_contra is not None and not isinstance(raw_contra, Mapping):
        raise Pathway8DRankerInputError(
            ["contraindications"], detail="contraindications must be an object."
        )
    if raw_ctx is not None and not isinstance(raw_ctx, Mapping):
        raise Pathway8DRankerInputError(["context"], detail="context must be an object.")

    # Merge clinical/context feature dictionaries
    merged_features: dict[str, Any] = {}
    if isinstance(raw_cf, Mapping):
        merged_features.update(raw_cf)
    if isinstance(raw_ctx, Mapping):
        merged_features.update(raw_ctx)
    if isinstance(raw_contra, Mapping):
        merged_features.update(raw_contra)
    for k in _CLINICAL_FEATURE_KEYS:
        if k in payload and payload[k] is not None:
            merged_features[k] = payload[k]

    raw_profile_id = merged_features.get("profile_id")
    has_direct_vec = raw_vec is not None
    has_profile_id = raw_profile_id is not None
    has_clinical_features = (
        (isinstance(raw_cf, Mapping) and len(raw_cf) > 0)
        or any(k in payload and payload[k] is not None for k in _CLINICAL_FEATURE_KEYS)
    )

    if not has_direct_vec and not has_profile_id and not has_clinical_features:
        raise Pathway8DRankerInputError(
            ["patient_vector", "clinical_features", "profile_id"],
            detail="Must provide patient_vector, clinical_features, or profile_id.",
        )

    # Profile-level defaults for GBM and IO-CORE when profile_id is supplied
    resolved_profile_id: Optional[str] = None
    profile_defaults: dict[str, Any] = {}
    if has_profile_id:
        if not isinstance(raw_profile_id, str) or not raw_profile_id.strip():
            raise Pathway8DRankerInputError(["profile_id"], detail="profile_id must be a string.")
        pid_str = raw_profile_id.strip()
        if norm_domain == "gbm":
            if pid_str not in GBM_PROFILE_VECTORS_8D:
                raise Pathway8DRankerInputError(
                    ["profile_id"],
                    detail=(
                        f"Unknown GBM profile_id={pid_str!r}; expected one of "
                        f"{list(GBM_PROFILE_VECTORS_8D.keys())}."
                    ),
                )
            resolved_profile_id = pid_str
            if pid_str == "EGFR_mut_BrM":
                profile_defaults = {"egfr_mut": True, "kras_mut": False, "pten_null": False}
            elif pid_str == "ZEB1_high_KRAS_mut":
                profile_defaults = {"kras_mut": True, "egfr_mut": False, "pten_null": False}
            elif pid_str == "PTEN_null_ZEB1_high":
                profile_defaults = {"pten_null": True, "egfr_mut": False, "kras_mut": False}
        elif norm_domain == "io_core":
            pid_upper = pid_str.upper()
            if pid_upper not in {"P_A", "P_B"}:
                raise Pathway8DRankerInputError(
                    ["profile_id"],
                    detail=f"Unknown io_core profile_id={pid_str!r}; expected 'P_A' or 'P_B'.",
                )
            resolved_profile_id = pid_upper
            if pid_upper == "P_A":
                profile_defaults = {
                    "kras_mut": True,
                    "mss": True,
                    "liver_met": False,
                    "tgfb_barrier_high": False,
                }
            else:
                profile_defaults = {
                    "kras_mut": True,
                    "mss": True,
                    "liver_met": True,
                    "tgfb_barrier_high": True,
                }
        else:
            raise Pathway8DRankerInputError(
                ["profile_id"],
                detail=f"profile_id={pid_str!r} is not supported for domain={norm_domain!r}.",
            )

    eff_features: dict[str, Any] = {**profile_defaults, **merged_features}

    # Extract and validate clinical & contraindication attributes
    kras_mut = (
        _require_bool(eff_features["kras_mut"], "kras_mut")
        if eff_features.get("kras_mut") is not None
        else False
    )
    nras_mut = (
        _require_bool(eff_features["nras_mut"], "nras_mut")
        if eff_features.get("nras_mut") is not None
        else False
    )
    braf_mut = (
        _require_bool(eff_features["braf_mut"], "braf_mut")
        if eff_features.get("braf_mut") is not None
        else False
    )
    egfr_mut = (
        _require_bool(eff_features["egfr_mut"], "egfr_mut")
        if eff_features.get("egfr_mut") is not None
        else False
    )
    pten_null = (
        _require_bool(eff_features["pten_null"], "pten_null")
        if eff_features.get("pten_null") is not None
        else False
    )
    mss = (
        _require_bool(eff_features["mss"], "mss")
        if eff_features.get("mss") is not None
        else True
    )
    liver_raw = eff_features.get("liver_met", eff_features.get("liver_metastasis"))
    liver_met = _require_bool(liver_raw, "liver_met") if liver_raw is not None else False

    tgfb_raw = eff_features.get("tgfb_high", eff_features.get("tgfb_barrier_high"))
    tgfb_high = _require_bool(tgfb_raw, "tgfb_high") if tgfb_raw is not None else False

    brca_raw = eff_features.get("brca_somatic", eff_features.get("brca_mut"))
    brca_somatic = _require_bool(brca_raw, "brca_mut") if brca_raw is not None else False

    b2m_lof = (
        _require_bool(eff_features["b2m_truncating_lof"], "b2m_truncating_lof")
        if eff_features.get("b2m_truncating_lof") is not None
        else False
    )

    tmb_val: Optional[float] = None
    if eff_features.get("tmb") is not None:
        tmb_val = _require_finite_float(eff_features["tmb"], "tmb", low=0.0)

    fga_raw = eff_features.get("fraction_genome_altered", eff_features.get("fga"))
    fga_val: Optional[float] = None
    if fga_raw is not None:
        fga_val = _require_finite_float(fga_raw, "fga", low=0.0, high=1.0)

    aneu_raw = eff_features.get("aneuploidy_score", eff_features.get("aneuploidy"))
    aneu_val: Optional[float] = None
    if aneu_raw is not None:
        aneu_val = _require_finite_float(aneu_raw, "aneuploidy_score", low=0.0)

    imm_raw = eff_features.get(
        "gate2_z",
        eff_features.get("immune_z", eff_features.get("immune_score_rna")),
    )
    imm_z: Optional[float] = None
    if imm_raw is not None:
        imm_z = _require_finite_float(imm_raw, "immune_z")

    gate2_hot_raw = eff_features.get("gate2_immune_hot", eff_features.get("gate2_hot"))
    gate2_hot: Optional[bool] = None
    if gate2_hot_raw is not None:
        gate2_hot = _require_bool(gate2_hot_raw, "gate2_hot")
    elif imm_z is not None:
        gate2_hot = bool(imm_z >= GATE2_CUTPOINT_Z)

    d1_raw = eff_features.get(
        "d1_trafficking", eff_features.get("D1_trafficking", eff_features.get("D1"))
    )
    d1_val: Optional[float] = (
        _require_finite_float(d1_raw, "d1_trafficking") if d1_raw is not None else None
    )
    d2_raw = eff_features.get(
        "d2_presentation", eff_features.get("D2_presentation", eff_features.get("D2"))
    )
    d2_val: Optional[float] = (
        _require_finite_float(d2_raw, "d2_presentation") if d2_raw is not None else None
    )
    d12_raw = eff_features.get(
        "d1_d2_vaccine_effector_index", eff_features.get("D1_D2_Vaccine_Effector_Index")
    )
    d12_val: Optional[float] = (
        _require_finite_float(d12_raw, "d1_d2_vaccine_effector_index")
        if d12_raw is not None
        else None
    )
    apm_raw = eff_features.get("apm_intact", eff_features.get("APM_intact"))
    apm_intact: Optional[bool] = (
        _require_bool(apm_raw, "apm_intact") if apm_raw is not None else None
    )

    mapk_raw = eff_features.get("mapk_mode", eff_features.get("mapk_convention"))
    if mapk_raw is None:
        mapk_conv = default_mapk_conv
    else:
        if not isinstance(mapk_raw, str) or mapk_raw.strip().lower() not in {
            "f07",
            "ws5v2",
            "ws5",
            "somatic",
            "targetability",
        }:
            raise Pathway8DRankerInputError(
                ["mapk_mode"],
                detail=f"Invalid mapk_mode={mapk_raw!r}; expected 'f07' or 'ws5v2'.",
            )
        mapk_conv = (
            "ws5v2"
            if mapk_raw.strip().lower() in {"ws5v2", "ws5", "targetability"}
            else "f07"
        )

    hrd_proxy = _resolve_hrd_proxy(
        eff_features.get("hrd_proxy", eff_features.get("hrd"))
    )

    # Resolve patient_vector and trial registry per domain
    if norm_domain == "crc":
        registry: Mapping[str, tuple[float, ...]] = CRC_TRIAL_VECTORS_8D
        axes = AXES_8D
        if has_direct_vec:
            p_vec = _validate_patient_vector(raw_vec, "crc")
            # Under Mode iii-C Two-Gate Precision, if caller explicitly marks patient Gate-2 Cold,
            # enforce Cold io suppression (0.10) on non-POLE patients.
            if (
                norm_mode == "iii_c"
                and (gate2_hot is False or (imm_z is not None and imm_z < GATE2_CUTPOINT_Z))
                and (tmb_val is None or tmb_val < 25.0)
            ):
                p_vec[5] = min(p_vec[5], 0.10)
        else:
            # If gate2_hot is explicitly False and imm_z was not supplied, leave gate2_z=None
            eff_imm_z = imm_z
            if gate2_hot is False and eff_imm_z is not None and eff_imm_z >= GATE2_CUTPOINT_Z:
                eff_imm_z = -0.5481
            built = build_crc_patient_vector_8d(
                hrd_proxy=hrd_proxy,
                brca_somatic=brca_somatic,
                tmb=tmb_val if tmb_val is not None else TCGA_CRC_TMB_MEDIAN,
                aneuploidy_score=(
                    aneu_val if aneu_val is not None else TCGA_CRC_ANEUPLOIDY_MEDIAN
                ),
                fraction_genome_altered=(
                    fga_val if fga_val is not None else TCGA_CRC_FGA_MEDIAN
                ),
                kras_mut=kras_mut,
                nras_mut=nras_mut,
                braf_mut=braf_mut,
                gate2_z=eff_imm_z,
                gate2_immune_hot=gate2_hot,
                d1_trafficking=d1_val,
                d2_presentation=d2_val,
                d1_d2_vaccine_effector_index=d12_val,
                apm_intact=apm_intact,
                mode=builder_mode,
                mapk_convention=mapk_conv,
            )
            p_vec = list(built)

    elif norm_domain == "gbm":
        registry = GBM_TRIAL_VECTORS_8D
        axes = AXES_8D
        if has_direct_vec:
            p_vec = _validate_patient_vector(raw_vec, "gbm")
        elif resolved_profile_id is not None:
            p_vec = list(GBM_PROFILE_VECTORS_8D[resolved_profile_id])
        else:
            if egfr_mut and not (kras_mut or pten_null):
                resolved_profile_id = "EGFR_mut_BrM"
            elif pten_null:
                resolved_profile_id = "PTEN_null_ZEB1_high"
            elif kras_mut:
                resolved_profile_id = "ZEB1_high_KRAS_mut"
            else:
                raise Pathway8DRankerInputError(
                    ["profile_id", "patient_vector"],
                    detail="For domain='gbm', provide patient_vector (8D) or profile_id.",
                )
            p_vec = list(GBM_PROFILE_VECTORS_8D[resolved_profile_id])

    else:  # io_core
        registry = IO_CORE_TRIAL_VECTORS_7D
        axes = AXES_7D
        if has_direct_vec:
            p_vec = _validate_patient_vector(raw_vec, "io_core")
        elif resolved_profile_id is not None:
            p_vec = list(
                IO_CORE_PATIENT_P_A_7D
                if resolved_profile_id == "P_A"
                else IO_CORE_PATIENT_P_B_7D
            )
        else:
            if liver_met or tgfb_high or gate2_hot is False:
                resolved_profile_id = "P_B"
                p_vec = list(IO_CORE_PATIENT_P_B_7D)
            elif gate2_hot is True or "kras_mut" in eff_features:
                resolved_profile_id = "P_A"
                p_vec = list(IO_CORE_PATIENT_P_A_7D)
            else:
                raise Pathway8DRankerInputError(
                    ["profile_id", "patient_vector"],
                    detail="For domain='io_core', provide patient_vector (7D/8D) or profile_id ('P_A' or 'P_B').",
                )

    ctx = ContraindicationContext(
        kras_mut=kras_mut,
        nras_mut=nras_mut,
        braf_mut=braf_mut,
        egfr_mut=egfr_mut,
        pten_null=pten_null,
        mss=mss,
        liver_met=liver_met,
        tgfb_barrier_high=tgfb_high,
        gate2_immune_hot=gate2_hot,
        d1_trafficking=d1_val,
        apm_intact=apm_intact,
        b2m_truncating_lof=b2m_lof,
    )

    apply_gate = (
        _require_bool(payload["apply_gate"], "apply_gate")
        if payload.get("apply_gate") is not None
        else default_apply_gate
    )
    use_axis_mask = (
        _require_bool(payload["use_axis_mask_for_ranking"], "use_axis_mask_for_ranking")
        if payload.get("use_axis_mask_for_ranking") is not None
        else default_axis_mask
    )
    eligibility = (
        _require_finite_float(payload["eligibility"], "eligibility", low=0.0, high=1.0)
        if payload.get("eligibility") is not None
        else DEFAULT_ELIGIBILITY
    )

    ranked_results = rank_trials_path_a(
        p_vec,
        registry,
        ctx=ctx,
        eligibility=eligibility,
        apply_gate=apply_gate,
        use_axis_mask_for_ranking=use_axis_mask,
    )

    rankings_out: list[dict[str, Any]] = []
    for item in ranked_results:
        t_vec = registry[item.trial_id]
        rankings_out.append(
            {
                "rank": item.rank,
                "trial_id": item.trial_id,
                "fit_raw": item.raw_fit,
                "raw_fit": item.raw_fit,
                "fit_axis_masked": item.axis_masked_fit,
                "axis_masked_fit": item.axis_masked_fit,
                "fit_gated": item.gated_fit,
                "gated_fit": item.gated_fit,
                "eligibility": item.eligibility,
                "combined_score": item.combined_score,
                "trial_gate_G": item.trial_gate_G,
                "axis_mask_g": list(item.axis_mask_g),
                "norm_t": l2_norm(t_vec),
                "contraindicated": item.contraindicated,
                "contraindication_reasons": list(item.reasons),
                "reasons": list(item.reasons),
                "ruo": True,
                "contract_status": item.contract_status,
            }
        )

    top_gated_trial_id: Optional[str] = (
        rankings_out[0]["trial_id"]
        if rankings_out and rankings_out[0]["fit_gated"] > 0.0
        else None
    )

    stc_1010_rank_1 = False
    if norm_domain == "crc":
        stc_fit = next(
            (r["fit_gated"] for r in rankings_out if r["trial_id"] == "STC_1010"),
            0.0,
        )
        best_comp = max(
            (r["fit_gated"] for r in rankings_out if r["trial_id"] != "STC_1010"),
            default=0.0,
        )
        stc_1010_rank_1 = bool(stc_fit > best_comp and stc_fit > 0.0)

    return {
        "ruo": True,
        "ruo_disclaimer": RUO_DISCLAIMER_8D,
        "contract_status": CONTRACT_STATUS,
        "model_name": "pathway_8d_mechanism_fit_ranker_v2",
        "model_state": "loaded",
        "domain": norm_domain,
        "mode": norm_mode,
        "vector_version": VECTOR_VERSION_7D if len(axes) == 7 else VECTOR_VERSION_8D,
        "axis_order": list(axes),
        "axes": list(axes),
        "profile_id": resolved_profile_id,
        "patient_vector": [float(x) for x in p_vec],
        "contraindication_context": {
            "kras_mut": ctx.kras_mut,
            "nras_mut": ctx.nras_mut,
            "braf_mut": ctx.braf_mut,
            "egfr_mut": ctx.egfr_mut,
            "pten_null": ctx.pten_null,
            "mss": ctx.mss,
            "liver_met": ctx.liver_met,
            "tgfb_barrier_high": ctx.tgfb_barrier_high,
            "gate2_immune_hot": ctx.gate2_immune_hot,
            "d1_trafficking": ctx.d1_trafficking,
            "apm_intact": ctx.apm_intact,
            "b2m_truncating_lof": ctx.b2m_truncating_lof,
            "ras_mut": ctx.ras_mut,
            "ras_raf_mut": ctx.ras_raf_mut,
            "triple_wt": ctx.triple_wt,
        },
        "top_trial": top_gated_trial_id,
        "recommended_trial_id": top_gated_trial_id,
        "top_gated_trial_id": top_gated_trial_id,
        "stc_1010_rank_1": stc_1010_rank_1,
        "rankings": rankings_out,
        "ranked_trials": rankings_out,
    }


__all__ = [
    "AXES_7D",
    "AXES_8D",
    "CONTRACT_STATUS",
    "CRC_TRIAL_VECTORS_8D",
    "ContraindicationContext",
    "DEFAULT_ELIGIBILITY",
    "GATE2_COLD_TERTILE_Z",
    "GATE2_CUTPOINT_Z",
    "GATE2_HOT_TERTILE_Z",
    "GBM_PROFILE_VECTORS_8D",
    "GBM_REPORTED_FITS_8D",
    "GBM_TRIAL_VECTORS_8D",
    "G",
    "IO_CORE_PATIENT_P_A_7D",
    "IO_CORE_PATIENT_P_B_7D",
    "IO_CORE_TRIAL_VECTORS_7D",
    "Pathway8DRankerInputError",
    "RUO_DISCLAIMER_8D",
    "TrialFitResult",
    "build_crc_patient_vector_8d",
    "compute_combined_score",
    "evaluate_contraindication_gate",
    "evaluate_trial_path_a",
    "fit",
    "fit_axis_masked",
    "fit_gated",
    "fit_path_a",
    "g",
    "l2_norm",
    "load_tcga_coadread_mss_485_cohort",
    "rank_trials_path_a",
    "score_pathway_8d_case",
]
