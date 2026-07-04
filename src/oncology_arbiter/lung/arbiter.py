"""Deterministic Fleischner-inspired risk-bucketing for NSCLC candidates.

**NOT** a calibrated ML model. This is a coarse translation of the 2017
Fleischner Society lung-nodule follow-up guidelines¹ into a
``LOW | MID | HIGH`` recommendation bucket that a downstream rules engine
can key on. Every response using this arbiter self-flags with
``model_state="placeholder"`` because the input candidates themselves come
from a threshold heuristic (see ``lung.pipeline``).

Dominant feature: ``max_diameter_mm`` from the pipeline output.

Fleischner buckets (approximated):
    d <  4 mm    → LOW    (no follow-up for solid nodules in low-risk pts)
    4 ≤ d < 8    → MID    (12-month follow-up)
    8 ≤ d ≤ 30   → HIGH   (3-month follow-up + PET/tissue diagnosis)
     d > 30      → HIGH   ("mass" — biopsy / tissue diagnosis)

We also include a small ``count bonus`` so multiple candidates nudge the
logit up slightly. This is *not* validated — see `docs/model_card_nsclc.md`.

¹ MacMahon H, Naidich DP, Goo JM, et al. Guidelines for Management of
Incidental Pulmonary Nodules Detected on CT Images: From the Fleischner
Society 2017. Radiology 2017;284(1):228-243. DOI: 10.1148/radiol.2017161659
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math


# Diameter → logit anchors. Piecewise linear between anchors.
DIAMETER_LOGIT_ANCHORS = [
    (0.0, -4.0),
    (4.0, -1.5),
    (8.0, 0.0),
    (15.0, 1.5),
    (30.0, 3.0),
    (60.0, 4.5),
]

# Count bonus applied to the logit.
COUNT_BONUS_GT_5 = 0.5
COUNT_BONUS_GT_2 = 0.25


# Diameter bucket boundaries (Fleischner-inspired). NB: these are RISK
# buckets, NOT follow-up interval labels — the follow-up decision lives in
# `models/nccn_nsclc_rules.py`.
BUCKET_LOW_MAX_MM = 4.0
BUCKET_MID_MAX_MM = 8.0


@dataclass
class NsclcArbiterFeatures:
    """Inputs the arbiter consumes."""
    max_diameter_mm: float
    n_candidates: int
    lung_voxel_fraction: float = 0.0

    @classmethod
    def from_lung_output(cls, out) -> "NsclcArbiterFeatures":
        return cls(
            max_diameter_mm=float(out.max_diameter_mm),
            n_candidates=int(len(out.candidates)),
            lung_voxel_fraction=float(out.lung_voxel_fraction),
        )


@dataclass
class ArbiterScore:
    """Result of `score_nsclc`."""
    risk_bucket: str  # NEGATIVE | LOW | MID | HIGH
    logit: float
    prob: float
    driving_feature: str
    max_diameter_mm: float
    n_candidates: int
    lung_voxel_fraction: float

    def as_dict(self) -> dict:
        return {
            "risk_bucket": self.risk_bucket,
            "logit": float(self.logit),
            "prob": float(self.prob),
            "driving_feature": self.driving_feature,
            "max_diameter_mm": float(self.max_diameter_mm),
            "n_candidates": int(self.n_candidates),
            "lung_voxel_fraction": float(self.lung_voxel_fraction),
        }


def _diameter_to_logit(d_mm: float) -> float:
    """Piecewise linear interpolation between anchor points."""
    if d_mm <= DIAMETER_LOGIT_ANCHORS[0][0]:
        return DIAMETER_LOGIT_ANCHORS[0][1]
    for (x0, y0), (x1, y1) in zip(DIAMETER_LOGIT_ANCHORS, DIAMETER_LOGIT_ANCHORS[1:]):
        if x0 <= d_mm <= x1:
            if x1 == x0:
                return y0
            frac = (d_mm - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return DIAMETER_LOGIT_ANCHORS[-1][1]


def _diameter_bucket(d_mm: float) -> str:
    if d_mm <= 0.0:
        return "NEGATIVE"
    if d_mm < BUCKET_LOW_MAX_MM:
        return "LOW"
    if d_mm < BUCKET_MID_MAX_MM:
        return "MID"
    return "HIGH"


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_nsclc(features: NsclcArbiterFeatures) -> ArbiterScore:
    """Score a case using the deterministic Fleischner-inspired mapping.

    Returns
    -------
    ArbiterScore with risk_bucket in {NEGATIVE, LOW, MID, HIGH}.
    """
    d = float(features.max_diameter_mm)
    n = int(features.n_candidates)

    logit = _diameter_to_logit(d)
    if n > 5:
        logit += COUNT_BONUS_GT_5
    elif n > 2:
        logit += COUNT_BONUS_GT_2

    bucket = _diameter_bucket(d)
    driving = "max_diameter_mm"
    if bucket == "HIGH" and d > 30.0:
        driving = "mass_diameter_gt_30mm"
    elif n > 5 and bucket in ("LOW", "MID"):
        driving = "multiple_candidates"

    return ArbiterScore(
        risk_bucket=bucket,
        logit=float(logit),
        prob=float(_sigmoid(logit)),
        driving_feature=driving,
        max_diameter_mm=float(d),
        n_candidates=n,
        lung_voxel_fraction=float(features.lung_voxel_fraction),
    )


__all__ = [
    "DIAMETER_LOGIT_ANCHORS",
    "COUNT_BONUS_GT_5",
    "COUNT_BONUS_GT_2",
    "BUCKET_LOW_MAX_MM",
    "BUCKET_MID_MAX_MM",
    "NsclcArbiterFeatures",
    "ArbiterScore",
    "score_nsclc",
]
