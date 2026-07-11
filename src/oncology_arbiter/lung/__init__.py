"""NSCLC-track CT pipeline (placeholder-grade Fleischner-inspired).

Public surface:
    CtSeries, read_ct_series               — from .ct_reader
    LungHeuristicOutput,
      NoduleCandidate,
      run_lung_heuristic,
      lung_mask_from_hu,
      nodule_candidate_blobs,
      summarize_candidates                 — from .pipeline
    NsclcArbiterFeatures, ArbiterScore,
      score_nsclc                          — from .arbiter

Every response using this module MUST self-flag as ``proxy_lung_heuristic``
in the envelope's model_state, per docs/model_card_nsclc.md.
"""
from __future__ import annotations

from .arbiter import (
    ArbiterScore,
    BUCKET_LOW_MAX_MM,
    BUCKET_MID_MAX_MM,
    DIAMETER_LOGIT_ANCHORS,
    NsclcArbiterFeatures,
    score_nsclc,
)
from .ct_reader import CtSeries, read_ct_series
from .resample import (
    LUNA16_TARGET_SPACING_MM,
    ResampledVolume,
    resample_for_luna16,
)
from .pipeline import (
    BODY_HU_MIN,
    LUNG_HU_MAX,
    LungHeuristicOutput,
    NODULE_HU_MAX,
    NODULE_HU_MIN,
    NoduleCandidate,
    lung_mask_from_hu,
    nodule_candidate_blobs,
    run_lung_heuristic,
    summarize_candidates,
)

__all__ = [
    "ArbiterScore",
    "BUCKET_LOW_MAX_MM",
    "BUCKET_MID_MAX_MM",
    "BODY_HU_MIN",
    "CtSeries",
    "DIAMETER_LOGIT_ANCHORS",
    "LUNA16_TARGET_SPACING_MM",
    "LUNG_HU_MAX",
    "LungHeuristicOutput",
    "ResampledVolume",
    "NODULE_HU_MAX",
    "NODULE_HU_MIN",
    "NoduleCandidate",
    "NsclcArbiterFeatures",
    "lung_mask_from_hu",
    "nodule_candidate_blobs",
    "read_ct_series",
    "resample_for_luna16",
    "run_lung_heuristic",
    "score_nsclc",
    "summarize_candidates",
]
