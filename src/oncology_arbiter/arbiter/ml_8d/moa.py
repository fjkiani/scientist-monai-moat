"""MoA weight vectors for K-dimensional mechanism columns."""
from __future__ import annotations

import numpy as np

from .constants import MOA_CLASSES_15, MOA_CLASSES_24

def moa_vec(moa_classes: tuple[str, ...] | None = None, **weights: float) -> np.ndarray:
    """Build MoA vector for arbitrary contract column set (default v1 K=15)."""
    classes = moa_classes or MOA_CLASSES_15
    vec = np.zeros(len(classes), dtype=float)
    idx_map = {name: i for i, name in enumerate(classes)}
    for k, v in weights.items():
        if k not in idx_map:
            raise KeyError(f"Unknown MoA class: {k}")
        vec[idx_map[k]] = float(v)
    return vec


def _moa_vec(**weights: float) -> np.ndarray:
    """v1 training default (K=15) — kept for fit/LOTO/cohort loaders."""
    return moa_vec(MOA_CLASSES_15, **weights)


def pad_moa_vec_15_to_24(moa_15: np.ndarray) -> np.ndarray:
    """Zero-pad 15D trial/cohort MoA for v2 scorer (extension dims inactive until tagged)."""
    m = np.asarray(moa_15, dtype=float).reshape(-1)
    if m.shape[0] != len(MOA_CLASSES_15):
        raise ValueError(f"moa_15 must be length {len(MOA_CLASSES_15)}, got {m.shape[0]}")
    ext = np.zeros(len(MOA_CLASSES_24) - len(MOA_CLASSES_15), dtype=float)
    return np.concatenate([m, ext])


def _moa_vec_v2(**weights: float) -> np.ndarray:
    return moa_vec(MOA_CLASSES_24, **weights)
