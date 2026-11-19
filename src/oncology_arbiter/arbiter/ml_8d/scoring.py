"""Signed bilinear projection scoring helpers."""
from __future__ import annotations

import numpy as np

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
