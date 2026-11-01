"""Generic image I/O helpers shared by the vision clients.

These helpers carry no model identity.  They previously lived in
``models/siglip_baseline.py`` (the ungated general-domain SigLIP proxy).  That
module was deleted when the proxy screening path was retired, so the two
model-agnostic pieces it held moved here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Neutral English prompt pair for a zero-shot breast screening probe. These are
# the prompts used across the SigLIP-family literature (Chen et al., 2023,
# arXiv:2303.15343) and are NOT clinical decision text. Zero-shot agreement
# scores against these prompts are uncalibrated and off-label.
DEFAULT_ZERO_SHOT_LABELS: tuple[str, str] = (
    "a mammogram of a breast with a malignant mass",
    "a mammogram of a breast without a malignant mass",
)


def to_pil_from_float01(arr: np.ndarray) -> Any:
    """Convert a preprocessed mammogram (float32 in [0,1], HxW) to a PIL image.

    * Grayscale [0,1] -> uint8 -> 3-channel RGB (SigLIP-family encoders expect 3ch).
    * No resizing here -- the processor handles resize/center-crop.
    """
    from PIL import Image

    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale mammogram, got shape {arr.shape}")
    a = np.clip(arr, 0.0, 1.0)
    a8 = (a * 255.0 + 0.5).astype(np.uint8)
    rgb = np.stack([a8, a8, a8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


# Back-compat alias: the private name is what medsiglip.py already imported.
_to_pil_from_float01 = to_pil_from_float01

__all__ = ["DEFAULT_ZERO_SHOT_LABELS", "to_pil_from_float01", "_to_pil_from_float01"]
