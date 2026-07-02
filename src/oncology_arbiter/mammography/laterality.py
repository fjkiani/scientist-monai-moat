"""Laterality detection and orientation normalization.

Laterality (LEFT / RIGHT breast) can be:
  * Declared in DICOM tags (`ImageLaterality`, `BodyPartExamined`).
  * Encoded in the filename (`..._LEFT_CC.dcm` in CBIS-DDSM).
  * Derived from image content — the breast is always on one side of the
    frame, the "chest wall" is on the other. The breast side is where the
    dense tissue is; the opposite side is background.

We combine metadata and content signals. Content-based detection is used
either standalone (no metadata) or as a sanity check on metadata.

Project orientation convention (DOCUMENT THIS — it differs from various
radiological software conventions):

  * LEFT breast → breast tissue on the LEFT half of the frame,
                  chest wall on the RIGHT half.
  * RIGHT breast → breast tissue on the RIGHT half of the frame,
                   chest wall on the LEFT half.

Rationale: the CBIS-DDSM Calc- subset ships in this orientation (verified
against 4 real fixtures). We adopt it as canonical for the downstream model
inputs so LEFT and RIGHT mammograms are consistently oriented and any
learned filter sees the same physical layout across sides.

NOTE: This is NOT the same as PACS/DICOM display convention, which usually
places the chest wall on the *outside* of a two-image pair (LEFT breast on
the right side of the pair). If you're comparing against a PACS station or
a radiologist's reading protocol, you may need to flip our output. This is
about consistency for model input, not about screen presentation.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np


class Laterality(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    UNKNOWN = "U"


def detect_laterality_from_metadata(
    metadata: dict,
    filename: str | Path | None = None,
) -> Laterality:
    """Return laterality from DICOM tags or filename hints; UNKNOWN otherwise."""
    lat = metadata.get("laterality_hint")
    if lat:
        u = lat.upper()
        if u.startswith("L"):
            return Laterality.LEFT
        if u.startswith("R"):
            return Laterality.RIGHT
    if filename is not None:
        stem = Path(filename).stem.upper()
        # CBIS-DDSM convention: "..._LEFT_CC" / "..._RIGHT_MLO"
        if "_LEFT_" in stem or stem.endswith("_LEFT") or "_LEFT." in stem:
            return Laterality.LEFT
        if "_RIGHT_" in stem or stem.endswith("_RIGHT") or "_RIGHT." in stem:
            return Laterality.RIGHT
    return Laterality.UNKNOWN


def detect_laterality_from_content(
    arr: np.ndarray,
    background_threshold: float | None = None,
) -> Laterality:
    """Return laterality by finding which side of the frame the breast is on.

    Algorithm:
      1. Threshold the image — if `background_threshold` is None, use Otsu
         (adapts to whatever the image dynamic range actually is). If given,
         use the fixed threshold.
      2. Compare foreground mass in the LEFT half vs the RIGHT half.
      3. The side with MORE foreground mass is where the breast sits;
         the opposite side is the chest wall side.

    Design note: CBIS-DDSM DICOMs declare BitsStored=16 but only use the
    bottom ~8 bits, so images have max ≈ 0.003 after 16-bit normalization.
    A hardcoded threshold like 0.02 would classify the whole image as
    background. Otsu adapts to the actual histogram.

    This returns laterality in the CURRENT image orientation. If the image
    is already in radiological display convention, LEFT means "chest wall
    on right, breast on left". If in acquisition orientation, the same
    return value means the same thing physically.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    if background_threshold is None:
        # Local Otsu import to keep this module dependency-free
        from .segmentation import _otsu_threshold
        threshold = _otsu_threshold(arr)
    else:
        threshold = background_threshold
    fg = arr > threshold
    h, w = fg.shape
    left_half = fg[:, : w // 2].sum()
    right_half = fg[:, w - w // 2 :].sum()
    if left_half == 0 and right_half == 0:
        return Laterality.UNKNOWN
    if left_half > right_half:
        return Laterality.LEFT
    return Laterality.RIGHT


def orient_to_radiological_convention(
    arr: np.ndarray,
    laterality: Laterality,
) -> np.ndarray:
    """Flip the image so it matches the project orientation convention.

      LEFT breast  → tissue on the LEFT of frame (chest wall on right)
      RIGHT breast → tissue on the RIGHT of frame (chest wall on left)

    See module docstring for a caveat about this NOT matching every PACS
    display convention.

    Algorithm: locate the breast in the current image (content-based).
    If tissue is on the wrong side for the given laterality label, mirror.
    """
    if laterality == Laterality.UNKNOWN:
        return arr
    # Where is the tissue currently?
    content_side = detect_laterality_from_content(arr)
    if content_side == Laterality.UNKNOWN:
        return arr
    # Project convention: breast tissue on the same-side half as the laterality
    desired_side = Laterality.LEFT if laterality == Laterality.LEFT else Laterality.RIGHT
    if content_side != desired_side:
        return np.fliplr(arr)
    return arr
