"""Synthetic-pectoral tests for `remove_pectoral_mlo`.

The 5 real CBIS-DDSM MLO fixtures we downloaded are cropped tightly enough
that the pectoral muscle is not visible in the top corners. This makes them
useful for reader/orientation/mask tests but not for validating that the
pectoral-removal algorithm actually removes anything.

Here we build controlled synthetic images with a known pectoral triangle
and confirm the algorithm removes it. This is a unit test — no real DICOMs.
"""
from __future__ import annotations

import numpy as np
import pytest

from oncology_arbiter.mammography.segmentation import remove_pectoral_mlo


def _synthetic_mlo(
    laterality: str,
    shape: tuple[int, int] = (800, 500),
    breast_intensity: float = 0.3,
    pectoral_intensity: float = 0.9,
) -> np.ndarray:
    """Build a synthetic MLO mammogram.

    Layout for laterality='L' (radiological convention: chest wall on right,
    pectoral triangle in top-right):
        - Breast ellipse on the LEFT half
        - Pectoral triangle at TOP-RIGHT

    For laterality='R' this is mirrored.
    """
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)

    # Breast ellipse
    yy, xx = np.mgrid[:h, :w]
    if laterality == "L":
        cy, cx = h // 2, w // 4     # left half
        pect_row_slope = 1.5        # top-right triangle
    else:
        cy, cx = h // 2, 3 * w // 4  # right half
        pect_row_slope = -1.5       # top-left triangle
    breast = ((xx - cx) / (w / 3)) ** 2 + ((yy - cy) / (h / 3)) ** 2 < 1
    img[breast] = breast_intensity

    # Pectoral triangle: bright region in top corner
    # LEFT breast: triangle from (0, w//2) to (0, w-1) to (h//3, w-1)
    # RIGHT breast: triangle from (0, 0) to (0, w//2) to (h//3, 0)
    if laterality == "L":
        for r in range(h // 3):
            col_start = w - int((h // 3 - r) * pect_row_slope)
            img[r, max(col_start, w // 2):] = pectoral_intensity
    else:
        for r in range(h // 3):
            col_end = int((h // 3 - r) * abs(pect_row_slope))
            img[r, : min(col_end, w // 2)] = pectoral_intensity
    return img


@pytest.mark.parametrize("laterality", ["L", "R"])
def test_synthetic_pectoral_is_actually_removed(laterality: str) -> None:
    """On a synthetic MLO with a known pectoral triangle, removal should
    zero out most of the pectoral pixels and leave the breast alone."""
    img = _synthetic_mlo(laterality)
    h, w = img.shape

    # Where is the pectoral?
    if laterality == "L":
        corner = (slice(None, h // 3), slice(w // 2, None))
    else:
        corner = (slice(None, h // 3), slice(None, w // 2))
    pect_mask_before = img[corner] > 0.5
    pect_pixels_before = int(pect_mask_before.sum())
    assert pect_pixels_before > 0, "test setup: synthetic pectoral should exist"

    removed = remove_pectoral_mlo(img, laterality=laterality)

    # Corner should be much darker after
    corner_before = img[corner].mean()
    corner_after = removed[corner].mean()
    assert corner_after < corner_before * 0.5, (
        f"pectoral corner should darken by >50%. "
        f"before={corner_before:.3f}, after={corner_after:.3f}"
    )

    # Most of the pectoral should be zeroed
    pect_pixels_still_bright = int((removed[corner] > 0.5).sum())
    fraction_removed = 1 - pect_pixels_still_bright / pect_pixels_before
    assert fraction_removed > 0.5, (
        f"expected >50% of pectoral pixels removed, got {fraction_removed:.1%}"
    )

    # Breast should NOT have been touched (or touched only trivially)
    if laterality == "L":
        breast_region = (slice(None), slice(None, w // 2))
    else:
        breast_region = (slice(None), slice(w // 2, None))
    breast_before = img[breast_region]
    breast_after = removed[breast_region]
    breast_diff = np.abs(breast_after - breast_before).mean()
    assert breast_diff < 0.01, (
        f"breast tissue should not be substantially modified, mean_diff={breast_diff}"
    )


def test_pectoral_removal_on_non_mlo_is_safe() -> None:
    """Calling remove_pectoral_mlo on a CC-view image (no pectoral) shouldn't
    crash. It may or may not modify pixels depending on the image, but it
    must return a same-shape float32 array."""
    img = _synthetic_mlo("L")
    # Zero the top corner so there's nothing to grow into
    h, w = img.shape
    img[: h // 3, w // 2 :] = 0.0
    out = remove_pectoral_mlo(img, laterality="L")
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # Should have removed nothing (nothing bright in corner)
    assert np.array_equal(out, img)


def test_pectoral_removal_unknown_laterality_returns_copy() -> None:
    """Unknown laterality → return a copy of the input, no modifications."""
    img = _synthetic_mlo("L")
    out = remove_pectoral_mlo(img, laterality="?")
    assert np.array_equal(out, img)
    assert out is not img  # must be a copy
