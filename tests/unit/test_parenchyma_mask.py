"""Unit tests for `oncology_arbiter.nsclc.parenchyma_mask`.

Coverage:
  - `_otsu` bimodal split (synthetic).
  - `build_parenchyma_mask` on a synthetic lung phantom (deterministic).
  - `is_detection_in_parenchyma` for in-mask / out-of-mask / OOB.
  - `apply_parenchyma_filter` mutates in-place and sets attr correctly.

Fixture-gated regression:
  - TCGA-24-1423 prior top-1 detection flips to `in_lung_parenchyma=False`.
    Skipped when the DICOMs are not on disk (CI / local fresh checkout).

The mask logic is intentionally covered on synthetic geometry so the
fast tests remain deterministic and CI-runnable without any TCIA data.
The TCGA-24-1423 regression is a soft-gated real-world sanity check.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oncology_arbiter.nsclc.parenchyma_mask import (
    LUNG_HU_MAX,
    LUNG_HU_MIN,
    _otsu,
    apply_parenchyma_filter,
    build_parenchyma_mask,
    is_detection_in_parenchyma,
)


# ---------------------------------------------------------------------------
# _otsu
# ---------------------------------------------------------------------------


def test_otsu_bimodal_split_between_modes():
    """Otsu on two well-separated modes should land strictly between them.

    We don't require the exact midpoint — Otsu on a 256-bin histogram
    over a wide HU range quantizes coarsely and can pin toward the
    smaller mode. What matters for the parenchyma mask is that the
    split is a valid separator (between the two modes).
    """
    rng = np.random.default_rng(42)
    left = rng.normal(-900.0, 20.0, size=5000)
    right = rng.normal(-200.0, 20.0, size=5000)
    values = np.concatenate([left, right])
    t = _otsu(values)
    assert -900.0 < t < -200.0, f"Otsu {t} outside expected split range"


def test_otsu_degenerate_all_zeros_returns_mean():
    """Degenerate single-valued input: threshold should sit essentially at 0."""
    values = np.zeros(1000, dtype=np.float32)
    t = _otsu(values)
    assert t == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# build_parenchyma_mask on a synthetic phantom
# ---------------------------------------------------------------------------


def _make_synthetic_chest_ct(
    shape: tuple[int, int, int] = (40, 60, 60),
    spacing: tuple[float, float, float] = (2.5, 0.7, 0.7),
    seed: int = 0,
) -> np.ndarray:
    """A tiny synthetic 'chest CT' with two lung blobs + mediastinum + body.

    Geometry:
      - Whole volume is HU=-2000 (below aerated window, matches how
        real DICOM pads exterior on many scanners).
      - Body silhouette: HU=0 (soft tissue) inside a central rectangular
        ROI covering most of the volume.
      - Two lung blobs: HU~-800 (in aerated window) in the left/right
        thirds inside the body.
      - Mediastinum: HU=50 rectangular block between the lungs.
      - Spine: HU=800 block at back of body.
    """
    rng = np.random.default_rng(seed)
    vol = np.full(shape, -2000.0, dtype=np.float32)
    D, H, W = shape
    # Body: shrinks from full frame by 5 vox on Y and X, all Z.
    vol[:, 5:-5, 5:-5] = 0.0
    # Left lung: axial y-center, x in [10..25].
    vol[5:-5, 15:-15, 10:25] = rng.normal(-800.0, 20.0, size=(D - 10, H - 30, 15))
    # Right lung: axial y-center, x in [35..50].
    vol[5:-5, 15:-15, 35:50] = rng.normal(-800.0, 20.0, size=(D - 10, H - 30, 15))
    # Mediastinum block between lungs (HU ~ soft tissue+).
    vol[5:-5, 15:-15, 25:35] = 50.0
    # Spine at back.
    vol[5:-5, -15:-8, 20:40] = 800.0
    return vol


def test_build_parenchyma_mask_on_synthetic_phantom():
    vol = _make_synthetic_chest_ct()
    spacing = (2.5, 0.7, 0.7)
    pm = build_parenchyma_mask(vol, spacing)

    # Otsu should have chosen a threshold well below -400 HU.
    assert pm.otsu_threshold_hu is not None
    assert LUNG_HU_MIN < pm.otsu_threshold_hu < LUNG_HU_MAX

    # Parenchyma fraction should be meaningful (>0) but not the whole
    # volume; two lungs occupy roughly (30 * 30 * 15 * 2) / (40*60*60) ~ 0.19.
    assert 0.05 < pm.fraction_of_volume < 0.40

    # Deep-lung points on the left and right lung: expected True.
    # (z=20, y=30, x=17) is in the middle of the left lung.
    left_lung = is_detection_in_parenchyma(
        (20 * spacing[0], 30 * spacing[1], 17 * spacing[2]),
        pm.mask, spacing,
    )
    right_lung = is_detection_in_parenchyma(
        (20 * spacing[0], 30 * spacing[1], 42 * spacing[2]),
        pm.mask, spacing,
    )
    assert left_lung.in_parenchyma is True, f"Left lung: {left_lung.reason}"
    assert right_lung.in_parenchyma is True, f"Right lung: {right_lung.reason}"

    # Mediastinum: expected False.
    med = is_detection_in_parenchyma(
        (20 * spacing[0], 30 * spacing[1], 30 * spacing[2]),
        pm.mask, spacing,
    )
    assert med.in_parenchyma is False, f"Mediastinum: {med.reason}"

    # Spine: expected False.
    spine = is_detection_in_parenchyma(
        (20 * spacing[0], 50 * spacing[1], 30 * spacing[2]),
        pm.mask, spacing,
    )
    assert spine.in_parenchyma is False, f"Spine: {spine.reason}"

    # Exterior (outside body): expected False.
    ext = is_detection_in_parenchyma(
        (20 * spacing[0], 2 * spacing[1], 2 * spacing[2]),
        pm.mask, spacing,
    )
    assert ext.in_parenchyma is False, f"Exterior: {ext.reason}"


# ---------------------------------------------------------------------------
# is_detection_in_parenchyma edge cases
# ---------------------------------------------------------------------------


def test_is_detection_in_parenchyma_out_of_bounds():
    mask = np.ones((10, 10, 10), dtype=bool)
    check = is_detection_in_parenchyma(
        (999.0, 999.0, 999.0), mask, (1.0, 1.0, 1.0),
    )
    assert check.in_parenchyma is False
    assert check.out_of_bounds is True
    assert "out of volume" in check.reason


def test_is_detection_in_parenchyma_negative_coord_out_of_bounds():
    mask = np.ones((10, 10, 10), dtype=bool)
    check = is_detection_in_parenchyma(
        (-1.0, 5.0, 5.0), mask, (1.0, 1.0, 1.0),
    )
    assert check.in_parenchyma is False
    assert check.out_of_bounds is True


# ---------------------------------------------------------------------------
# apply_parenchyma_filter mutates in place
# ---------------------------------------------------------------------------


def test_apply_parenchyma_filter_mutates_in_place():
    vol = _make_synthetic_chest_ct()
    spacing = (2.5, 0.7, 0.7)

    # Two detections: one in left lung, one in mediastinum.
    d_lung = SimpleNamespace(
        center_z_mm=20 * spacing[0],
        center_y_mm=30 * spacing[1],
        center_x_mm=17 * spacing[2],
    )
    d_med = SimpleNamespace(
        center_z_mm=20 * spacing[0],
        center_y_mm=30 * spacing[1],
        center_x_mm=30 * spacing[2],
    )
    detections = [d_lung, d_med]

    apply_parenchyma_filter(detections, vol, spacing)

    assert d_lung.in_lung_parenchyma is True
    assert d_med.in_lung_parenchyma is False


# ---------------------------------------------------------------------------
# TCGA-24-1423 fixture-gated regression
# ---------------------------------------------------------------------------


TCGA_24_1423_DIR = Path("/workspace/data/tcga_24_1423")


@pytest.mark.skipif(
    not TCGA_24_1423_DIR.exists() or not any(TCGA_24_1423_DIR.iterdir()),
    reason="TCGA-24-1423 CT not on disk; skipping real-world regression",
)
def test_tcga_24_1423_top_detection_flips_out_of_parenchyma():
    """Verify the LUNA16 top-1 anchor from TCGA-24-1423 is out-of-parenchyma.

    Anchor (transcript-verified):
        scanner-frame center (z=-238.74, y=313.94, x=188.31) mm
        score=0.8962, diameter=8.04 mm

    Full-scan translation:
        scanner voxel 0 → z=-680.0 mm (from ImagePositionPatient)
        volume-frame z = -238.74 - (-680.0) = 441.26 mm
                       ≈ voxel 88 at dz=5.0 mm spacing.

    That voxel is on the ribs/chest wall (HU > 1000), NOT lung.
    Path C must flip it to `in_lung_parenchyma=False`.
    """
    from oncology_arbiter.lung.ct_reader import read_ct_series

    ct = read_ct_series(TCGA_24_1423_DIR)
    spacing = (
        float(ct.slice_thickness_mm),
        float(ct.pixel_spacing_mm[0]),
        float(ct.pixel_spacing_mm[1]),
    )
    scanner_z_min = min(ct.z_positions_mm)

    # Scanner-frame report → volume-frame mm.
    scanner_z = -238.74
    y_mm = 313.94
    x_mm = 188.31
    volume_z_mm = scanner_z - scanner_z_min

    pm = build_parenchyma_mask(ct.volume, spacing)
    check = is_detection_in_parenchyma((volume_z_mm, y_mm, x_mm), pm.mask, spacing)

    # HU value at that voxel to prove it's not lung.
    iz, iy, ix = check.center_vox
    hu_here = float(ct.volume[iz, iy, ix])
    assert hu_here > 100.0, (
        f"expected non-lung HU (bone/soft tissue); got HU={hu_here} at "
        f"voxel {check.center_vox}"
    )
    assert check.in_parenchyma is False, (
        f"Regression: TCGA-24-1423 anchor should be out-of-parenchyma; "
        f"got {check.reason}"
    )


@pytest.mark.skipif(
    not TCGA_24_1423_DIR.exists() or not any(TCGA_24_1423_DIR.iterdir()),
    reason="TCGA-24-1423 CT not on disk; skipping z-resample regression",
)
def test_tcga_24_1423_z_resampled_anchor_still_out_of_parenchyma():
    """After z-resample from 5 mm -> 1.25 mm the anchor stays out-of-parenchyma.

    This mirrors the production inference wire in
    `api/app.py::/v1/case/full` when
    ``ONCOLOGY_ARBITER_ENABLE_LUNA16_RETINANET`` is on:

        read_ct_series(...)               # native dz=5.0 mm
        resample_for_luna16(...)          # -> dz=1.25 mm
        LungNoduleDetector.detect(...)    # runs on resampled volume
        apply_parenchyma_filter(...)      # queries the resampled volume

    We assert the anchor voxel, expressed in the RESAMPLED volume frame,
    lands on a non-parenchyma voxel (HU >> aerated-lung window).
    """
    from oncology_arbiter.lung.ct_reader import read_ct_series
    from oncology_arbiter.lung.resample import resample_for_luna16

    ct = read_ct_series(TCGA_24_1423_DIR)
    src_spacing = (
        float(ct.slice_thickness_mm),
        float(ct.pixel_spacing_mm[0]),
        float(ct.pixel_spacing_mm[1]),
    )
    resamp = resample_for_luna16(ct.volume, src_spacing, z_only=True)
    assert resamp.was_resampled is True, (
        f"expected resample from {src_spacing} to LUNA16 target; "
        f"got was_resampled=False"
    )
    assert resamp.spacing_mm[0] == pytest.approx(1.25, rel=1e-6)
    assert resamp.z_scale_factor == pytest.approx(4.0, rel=1e-6)

    # Anchor volume-frame z is invariant across resample (same mm origin).
    scanner_z = -238.74
    y_mm = 313.94
    x_mm = 188.31
    volume_z_mm = scanner_z - min(ct.z_positions_mm)

    pm = build_parenchyma_mask(resamp.volume, resamp.spacing_mm)
    check = is_detection_in_parenchyma(
        (volume_z_mm, y_mm, x_mm), pm.mask, resamp.spacing_mm,
    )

    iz, iy, ix = check.center_vox
    hu_here = float(resamp.volume[iz, iy, ix])
    assert hu_here > 100.0, (
        f"expected non-lung HU on resampled volume; got HU={hu_here} at "
        f"voxel {check.center_vox}"
    )
    assert check.in_parenchyma is False, (
        f"Regression: TCGA-24-1423 anchor should stay out-of-parenchyma "
        f"after z-resample; got {check.reason}"
    )
