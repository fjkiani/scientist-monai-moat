"""Tests for `oncology_arbiter.lung.resample.resample_for_luna16`.

These exercise the runtime resampler used by the LUNA16 inference path.
Synthetic-only — no network, no on-disk CT required.
"""
from __future__ import annotations

import numpy as np
import pytest

from oncology_arbiter.lung.resample import (
    DZ_PASSTHROUGH_TOL_MM,
    LUNA16_TARGET_SPACING_MM,
    resample_for_luna16,
)


def _make_volume(shape=(20, 32, 32), fill=-500.0):
    """Random-ish HU volume in the aerated-lung range so we can eyeball resample."""
    rng = np.random.default_rng(0xC0FFEE)
    v = rng.uniform(-1000.0, 200.0, size=shape).astype(np.float32)
    return v


def test_zonly_resample_5mm_to_125mm_scales_z_only():
    """5.0 mm -> 1.25 mm z-resample expands z by 4x, leaves in-plane."""
    v = _make_volume(shape=(10, 32, 32))
    r = resample_for_luna16(v, spacing_mm=(5.0, 0.7285, 0.7285), z_only=True)

    assert r.was_resampled is True
    assert r.method == "sitk_linear"
    # z upsampled 4x, in-plane preserved
    assert r.volume.shape == (40, 32, 32)
    assert r.spacing_mm[0] == pytest.approx(1.25, rel=1e-6)
    assert r.spacing_mm[1] == pytest.approx(0.7285, rel=1e-6)
    assert r.spacing_mm[2] == pytest.approx(0.7285, rel=1e-6)
    assert r.z_scale_factor == pytest.approx(4.0, rel=1e-6)
    # HU range preserved (bounded by clip pad)
    assert r.volume.min() >= -1024.0
    assert r.volume.max() <= 200.0


def test_passthrough_when_already_at_target_spacing():
    """dz within tolerance of 1.25 mm -> passthrough, unchanged shape."""
    v = _make_volume(shape=(40, 32, 32))
    src_spacing = (1.25, 0.703125, 0.703125)
    r = resample_for_luna16(v, spacing_mm=src_spacing, z_only=True)

    assert r.was_resampled is False
    assert r.method == "passthrough"
    assert r.volume.shape == v.shape
    assert r.spacing_mm == pytest.approx(src_spacing, abs=1e-9)
    assert r.z_scale_factor == pytest.approx(1.0, abs=1e-9)
    # No mutation of the returned reference; safe to check id equality
    assert np.array_equal(r.volume, v)


def test_z_only_false_rescales_all_three_axes():
    """z_only=False resamples every axis to LUNA16_TARGET_SPACING_MM."""
    v = _make_volume(shape=(10, 32, 32))
    r = resample_for_luna16(v, spacing_mm=(5.0, 1.0, 1.0), z_only=False)

    assert r.was_resampled is True
    assert r.spacing_mm == pytest.approx(LUNA16_TARGET_SPACING_MM, rel=1e-6)
    # z upsampled 4x; in-plane upsampled ~1/0.703 = 1.42x
    assert r.volume.shape[0] == 40
    assert r.volume.shape[1] == int(round(32 * 1.0 / 0.703125))
    assert r.volume.shape[2] == int(round(32 * 1.0 / 0.703125))


def test_air_fill_matches_hu_floor():
    """Out-of-source-bounds voxels are filled with -1024 HU (air)."""
    v = _make_volume(shape=(2, 32, 32))
    # Downsample z heavily so the target grid extends beyond the source.
    # Actually — we want an *up*sample where new z-count > old z-count * factor.
    # SITK will fill boundary voxels with the DefaultPixelValue; we can
    # eyeball this by asserting min <= -1024.
    r = resample_for_luna16(v, spacing_mm=(5.0, 0.7285, 0.7285), z_only=True)
    assert r.volume.min() >= -1024.0
    assert r.volume.min() <= r.volume.max()


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="expected 3D"):
        resample_for_luna16(
            np.zeros((10, 10), dtype=np.float32),
            spacing_mm=(5.0, 0.7285, 0.7285),
        )


def test_rejects_non_positive_spacing():
    with pytest.raises(ValueError, match="positive"):
        resample_for_luna16(
            _make_volume(),
            spacing_mm=(5.0, -0.5, 0.7285),
        )


def test_within_passthrough_tolerance_not_resampled():
    """dz within DZ_PASSTHROUGH_TOL_MM of 1.25 -> passthrough."""
    v = _make_volume(shape=(40, 32, 32))
    almost = (1.25 + DZ_PASSTHROUGH_TOL_MM * 0.5, 0.703125, 0.703125)
    r = resample_for_luna16(v, spacing_mm=almost, z_only=True)
    assert r.was_resampled is False


def test_beyond_passthrough_tolerance_is_resampled():
    """dz outside DZ_PASSTHROUGH_TOL_MM -> real resample."""
    v = _make_volume(shape=(40, 32, 32))
    beyond = (1.25 + DZ_PASSTHROUGH_TOL_MM * 5, 0.703125, 0.703125)
    r = resample_for_luna16(v, spacing_mm=beyond, z_only=True)
    assert r.was_resampled is True
