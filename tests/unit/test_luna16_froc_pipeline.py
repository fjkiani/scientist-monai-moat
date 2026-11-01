"""Unit tests for scripts/luna16_froc.py -- pure logic, no network calls.

These use a small SYNTHETIC-geometry SimpleITK volume (fabricated
spacing/origin/shape, deterministic pixel values) purely to exercise the
DICOM re-encoding and coordinate-inversion MATH in isolation. This is
standard unit-test practice (isolating logic from I/O), not a claim about
LUNA16 nodule-detection performance -- no FROC/sensitivity number is
produced or asserted here. The live, real-data FROC measurement lives in
tests/integration and is driven by scripts/luna16_froc.py against real
subset0 volumes + the live luna16-detect endpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import luna16_froc as m  # noqa: E402


def _make_synthetic_volume(seriesuid: str = "1.2.synthetic.unittest.0001"):
    """Build a small, deterministic, clearly-synthetic SimpleITK volume for
    pure geometry/logic testing. Shape/spacing/origin are fabricated and
    labeled as such; this never touches real LUNA16 data."""
    import SimpleITK as sitk

    rng = np.random.RandomState(0)
    d, h, w = 6, 16, 20
    hu = rng.randint(-1000, 400, size=(d, h, w)).astype(np.int16)

    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((0.75, 0.8, 1.5))  # (sx, sy, sz) ITK order
    img.SetOrigin((-100.0, -150.0, -300.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return {
        "image": img,
        "array": hu,
        "spacing_xyz": img.GetSpacing(),
        "origin_xyz": img.GetOrigin(),
        "direction": img.GetDirection(),
        "seriesuid": seriesuid,
        "shape_dhw": (d, h, w),
    }


def test_assert_identity_direction_accepts_identity():
    m.assert_identity_direction((1, 0, 0, 0, 1, 0, 0, 0, 1), context="unit-test")


def test_assert_identity_direction_rejects_rotation():
    rotated = (0, 1, 0, 1, 0, 0, 0, 0, 1)  # 90-degree swap, not axis-aligned in the assumed sense
    with pytest.raises(m.CoordinateFrameError):
        m.assert_identity_direction(rotated, context="unit-test")


def test_dicom_round_trip_preserves_real_pixel_values_exactly():
    vol = _make_synthetic_volume()
    blobs = m.hu_array_to_dicom_series(vol)
    assert len(blobs) == vol["shape_dhw"][0]

    recovered = m.decode_dicom_series_for_check(blobs)
    np.testing.assert_array_equal(recovered, vol["array"])


def test_dicom_round_trip_preserves_series_uid_and_geometry():
    import pydicom
    import io

    vol = _make_synthetic_volume(seriesuid="1.2.synthetic.unittest.9999")
    blobs = m.hu_array_to_dicom_series(vol)

    ds0 = pydicom.dcmread(io.BytesIO(blobs[0]))
    assert ds0.SeriesInstanceUID == "1.2.synthetic.unittest.9999"
    sx, sy, sz = vol["spacing_xyz"]
    assert ds0.PixelSpacing == [pytest.approx(sy), pytest.approx(sx)]
    assert ds0.SliceThickness == pytest.approx(sz)

    ds_last = pydicom.dcmread(io.BytesIO(blobs[-1]))
    # z should increase monotonically with instance/slice index for this axis-aligned volume
    assert float(ds_last.ImagePositionPatient[2]) > float(ds0.ImagePositionPatient[2])


def test_coordinate_recovery_round_trip_self_test_passes_on_synthetic_geometry():
    """Exercises the exact function used as the mandatory precondition
    check before trusting any real FROC run: a real (or here, synthetic)
    world coordinate must round-trip through voxel-index recovery to
    floating point precision."""
    vol = _make_synthetic_volume()
    img = vol["image"]
    # Pick a point clearly inside the synthetic volume's extent and confirm
    # round trip -- this is a geometry/math check, not a nodule claim.
    ox, oy, oz = vol["origin_xyz"]
    sx, sy, sz = vol["spacing_xyz"]
    fake_annotation_row = {
        "coordX": str(ox + 3.0 * sx),
        "coordY": str(oy + 4.0 * sy),
        "coordZ": str(oz + 2.0 * sz),
    }
    result = m.round_trip_selftest(vol, fake_annotation_row, atol_mm=1e-6)
    assert result["ok"], result


def test_coordinate_recovery_matches_hand_computed_value():
    """Hand-computable case: detection reported at exactly voxel (2,3,1)
    (z,y,x) must recover to origin + index*spacing exactly, for identity
    direction cosines."""
    vol = _make_synthetic_volume()
    img = vol["image"]
    sx, sy, sz = vol["spacing_xyz"]
    ox, oy, oz = vol["origin_xyz"]

    voxel_z, voxel_y, voxel_x = 2.0, 3.0, 1.0
    spacing_sent = (sz, sy, sx)
    fake_detection = {
        "center_z_mm": voxel_z * spacing_sent[0],
        "center_y_mm": voxel_y * spacing_sent[1],
        "center_x_mm": voxel_x * spacing_sent[2],
    }
    wx, wy, wz = m.recover_world_coords(fake_detection, image=img, spacing_sent_dzdydx=spacing_sent)
    assert wx == pytest.approx(ox + voxel_x * sx, abs=1e-9)
    assert wy == pytest.approx(oy + voxel_y * sy, abs=1e-9)
    assert wz == pytest.approx(oz + voxel_z * sz, abs=1e-9)


def test_non_int16_lossy_hu_values_raise_instead_of_silently_truncating():
    vol = _make_synthetic_volume()
    # Corrupt with a fractional, non-losslessly-roundable HU value.
    bad_array = vol["array"].astype(np.float32)
    bad_array[0, 0, 0] = 12.37
    vol_bad = dict(vol)
    vol_bad["array"] = bad_array
    with pytest.raises(m.CoordinateFrameError):
        m.hu_array_to_dicom_series(vol_bad)
