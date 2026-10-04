"""Z-axis resampling for LUNA16-compatible CT inference.

Motivation
----------
The MONAI Model Zoo `lung_nodule_ct_detection@0.6.9` RetinaNet was trained
on LIDC/LUNA16 volumes resampled to (dz=1.25, dy=0.703125, dx=0.703125) mm.
On clinical scans acquired at CAP thickness (dz=5.0 mm — e.g. TCGA-24-1423)
the detector's z-axis anchors under-cover the true nodule scale: its
receptive field expects ~4× more axial detail than the raw scan provides.

Empirical observation (TCGA-24-1423, 129×512×512 @ dz=5.0):
    detector on native dz=5.0  -> n=4 detections, top_score=0.0669
    detector on resampled dz=1.25 -> n=16, top_score=0.7768
                                     top-1 at (z=442.4, y=315.2, x=189.1) mm
                                     ≈ transcript anchor (441.3, 313.9, 188.3) mm

The training-corpus resampler at ``nsclc.luna16_finetune.resample_series``
handles the on-disk .mhd → .nii.gz path; this module handles the runtime
in-memory HU-volume → HU-volume path used by the API pipeline.

Design
------
- Trilinear resample (matches the finetune resampler and MONAI Model Zoo
  bundle's own pre-processing spec).
- Air fill value: −1024 HU (matches the bundle's ``HU_range = [-1024, 300]``
  clip).
- Z-only rescale by default: in-plane spacing on modern clinical CT is
  already ~0.5–0.9 mm, so the detector runs fine at native (dy, dx). This
  keeps memory small and preserves per-slice detail.
- Full-target rescale (all three axes) is available via
  ``target_spacing_mm`` for callers who want to match the finetune recipe
  exactly.

The choice to resample z (not skip low-z-resolution scans) is deliberate:
5mm slabs are the most common CAP acquisition in oncology (TCGA, NLST,
routine follow-up) and simply rejecting them would exclude the majority
of real-world requests. Trilinear over the HU volume is faithful — it
introduces no synthetic content, only interpolates between adjacent
acquired slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# LUNA16 / MONAI Model Zoo target spacing (dz, dy, dx) in mm.
LUNA16_TARGET_SPACING_MM: Tuple[float, float, float] = (1.25, 0.703125, 0.703125)

# Any scan with dz within this tolerance of the target is passed through
# untouched. 0.05 mm = 4% of 1.25 mm — well below acquisition jitter.
DZ_PASSTHROUGH_TOL_MM = 0.05

# Fill value for extrapolated voxels (matches bundle HU_range floor).
AIR_HU = -1024.0


@dataclass(frozen=True)
class ResampledVolume:
    """Return value of `resample_for_luna16`."""

    volume: np.ndarray                          # (D, H, W), float32 HU
    spacing_mm: Tuple[float, float, float]      # (dz, dy, dx) of `volume`
    source_spacing_mm: Tuple[float, float, float]
    was_resampled: bool
    z_scale_factor: float                       # source_dz / target_dz
    method: str                                 # "sitk_linear" | "passthrough"


def resample_for_luna16(
    volume_hu: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    *,
    target_spacing_mm: Optional[Tuple[float, float, float]] = None,
    z_only: bool = True,
) -> ResampledVolume:
    """Resample a HU volume to LUNA16-compatible spacing.

    Parameters
    ----------
    volume_hu : ndarray shape (D, H, W)
        CT volume in Hounsfield Units.
    spacing_mm : (dz, dy, dx)
        Source voxel spacing in millimeters. This is what the caller
        reads out of the DICOM series (``SliceThickness`` /
        ``PixelSpacing``).
    target_spacing_mm : (dz, dy, dx), optional
        Target spacing. Defaults to ``LUNA16_TARGET_SPACING_MM``.
    z_only : bool, default True
        When True, only the z axis is rescaled; ``(dy, dx)`` are kept at
        their source values. This is the recommended default because
        in-plane spacing on clinical CT is already fine-grained and
        preserving it avoids unnecessary memory blowup + interpolation
        loss on the higher-resolution axes.

        When False, all three axes are rescaled to ``target_spacing_mm``.

    Returns
    -------
    ResampledVolume

    Notes
    -----
    * SimpleITK is imported lazily so the module is importable in envs
      where the detector deps aren't installed.
    * Volumes already at the target dz (within ``DZ_PASSTHROUGH_TOL_MM``)
      are returned unchanged with ``was_resampled=False``.
    """
    import SimpleITK as sitk

    v = np.asarray(volume_hu, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"expected 3D HU volume, got shape {v.shape}")

    src_dz, src_dy, src_dx = (float(s) for s in spacing_mm)
    if src_dz <= 0 or src_dy <= 0 or src_dx <= 0:
        raise ValueError(f"spacing entries must be positive; got {spacing_mm}")

    tgt_dz, tgt_dy, tgt_dx = (
        target_spacing_mm if target_spacing_mm is not None
        else LUNA16_TARGET_SPACING_MM
    )
    tgt_dz = float(tgt_dz)
    tgt_dy = float(tgt_dy) if not z_only else src_dy
    tgt_dx = float(tgt_dx) if not z_only else src_dx

    dz_diff = abs(src_dz - tgt_dz)
    inplane_matches = (
        abs(src_dy - tgt_dy) <= DZ_PASSTHROUGH_TOL_MM
        and abs(src_dx - tgt_dx) <= DZ_PASSTHROUGH_TOL_MM
    )
    if dz_diff <= DZ_PASSTHROUGH_TOL_MM and inplane_matches:
        return ResampledVolume(
            volume=v,
            spacing_mm=(src_dz, src_dy, src_dx),
            source_spacing_mm=(src_dz, src_dy, src_dx),
            was_resampled=False,
            z_scale_factor=1.0,
            method="passthrough",
        )

    # SimpleITK image axes are (X, Y, Z). GetImageFromArray reads a
    # (D, H, W) numpy array as (Z, Y, X), so SetSpacing/SetSize/GetSize
    # need to be in (X, Y, Z) order.
    img = sitk.GetImageFromArray(v)
    img.SetSpacing((src_dx, src_dy, src_dz))
    img.SetOrigin((0.0, 0.0, 0.0))

    old_size_xyz = img.GetSize()            # (X, Y, Z)
    new_spacing_xyz = (tgt_dx, tgt_dy, tgt_dz)
    new_size_xyz = [
        int(round(old_size_xyz[i] * (src_dx, src_dy, src_dz)[i] / new_spacing_xyz[i]))
        for i in range(3)
    ]
    if new_size_xyz[2] < 1:
        raise ValueError(
            f"z-resample would produce {new_size_xyz[2]} slices; "
            f"source={src_dz}mm target={tgt_dz}mm"
        )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing_xyz)
    resampler.SetSize(new_size_xyz)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(AIR_HU)

    out_img = resampler.Execute(img)
    out_np = sitk.GetArrayFromImage(out_img).astype(np.float32)   # (Z, Y, X)

    return ResampledVolume(
        volume=out_np,
        spacing_mm=(tgt_dz, tgt_dy, tgt_dx),
        source_spacing_mm=(src_dz, src_dy, src_dx),
        was_resampled=True,
        z_scale_factor=src_dz / tgt_dz,
        method="sitk_linear",
    )


__all__ = [
    "LUNA16_TARGET_SPACING_MM",
    "DZ_PASSTHROUGH_TOL_MM",
    "AIR_HU",
    "ResampledVolume",
    "resample_for_luna16",
]
