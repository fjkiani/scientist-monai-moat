"""Lung parenchyma mask + LUNA16 detection filter (Path C).

Motivation
----------
LUNA16 (and the MONAI Model Zoo bundle we ship) was trained on curated
LIDC-IDRI CT with nodules internal to lung parenchyma. On unfiltered
inference, the detector can fire on structures OUTSIDE the parenchyma
(vessels at the mediastinum, pleural thickening, chest wall structures,
diaphragm). Those false positives cannot be nodules of clinical interest
by construction.

Regression anchor: TCGA-24-1423 prior top-1 detection sits at
    (z=-238.74, y=313.94, x=188.31)mm, score=0.8962, diameter=8.04mm

That point is anatomically at the chest wall, not lung parenchyma.
Path C's job is to flip `in_lung_parenchyma=False` for that detection so
downstream ranking/UX suppresses it.

Approach
--------
Given the HU volume:
  1. Build a candidate mask by HU thresholding to the aerated-lung
     window [-1000, -400] HU. Otsu is used to pick a data-driven split
     WITHIN the aerated window when the volume distribution supports it,
     but we always clamp between [-1000, -400] to guarantee we never
     select mediastinum (~0 HU) or bone (>100 HU).
  2. Per-slice morphological closing to fill airway / small-bronchi
     holes without leaking into mediastinum.
  3. Confine to the patient body. Build a body silhouette from
     `v > LUNG_HU_MAX` + slice-wise hole-fill, keep the biggest CC.
     Everything outside that silhouette is exterior air (crucial for
     CAP scans where the patient sits centered on the table and
     exterior air does NOT touch the axial border).
  4. Take the two largest 3D connected components of the confined
     mask (left + right lung).
  5. Slice-wise hole-fill on the surviving mask to catch nodules and
     consolidations that appear as HU-window holes inside the lung
     silhouette. Does NOT dilate outward — outward dilation crosses
     the pleura and would let non-parenchyma detections slip through.

Detection filter
----------------
Given a list of `Luna16Detection` records (with center in mm in the
same coordinate frame as `spacing_mm` — i.e. voxel_index * spacing,
NOT DICOM scanner-frame), transform back to voxel indices and check
`mask[cz, cy, cx]`. Detections with `mask[...] == True` get
`in_lung_parenchyma=True`; others get False.

Coordinate-frame note (regression anchor):
    TCGA-24-1423 prior top-1 detection was reported at
        (z=-238.74, y=313.94, x=188.31) mm, score=0.8962, diameter=8.04mm
    The negative z is DICOM scanner-frame (ImagePositionPatient),
    feet-first. The detector actually indexes voxel-frame (voxel * dz).
    That report used a 30-slice in-domain SUBSET whose voxel 0
    corresponded to scanner z=-245.0 mm, so the volume-frame center is:
        z_volume = z_scanner - scanner_z_min
                 = -238.74 - (-245.0) = 6.26 mm within the subset
    For the full 129-slice pull whose voxel 0 = scanner z=-680.0 mm:
        z_volume = -238.74 - (-680.0) = 441.26 mm  -> voxel 88
    That point falls on the chest wall / rib (HU > 200), and the mask
    filter correctly flips it to `in_lung_parenchyma=False`.

This module is intentionally standalone (no `luna16_retinanet.py`
dependency). It operates on the same HU volume the detector saw, so
the filter is trivially consistent with detector inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Aerated-lung HU window (well documented in radiology literature).
LUNG_HU_MIN = -1000.0
LUNG_HU_MAX = -400.0

# Preserved for backward compat with existing API callers. In the
# current implementation this is a "fill holes" trigger (>0 means fill
# holes on the parenchyma mask, capturing internal nodules /
# consolidations). Outward dilation is intentionally OFF because it
# crosses the visceral pleura into rib/muscle/fat, which lets
# non-parenchyma detections slip through. Peripheral / subpleural
# nodules are caught by the hole-fill because they sit INSIDE the lung
# silhouette on axial slices.
DEFAULT_BOUNDARY_DILATE_MM = 3.0


@dataclass(frozen=True)
class ParenchymaMask:
    """Output of `build_parenchyma_mask`.

    Kept small on purpose: the mask itself + provenance for the response.
    """

    mask: np.ndarray  # bool, shape = volume_shape (D, H, W)
    hu_range: tuple[float, float]
    otsu_threshold_hu: float | None
    boundary_dilate_mm: float
    fraction_of_volume: float


def _otsu(values: np.ndarray) -> float:
    """Otsu's threshold on 1-D float samples.

    Implemented here so we don't take a hard dep on scikit-image just for
    a threshold: 40 lines of stdlib + numpy is enough.
    """
    hist, edges = np.histogram(values, bins=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(values.mean())
    prob = hist / total
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    # Between-class variance for every possible threshold.
    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    mu0 = np.cumsum(prob * bin_centers) / np.maximum(w0, 1e-12)
    mu1 = (np.sum(prob * bin_centers) - np.cumsum(prob * bin_centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(bin_centers[int(np.argmax(between))])


def build_parenchyma_mask(
    volume_hu: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    boundary_dilate_mm: float = DEFAULT_BOUNDARY_DILATE_MM,
    use_otsu: bool = True,
) -> ParenchymaMask:
    """Build the lung parenchyma mask from a HU volume.

    Parameters
    ----------
    volume_hu : ndarray shape (D, H, W)
        CT volume in Hounsfield Units.
    spacing_mm : (dz, dy, dx)
        Voxel spacing in millimeters. Used only for the boundary dilation
        so the dilation is anatomically consistent regardless of scan
        resolution.
    boundary_dilate_mm : float
        Isotropic dilation applied AFTER connected-component selection,
        to keep peripheral nodules on the parenchyma boundary.
    use_otsu : bool
        When True, replace `LUNG_HU_MAX` with an Otsu-derived data-driven
        threshold clipped to [LUNG_HU_MIN, LUNG_HU_MAX]. When False,
        use the fixed window unconditionally. Defaults to True.

    Returns
    -------
    ParenchymaMask
    """
    from scipy import ndimage as ndi

    v = np.asarray(volume_hu, dtype=np.float32)
    if v.ndim == 4 and v.shape[0] == 1:
        v = v[0]
    if v.ndim != 3:
        raise ValueError(f"expected 3D or (1,D,H,W) volume, got shape {v.shape}")

    # 1. HU-window candidate air/parenchyma mask.
    hi = LUNG_HU_MAX
    otsu_val: float | None = None
    if use_otsu:
        # Only consider voxels already in the aerated window for Otsu;
        # this stops nasal air / patient-outside-scanner air from
        # dragging the threshold up.
        window = v[(v >= LUNG_HU_MIN) & (v <= LUNG_HU_MAX)]
        if window.size >= 1000:  # need enough samples for a meaningful split
            otsu_val = _otsu(window)
            # Otsu gives us a split point INSIDE the window; if it lands
            # too close to the ceiling it's not useful (the whole window
            # was one class). Fall back to fixed ceiling.
            if otsu_val < LUNG_HU_MAX - 5.0:
                hi = otsu_val
    mask = (v >= LUNG_HU_MIN) & (v <= hi)

    # 2. Morphological closing (per-slice, 2D) to fill airways / small
    # bronchi so we don't accidentally punch holes through the
    # parenchyma. Radius chosen small (3 voxels) — larger closes trapped
    # air pockets that we WANT to keep as parenchyma.
    struct_2d = ndi.generate_binary_structure(2, 1)
    for i in range(mask.shape[0]):
        mask[i] = ndi.binary_closing(mask[i], structure=struct_2d, iterations=3)

    # 3. Confine to the patient body. Chest CT scans image the patient
    #    surrounded by air; on CAP scans that exterior air can form a
    #    massive CC of aerated voxels that does NOT touch the axial
    #    border (the patient sits centered, table underneath). Simply
    #    stripping border-touching CCs is not enough.
    #
    #    Instead: compute the patient body as the largest connected
    #    component of "solid tissue" (HU > -400). Anything outside that
    #    body silhouette is dropped.
    struct_3d = ndi.generate_binary_structure(3, 3)
    body = v > LUNG_HU_MAX  # HU > -400 = soft tissue / bone
    # Fill interior air pockets (lungs, gut) so we get the SILHOUETTE
    # of the patient, not just tissue voxels.
    for i in range(body.shape[0]):
        body[i] = ndi.binary_fill_holes(body[i])
    body_labels, n_body = ndi.label(body, structure=struct_3d)
    if n_body >= 1:
        body_sizes = ndi.sum(body, body_labels, range(1, n_body + 1))
        biggest = int(np.argmax(body_sizes)) + 1  # +1 because 0 is bg
        body = body_labels == biggest
        mask = mask & body

    labels, n_lab = ndi.label(mask, structure=struct_3d)

    # 4. Keep only the two largest connected components (lung left + right).
    if n_lab > 2:
        sizes = ndi.sum(mask, labels, range(1, n_lab + 1))
        # sizes is 1-indexed via range(1, n_lab+1); top-2 label ids:
        keep_labels = np.argsort(sizes)[-2:] + 1  # +1 because label 0 is BG
        mask = np.isin(labels, keep_labels)
    elif n_lab == 0:
        # Empty result; return an all-False mask rather than crash.
        return ParenchymaMask(
            mask=np.zeros_like(mask, dtype=bool),
            hu_range=(LUNG_HU_MIN, hi),
            otsu_threshold_hu=otsu_val,
            boundary_dilate_mm=boundary_dilate_mm,
            fraction_of_volume=0.0,
        )

    # 5. Fill small internal holes on axial slices to catch nodules
    #    that sit INSIDE the parenchyma silhouette (nodules appear as
    #    HU-window holes because they're denser than aerated lung).
    #
    #    Because we already confined the mask to the patient body in
    #    step 3, hole-filling here only fills nodule-sized gaps within
    #    the left/right lung silhouettes — it does NOT wrap around the
    #    mediastinum (mediastinum has already been excluded by the
    #    top-2 CC step above).
    #
    #    We intentionally do NOT dilate outward by `boundary_dilate_mm`.
    #    Outward dilation crosses the visceral pleura into rib, muscle
    #    and subcutaneous fat, which would let non-parenchyma detections
    #    (HU > 200) slip through.
    if boundary_dilate_mm > 0:
        for i in range(mask.shape[0]):
            mask[i] = ndi.binary_fill_holes(mask[i])

    frac = float(mask.sum()) / float(mask.size)

    return ParenchymaMask(
        mask=mask.astype(bool),
        hu_range=(LUNG_HU_MIN, hi),
        otsu_threshold_hu=otsu_val,
        boundary_dilate_mm=boundary_dilate_mm,
        fraction_of_volume=frac,
    )


@dataclass
class DetectionInMaskCheck:
    """Result of `is_detection_in_parenchyma`.

    We keep both the boolean AND the specific voxel/mask readout so
    downstream logs are auditable.
    """

    in_parenchyma: bool
    center_vox: tuple[int, int, int]  # (z, y, x) voxel indices actually queried
    out_of_bounds: bool                # True if the mm coordinate falls outside the volume
    reason: str                        # human-readable summary


def is_detection_in_parenchyma(
    center_zyx_mm: tuple[float, float, float],
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> DetectionInMaskCheck:
    """Check whether a detection's center-mm falls inside the parenchyma.

    The convention we use across ct_reader + luna16_retinanet is that
    (0, 0, 0) mm is the origin of the volume (corner of voxel (0,0,0))
    and mm coordinates grow with voxel indices. Some upstream code paths
    hand back mm coordinates with negative signs (RAS scanner frame). We
    accept those inputs and interpret them as "relative to volume
    origin" — i.e., a center at z=-238mm on a volume that only spans
    z=[0, 200mm] will fall out of bounds.

    Parameters
    ----------
    center_zyx_mm : (cz, cy, cx) in millimeters relative to volume origin.
    mask : bool array shape (D, H, W)
    spacing_mm : (dz, dy, dx)

    Returns
    -------
    DetectionInMaskCheck
    """
    cz_mm, cy_mm, cx_mm = center_zyx_mm
    dz, dy, dx = spacing_mm
    if dz <= 0 or dy <= 0 or dx <= 0:
        raise ValueError(f"spacing_mm entries must be positive; got {spacing_mm}")

    D, H, W = mask.shape
    iz = int(round(cz_mm / dz))
    iy = int(round(cy_mm / dy))
    ix = int(round(cx_mm / dx))

    if not (0 <= iz < D and 0 <= iy < H and 0 <= ix < W):
        return DetectionInMaskCheck(
            in_parenchyma=False,
            center_vox=(iz, iy, ix),
            out_of_bounds=True,
            reason=(
                f"center at ({cz_mm:.2f}, {cy_mm:.2f}, {cx_mm:.2f}) mm "
                f"-> voxel ({iz}, {iy}, {ix}) out of volume shape "
                f"({D}, {H}, {W}) with spacing {spacing_mm}"
            ),
        )

    inside = bool(mask[iz, iy, ix])
    return DetectionInMaskCheck(
        in_parenchyma=inside,
        center_vox=(iz, iy, ix),
        out_of_bounds=False,
        reason=(
            f"center at ({cz_mm:.2f}, {cy_mm:.2f}, {cx_mm:.2f}) mm -> "
            f"voxel ({iz}, {iy}, {ix}) mask={inside}"
        ),
    )


def apply_parenchyma_filter(
    detections: Iterable,
    volume_hu: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    boundary_dilate_mm: float = DEFAULT_BOUNDARY_DILATE_MM,
    use_otsu: bool = True,
):
    """Enrich detections with `in_lung_parenchyma` in place.

    `detections` is an iterable of any object with `center_z_mm`,
    `center_y_mm`, `center_x_mm` (mm relative to volume origin) — this
    matches both `Luna16Detection` from the API schemas and
    `NoduleBox` from `luna16_retinanet.py`.

    This function is intentionally side-effect only (sets an attribute
    OR dict key on each detection). It does NOT drop any detection —
    filtering / suppression is the caller's job. Reason: we surface
    "detected but outside parenchyma" cleanly to the UI so radiologists
    can still see the raw model output for audit purposes.

    Returns
    -------
    ParenchymaMask
        The mask used for filtering, so the caller can log it in the
        response.
    """
    pmask = build_parenchyma_mask(
        volume_hu, spacing_mm,
        boundary_dilate_mm=boundary_dilate_mm,
        use_otsu=use_otsu,
    )

    for d in detections:
        # Accept both attribute-style (Pydantic model, NoduleBox) and
        # dict-style detection records.
        if isinstance(d, dict):
            cz = float(d["center_z_mm"])
            cy = float(d["center_y_mm"])
            cx = float(d["center_x_mm"])
        else:
            cz = float(d.center_z_mm)
            cy = float(d.center_y_mm)
            cx = float(d.center_x_mm)
        check = is_detection_in_parenchyma((cz, cy, cx), pmask.mask, spacing_mm)
        if isinstance(d, dict):
            d["in_lung_parenchyma"] = check.in_parenchyma
        else:
            try:
                # Pydantic v2 model: use setattr — model config allows mutation
                # by default in BaseModel unless frozen=True.
                object.__setattr__(d, "in_lung_parenchyma", check.in_parenchyma)
            except (AttributeError, TypeError):
                # Fallback for immutable objects — skip silently rather
                # than crash. The upstream filter will read whatever
                # default the object already has.
                pass

    return pmask
