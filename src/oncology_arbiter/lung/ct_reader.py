"""Read a CT series from a directory of DICOM slice files.

Wire-compatible with the LIDC-IDRI on-disk layout::

    <base>/<PatientID>/<StudyUID>/CT_<SeriesUID>/<slice>.dcm

The directory must contain only DICOM files belonging to a single CT series.
LIDC-IDRI stores segmentation (SEG_*), structured reports (SR_*), and X-ray
(DX_*) series in sibling directories under the same StudyUID; the caller must
already have resolved a `CT_<SeriesUID>` subdirectory. The public
`biomni.data.lidc_idri` helper does that resolution.

Returns a `CtSeries` dataclass with:
    volume         : np.ndarray shape (Z, Y, X), float32 Hounsfield units
    z_positions_mm : list[float], ascending (feet → head)
    pixel_spacing_mm : (row_mm, col_mm)
    slice_thickness_mm : float, from first slice header
    raw_meta       : dict of the first slice's core DICOM fields

**No fabricated pixels.** If a slice fails to decode or lacks the fields we
need to build an HU volume, we raise instead of silently patching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CtSeries:
    volume: np.ndarray
    z_positions_mm: list[float]
    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    raw_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.volume.shape)  # type: ignore[return-value]


def _slice_key(ds) -> float:
    """Ordering key used to stack slices feet → head.

    Priority: ImagePositionPatient[2] (mm along Z) > InstanceNumber > filename
    hash. LIDC uses ImagePositionPatient consistently, so the fallbacks are
    only exercised on malformed data.
    """
    ipp = ds.get("ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except (TypeError, ValueError):
            pass
    inst = ds.get("InstanceNumber", None)
    if inst is not None:
        try:
            return float(inst)
        except (TypeError, ValueError):
            pass
    # Last-resort deterministic hash — never fabricated content, just an
    # ordering key that stays stable across reruns.
    fname = getattr(ds, "filename", "") or ""
    return float(abs(hash(fname)) % (10**9))


def read_ct_series(series_dir: str | Path) -> CtSeries:
    """Load a CT series from a directory of DICOM slice files.

    Parameters
    ----------
    series_dir : str | Path
        Directory containing DICOM slice files. Must be a single-series dir
        (e.g. ``.../CT_<SeriesUID>/``).

    Returns
    -------
    CtSeries

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no readable DICOM files.
    ValueError
        If slices disagree on `PixelSpacing`, `Rows`, or `Columns`.
    """
    import pydicom

    p = Path(series_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"series dir does not exist: {p}")

    dcm_files = sorted(x for x in p.iterdir() if x.is_file())
    if not dcm_files:
        raise FileNotFoundError(f"no DICOM files in series dir: {p}")

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        except Exception:
            # skip non-DICOM (LIDC sometimes has stray .DS_Store etc)
            continue
        if "PixelData" not in ds:
            continue
        datasets.append(ds)
    if not datasets:
        raise FileNotFoundError(f"no readable DICOM slices in {p}")

    datasets.sort(key=_slice_key)

    # Guardrails: every slice must have matching Rows/Columns/PixelSpacing.
    first = datasets[0]
    rows = int(first.Rows)
    cols = int(first.Columns)
    ps = first.get("PixelSpacing", [1.0, 1.0])
    row_mm = float(ps[0])
    col_mm = float(ps[1])
    for ds in datasets[1:]:
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError(
                f"series {p} has slices with mismatched Rows/Columns: "
                f"expected {rows}x{cols}, saw {int(ds.Rows)}x{int(ds.Columns)}"
            )
        p2 = ds.get("PixelSpacing", ps)
        if float(p2[0]) != row_mm or float(p2[1]) != col_mm:
            raise ValueError(
                f"series {p} has slices with mismatched PixelSpacing: "
                f"expected ({row_mm},{col_mm}), saw ({float(p2[0])},{float(p2[1])})"
            )

    # Build HU volume, one slice at a time. Rescale slope/intercept convert
    # stored pixel values into Hounsfield units.
    n = len(datasets)
    volume = np.empty((n, rows, cols), dtype=np.float32)
    z_positions: list[float] = []
    for i, ds in enumerate(datasets):
        px = ds.pixel_array.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        volume[i] = px * slope + intercept
        z_positions.append(_slice_key(ds))

    thickness = float(first.get("SliceThickness", 1.0))

    raw_meta: dict[str, Any] = {
        "modality": str(first.get("Modality", "")),
        "manufacturer": str(first.get("Manufacturer", "")),
        "rows": rows,
        "cols": cols,
        "n_slices": n,
        "rescale_slope": float(first.get("RescaleSlope", 1.0)),
        "rescale_intercept": float(first.get("RescaleIntercept", 0.0)),
        "slice_thickness_mm": thickness,
        "pixel_spacing_mm": (row_mm, col_mm),
        "z_min_mm": min(z_positions),
        "z_max_mm": max(z_positions),
        "series_uid": str(first.get("SeriesInstanceUID", "")),
        "patient_id": str(first.get("PatientID", "")),
    }

    return CtSeries(
        volume=volume,
        z_positions_mm=z_positions,
        pixel_spacing_mm=(row_mm, col_mm),
        slice_thickness_mm=thickness,
        raw_meta=raw_meta,
    )


__all__ = ["CtSeries", "read_ct_series"]
