"""DICOM + PNG readers for mammography inputs.

Returns a `(pixel_array, metadata_dict)` tuple. The array is always float32
in [0, 1] scaled from the full stored range so downstream logic does not
have to remember whether the source was 12-bit, 14-bit, or 16-bit.

Metadata is a plain dict with the keys we actually use downstream:
    modality, body_part, laterality_hint, view_hint, orientation_hint,
    manufacturer, rows, cols, bits_stored, photometric, source_path
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_mammogram_dicom(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a mammography DICOM. Returns (float32 [0,1] array, metadata)."""
    import pydicom  # local import so this module can be imported without pydicom
    p = Path(path)
    ds = pydicom.dcmread(str(p))
    arr = ds.pixel_array
    if arr.ndim != 2:
        raise ValueError(
            f"expected 2D mammogram, got shape {arr.shape} for {p}"
        )
    # MONOCHROME1 means high pixel value = dark, invert to MONOCHROME2 convention.
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    arr_f = arr.astype(np.float32)
    if photometric.upper() == "MONOCHROME1":
        arr_f = arr_f.max() - arr_f

    # Normalize to [0, 1] by dividing by the max representable value at
    # BitsStored, NOT the observed max — this preserves relative brightness
    # across images of the same acquisition family.
    bits_stored = int(ds.get("BitsStored", 16))
    max_val = float(2 ** bits_stored - 1)
    # But if observed max exceeds representable range (rare, corrupted headers)
    # fall back to observed max.
    if arr_f.max() > max_val:
        max_val = float(arr_f.max())
    arr_f = arr_f / max(max_val, 1.0)
    arr_f = np.clip(arr_f, 0.0, 1.0)

    meta = _extract_dicom_metadata(ds, str(p))
    return arr_f, meta


def read_mammogram_png(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a mammography PNG/JPEG. Returns (float32 [0,1] array, metadata)."""
    from PIL import Image
    p = Path(path)
    img = Image.open(p)
    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)  # RGB -> grey
    max_val = float(arr.max()) if arr.max() > 0 else 1.0
    if max_val <= 255:
        arr = arr / 255.0
    elif max_val <= 4095:
        arr = arr / 4095.0
    else:
        arr = arr / max_val
    arr = np.clip(arr, 0.0, 1.0)
    return arr, {
        "modality": "MG",
        "body_part": None,
        "laterality_hint": None,
        "view_hint": None,
        "orientation_hint": None,
        "manufacturer": None,
        "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1]),
        "bits_stored": 8 if max_val <= 255 else 12 if max_val <= 4095 else 16,
        "photometric": "MONOCHROME2",
        "source_path": str(p),
    }


def _extract_dicom_metadata(ds: Any, source_path: str) -> dict[str, Any]:
    """Pull the tags we care about + normalize known aliases.

    CBIS-DDSM curated data commonly has:
      * BodyPartExamined = "Left Breast" / "Right Breast"  (laterality hint)
      * PatientOrientation = "CC" / "MLO"                  (view hint)
      * ImageLaterality / ViewPosition often ABSENT
    Native DICOM has:
      * ImageLaterality = "L" / "R"
      * ViewPosition = "CC" / "MLO" / "ML" / "LM" / etc.
    Both are respected — DICOM tags take priority over derived hints.
    """
    lat = ds.get("ImageLaterality", None)
    body_part = ds.get("BodyPartExamined", None)
    if lat is None and body_part:
        bp = str(body_part).lower()
        if "left" in bp:
            lat = "L"
        elif "right" in bp:
            lat = "R"

    view = ds.get("ViewPosition", None)
    pat_orient = ds.get("PatientOrientation", None)
    if view is None and pat_orient:
        po = str(pat_orient).upper()
        if "CC" in po:
            view = "CC"
        elif "MLO" in po:
            view = "MLO"

    return {
        "modality": str(ds.get("Modality", "")) or None,
        "body_part": str(body_part) if body_part else None,
        "laterality_hint": str(lat).upper() if lat else None,
        "view_hint": str(view).upper() if view else None,
        "orientation_hint": str(pat_orient) if pat_orient else None,
        "manufacturer": str(ds.get("Manufacturer", "")) or None,
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
        "bits_stored": int(ds.get("BitsStored", 16)),
        "photometric": str(ds.get("PhotometricInterpretation", "MONOCHROME2")),
        "source_path": source_path,
    }
