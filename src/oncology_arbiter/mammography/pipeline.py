"""End-to-end mammography preprocessing pipeline.

`preprocess_mammogram(path)` returns:
    PreprocessedMammogram(
        image: np.ndarray,           # float32 [0, 1], radiological orientation
        breast_mask: np.ndarray,     # bool, same shape as image
        pectoral_removed: bool,      # True on MLO views
        metadata: MammogramMetadata, # laterality, view, source metadata
    )

This is the shape a downstream screening detector, biopsy tool, or
therapy-recommender ingests. The image is deliberately kept at native
resolution — we don't resize here because the target model chooses its
own input size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .laterality import (
    Laterality,
    detect_laterality_from_content,
    detect_laterality_from_metadata,
    orient_to_radiological_convention,
)
from .reader import read_mammogram_dicom, read_mammogram_png
from .segmentation import breast_mask_otsu, remove_pectoral_mlo
from .view import View, detect_view_from_metadata


@dataclass
class MammogramMetadata:
    laterality: Laterality
    view: View
    laterality_source: str        # "dicom_tag" | "filename" | "content" | "unknown"
    view_source: str              # "dicom_tag" | "filename" | "unknown"
    orientation_flipped: bool     # True if we mirrored for radiological convention
    original_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedMammogram:
    image: np.ndarray
    breast_mask: np.ndarray
    pectoral_removed: bool
    metadata: MammogramMetadata


def preprocess_mammogram(
    path: str | Path,
    *,
    remove_pectoral: bool = True,
    normalize_orientation: bool = True,
    laterality_hint: str | Laterality | None = None,
    view_hint: str | View | None = None,
) -> PreprocessedMammogram:
    """Full pipeline: read → orient → mask → (optional) remove pectoral.

    Args:
      path: DICOM or PNG file.
      remove_pectoral: apply pectoral removal on MLO views.
      normalize_orientation: flip to project orientation convention.
      laterality_hint: if provided, this takes priority over DICOM tags,
        filename, and content detection. Useful when the API caller already
        knows the laterality (e.g., from a PACS system) or when the on-disk
        filename does not carry the CBIS-DDSM convention (HTTP upload path).
      view_hint: same priority story for CC/MLO.
    """
    p = Path(path)

    # 1. Read
    if p.suffix.lower() in (".dcm", ".dicom"):
        arr, raw_meta = read_mammogram_dicom(p)
    elif p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        arr, raw_meta = read_mammogram_png(p)
    else:
        raise ValueError(
            f"unsupported extension {p.suffix!r} — expected .dcm, .png, .jpg, .tif"
        )

    # 2. Laterality — hint > metadata > filename > content
    if laterality_hint is not None:
        hint = str(laterality_hint).upper()[:1]
        lat = {"L": Laterality.LEFT, "R": Laterality.RIGHT}.get(hint, Laterality.UNKNOWN)
        lat_src = "hint" if lat != Laterality.UNKNOWN else "unknown"
        if lat == Laterality.UNKNOWN:
            # Bad hint — fall through to normal detection
            lat = detect_laterality_from_metadata(raw_meta, filename=p)
            lat_src = "dicom_tag" if raw_meta.get("laterality_hint") else \
                      ("filename" if lat != Laterality.UNKNOWN else "unknown")
    else:
        lat = detect_laterality_from_metadata(raw_meta, filename=p)
        lat_src = "dicom_tag" if raw_meta.get("laterality_hint") else "unknown"
        if lat == Laterality.UNKNOWN:
            lat = detect_laterality_from_content(arr)
            lat_src = "content" if lat != Laterality.UNKNOWN else "unknown"
        elif lat_src == "unknown":
            lat_src = "filename"

    # 3. View — hint > metadata > filename
    if view_hint is not None:
        v = str(view_hint).upper()
        view = {"CC": View.CC, "MLO": View.MLO}.get(v, View.UNKNOWN)
        view_src = "hint" if view != View.UNKNOWN else "unknown"
        if view == View.UNKNOWN:
            view = detect_view_from_metadata(raw_meta, filename=p)
            view_src = "dicom_tag" if raw_meta.get("view_hint") else (
                "filename" if view != View.UNKNOWN else "unknown"
            )
    else:
        view = detect_view_from_metadata(raw_meta, filename=p)
        view_src = "dicom_tag" if raw_meta.get("view_hint") else (
            "filename" if view != View.UNKNOWN else "unknown"
        )

    # 4. Normalize orientation to radiological display
    flipped = False
    if normalize_orientation and lat != Laterality.UNKNOWN:
        content_side_before = detect_laterality_from_content(arr)
        arr_oriented = orient_to_radiological_convention(arr, lat)
        if content_side_before != Laterality.UNKNOWN:
            # We know we flipped if the tissue moved sides
            content_side_after = detect_laterality_from_content(arr_oriented)
            flipped = content_side_before != content_side_after
        arr = arr_oriented

    # 5. Breast mask
    mask = breast_mask_otsu(arr)

    # 6. Pectoral removal (MLO only)
    pect_removed = False
    if remove_pectoral and view == View.MLO and lat != Laterality.UNKNOWN:
        arr = remove_pectoral_mlo(arr, laterality=lat.value)
        # Recompute mask after pectoral removal.
        mask = breast_mask_otsu(arr)
        pect_removed = True

    return PreprocessedMammogram(
        image=arr.astype(np.float32),
        breast_mask=mask,
        pectoral_removed=pect_removed,
        metadata=MammogramMetadata(
            laterality=lat,
            view=view,
            laterality_source=lat_src,
            view_source=view_src,
            orientation_flipped=flipped,
            original_metadata=raw_meta,
        ),
    )
