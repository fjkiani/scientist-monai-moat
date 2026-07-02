"""Mammography-specific preprocessing.

The generic 3D CT/MRI pipeline in `ai-training/domains/medical/pipeline.py`
(HU windowing, resample to isotropic, center-crop-or-pad 64x64x64) is the
right shape for whole-volume classification but useless for 2D screening
mammography. Mammograms need a different set of primitives:

  * Read a mammography DICOM (2D, MG modality, 16-bit MONOCHROME2) or a PNG
    mirror; return a normalized float32 [0, 1] array and rich metadata.
  * Detect laterality (L/R) — from DICOM tags, filename, or image content
    (which side of the frame the breast is on).
  * Detect view (CC/MLO) — from tags, path, or by pectoral-muscle detection.
  * Normalize to radiological display convention (LEFT breast chest wall on
    the RIGHT of the frame, RIGHT breast chest wall on the LEFT). CBIS-DDSM
    is acquisition-orientation by default; without this step, downstream
    laterality-dependent models will silently see mirrored data.
  * Segment the breast tissue (Otsu threshold + largest connected component)
    and produce a binary breast mask.
  * Remove the pectoral muscle triangle (MLO views only) so it does not
    dominate downstream feature extractors.

All primitives operate on 2D uint16 or float arrays and return either a
transformed array or a small dict of extracted features. Nothing here calls
a neural network — that's Phase 5 (screening detector) territory.

Ground-truth notes for future work:
  * CBIS-DDSM DICOMs use `PatientOrientation` (CC/MLO) and `BodyPartExamined`
    ("Left Breast" / "Right Breast") rather than the standard `ViewPosition`
    and `ImageLaterality` tags. `read_mammogram_dicom` tries both.
  * Full CBIS-DDSM DICOMs are ~25 MB uint16 4000+ px on the long side.
    Preprocessing runs on numpy, not torch, so it is CPU-bound; typical
    single-image preprocessing time is under 1 second on one core.
"""
from __future__ import annotations

from .laterality import (
    Laterality,
    detect_laterality_from_content,
    detect_laterality_from_metadata,
    orient_to_radiological_convention,
)
from .pipeline import MammogramMetadata, PreprocessedMammogram, preprocess_mammogram
from .reader import read_mammogram_dicom, read_mammogram_png
from .segmentation import breast_mask_otsu, remove_pectoral_mlo
from .view import View, detect_view_from_metadata

__all__ = [
    # readers
    "read_mammogram_dicom",
    "read_mammogram_png",
    # laterality
    "Laterality",
    "detect_laterality_from_metadata",
    "detect_laterality_from_content",
    "orient_to_radiological_convention",
    # view
    "View",
    "detect_view_from_metadata",
    # segmentation
    "breast_mask_otsu",
    "remove_pectoral_mlo",
    # pipeline
    "MammogramMetadata",
    "PreprocessedMammogram",
    "preprocess_mammogram",
]
