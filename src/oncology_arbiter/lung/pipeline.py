"""NSCLC-track CT preprocessing + nodule-candidate heuristic.

**PLACEHOLDER-GRADE.** This is a threshold-based blob detector, not a
trained lung-nodule detector. Every response using it MUST self-flag with
`proxy_lung_heuristic` and a warning that names the placeholder status.
The MONAI lung-nodule detector will replace it later.

Pipeline stages:
    1. read_ct_series()                (CT DICOM directory  →  HU volume)
    2. lung_mask_from_hu(volume)        (body silhouette ∩ HU<-500 lung air)
    3. nodule_candidate_blobs(volume, lung_mask)
       (dilate lung mask by 3 voxels, then find connected components in
        the soft-tissue HU window intersected with the dilated envelope)
    4. summarize_candidates(labels, volume, spacing, top_n)
       (vectorized per-blob voxel count / mean HU / centroid; return the
        top N blobs by voxel count)

HU thresholds pinned to conventional pulmonary CT ranges (Fleischner /
LIDC-IDRI literature). Lung parenchyma sits in roughly ``[-1000, -500]``
HU; body soft tissue is above ``-500`` and below ``+200``; nodules are
soft-tissue voxels adjacent to aerated lung.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---- HU thresholds -------------------------------------------------------
#
# `LUNG_HU_MAX`: everything above this value is NOT lung air. We keep -500
#   rather than -400 because the transition band from parenchyma to
#   emphysematous soft tissue is centred near -500; a stricter -400 cutoff
#   throws away visible aerated lung.
#
# `BODY_HU_MIN`: below this is gantry air (or truncation artefact). Using
#   -900 puts the body silhouette threshold at the standard CT scanner
#   soft-tissue boundary for anthropomorphic segmentation.
#
# `NODULE_HU_MIN` / `NODULE_HU_MAX`: soft-tissue window used to search for
#   nodule candidates. Ground-glass nodules extend down to about -300;
#   solid nodules go up to about +200. Higher HU (contrast enhancement,
#   bone) is excluded.
LUNG_HU_MAX = -500.0
BODY_HU_MIN = -900.0
NODULE_HU_MIN = -300.0
NODULE_HU_MAX = 200.0

# Blob filter defaults. These are voxel-count bounds against the isotropic
# blob detector output — deliberately loose so the heuristic can surface
# candidates even on thick-slice acquisitions. Diameter conversion happens
# in `summarize_candidates` using the true spacing.
DEFAULT_MIN_VOXELS = 8
DEFAULT_MAX_VOXELS = 20_000
DEFAULT_DILATE_ITER = 3
DEFAULT_TOP_N = 10


@dataclass
class NoduleCandidate:
    """One blob returned by the heuristic."""
    label: int
    voxel_count: int
    diameter_mm: float
    mean_hu: float
    centroid_zyx_vox: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "voxel_count": self.voxel_count,
            "diameter_mm": float(self.diameter_mm),
            "mean_hu": float(self.mean_hu),
            "centroid_z_vox": float(self.centroid_zyx_vox[0]),
            "centroid_y_vox": float(self.centroid_zyx_vox[1]),
            "centroid_x_vox": float(self.centroid_zyx_vox[2]),
        }


@dataclass
class LungHeuristicOutput:
    """Return value of `run_lung_heuristic()`. Placeholder-grade."""
    lung_voxel_fraction: float
    n_candidates_total: int
    n_candidates_kept: int
    candidates: list[NoduleCandidate]
    max_diameter_mm: float
    spacing_mm: tuple[float, float, float]
    hu_thresholds: dict = field(default_factory=lambda: {
        "lung_hu_max": LUNG_HU_MAX,
        "body_hu_min": BODY_HU_MIN,
        "nodule_hu_min": NODULE_HU_MIN,
        "nodule_hu_max": NODULE_HU_MAX,
    })

    def as_dict(self) -> dict:
        return {
            "lung_voxel_fraction": float(self.lung_voxel_fraction),
            "n_candidates_total": int(self.n_candidates_total),
            "n_candidates_kept": int(self.n_candidates_kept),
            "max_diameter_mm": float(self.max_diameter_mm),
            "spacing_mm": list(self.spacing_mm),
            "hu_thresholds": dict(self.hu_thresholds),
            "candidates": [c.as_dict() for c in self.candidates],
        }


# ------------------------------------------------------------------ masks
def _body_silhouette_2d(slice_hu: np.ndarray) -> np.ndarray:
    """Per-slice body silhouette: HU>BODY_HU_MIN, largest CC, hole-filled.

    This is what a radiologist would draw around the patient's torso —
    everything that isn't gantry air. Hole-filling means the lung interior
    (which is HU < -500) is treated as INSIDE the body, so the subsequent
    lung mask can be built as `body & (HU < LUNG_HU_MAX)`.
    """
    from scipy import ndimage as ndi
    raw = slice_hu > BODY_HU_MIN
    if not raw.any():
        return np.zeros_like(raw, dtype=bool)
    labels, n = ndi.label(raw)
    if n == 0:
        return np.zeros_like(raw, dtype=bool)
    # largest connected component
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    biggest = int(np.argmax(sizes))
    body = labels == biggest
    body = ndi.binary_fill_holes(body)
    return body


def lung_mask_from_hu(volume: np.ndarray) -> np.ndarray:
    """Return a boolean mask of aerated lung inside the body silhouette.

    volume : (Z, Y, X) float32 HU array
    returns: (Z, Y, X) bool array
    """
    z = volume.shape[0]
    mask = np.zeros_like(volume, dtype=bool)
    for i in range(z):
        body = _body_silhouette_2d(volume[i])
        mask[i] = body & (volume[i] < LUNG_HU_MAX)
    return mask


def _dilate_mask(mask: np.ndarray, iterations: int = DEFAULT_DILATE_ITER) -> np.ndarray:
    from scipy import ndimage as ndi
    if iterations <= 0:
        return mask
    return ndi.binary_dilation(mask, iterations=iterations)


# ---------------------------------------------------------------- blob detect
def nodule_candidate_blobs(
    volume: np.ndarray,
    lung_mask: np.ndarray,
    *,
    dilate_iter: int = DEFAULT_DILATE_ITER,
) -> tuple[np.ndarray, int]:
    """Return (labels, n_labels) for candidate blobs.

    We dilate the lung mask by `dilate_iter` voxels so nodules touching the
    pleural surface or adjacent to a vessel are still inside the search
    envelope, then intersect with the soft-tissue HU window and label
    connected components with 6-connectivity.
    """
    from scipy import ndimage as ndi
    dilated_lung = _dilate_mask(lung_mask, iterations=dilate_iter)
    cand = dilated_lung & (volume > NODULE_HU_MIN) & (volume < NODULE_HU_MAX)
    labels, n = ndi.label(cand)
    return labels, int(n)


def _isotropic_diameter_mm(n_voxels: int, spacing_mm: tuple[float, float, float]) -> float:
    """Convert voxel count → equivalent-sphere diameter in mm.

    volume_mm3 = n_voxels * dz*dy*dx ;  d = 2 * (3V / 4π)^(1/3)
    """
    dz, dy, dx = spacing_mm
    v_mm3 = float(n_voxels) * float(dz) * float(dy) * float(dx)
    if v_mm3 <= 0.0:
        return 0.0
    r = (3.0 * v_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return 2.0 * r


def summarize_candidates(
    labels: np.ndarray,
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    min_voxels: int = DEFAULT_MIN_VOXELS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[list[NoduleCandidate], int, int]:
    """Vectorized per-blob summary.

    Returns (candidates_top_n, n_total_blobs, n_kept_after_filter).
    """
    from scipy.ndimage import center_of_mass, mean as ndi_mean

    if labels.size == 0:
        return [], 0, 0

    counts = np.bincount(labels.ravel())
    n_total = int((counts > 0).sum() - 1)  # exclude background label 0
    if n_total <= 0:
        return [], 0, 0

    # Filter by voxel-count bounds BEFORE the expensive COM/mean calls.
    keep_indices = np.where((counts >= min_voxels) & (counts <= max_voxels))[0]
    keep_labels = [int(l) for l in keep_indices if l != 0]
    n_kept = len(keep_labels)
    if n_kept == 0:
        return [], n_total, 0

    # Sort kept labels by voxel count (desc) and keep the top_n.
    keep_labels.sort(key=lambda l: int(counts[l]), reverse=True)
    keep_labels = keep_labels[: max(top_n, 0)]
    if not keep_labels:
        return [], n_total, n_kept

    # Vectorized: one call each computes properties for all requested labels.
    coms = center_of_mass(labels > 0, labels=labels, index=keep_labels)
    mean_hus = ndi_mean(volume, labels=labels, index=keep_labels)

    # scipy returns a single tuple when index has 1 element — normalize.
    if not isinstance(coms, list):
        coms = [coms] if len(keep_labels) == 1 else list(coms)
    if np.ndim(mean_hus) == 0:
        mean_hus = [float(mean_hus)]

    out: list[NoduleCandidate] = []
    for label, com, mhu in zip(keep_labels, coms, mean_hus):
        n_vox = int(counts[label])
        d_mm = _isotropic_diameter_mm(n_vox, spacing_mm)
        centroid = (float(com[0]), float(com[1]), float(com[2]))
        out.append(
            NoduleCandidate(
                label=int(label),
                voxel_count=n_vox,
                diameter_mm=d_mm,
                mean_hu=float(mhu),
                centroid_zyx_vox=centroid,
            )
        )
    return out, n_total, n_kept


# ----------------------------------------------------------------- top-level
def run_lung_heuristic(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    dilate_iter: int = DEFAULT_DILATE_ITER,
    min_voxels: int = DEFAULT_MIN_VOXELS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    top_n: int = DEFAULT_TOP_N,
    lung_mask: Optional[np.ndarray] = None,
) -> LungHeuristicOutput:
    """End-to-end placeholder heuristic.

    Parameters
    ----------
    volume : (Z, Y, X) float32 array of Hounsfield units
    spacing_mm : (dz, dy, dx) in mm
    dilate_iter : voxels to dilate the lung mask before candidate search
    min_voxels / max_voxels : voxel-count filter for candidate blobs
    top_n : keep at most this many candidates (sorted by voxel count desc)
    lung_mask : optional pre-computed lung mask; if not given we build one

    Returns
    -------
    LungHeuristicOutput
    """
    if lung_mask is None:
        lung_mask = lung_mask_from_hu(volume)
    total_voxels = volume.size
    lung_frac = float(lung_mask.sum()) / float(max(total_voxels, 1))
    labels, _n_labels = nodule_candidate_blobs(
        volume, lung_mask, dilate_iter=dilate_iter
    )
    cands, n_total, n_kept = summarize_candidates(
        labels,
        volume,
        spacing_mm,
        min_voxels=min_voxels,
        max_voxels=max_voxels,
        top_n=top_n,
    )
    max_d = max((c.diameter_mm for c in cands), default=0.0)
    return LungHeuristicOutput(
        lung_voxel_fraction=lung_frac,
        n_candidates_total=n_total,
        n_candidates_kept=n_kept,
        candidates=cands,
        max_diameter_mm=max_d,
        spacing_mm=spacing_mm,
    )


__all__ = [
    "LUNG_HU_MAX",
    "BODY_HU_MIN",
    "NODULE_HU_MIN",
    "NODULE_HU_MAX",
    "DEFAULT_MIN_VOXELS",
    "DEFAULT_MAX_VOXELS",
    "DEFAULT_DILATE_ITER",
    "DEFAULT_TOP_N",
    "NoduleCandidate",
    "LungHeuristicOutput",
    "lung_mask_from_hu",
    "nodule_candidate_blobs",
    "summarize_candidates",
    "run_lung_heuristic",
]
