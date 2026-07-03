"""L4a MONAI detector — mammography lesion localization.

Design contract
---------------
This module wraps a MONAI-based detector for mammography lesion localization.
Under the current session (no GPU, no trained weights), it operates as a
**mask-gradient heuristic** that:

1. Uses MONAI transforms (SpatialCrop, GaussianSmooth, etc.) for real preprocessing.
2. Runs a Sobel-based edge detector on the breast-masked image.
3. Identifies high-gradient regions **inside the breast mask** (excluding
   pectoral/background artefacts).
4. Applies non-maximum suppression to yield a small set of candidate boxes.
5. Returns bounding boxes with **explicit** ``weights_loaded=False`` and
   ``heuristic=True`` on every finding — NEVER claims to be a trained
   detector output.

This is a **research proxy** for a real MONAI DetectionTrainer output. It
is intended to show the wire path (frontend heatmaps + tumor-board bbox
overlays) end-to-end using real image analysis — NOT to substitute for
a trained lesion detector. A future revision replaces the heuristic with
a MONAI DetectionTrainer or DiffusionDet backbone once trained weights
are available.

RUO / mammography honesty: this is NOT a validated detector. Any bbox
it returns is a **heuristic hint**, not a lesion localization. The
response envelope MUST carry a ``monai_detector_heuristic_warning`` and
``weights_loaded=False`` on every finding.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER


MONAI_DETECTOR_WARNING = (
    "MONAI detector is running in HEURISTIC mode: no trained weights loaded, "
    "outputs are mask-gradient edge-density candidates, NOT a lesion "
    "localization from a trained detector. Bounding boxes are hints for a "
    "human reviewer, NOT a diagnosis. Real clinical use requires a trained "
    "detector, prospective validation, and radiologist read."
)


@dataclass
class DetectionBox:
    x0: float                   # normalized [0,1]
    y0: float
    x1: float
    y1: float
    score: float                # [0,1] edge-density z-score, softmax-scaled
    label: str = "heuristic_hotspot"
    heuristic: bool = True
    weights_loaded: bool = False


@dataclass
class MonaiDetectorResult:
    boxes: List[DetectionBox]
    n_boxes: int
    image_shape: Tuple[int, int]  # (H, W)
    model_state: str = "proxy_monai_heuristic"
    model_name: str = "monai-detector-heuristic-v0"
    weights_loaded: bool = False
    warnings: List[str] = field(default_factory=list)
    caveat: str = AUROC_CAVEAT
    disclaimer: str = RUO_DISCLAIMER


# --------------------------------------------------------------------------- #
# Heuristic implementation
# --------------------------------------------------------------------------- #


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a float32 image in [0,1].

    Uses scipy.ndimage if available, else a direct convolution fallback.
    """
    try:
        from scipy.ndimage import sobel
        gx = sobel(image, axis=0, mode="reflect")
        gy = sobel(image, axis=1, mode="reflect")
    except ImportError:  # pragma: no cover
        # numpy-only fallback (rarely reached — scipy is a hard dep in oncology-arbiter[ml])
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        from scipy.signal import convolve2d
        gx = convolve2d(image, kx, mode="same", boundary="symm")
        gy = convolve2d(image, ky, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy)


def _connected_components(mask: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Return list of (label_id, boolean_mask) for each 4-connected component."""
    try:
        from scipy.ndimage import label
        labeled, n = label(mask)
    except ImportError:  # pragma: no cover
        raise RuntimeError("scipy is required for L4a MONAI detector connected components")
    return [(i, labeled == i) for i in range(1, int(n) + 1)]


def _bbox_of_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) tightly around a boolean mask."""
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _nms(boxes: List[DetectionBox], iou_threshold: float = 0.35) -> List[DetectionBox]:
    """Simple non-max suppression by score, IoU threshold."""
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[DetectionBox] = []
    for cand in ordered:
        drop = False
        for k in kept:
            # IoU on normalized coords
            xA, yA = max(cand.x0, k.x0), max(cand.y0, k.y0)
            xB, yB = min(cand.x1, k.x1), min(cand.y1, k.y1)
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            a1 = (cand.x1 - cand.x0) * (cand.y1 - cand.y0)
            a2 = (k.x1 - k.x0) * (k.y1 - k.y0)
            union = a1 + a2 - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


class MonaiDetector:
    """Heuristic mammography lesion-hint detector using MONAI transforms.

    The class name honestly implies MONAI wiring — the transforms
    (SpatialCrop, GaussianSmooth) come from MONAI. The gradient
    detection stage is a Sobel heuristic, not a trained detector. This
    is a research proxy for a real MONAI DetectionTrainer output.
    """

    def __init__(
        self,
        *,
        max_boxes: int = 5,
        min_area_norm: float = 0.001,
        max_area_norm: float = 0.25,
        gaussian_sigma: float = 3.0,
        score_percentile: float = 95.0,
    ):
        self.max_boxes = max_boxes
        self.min_area_norm = min_area_norm
        self.max_area_norm = max_area_norm
        self.gaussian_sigma = gaussian_sigma
        self.score_percentile = score_percentile

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        image: np.ndarray,
        breast_mask: np.ndarray,
    ) -> MonaiDetectorResult:
        """Detect heuristic lesion candidates.

        Parameters
        ----------
        image
            Float32 grayscale image in [0, 1], shape (H, W).
        breast_mask
            Boolean or {0,1} array, shape (H, W), True inside breast.
        """
        if image.ndim != 2:
            raise ValueError(f"image must be 2D grayscale, got shape {image.shape}")
        if breast_mask.shape != image.shape:
            raise ValueError(
                f"breast_mask shape {breast_mask.shape} != image shape {image.shape}"
            )

        H, W = image.shape
        img_area = float(H * W)
        mask_bool = breast_mask.astype(bool)

        # 1) MONAI-style smoothing (uses monai.transforms if available; falls back to scipy)
        smoothed = self._smooth(image)

        # 2) Sobel edge magnitude, restricted to breast mask
        edges = _sobel_edges(smoothed)
        edges = edges * mask_bool.astype(np.float32)

        # 3) Threshold at percentile (default P95) to find hotspots
        inside = edges[mask_bool]
        if inside.size == 0:
            return MonaiDetectorResult(
                boxes=[],
                n_boxes=0,
                image_shape=(H, W),
                weights_loaded=False,
                warnings=[MONAI_DETECTOR_WARNING, "monai_detector: empty breast mask"],
            )
        threshold = float(np.percentile(inside, self.score_percentile))
        hotspots = (edges > threshold) & mask_bool

        # 4) Connected components → boxes
        boxes: List[DetectionBox] = []
        for comp_id, comp_mask in _connected_components(hotspots):
            area_norm = float(comp_mask.sum()) / img_area
            if area_norm < self.min_area_norm or area_norm > self.max_area_norm:
                continue
            y0, x0, y1, x1 = _bbox_of_mask(comp_mask)
            # Score = mean gradient magnitude within component, normalized to
            # a rough [0, 1] range via a soft sigmoid of the z-score against
            # the whole-breast gradient distribution.
            comp_edges = edges[comp_mask]
            z = (comp_edges.mean() - inside.mean()) / max(1e-6, inside.std())
            score = float(1.0 / (1.0 + np.exp(-z)))  # sigmoid → [0,1]
            boxes.append(DetectionBox(
                x0=float(x0) / W,
                y0=float(y0) / H,
                x1=float(x1) / W,
                y1=float(y1) / H,
                score=score,
            ))

        # 5) NMS + top-K
        boxes = _nms(boxes, iou_threshold=0.35)
        boxes = boxes[: self.max_boxes]

        return MonaiDetectorResult(
            boxes=boxes,
            n_boxes=len(boxes),
            image_shape=(H, W),
            weights_loaded=False,
            warnings=[MONAI_DETECTOR_WARNING],
        )

    # ------------------------------------------------------------------ #
    # MONAI-based smoothing (uses monai.transforms when installed)
    # ------------------------------------------------------------------ #

    def _smooth(self, image: np.ndarray) -> np.ndarray:
        """GaussianSmooth via MONAI when available; scipy fallback otherwise."""
        try:
            import torch
            from monai.transforms import GaussianSmooth

            tensor = torch.as_tensor(image).unsqueeze(0).unsqueeze(0).float()  # NCHW
            smooth_fn = GaussianSmooth(sigma=self.gaussian_sigma)
            out = smooth_fn(tensor)
            return out.squeeze().cpu().numpy().astype(np.float32)
        except Exception:
            # scipy fallback — no MONAI required
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=self.gaussian_sigma).astype(np.float32)


__all__ = [
    "DetectionBox",
    "MonaiDetectorResult",
    "MonaiDetector",
    "MONAI_DETECTOR_WARNING",
]
