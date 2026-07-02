"""Breast tissue segmentation + pectoral muscle removal (MLO views).

Both routines are pure-numpy so they run without a GPU. They intentionally
use classical algorithms (Otsu threshold + largest connected component,
region-growing from an MLO corner) rather than a learned model because:

  1. A first-line preprocessing step should be deterministic, fast, and
     inspectable. Every downstream model will re-crop or re-mask anyway.
  2. Otsu + largest-CC is a well-established mammography baseline that
     works reliably on properly-oriented images.
  3. Pectoral removal on MLO views is a geometric problem; a learned
     segmenter here would be overkill.
"""
from __future__ import annotations

import numpy as np


def breast_mask_otsu(arr: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the breast tissue.

    Steps:
      1. Otsu threshold on the pixel histogram.
      2. Keep only the largest connected component (excludes label bars,
         speckle noise, and small foreign objects like acquisition markers).

    Returns:
        Boolean mask, True where tissue is.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    threshold = _otsu_threshold(arr)
    binary = arr > threshold
    return _largest_connected_component(binary)


def remove_pectoral_mlo(
    arr: np.ndarray,
    laterality: str = "L",
    *,
    corner_fraction: float = 0.35,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Zero out the pectoral muscle triangle on an MLO view.

    Precondition: `arr` is in the project orientation convention (see
    `oncology_arbiter.mammography.laterality`):
      * LEFT breast: tissue on LEFT half → chest wall/pectoral on RIGHT
        → pectoral triangle at TOP-RIGHT corner.
      * RIGHT breast: tissue on RIGHT half → chest wall/pectoral on LEFT
        → pectoral triangle at TOP-LEFT corner.

    We use a simple region-growing algorithm rooted at the corner:
      1. Compute a bright-pixel mask (top 25th percentile of tissue).
      2. Flood-fill from the appropriate top corner within the mask.
      3. Zero the filled region.

    Args:
        arr: 2D float32 image in [0, 1].
        laterality: "L" or "R" — controls which corner is filled from.
        corner_fraction: Only pixels within this fraction of the top and
            of the chest-wall side are eligible for flood-fill. This
            bounds the pectoral region so we cannot accidentally erase
            the whole breast.
        fill_value: value written into the removed region.

    Returns:
        A copy of `arr` with the pectoral region set to `fill_value`.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    h, w = arr.shape
    laterality = str(laterality).upper()[:1]
    if laterality not in ("L", "R"):
        return arr.copy()

    # Bright-pixel mask: pectoral muscle is brighter than surrounding tissue
    tissue = arr > 0.02
    if tissue.sum() == 0:
        return arr.copy()
    tissue_vals = arr[tissue]
    bright_thresh = float(np.percentile(tissue_vals, 75))
    bright = arr >= bright_thresh

    # Restrict candidacy to the pectoral corner region
    corner = np.zeros_like(bright, dtype=bool)
    top_h = int(h * corner_fraction)
    side_w = int(w * corner_fraction)
    if laterality == "L":
        # Pectoral triangle is TOP-RIGHT
        corner[:top_h, w - side_w :] = True
        seed = (0, w - 1)
    else:
        # Pectoral triangle is TOP-LEFT
        corner[:top_h, :side_w] = True
        seed = (0, 0)

    candidate = bright & corner
    if not candidate[seed]:
        # The corner pixel itself might not be bright; look for the closest
        # bright pixel within a small neighborhood of the seed.
        srow_max = min(top_h, 50)
        scol_range = (
            range(w - side_w, w) if laterality == "L" else range(0, side_w)
        )
        found = False
        for r in range(srow_max):
            for c in scol_range:
                if candidate[r, c]:
                    seed = (r, c)
                    found = True
                    break
            if found:
                break
        if not found:
            return arr.copy()

    # BFS flood-fill from seed within `candidate`
    filled = _bfs_flood(candidate, seed)
    out = arr.copy()
    out[filled] = fill_value
    return out


# --------------------------------------------------------------------------- #
# Internal helpers


def _otsu_threshold(arr: np.ndarray, nbins: int = 256) -> float:
    """Otsu's method — no scikit-image dependency."""
    a = arr.ravel()
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(a, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(nbins))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b2))
    return float(edges[k])


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """Return the largest True-region as a bool mask; scipy.ndimage-free."""
    h, w = binary.shape
    label = np.zeros(binary.shape, dtype=np.int32)
    next_label = 1
    sizes: dict[int, int] = {}
    from collections import deque

    for i in range(h):
        for j in range(w):
            if binary[i, j] and label[i, j] == 0:
                # BFS
                queue: deque[tuple[int, int]] = deque([(i, j)])
                label[i, j] = next_label
                count = 0
                while queue:
                    y, x = queue.popleft()
                    count += 1
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary[ny, nx]
                            and label[ny, nx] == 0
                        ):
                            label[ny, nx] = next_label
                            queue.append((ny, nx))
                sizes[next_label] = count
                next_label += 1

    if not sizes:
        return np.zeros_like(binary, dtype=bool)
    biggest = max(sizes.items(), key=lambda kv: kv[1])[0]
    return label == biggest


def _bfs_flood(candidate: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """4-connected BFS from `seed` within `candidate` (True pixels only)."""
    from collections import deque

    h, w = candidate.shape
    out = np.zeros_like(candidate, dtype=bool)
    if not candidate[seed]:
        return out
    queue: deque[tuple[int, int]] = deque([seed])
    out[seed] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((ny, nx))
    return out
