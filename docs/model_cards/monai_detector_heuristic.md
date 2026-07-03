# Model card: monai-detector-heuristic (L4a)

**Version**: `monai-detector-heuristic-v0` (2026-07-03)
**Backbone**: MONAI `transforms` (SpatialCrop, GaussianSmooth) + Sobel edges
**Head**: connected-components + non-max suppression on gradient hotspots
**Status**: **RESEARCH USE ONLY** — NOT FDA cleared. `weights_loaded=False`.
Heuristic mode ONLY under this build.

## Intended use

Given a preprocessed mammogram + breast mask (already produced by
`oncology_arbiter.mammography.preprocess_mammogram`), return a small set
of **bounding-box hints** identifying high-edge-density regions inside
the breast. These are *hints for a human reviewer*, NOT a trained
detector's lesion localization.

Feeds into the tumor-board UX so a radiologist can quickly triage
possible focal densities / masses on the screening view.

## What this model is NOT

* **Not a trained lesion detector**. There are no learned weights. The
  algorithm is: MONAI GaussianSmooth → Sobel gradient → percentile
  threshold inside the breast mask → connected components → area
  filter → non-max suppression → top-K.
* **Not a substitute for CAD or radiologist read**. Any bbox is a
  hint; it may hit calcifications, dense parenchyma, imaging
  artefacts, or pectoral edges that leaked past the mask.
* **Not validated on any breast-cancer benchmark**. No AUROC / mAP /
  sensitivity is claimed. The AUROC caveat constant applies verbatim.

## Wire contract

* Env flag: `ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR=1`
* Endpoint: `POST /v1/screening/analyze`
* Output: adds `ScreeningFinding` entries with:
  * `label` prefixed by `monai_heuristic:` (typically `monai_heuristic:heuristic_hotspot`)
  * `score ∈ [0, 1]` — sigmoid of the region's edge-density z-score
  * `location_bbox_normalized: [x0, y0, x1, y1]` in [0, 1]
* If MedSigLIP + SigLIP-proxy both off, `provenance.model_state`
  becomes `proxy_monai_heuristic`; otherwise it keeps the primary
  backend's state and adds MONAI findings alongside.

## Failure modes

| Scenario | Response |
|----------|----------|
| Flag off | No `monai_heuristic:*` findings; provenance unchanged |
| Flag on + empty mask | `monai_detector: empty breast mask` warning; 0 findings |
| Flag on + normal mammogram | 1-5 heuristic findings with bboxes and heuristic warning |
| Flag on + exception | `monai_detector_error:<Type>:<msg>` warning; no findings |

## Honesty warning (verbatim, emitted on every run)

> MONAI detector is running in HEURISTIC mode: no trained weights
> loaded, outputs are mask-gradient edge-density candidates, NOT a
> lesion localization from a trained detector. Bounding boxes are
> hints for a human reviewer, NOT a diagnosis. Real clinical use
> requires a trained detector, prospective validation, and radiologist
> read.

## Reproducibility

* Deterministic: same `(image, breast_mask)` → same boxes.
* Default parameters:
  * `max_boxes=5`, `gaussian_sigma=3.0`, `score_percentile=95.0`
  * `min_area_norm=0.001`, `max_area_norm=0.25` (area as fraction of image)
  * NMS IoU threshold: 0.35

RESEARCH USE ONLY. See `AUROC_CAVEAT` and `RUO_DISCLAIMER` in
`src/oncology_arbiter/__init__.py`.
