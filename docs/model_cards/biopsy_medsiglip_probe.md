# Model card: biopsy-medsiglip-probe (L4b)

**Version**: `biopsy_probe_v0` (2026-07-03)
**Backbone**: `google/medsiglip-448` (HAI-DEF gated, requires HF_TOKEN + accepted terms)
**Head**: linear (3-class) + Platt temperature scaling
**Status**: **RESEARCH USE ONLY** — NOT FDA cleared. Weights are **synthetic**.

## Intended use

Given a breast biopsy image (WSI patch or gross photo), return one of three
subtype labels:

| Class  | Meaning                        |
|--------|--------------------------------|
| IDC    | Invasive Ductal Carcinoma      |
| DCIS   | Ductal Carcinoma In Situ       |
| benign | Non-malignant / normal tissue  |

The response includes a calibrated probability for each class, an
inline honesty warning that the head is synthetic, and a HAI-DEF
`gate_report` documenting whether the encoder actually loaded.

## What this model is NOT

* **Not a WSI reader**. There is no OpenSlide integration. The image is
  fed directly to the MedSigLIP-448 vision encoder. For a real slide-level
  workflow you must tile the WSI upstream and aggregate predictions.
* **Not trained on real patient data**. The linear head's weights are
  drawn from `numpy.random.default_rng(seed=20260703)`, orthogonalized via
  Gram-Schmidt, and biased by class priors. `n_training=48` is a placeholder;
  `n_training_synthetic=True` is recorded verbatim in
  `arbiter/models/biopsy_probe_v0.json`.
* **Not a substitute for pathology review**. Every recommendation MUST be
  reviewed by a certified pathologist.

## AUROC

The response envelope carries the shared `AUROC_CAVEAT` constant. **No
mammography AUROC is claimed for MedSigLIP-448**; the only Google-published
breast-related AUROC for MedSigLIP is on **histopathology** (Invasive
Breast Cancer, n=5000, 3 classes, zero-shot 0.933 / linear-probe 0.930 /
HAI-DEF LP 0.943 — Google MedSigLIP model card).

## Wire contract

* Env flag: `ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP=1`
* Endpoint: `POST /v1/biopsy/analyze`
* Input: `wsi_url` OR `wsi_bytes_b64` OR `report_text` (must supply at least one)
* Output: `BiopsyResponse` with
  * `subtype_prediction ∈ {IDC, DCIS, benign, null}`
  * `confidence ∈ [0, 1]`
  * `provenance.model_state ∈ {placeholder, loaded_biopsy_probe, gated}`
  * `warnings` — always includes `biopsy_probe_synthetic_weights: n_training=48 synthetic=True`

## Failure modes

| Preflight outcome         | Response                                  |
|---------------------------|-------------------------------------------|
| HAI-DEF ALLOWED           | `loaded_biopsy_probe` + subtype           |
| HAI-DEF FORBIDDEN (403)   | `gated` + `biopsy_medsiglip_gated:forbidden:...` warning |
| HAI-DEF UNAUTHENTICATED   | `gated` + `biopsy_medsiglip_gated:unauthenticated:...` warning |
| Report-text-only input    | `placeholder` + `biopsy_medsiglip_skipped:report_text_only:no_image_provided` warning |
| Any other exception       | `placeholder` + `biopsy_medsiglip_error:<Type>:<msg>` warning |

## Reproducibility

* Weights file: `src/oncology_arbiter/arbiter/models/biopsy_probe_v0.json`
  (64,424 bytes; sha256 of committed file recorded in git)
* Weight generation seed: `numpy.random.default_rng(seed=20260703)`
* Temperature (Platt): 1.2
* Class biases: `[IDC=-0.5, DCIS=-0.2, benign=+0.3]`

RESEARCH USE ONLY. See `AUROC_CAVEAT` and `RUO_DISCLAIMER` in
`src/oncology_arbiter/__init__.py`.
