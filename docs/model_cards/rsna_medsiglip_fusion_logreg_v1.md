# RSNA MedSigLIP Fusion LogReg v1 (RUO)

## Summary

This card documents a **supervised fusion head** (logistic regression) fit on the **RSNA Breast Cancer Detection** cohort using:

- a **1,152‑D MedSigLIP‑448 embedding** (off‑label backbone), and
- a small **clinical/technical metadata vector** (BI‑RADS density + view + laterality + implant + site_id + machine_id + age/age²).

This is **not** the Google MedSigLIP base model card. For the foundation model card, see `docs/model_cards/medsiglip_448.md`.

**Honesty markers:** `RUO_DISCLAIMER`, `AUROC_CAVEAT`, `ModelState`.

## Model artifact (frozen weights)

- **Weights**: `models/rsna_medsiglip_fusion_logreg_v1.json`
- **Scorer**: `src/oncology_arbiter/models/rsna_medsiglip_fusion_logreg.py`
- **API route**: `POST /v1/screening/fusion` (RUO)

The JSON artifact contains the exact `StandardScaler` parameters and `LogisticRegression` coefficients used to compute the fused probability without requiring scikit‑learn at runtime.

## Data + split

- **Dataset**: RSNA Breast Cancer Detection (Kaggle) `train.csv`
- **Split**: patient‑disjoint via `GroupShuffleSplit(groups=patient_id, test_size=0.2, random_state=42)`
- **Scale**: 54,706 images total; test fold 10,962 images (192 positive)

## Performance (retrospective; image-level; RUO)

All numbers below are taken from the frozen artifact `models/rsna_medsiglip_fusion_logreg_v1.json` (`metrics` field).

- **Metadata-only AUROC**: 0.6191
- **MedSigLIP embedding-only AUROC**: 0.6650
- **Fused AUROC**: 0.6898

### Paired superiority test (patient-level paired bootstrap)

We report a **paired** patient-level bootstrap over the *same test fold* (B=2000 resamples, resampling patients with replacement; all images for a sampled patient are included):

- **ΔAUROC (fused − metadata)**: mean 0.0709, 95% CI [0.0251, 0.1174]
- **0/2000** bootstrap deltas ≤ 0 (artifact field `p_delta_le_zero = 0.0`) → **p < 0.0005** at this bootstrap resolution

## Intended use

- **Research use only**: comparative prototyping for screening triage pipelines.
- **Not** a diagnostic claim, not a treatment decision tool, not FDA‑cleared / CE‑marked.

## Key limitations

- **Off-label backbone**: MedSigLIP‑448 is not pretrained on screening mammography.
- **Image-level**: scoring is per image, not per breast/per exam/per patient clinical workflow.
- **Retrospective**: no prospective validation and no transportability guarantee across sites/devices.
- **Metadata dependence**: categorical covariates (site_id/machine_id/view) can encode acquisition confounding.

