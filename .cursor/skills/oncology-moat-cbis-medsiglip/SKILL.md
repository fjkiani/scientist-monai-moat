---
name: oncology-moat-cbis-medsiglip
description: >-
  Forces CBIS/MedSigLIP real retraining tonight past the frozen Jul-10 joblib.
  Aligns arbriter 0.75 AUC tables and
  WEIGHT_REDISTRIBUTION 0.85 floor. Bans NBIA rescoring as training.
---

# Oncology Moat — CBIS / MedSigLIP (TONIGHT)

**RUO.** Floor 0.85 for word "screening". Live ~0.73–0.76 = **BELOW**.

## Canon numbers (do not headline higher)

| Source | AUC |
|--------|-----|
| `docs/proofs/cbis_ddsm_logreg_v1_metrics.json` | ~0.753 |
| `docs/proofs/cbis_ddsm_nbia_dicom_v1_metrics.json` | ~0.730 frozen |
| `arbriter/report_wave_reports_index.md` | v2 0.7532 cross-check |
| Finding I | train script `C=1.0` ≠ shipped `C=0.01` — reproducibility gap |

## Forbidden

- Frozen rescoring = training
- Chasing 0.85 with threshold games tonight without new weights
- Leaving API/docs saying screening while floor FAIL

## Tonight bar — train, wire, evaluate, push

1. Acquire/verify the CBIS training corpus and patient-level split manifest.
   Missing data must be downloaded immediately; it is not a blocker label.
2. Regenerate 1152-d MedSigLIP embeddings for train and held-out patients.
3. Fix `scripts/train_cbis_ddsm_logreg.py` so its hyperparameters reproduce
   the selected fit; run a genuinely new fit.
4. Commit `models/cbis_ddsm_logreg_v2.joblib`, weight SHA, corpus/split SHA,
   patient-grouped CV, and held-out AUROC with CI.
5. Wire the product to v2 and prove the loaded SHA in an integration test.
6. Push. If AUROC remains below 0.85, report the failure honestly **while
   preserving the trained capability** and continue optimization—do not
   delete, gate, park, or relabel the capability as the deliverable.

Forbidden exits: claim deletion, alternate no-train path, status-label parking,
frozen rescoring, or threshold tuning presented as model improvement.

## DONE

| Gate | Evidence |
|------|----------|
| New v2 weight + sha256 | |
| Corpus/split sha + patient-disjoint held-out metric | |
| Product loads v2 SHA | integration test |
| `src/`/`deploy/`/`models/` in `git show --stat` | |
| tip moved | |
