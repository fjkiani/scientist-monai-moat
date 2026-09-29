---
name: oncology-moat-biopsy
description: >-
  Forces real MedSigLIP-1152 biopsy head tonight on scientist-monai-moat.
  Aligns quarantine_receipt + arbriter ClinicalBERT/biopsy honesty tables.
  Bans synthetic biopsy_probe_v0 and Wave report AUROC without weights on HEAD.
  Use when biopsy probe, L4b, IDC/DCIS, or n_training_synthetic.
---

# Oncology Moat — Biopsy Probe (TONIGHT)

**RUO.** Deadline: **tonight**. Quarantine ≠ product.

## Canon

- `artifacts/audit/quarantine_receipt.json` — synthetic 768-d, 0 real slides
- `src/oncology_arbiter/arbiter/models/biopsy_probe_v0.json` — still on HEAD
- Wave reports' biopsy AUROC 0.92 refer to **stage logistic** or missing artifacts — **not** this probe file

## Forbidden

- Quarantine receipt as DONE
- Another 768-d head
- Metrics without `biopsy_probe_v1` on git
- Five-month "Phase 4 TCGA-BRCA" deferral when Alpha said tonight

## Tonight bar — one terminal condition

1. Confirm corpus on disk. If absent, acquire it immediately and continue in
   the same run; dataset acquisition is a step, never a status or deliverable.
2. Embed MedSigLIP **1152-d**; train 3-class head; patient-disjoint split.
3. Commit `src/oncology_arbiter/arbiter/models/biopsy_probe_v1.json` (+ wire loader).
4. `n_training_synthetic: false`, `embed_dim: 1152`, sha256 in reply.
5. Run held-out evaluation and integration test proving the product loads v1.
6. Push (`anti-fortress`).

Forbidden exits: missing-corpus status, download-PID-as-delivery, tombstone,
capability removal, v0 rejection without v1, or synthetic fallback.

## DONE table

| Gate | Evidence |
|------|----------|
| v1 on HEAD and product loads v1 | |
| embed_dim 1152 / synthetic false | |
| n_train_real + patient-disjoint held-out metrics | |
| sha256 | |
| tip moved | `git ls-remote` |
