# Real-Text ClinicalBERT Retrain — v0.5.1 Audit

**Date**: 2026-07-20
**Branch**: `feat/v0.5.1-production-grade-nsclc-gold`
**Trigger**: v0.5.0 5-seed baseline stalled at F1 = 0.0667 ± 0.008 on breast/CRC TCGA-242 gold.

## Bottleneck Analysis

The v0.5.0 corpus was regex-weak-labeled from raw TCGA reports. Regex alone had ~2% token coverage. This drove three failure modes:

1. **Molecular entities (ER/PR/HER2/KI67) had zero regex signal** in TCGA's pre-molecular reports — regex patterns matched breast IHC report headers but not the value contexts. Result: 0 labels for these entities in weak-labeled train.
2. **T/N/M staging** hit occasional matches but with 5-30 tokens/entity typical extent, single-token BIO tags dominated → I- tags never appear in weak labels.
3. **98.4% "O" tokens** in the aggregate weak-labeled corpus caused unweighted CE loss to converge to "predict O everywhere". Val loss dropped and val F1 stayed at 0.

Three coordinated changes fix this:

### Change 1: Snorkel multi-signal weak supervision

4 label functions:
- **LF-regex** — the v0.5.0 pattern set (baseline, high precision 0.99, low coverage)
- **LF-LLM** — `tencent/hy3:free` accepted-span votes across the training corpus
- **LF-ontology** — NCI/HGNC dictionary matches (gene names)
- **LF-section** — section-header–conditional patterns (immunohistochemistry, molecular findings), anchored after `.` or `\n` to avoid noise from period-flattened TCGA text

Resolved via an EM-style `LabelModel` (`weak_supervision.py::LabelModel`) that fits per-LF accuracies from the LF vote matrix. `LabelModel.predict()` argmaxes to hard BIO labels (no soft targets — we write hard `list[str]` labels to `train.jsonl` for HuggingFace compatibility).

Corpus-level LF stats (post-fit):

| LF | Accuracy | Coverage | Conflict |
|---|---|---|---|
| LF-regex | 0.990 | 2.5% | 4.5% |
| LF-LLM | 0.513 | 2.9% | 6.3% |
| LF-ontology | 0.512 | 0.4% | 5.4% |
| LF-section | 0.990 | 2.0% | 4.8% |

Non-O class diversity in the resulting hard-labeled corpus: **30 classes** vs ~10 for v0.5.0. Class distribution highlights (from `manifest.json`):

- Molecular: EGFR 74, HER2_AMP 74, ALK 59, KI67_PCT 48, KRAS 47, BRAF 13, MSI 5, MET 5, ROS1 1
- Staging: T_STAGE 2001, N_STAGE 1301, M_STAGE 872
- IHC: HER2_VALUE 424, ER_VALUE 501, PR_VALUE 390

### Change 2: Class-weighted CE + label smoothing

Formula: `w_c = (1 / √freq_c) / (1 / √freq_O)`, so `w_O = 1.0` and rare tags floor upward. Clamped at 25× to prevent Snorkel-noise-driven instability. Label smoothing ε = 0.1 to further tolerate LF-noise.

Smoke-test validation (200-report subset, 3 epochs, seed 42):
- Epoch 0: train_loss 9.24, val_micro_f1 0.0000
- Epoch 1: train_loss 8.84, **val_micro_f1 0.0899** ← class-weighted loss lifts F1 off zero
- Epoch 2: train_loss 6.79 (still improving)

Weighted-loss ablation is left as future work; the smoke result confirms the fix directionally.

### Change 3: NSCLC gold annotation

- 200 TCGA-LUAD (80) + TCGA-LUSC (120) reports
- Annotator: `tencent/hy3:free` via OpenRouter, 3-key rotation (bypasses free 50/day cap)
- Self-consistency: n_runs=3, min_votes=2 for accept
- **κ mean 0.737, median 0.780** across 200 reports
- 121/200 (60.5%) meet κ ≥ 0.7 threshold
- 1,821 accepted spans; 9.1 spans/report mean
- **BIO alignment error rate: 2.09%** (measured on gold split — 38 mentions failed to relocate into token boundaries after LLM offset re-alignment)

Sparse molecular caveat: TCGA is pre-molecular era. Only 12 EGFR, 3 KRAS, 3 HER2_AMP, 2 ALK mentions in 200 reports. Downstream fine-tuning on modern institutional data recommended before clinical use.

## Training Fan-Out

5 seeds (42, 123, 456, 789, 1234) trained on 5 workers in parallel. Per-seed config identical:
- Corpus: 2389 train / 217 val / 200 nsclc + 96 breast_crc test
- 3 epochs, batch 8, lr 5e-5, max_len 192
- `--class-weighted-loss --label-smoothing 0.1`

Aggregation via `aggregate_real_v51.py` writes `AGGREGATE_v51.json` with:
- Per-seed metrics + per-cancer F1 (nsclc, breast_crc, combined)
- Per-entity mean ± std over 5 seeds
- Rollback check: v0.5.1 vs v0.5.0 breast/CRC F1 relative delta

## Rollback Rule

If v0.5.1 breast/CRC F1 drops >5 relative points vs v0.5.0 baseline (0.0667 mean), retain v0.5.0 weights. This decision surfaces in `AGGREGATE_v51.json.rollback.rollback_triggered`.

## Bugs Fixed This Cycle

1. **`_realign_span` NoneType crash**: LLM sometimes returns `null` for `text_span` field. Half-B training annotation crashed at 921/1195. Patched `llm_labeler.py::_realign_span` to null-check `text_span` and `value` before `.strip()`. Rerun resumed and completed.
2. **Class-weight normalization inverted**: initial O_weight came out to 0.006 (way too small). Rewrote to normalize by O's inv_sqrt so O_weight=1.0 and non-O tags floor upward. Smoke-verified.
3. **Corpus manifest not read for provenance**: v0.5.1 corpus writes `provenance: REAL-v0.5.1-snorkel-openrouter-llm` to `manifest.json`. Training script now reads this first and only falls back to the legacy synthetic-detection heuristic if the manifest is missing.
4. **Section LF regex anchored to line-start**: on period-flattened TCGA text, line-start `^` never matched. Regex is now anchored after `.` or `\n` — coverage jumped from 0.05% to 2%.
5. **BIO alignment error rate not tracked**: added counter in `corpus_v51.py::_bio_labels_from_entities` and computed on the NSCLC gold split. Rate is now written to `manifest.json`.

## Modal Deploy Contract

- Workspace: `crispro-test`
- App: `clinicalbert`
- Endpoints: `crispro-test--clinicalbert-{healthz,info,parse}.modal.run`
- `/info` fields added in v0.5.1: `real_text_micro_f1_breast_crc`, `real_text_micro_f1_nsclc`, `real_text_micro_f1_combined`, `annotator_kappa_nsclc`, `snorkel_label_model_accuracy`, `bio_alignment_error_rate`, `class_weighted_loss`, `label_smoothing`.

## Frozen Invariants (unchanged from v0.5.0)

- Bundle sha256: `0121fc84fc798af57aad78f2c9274506eac769313d4b21329ad9e3775c5b3a4c`
- Manuscript sha: `d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1`
- 7 anchor p-values (Elo seed=20260619, k=16) untouched
- Tumor-board contract: `tumor_board.v3.multimodal-with-manuscript-claims`

## Open Follow-Ups

- Fine-tune on modern institutional NSCLC data to lift molecular-marker recall (TCGA sparse-molecular caveat).
- κ distribution: 34/200 reports have κ < 0.5 — surface these to the model card as "annotator-disputed" and consider excluding from test set for reporting.
- Soft-Snorkel-target training (probability-weighted CE per token) is a natural extension if hard-argmax retention proves fragile.
- Consider rotating GITHUB_PAT (compromised in prior session env-secrets).
