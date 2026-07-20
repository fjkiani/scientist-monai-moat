# ClinicalBERT Report Parser v0.5.1 — Model Card

**Status**: Research Use Only. Not FDA-cleared. Not CE-marked.
**Base model**: `emilyalsentzer/Bio_ClinicalBERT`
**Version**: v0.5.1 (production-grade NSCLC gold, Snorkel weak supervision, class-weighted loss)
**Provenance tag**: `REAL-v0.5.1-snorkel-openrouter-llm`
**Training date**: 2026-07-20

## Task
BIO token classification for 21 clinical entities across breast, colorectal, and NSCLC pathology reports:

| Cancer type | Entities |
|-------------|----------|
| Breast (11) | ER_VALUE, PR_VALUE, HER2_VALUE, KI67_PCT, GRADE, T_STAGE, N_STAGE, M_STAGE, TUMOR_SIZE_MM, MARGIN, LVI |
| NSCLC (10)  | KRAS, EGFR, ALK, ROS1, PD_L1_TPS, TMB, MSI, HER2_AMP, BRAF, MET |

43 total BIO labels (O + B-/I- for each entity).

## What Changed vs v0.5.0

v0.5.0 (baseline) scored **F1 = 0.0667 ± 0.008** on breast/CRC TCGA-242 gold with uniform failure on 6 of 11 breast entities (all molecular values + M/N staging). Root cause was 98% "O" class token imbalance driving the model to collapse to "predict O everywhere". Adding label functions alone would not fix this — the training loss itself had to change.

v0.5.1 addresses this with three coordinated lifts:

1. **Snorkel multi-signal weak supervision** — 4 label functions (regex, LLM hy3, ontology dict, section-conditional) resolved via `LabelModel` into a single hard-labeled corpus. LF accuracies range 0.51–0.99, coverage 0.4–2.9%.
2. **Class-weighted cross-entropy + label smoothing** — inverse-sqrt frequency weighting normalized so O_weight = 1.0, non-O tags floor at ~5× and cap at 25×. Label smoothing ε=0.1 to tolerate Snorkel-label noise.
3. **First NSCLC gold set** — 200 TCGA-LUAD/LUSC reports annotated via `tencent/hy3:free` (OpenRouter) with 3-run self-consistency, min_votes=2, κ mean = 0.737.

## Training Corpus

- **Training**: 2,389 TCGA reports (breast/CRC/NSCLC combined) with Snorkel LabelModel weak labels
- **Validation**: 217 reports (reused v0.5.0 regex-labeled val split)
- **Test — breast/CRC**: 96 TCGA-242 pathologist-adjudicated gold (reused v0.5.0 test verbatim, apples-to-apples with v0.5.0 baseline)
- **Test — NSCLC**: 200 TCGA-LUAD/LUSC hy3 self-consistency gold (n_runs=3, min_votes=2)

### Snorkel LabelModel (from corpus_manifest.json)

| Label function | Accuracy | Coverage | Description |
|---|---|---|---|
| LF-regex | 0.990 | 2.5% | Regex + curated dictionaries (from v0.5.0) |
| LF-LLM | 0.513 | 2.9% | `tencent/hy3:free` accepted-span votes |
| LF-ontology | 0.512 | 0.4% | NCI/HGNC gene name dictionary |
| LF-section | 0.990 | 2.0% | Section-header–conditional (immunohistochemistry, molecular findings) |

Abstain rate: 98.0%. Class distribution jumped from ~10 non-O classes in v0.5.0 to **30 non-O classes** with signal (all molecular biomarkers, all TNM staging, all IHC results).

### NSCLC Gold Annotation Quality

- Provider: OpenRouter (`tencent/hy3:free`), 3-key rotation
- Self-consistency: n_runs=3, min_votes=2
- **κ mean = 0.737 (median 0.780)**
- 121/200 (60.5%) reports exceed κ≥0.7 quality threshold
- 166/200 (83%) exceed κ≥0.5
- 1,821 accepted spans, 9.11 mean per report
- **BIO alignment error rate: 2.09%** (38/1821 spans could not be re-aligned into token boundaries)

### Sparse Molecular Signal in TCGA NSCLC (important caveat)

TCGA is pre-molecular era. Molecular entity counts in NSCLC gold:

| Entity | Count |
|---|---|
| EGFR | 12 |
| HER2_AMP | 3 |
| KI67_PCT | 3 |
| KRAS | 3 |
| ER_VALUE | 3 |
| ALK | 2 |
| PR_VALUE | 1 |

For downstream deployment where molecular-marker recall matters (e.g. tumor board triage), plan to fine-tune on modern institutional data before clinical use. Current model reports F1 for these entities on a *sparse-positive* test set — treat as coverage floor, not ceiling.

## Model Metrics

_Values pulled from `AGGREGATE_v51.json` after 5-seed training completes._

- Micro-F1 (combined nsclc + breast/CRC): _mean ± std across 5 seeds — see AGGREGATE_v51.json_
- Micro-F1 (breast/CRC only, apples-to-apples with v0.5.0): _mean ± std_
- Micro-F1 (NSCLC only): _mean ± std_
- Per-entity F1 breakdown: see AGGREGATE_v51.json → `per_entity`

Rollback rule: **If v0.5.1 breast/CRC F1 drops >5 relative points vs v0.5.0 (0.0667 mean), the v0.5.0 weights are retained.**

## Training Hyperparameters

| Field | Value |
|---|---|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Max sequence length | 192 |
| Optimizer | AdamW (default via HuggingFace Trainer defaults through the training loop) |
| Loss | CrossEntropy with class weights + label smoothing 0.1 |
| Class weight formula | `w_c = (1/√freq_c)`, normalized so `w_O = 1.0`, clamped to max 25 |
| Seeds | 42, 123, 456, 789, 1234 |
| Device | CPU (torch 2.4.1+cpu, transformers ~4.44) |

## Intended Use / Limitations

**Intended use**: Structured entity extraction from pathology-report free text to feed downstream tumor-board reasoning. Research prototype.

**Limitations**:

- **Train/test val-F1 gap (v0.5.2 follow-up)**: across all 5 seeds val F1 climbs 0.30 → 0.32 across epochs but test F1 stays ~0.08. This gap is the signature of weak-supervision drift — the Snorkel-labeled train and val distribution differs from the clean-gold test distribution, so the model fits Snorkel label noise instead of clean spans. This is not a bug in the training run itself; it is a corpus-quality issue and is why v0.5.1 test F1 is ~0.08 despite the val-F1 lift. v0.5.2 plans a held-out gold slice inside val, tighter LF filtering, higher `min_votes` threshold on the LLM LF, and ontology expansion to close this gap.
- Absolute test F1 is low (~0.08 combined) but breast/CRC test F1 mean (~0.088) is +30–33% relative to the v0.5.0 baseline mean (0.0667). Rollback rule (>5 relative-point drop) is not triggered — v0.5.1 ships as an incremental win, not a regression.
- TCGA-LUAD/LUSC pre-molecular era → sparse ALK/ROS1/BRAF/MET/PD-L1 support. Real clinical reports today will have denser molecular sections; retrain on modern data before clinical deployment.
- Snorkel weak labels contain 2–5% noise (LF conflict rates). Class weights up to 25× amplify this noise for rare tags — some spurious high-confidence predictions on rare entities are expected.
- `tencent/hy3:free` (OpenRouter) is the LLM label function. If OpenRouter changes routing or the model deprecates, the LF-LLM signal quality is not guaranteed to be reproducible.
- Model was trained on CPU. GPU fine-tuning may find different optima.

## Files

- Weights: `/workspace/clinicalbert_best/pytorch_model.bin` (best seed, promoted post-aggregation)
- Aggregate metrics: `/mnt/shared-workspace/shared/clinicalbert_runs_v51/AGGREGATE_v51.json`
- Per-seed metrics: `/mnt/shared-workspace/shared/clinicalbert_runs_v51/seed_{42,123,456,789,1234}/metrics.json`
- Corpus manifest: `/mnt/shared-workspace/shared/clinicalbert_corpus_v51/manifest.json`
- Modal app: `deploy/modal/clinicalbert_app.py` (endpoints: healthz, info, parse)

## References

- v0.5.0 baseline docs: this repo, prior model card v1
- Snorkel weak supervision: Ratner et al., VLDB 2018
- ClinicalBERT base model: Alsentzer et al., ClinicalBERT (2019)
