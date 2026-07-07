# Model card — report_parser_clinicalbert_v1

## Summary

Token-classification head fine-tuned on top of `emilyalsentzer/Bio_ClinicalBERT`
to extract 11 entity types from breast-cancer pathology reports and roll them
up into the schema used by `oncology_arbiter`'s regex parser. Used inside the
fused-mode report parser to rescue reports where the regex layer fails on
long-form or non-standard phrasing.

- **Version**: `v1` (v0.3.0-alpha)
- **Provenance**: `SYNTHETIC-v0.3.0`
- **License**: same as this repository
- **Trained by**: `oncology_arbiter/nlp/train_clinicalbert_report_parser.py`
- **Weights host (dev)**: `/workspace/models/report_parser_clinicalbert_v1/`
- **Weights mirror (deliverable)**: `/mnt/results/models/report_parser_clinicalbert_v1/`
- **Not deployed to Render** (dyno = 512 MB; ClinicalBERT weights = ~430 MB).
  Fusion runs on the client / dev fleet only for this release.

## Task

**Input**: free-text pathology report.
**Output**: BIO span labels over the 11 entity types below, rolled up into
per-field values matching the regex parser's schema.

### Entity types (11) and BIO labels (23)

```
O + {B-, I-} × {
  ER_VALUE, PR_VALUE, HER2_VALUE, KI67_PCT, GRADE,
  T_STAGE, N_STAGE, M_STAGE, TUMOR_SIZE_MM, MARGIN, LVI
}
```

### Per-entity confidence thresholds

The parser exposes both `value` (the canonicalized value) and `match_state`
(`matched` / `ambiguous` / `no_match`). If the model's mean per-span softmax
confidence is below the entity threshold, `match_state=ambiguous` and `value=None`
— the span is surfaced via `matched_text` for downstream consumers.

| Entity | Threshold |
|---|---:|
| ER_VALUE, PR_VALUE, HER2_VALUE | 0.7 |
| KI67_PCT | 0.7 |
| GRADE | 0.7 |
| T_STAGE, N_STAGE, M_STAGE | 0.6 |
| TUMOR_SIZE_MM | 0.6 |
| MARGIN, LVI | 0.7 |

## Training data

**Corpus**: `oncology_arbiter/nlp/corpus_synth.py` — a rule-based generator
that produces synthetic breast-cancer pathology reports with paired
BIO-labeled entities and field-level `ground_truth` rollup.

- Train: **1400 reports** (`train.jsonl`, 4.0 MB)
- Val: **300 reports** (`val.jsonl`, 866 KB)
- Test: **300 reports** (`test.jsonl`, 866 KB, this held-out split)
- Total: **2000 synthetic reports**

**Provenance labeling**: every corpus JSONL line carries `provenance="SYNTHETIC-v0.3.0"`
which flows through to `ClinicalBertParsedReport.provenance`. Fused-mode API
responses that consumed a ClinicalBERT prediction carry this string so the UI
can badge them.

**Important**: this corpus is synthetic. Numbers below characterize the model
against the distribution the corpus generates, NOT against real hospital
pathology reports. The purpose of `v1` is (a) prove the fusion pipeline
works end-to-end and (b) establish an eval harness that a future real-data
`v2` can be dropped into without changing consumers.

## Training

- **Base**: `emilyalsentzer/Bio_ClinicalBERT` (masked-LM warm start)
- **Head**: token-classification, 23 output labels
- **Optimizer**: AdamW, lr=5e-5, weight_decay=0.01
- **Batch**: 16
- **Max seq len**: 512 word-piece tokens
- **Epochs**: 3
- **Loss**: cross-entropy on active BIO labels (padding masked)
- **Hardware**: 16 vCPU / 64 GB RAM CPU-only worker (`worker-nsclc`)
- **Wall-clock**: ~24 min end-to-end training + eval

### Checkpoint scores

| Epoch | Val loss | Val micro F1 |
|---:|---:|---:|
| 0 | 0.0831 | 0.9490 |
| 1 | 0.0119 | 0.9920 |
| 2 | **0.0088** | **0.9933** |

`epoch_2` is promoted to top-level and shipped as the release checkpoint.

## Held-out test set — full metrics

Evaluated by `scripts/eval_report_parser_clinicalbert.py` on the
300-report `test.jsonl` split.

### Micro (all fields pooled)

- **Micro F1 (relaxed): 0.9546** (P=1.0000, R=0.9132)
- **Value accuracy given match: 0.9520** (2853/2997)
- Support: 3282 gold field values across 300 reports
- **Micro F1 (strict): 0.8733** (P=1.0000, R=0.7751)  — see note below
- **Value accuracy (strict, given matched): 0.9697**
- Eval wall-clock: 261.9 s (~0.87 s/report, CPU)

### Per-field (relaxed: matched | ambiguous)

| Field | Support | P | R | F1 | val_acc |
|---|---:|---:|---:|---:|---:|
| er | 300 | 1.0000 | 1.0000 | 1.0000 | 0.9133 (274/300) |
| pr | 300 | 1.0000 | 1.0000 | 1.0000 | 0.9367 (281/300) |
| her2 | 300 | 1.0000 | 0.9867 | 0.9933 | 1.0000 (296/296) |
| grade | 300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (300/300) |
| tumor_size_mm | 300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (300/300) |
| t_stage | 300 | 1.0000 | 1.0000 | 1.0000 | 0.8000 (240/300) |
| n_stage | 300 | 1.0000 | 1.0000 | 1.0000 | 0.8700 (261/300) |
| m_stage | 300 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (300/300) |
| margin | 300 | 1.0000 | 0.6533 | 0.7903 | 1.0000 (196/196) |
| lvi | 300 | 1.0000 | 0.4100 | 0.5816 | 1.0000 (123/123) |
| ki67_pct | 282 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (282/282) |

### Per-field (strict: matched-only — model commits above threshold)

| Field | Strict F1 | Strict Recall | Strict val_acc |
|---|---:|---:|---:|
| er | 0.9437 | 0.8933 | 1.0000 |
| pr | 0.9547 | 0.9133 | 1.0000 |
| her2 | 0.9933 | 0.9867 | 1.0000 |
| grade | 1.0000 | 1.0000 | 1.0000 |
| ki67_pct | 1.0000 | 1.0000 | 1.0000 |
| tumor_size_mm | 1.0000 | 1.0000 | 1.0000 |
| t_stage | 0.6726 | 0.5067 | 0.6053 |
| n_stage | 0.7124 | 0.5533 | 0.8976 |
| m_stage | 0.7680 | 0.6233 | 1.0000 |
| margin | 0.7903 | 0.6533 | 1.0000 |
| lvi | 0.5816 | 0.4100 | 1.0000 |

Read strict alongside relaxed. Stage strict recall is depressed (0.51–0.62)
because the corpus mixes staged (`T2 N1 M0`) and unstaged report styles;
when the T/N/M tokens appear in isolated summary lines their per-span
confidence sometimes falls below the 0.6 stage threshold. The regex layer
covers these in fused mode.

### Fused-vs-regex agreement (receptor panel, 4-cell tables)

On the 300-report test set, running the fused parser (regex + ClinicalBERT
rescue) vs the regex-only parser:

| Field | Fused ✓ Regex ✓ | Fused ✓ Regex ✗ | Fused ✗ Regex ✓ | Fused ✗ Regex ✗ | Net |
|---|---:|---:|---:|---:|---:|
| ER | 164 | **110** | 0 | 26 | **+110** |
| PR | 166 | **115** | 0 | 19 | **+115** |
| HER2 | 204 | **92** | 4 | 0 | **+88** |
| GRADE | 300 | 0 | 0 | 0 | ±0 |

- Fusion strictly improves ER (0 breakages).
- Fusion strictly improves PR (0 breakages).
- Fusion improves HER2 by +88 net despite 4 breakages (0.013× breakage rate).
- Fusion does nothing on GRADE (regex already 100% on this test set).

## Known failure modes

1. **Margin / LVI recall (0.65 / 0.41)** — the model rarely commits above
   threshold on these two fields. Corpus phrasing for margin ("close",
   "widely negative", "0.2 mm to the deep margin") and LVI ("focally present",
   "not identified in submitted tissue") is longer and less template-like
   than receptor phrasing. Regex layer handles most of these in fused mode.
2. **Stage strict recall depressed (0.51–0.62)** — see strict per-field
   table above. Relaxed F1 is 1.0 on all three (T/N/M) — the model
   correctly identifies stage spans — but strict-view recall is lower
   because the model often falls just below the 0.6 threshold on stage
   entities, especially in off-template phrasing like "Pathologic
   stage: pT2, N1, M0.". Regex layer handles these in fused mode.
3. **Test set is entirely synthetic.** Model performance on real hospital
   pathology reports is unmeasured for `v1`. Do not represent these numbers
   as real-world clinical accuracy.
4. **CPU-only inference.** 0.87 s/report is fine for dev / batch but not for
   the Render 0.1 CPU dyno; deploying this on the free tier would starve
   the API. Weights are deliberately excluded from the Docker image.
5. **Tumor size integer-mm phrasing gap.** The synthetic corpus always
   writes tumor size with `.0` suffix (e.g. `"Tumor size: 22.0 mm"`),
   so the model never learned to tag integer-mm surfaces. On post-training
   API smokes, `"Tumor size: 22 mm"` returned `no_match`; adding `.0`
   makes it match. Not visible in the held-out eval (which uses corpus
   phrasing) but a real gap for wild input. Regex layer handles integer
   surfaces in fused mode.

## Usage

### Standalone

```python
from pathlib import Path
from oncology_arbiter.nlp.clinicalbert_parser import ClinicalBertReportParser

parser = ClinicalBertReportParser(
    model_dir=Path("/workspace/models/report_parser_clinicalbert_v1"),
)
report = parser.parse("ER: strong nuclear positivity. PR: positive. HER2: 3+. Grade 2. Ki-67: 15%.")
print(report.as_dict())
# {'er': {'value': True, 'match_state': 'matched', ...},
#  'pr': {'value': True, ...},
#  'her2': {'value': 'positive', ...},
#  'grade': {'value': 2, ...},
#  'extended_fields': {...},
#  'parser_id': 'clinicalbert_v1',
#  'provenance': 'SYNTHETIC-v0.3.0'}
```

### Fused (regex + ClinicalBERT rescue)

```python
from oncology_arbiter.models.report_parser import parse_pathology_report

result = parse_pathology_report(text, enable_clinicalbert=True)
```

The fused parser's response contract is unchanged from the regex-only
parser — every field carries an added `source` key (`"regex_v1"` |
`"clinicalbert_v1"`) so a caller can attribute which layer produced each
value.

## Metadata dump (for downstream consumers)

Available in `test_metrics.json` alongside the weights:

- `ckpt_dir`, `test_jsonl`, `n_reports`, `eval_seconds`
- `micro` (relaxed + `micro.strict` sub-dict)
- `macro`
- `per_field[field]` with `.strict` sub-dict on each
- `fused_vs_regex_agreement` — 4-cell contingency per receptor field

## Provenance & reproducibility

- Corpus: deterministic seed in `corpus_synth.py` — regenerate via
  `python -m oncology_arbiter.nlp.corpus_synth`
- Training: `python scripts/train_report_parser_clinicalbert.py`
- Eval: `python scripts/eval_report_parser_clinicalbert.py --ckpt-dir <dir> --test-jsonl <path> --out-dir <dir>`
- Weights sha256 (`model.safetensors`):
  `3bead0422395211adf772edc57626934698da40cee1c40a6920a951dfd893851`
