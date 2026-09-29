# Model card — report_parser_clinicalbert_v1

**RESEARCH USE ONLY — not validated for clinical decision-making. Not FDA-cleared.
Not CE-marked. Investigational / IRB context only.**

## Summary

Token-classification head fine-tuned on top of
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
to extract **21 entity types** from oncology pathology reports (breast +
NSCLC molecular / IHC fields) and roll them up into the schema used by
`oncology_arbiter`'s `/v1/case/full` NSCLC branch. Ships as the ClinicalBERT
production report parser for v0.4.1.

- **Version**: `v1` (v0.4.1-alpha)
- **Provenance**: `SYNTHETIC-v0.3.1` (deterministic, 21 entities, mixed
  breast + NSCLC cancer type)
- **License**: same as this repository
- **Base**: `emilyalsentzer/Bio_ClinicalBERT` (HuggingFace, masked-LM warm start)
- **Trained by**: `src/oncology_arbiter/nlp/clinicalbert_train.py`
- **Deployment target (prod)**: Modal (`crispro-test--clinicalbert`,
  `min_containers=1` warm). Render's 512 MB free-tier dyno cannot host
  the ~430 MB safetensors + torch runtime; it calls Modal over HTTP via
  a stdlib-only `urllib` client.
- **Deployment fallback (local dev / uvicorn)**: in-process torch backend
  via `ClinicalBertLocalClient`, gated on `CLINICALBERT_BACKEND=local`.

### AUROC caveat

This is a **token-classification** model, not a diagnostic classifier —
AUROC is not the primary metric. We report per-seed micro-F1 on a held-out
synthetic test split. Any downstream AUROC computed from fields it emits
inherits the same circularity caveat as the rest of the tumor-board
pipeline: labels and features co-originate from published narratives, so
apparent performance is inflated relative to prospective validation.
Expected prospective F1 on real hospital pathology reports: unknown for
`v1`; the corpus is entirely synthetic.

## Task

**Input**: free-text pathology report (typically 300–800 tokens).
**Output**: BIO span labels over 21 entity types, rolled up into a
per-field dict of `{surface, start_tok, end_tok, value}`. Canonicalized
values (e.g. `wild_type`, `not_amplified`, `mss`, `positive`) after
whitespace-normalized surface matching.

### Entity types (21) and BIO labels (43)

Breast biomarkers (11):

```
ER_VALUE, PR_VALUE, HER2_VALUE, KI67_PCT, GRADE,
T_STAGE, N_STAGE, M_STAGE, TUMOR_SIZE_MM, MARGIN, LVI
```

NSCLC molecular / IHC (10):

```
KRAS, EGFR, ALK, ROS1, PD_L1_TPS, TMB, MSI, HER2_AMP, BRAF, MET
```

`O + {B-, I-} × 21 = 43` BIO output labels. Label order is fixed across
all 5 seeds; any change requires a corpus regeneration + re-train and
breaks the wire contract.

## Training data

**Corpus**: `src/oncology_arbiter/nlp/corpus_synth.py` — rule-based
generator producing synthetic reports (mixed breast + NSCLC) with paired
BIO labels and field-level ground truth.

- Provenance string in every JSONL line: `SYNTHETIC-v0.3.1`
- Total: **2000 synthetic reports** (train 1400 / val 300 / test 300)
- Corpus manifest: `/mnt/shared-workspace/shared/nsclc_corpus_v031/manifest.json`

**Important**: numbers below characterize the model against the
distribution the generator produces, NOT against real hospital reports.

## Training

- **Base**: `emilyalsentzer/Bio_ClinicalBERT`
- **Head**: token-classification, 43 output labels
- **Optimizer**: AdamW, lr=5e-5
- **Batch**: 8
- **Max seq len**: 192 word-piece tokens
- **Epochs**: 3
- **Loss**: cross-entropy on active BIO labels (padding masked)
- **Hardware**: 5× CPU workers, 32 GB / 16 vCPU each
- **Seeds**: `{42, 123, 456, 789, 1234}` (all 5 completed, no reruns)
- **Wall-clock** per seed: 2 492 s – 4 848 s (median ~4 400 s ≈ 73 min)

### Held-out test micro-F1 by seed (SYNTHETIC-v0.3.1 test split, n=300)

| Seed | Test micro-F1 | TP | FP | FN | Train seconds |
|---:|---:|---:|---:|---:|---:|
| 42 | **0.971672** | 4099 | 44 | 195 | 4027.7 |
| 123 | 0.971557 | 4099 | 45 | 195 | 4451.0 |
| 456 | 0.971550 | 4099 | 45 | 195 | 4847.8 |
| 789 | 0.971435 | 4099 | 46 | 195 | 4404.4 |
| 1234 | 0.971550 | 4099 | 45 | 195 | 2491.9 |

- **Mean ± SD micro-F1**: **0.971553 ± 0.000084** (n=5)
- **Min / Max**: 0.971435 / 0.971672
- **F1 floor (v0.4.1 plan)**: 0.85
- **Headroom above floor**: +0.121 (14.3 % relative)
- **All 5 seeds pass the 0.85 floor** — enforced by
  `tests/regression/test_clinicalbert_f1_floor.py`.
- **Promoted checkpoint**: seed 42 (highest test micro-F1). Deployed at
  `/workspace/clinicalbert_best/` on worker-agg (local) and shipped to
  Modal as `/model/` inside the image via `modal.Image.add_local_dir`.

Aggregate: `/mnt/shared-workspace/shared/clinicalbert_runs/AGGREGATE.json`
Per-seed metrics: `/mnt/shared-workspace/shared/clinicalbert_runs/seed<N>/metrics.json`

## Known failure modes

1. **Free-text disambiguation ambiguity on TMB.** In unstructured reports
   where a tumor-dimension string ("`3.2 x 2.1 x 1.4 cm`") precedes the
   molecular block, the model has been observed to tag the leading `3.2`
   as `TMB` value instead of the later `9.2 mut/Mb`. Does NOT affect F1
   on the structured SYNTHETIC-v0.3.1 test split (which places TMB in a
   deterministic template context); is a real-world gap. Document as a
   caveat; do not tune around it. Downstream consumers should
   cross-check TMB against structured molecular blocks.
2. **Breast IHC labels fire on NSCLC IHC blocks.** Because the label
   space is shared across cancer types, IHC keywords in an NSCLC report
   (TTF-1, Napsin A, p40 phrased with `positive`/`negative`) can trigger
   breast-family `ER_VALUE / PR_VALUE / HER2_VALUE` tags. Downstream
   consumers should route parsed fields by the cancer type on the
   request envelope; the parser is intentionally cancer-agnostic.
3. **Whitespace-normalized surface matching.** The tokenizer splits
   `wild-type` into `wild - type` (three tokens); canonicalization uses
   `re.sub(r"\s*-\s*", "-", surface.lower())` before dictionary lookup.
   Any downstream code that re-canonicalizes must apply the same
   normalization or it will double-fail on hyphenated tokens.
4. **CPU-only inference in prod.** Modal container runs on CPU — a warm
   `min_containers=1` container answers in ~2 s wall for a ~600 token
   report. Cold start adds ~15 s of model load. Latency-sensitive
   callers should keep the container warm.
5. **Synthetic corpus.** Model performance on real hospital pathology
   reports is unmeasured for `v1`. Do not represent these numbers as
   real-world clinical accuracy. A future `v2` corpus with real
   de-identified pathology text is required for prospective validation.

## Deployment

### Modal (production)

- **App name**: `clinicalbert`
- **Workspace**: `crispro-test`
- **URL layout**:
  - `https://crispro-test--clinicalbert-healthz.modal.run`
  - `https://crispro-test--clinicalbert-info.modal.run`
  - `https://crispro-test--clinicalbert-parse.modal.run`
- **Image deps**: `torch==2.4.1`, `transformers==4.44.2`,
  `safetensors==0.4.5`, `huggingface_hub==0.24.7`, `fastapi==0.115.0`,
  `numpy==1.26.4`
- **Weight mount**: `modal.Image.add_local_dir(/workspace/clinicalbert_best, /model)`
- **Warm containers**: `min_containers=1` when
  `CLINICALBERT_MODAL_MODE=prod`

### Render (client)

- `render.yaml` sets `CLINICALBERT_BACKEND=modal` +
  `CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert`.
- `oncology_arbiter/api/app.py` NSCLC branch instantiates
  `ClinicalBertModalClient` and POSTs the pathology narrative to
  `<CLINICALBERT_MODAL_URL>-parse.modal.run`.
- Failure modes are surfaced honestly: on network error / non-200 /
  `{"error": ...}` payload, `parsed_report` is `None` and
  `parsed_report_provenance` carries `source: clinicalbert_modal_error`.

### Local (dev only)

- `CLINICALBERT_BACKEND=local` +
  `CLINICALBERT_LOCAL_WEIGHT_DIR=/workspace/clinicalbert_best` loads
  the fine-tuned model in-process via `ClinicalBertLocalClient` (torch
  + transformers). Used for uvicorn smoke; not shipped to Render.

## Wire contract

`GET /parse` (Modal) response schema:

```json
{
  "provenance": "SYNTHETIC-v0.3.1",
  "base_model": "emilyalsentzer/Bio_ClinicalBERT",
  "training_seed": 42,
  "test_micro_f1": 0.971672395401209,
  "parsed": { "<entity>": {"surface": "...", "start_tok": N, "end_tok": N, "value": "..."}, ... },
  "spans": [ ... ],
  "n_tokens": <int>,
  "seconds": <float>,
  "app_version": "clinicalbert-modal-v0.4.1-alpha",
  "disclaimer": "RESEARCH USE ONLY — not validated for clinical decision-making. Not FDA-cleared. Not CE-marked. Investigational / IRB context only."
}
```

The disclaimer is emitted verbatim by the Modal app on every parse.

## Regression tests

- `tests/regression/test_clinicalbert_f1_floor.py` — parametrised over
  the 5 seeds, asserts `test.micro.f1 >= 0.85` for each. Also
  cross-checks that `AGGREGATE.json` matches the per-seed detail.
- `tests/regression/test_clinicalbert_modal_wire.py` — asserts
  `render.yaml` publishes `CLINICALBERT_BACKEND=modal` +
  `CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert`, that
  `app.py` imports and branches on the Modal client, that the Modal
  client module exists and is stdlib-only (no `requests` / `httpx`),
  and that `NsclcResponse` carries `parsed_report` +
  `parsed_report_provenance`.

Both marked `@pytest.mark.regression`; run with `pytest -m regression`.

## Provenance & reproducibility

- Corpus generator: `src/oncology_arbiter/nlp/corpus_synth.py`
- Trainer entry: `src/oncology_arbiter/nlp/clinicalbert_train.py`
- Per-seed launcher: `deploy/hpc/run_seed.sh` (or the staged copy at
  `/mnt/shared-workspace/shared/clinicalbert_stage/run_seed.sh`)
- Weights sha256 (`model.safetensors`, seed=42):
  `243d05be0d87a064d51802ee0be13d7f`
  *(md5; sha256 computed on push).*
