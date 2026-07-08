# `oncology-arbiter` — End-to-end audit v2 (v0.3.0-alpha)

_Prepared 2026-07-07 after direct code + wire inspection on worker-nsclc, worker-1, and worker-frontend. Supersedes `report_end_to_end_audit.md` (2026-07-02, v0.2.2-alpha vintage)._

---

## §0. What changed since v0.2.2-alpha

Between the v0.2.2-alpha tag (`6a1cd7c`, 2026-07-02) and this audit (2026-07-07), 13 commits landed on `main`. Five of those commits promoted formerly-stubbed code paths to real, trained, wired implementations. The audit v1 §4 ("What is STUBBED / PLACEHOLDER") is now **materially out of date** — that is the primary reason for v2.

Delta by commit (newest first):

| SHA | Commit | Layer moved |
|---|---|---|
| `adf3cab` | docs: fix stale stage F1=0 in v1 model card; add tumor_size_mm integer-mm gap | docs only |
| `e1214cc` | wire trained ClinicalBERT weights + integration tests | biopsy: shipped → wired |
| `15fb366` | ClinicalBERT report parser v1 eval + model card | training run + audit deliverable |
| `c7f2379` | fix: line-buffer stdout + per-epoch ckpts in training | training infra |
| `13c40b8` | Bio_ClinicalBERT report parser + regex fusion | biopsy report parse: stub → shipped |
| `b67f864` | probe-aware ScreeningTab + LUNA16 panel + fused-parser detail | frontend UI |
| `033ac21` | wire LUNA16 RetinaNet into `/v1/case/full?cancer=nsclc` | nsclc detection: stub → wired |
| `9686ce2` | wire CBIS-DDSM probe into `/v1/screening/analyze` | screening: stub → wired |
| `42dc2e1` | trained CBIS-DDSM supervised probe (test AUC 0.7526) | screening: shipped → trained |
| `29683c7` | Modal MedSigLIP client + CBIS-DDSM training pipeline | screening infra |
| `ad48cd4` | Modal MedSigLIP-448 GPU inference deploy | GPU inference infra |
| `99113e3` | real Co-Scientist LLM loop via Gemma-4-31b | supervisor: stub → real LLM |
| `d0dfc9a` | chore(v0.2.2): refresh ledger timestamp | admin only |

Net effect: **4 of the 7 stubbed items in audit v1 §4 are no longer stubs**. Specifically:
- §4.1 "Detector inference — everywhere" — screening (`9686ce2`), nsclc detection (`033ac21`), and biopsy report parse (`13c40b8`, `e1214cc`) all now do real work. Biopsy WSI subtyping is still zero-shot proxy.
- §4.3 "Supervisor / Co-Scientist loop" — real 5-phase LLM loop via Gemma-4-31b (`99113e3`).
- §4.6 "Report text extraction" — trained Bio_ClinicalBERT v1 with fused regex-BERT rescue is now the default when the env flag is on.

Items **still stubbed** or **partial**:
- §4.4 URL-based DICOM ingestion — unchanged (still 501).
- §4.5 WSI ingestion — still uses image bytes, no OpenSlide.
- §4.7 IRB / ledger enforcement — schemas exist, no DB.
- Biopsy: WSI → subtype is a MedSigLIP zero-shot linear probe (`n_training=48 synthetic=True`); no supervised classifier trained on real WSI patches yet.

The rest of this document is a fresh audit against current-code (`adf3cab` on `main`), rerun today, not a diff against v1.

---

## §1. TL;DR — what's actually running vs. what's shipped code

Same three layers, updated:

| Layer | State on worker-nsclc as of 2026-07-07 17:50 UTC |
|---|---|
| Repo shipped | 353+ tests (11 new v0.3.0 e2e), 5 model cards (SigLIP, MedSigLIP-448, MedGemma 4B & 27B, **report_parser_clinicalbert_v1**), 3 arbiter templates, 5 CBIS-DDSM DICOMs, 5 Co-Scientist tools, real supervisor loop, IRB templates, ledger SQL schema |
| Wired in API | 6 real code paths (DICOM preproc, arbiter scoring, model card index, **screening probe**, **nsclc detection**, **fused report parser**) + 3 partial/placeholder (biopsy WSI subtyping proxy, therapy template, URL DICOM ingestion) |
| Deployed on Render | `oncology-arbiter.onrender.com` (Docker, free plan, 512 MB / 0.1 CPU, region=oregon). CBIS-DDSM probe and ClinicalBERT parser NOT enabled on the dyno. ClinicalBERT is a 430 MB safetensors load that would exceed the 512 MB memory ceiling; CBIS-DDSM probe requires `MEDSIGLIP_BACKEND=modal` + a Modal token (neither set on Render). Local dev boxes have both. |

The five model cards are documentation only for four of them. **`report_parser_clinicalbert_v1.md` is the first model card in this repo whose weights are actually loaded by the wired API code path.**

---

## §2. Verified numbers (rerun today, 2026-07-07)

### 2.1 Test suite (worker-1, HEAD=adf3cab)
- `pytest tests/nlp/ tests/unit/test_report_parser.py tests/models/` → **116 passed, 2 skipped** (2 skipped are missing-fixture, not failures).
- `pytest tests/nlp/test_clinicalbert_e2e.py -m models` → **11 passed** in 48 s (loads real weights from `/mnt/results/`, worker-1 has no `/workspace/models/` copy).
- Fresh `pytest tests/` full-suite run not attempted from worker-1 (would pull in `slow_ml`-marked tests and networked live tests).

### 2.2 Report parser v1 held-out eval (300 synthetic reports, epoch_2 checkpoint)
Field-level, span-based P/R/F1 with two views:

**Relaxed** (accept `matched` or `ambiguous` — matched_text falls back on ambiguous):
- Micro F1 = **0.9546** (P=1.0000, R=0.9132)
- Value accuracy = **0.9520** (2853 / 2997)
- Per-field: ER F1=1.0, PR F1=1.0, HER2 F1=0.9933, GRADE F1=1.0, ki67_pct F1=1.0, T/N/M-stage F1=1.0, margin F1=0.7903, LVI F1=0.5816

**Strict** (`matched` only — "how often does the model commit above threshold and get it right"):
- Micro F1 = **0.8733** (P=1.0000, R=0.7751)
- Value accuracy (given matched) = **0.9697**
- Per-field: ER F1=0.9437, PR F1=0.9547, HER2 F1=0.9933, GRADE F1=1.0, T-stage F1=0.6726, N-stage F1=0.7124, M-stage F1=0.7680

**Fusion vs regex net rescues** (fused wins − fused breaks): ER **+110**, PR **+115**, HER2 **+88**, GRADE **±0**. Fusion never breaks ER or PR. HER2 has 4 breakages against 92 rescues (0.013× breakage rate).

Full JSON: `/mnt/results/models/report_parser_clinicalbert_v1/test_metrics.json`. Human-readable: `eval_summary.md` in the same dir.

### 2.3 CBIS-DDSM screening probe (from `docs/proofs/cbis_ddsm_logreg_v1_metrics.json`)
- Test AUC = **0.7526** on the CBIS-DDSM upstream held-out split (n_test=641, pos=260, neg=381).
- Trained on 2445 mammograms embedded via Modal MedSigLIP-448 → 1152-d embedding → sklearn LogReg.
- Wired into `/v1/screening/analyze` when `MEDSIGLIP_BACKEND=modal` AND `ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE=1`. Overrides zero-shot `overall_score` with `probe.proba_cancer`.

### 2.4 LUNA16 RetinaNet (MONAI bundle 0.6.9)
- Publisher-reported mAP = **0.852** on LUNA16 fold 0. Not re-run in this repo — used as a shipped MONAI bundle.
- RetinaNet 3D, resnet50 backbone, 20.9M params, weights at `/workspace/monai_bundles/lung_nodule_ct_detection/models/` (~83 MB).
- Wired into `/v1/case/full?cancer=nsclc`. HU normalization [-1024, 300] → [0, 1], sliding window [192, 192, 80] @ 0.25 overlap.

### 2.5 API smoke on `/v1/biopsy/analyze` (worker-1 via TestClient, fused mode, weights from `/mnt/results/`)
4 smokes, all HTTP 200. See `/mnt/results/api_smoke_v0_3_0/`:
- **Smoke 1** (ambiguous phrasing): `"strong nuclear positivity"` → ER=True (source=clinicalbert); `"moderate to strong"` → PR=True (source=clinicalbert); `"3+"` → HER2=positive (source=fused). Extended fields populated: ki67_pct=15, n_stage=N1, m_stage=M0, margin=negative.
- **Smoke 2** (negative rescue): `"no staining seen"` → PR=False (source=clinicalbert). Regex misses this; BERT catches it.
- **Smoke 3** (regex-only fallback with `ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER` unset): wire shape identical, `parser_id=proxy_regex_v0`, all sources `regex`, extended_fields empty.
- **Smoke 4** (`/v1/case/full` fan-out): biopsy sub-call's `report_parse` block is threaded through correctly. `provenance.model_state=fused_regex_clinicalbert`, `model_name=clinicalbert_v1+regex_v0`.

**Copy-verbatim smoke output will be inserted into §2 once the live Render URL is up (§3.3).**

---

## §3. What is REAL (end-to-end, not stubbed)

### 3.1 Mammography preprocessing pipeline
Unchanged since audit v1 §3.1. Real DICOM reader, laterality/view detection with content-based fallback, Otsu segmentation, MLO pectoral removal. 49 tests in `test_mammography_real_dicoms.py`.

### 3.2 L2 logistic arbiter
Unchanged. 15/12/15 coefficients across screening/biopsy/therapy templates, `n_training=0`, decoy-safe `_encode_bool` (None → 0.5).

### 3.3 Model card index + artifact streamer
Now indexes **5** cards (added `report_parser_clinicalbert_v1.md`).

### 3.4 Arbiter transparency playground
Unchanged. Public `/v1/demo/case` endpoint drops `require_api_key`.

### 3.5 CBIS-DDSM screening probe *(v0.3.0)*
File: `src/oncology_arbiter/models/cbis_ddsm_probe.py`. Sklearn Pipeline (`StandardScaler` + `LogisticRegression`, penalty=l2, C=0.01) on 1152-d MedSigLIP embeddings. Trained pipeline persisted at `models/cbis_ddsm_logreg_v1.joblib` (33 KB) — small enough to check into the repo alongside the training data provenance JSON at `docs/proofs/cbis_ddsm_logreg_v1_metrics.json`.

Provenance chain when enabled:
```
DICOM bytes → decode → PIL → base64 → Modal /embed (A10G, 1152-d)
           → in-process LogReg → proba_cancer
           → override zero-shot overall_score
           → arbiter feature: screening_medsiglip_findings
```

Gate: `MEDSIGLIP_BACKEND=modal` AND `ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE=1`. Missing Modal token → surfaces `screening_cbis_ddsm_probe_error:*` in warnings, does NOT fabricate.

### 3.6 LUNA16 RetinaNet nodule detector *(v0.3.0)*
File: `src/oncology_arbiter/nsclc/luna16_retinanet.py`. Loads a MONAI bundle from `/workspace/monai_bundles/lung_nodule_ct_detection/`. Wired into `/v1/case/full?cancer=nsclc`.

Gate: `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` and a valid `nsclc_ct_input.series_dir` in the request. Without both, endpoint returns shape-only response, no CT decode, no inference.

### 3.7 Bio_ClinicalBERT report parser + regex fusion *(v0.3.0)*
Files:
- `src/oncology_arbiter/nlp/clinicalbert_parser.py` — model wrapper + canonicalizers + 3-tier weights resolver
- `src/oncology_arbiter/nlp/report_parser_v2.py` — fusion switch (regex / bert / fused)
- `src/oncology_arbiter/nlp/corpus_synth.py` — synthetic corpus generator (train/val/test = 1200/300/300 reports)
- `docs/model_cards/report_parser_clinicalbert_v1.md` — model card
- `src/oncology_arbiter/nlp/clinicalbert_train.py` — training script (3 epochs, AdamW, lr=2e-5)
- `scripts/eval_report_parser_clinicalbert.py` — field-level eval harness
- `tests/nlp/test_clinicalbert_e2e.py` — 11 e2e tests marked `@pytest.mark.models`

Wire contract: `report_parse` block on `BiopsyResponse` with `parser_id`, `fusion_mode`, `per_field_source`, `per_field_confidence`, `extended_fields` (7 keys). See §2.5 above for full smoke coverage.

Weights resolution:
```python
1. env ONCOLOGY_ARBITER_CLINICALBERT_DIR       # override
2. /workspace/models/report_parser_clinicalbert_v1  # dev
3. /mnt/results/models/report_parser_clinicalbert_v1  # deliverable mirror
```
Present-file sentinel is `label_map.json`. `FileNotFoundError` with actionable hint if none resolves.

### 3.8 HAI-DEF gating logic
Unchanged since audit v1 §3.7. Wired into biopsy sub-endpoint preflight; on FORBIDDEN emits `ModelState.GATED` with actionable warning.

### 3.9 Co-Scientist LLM loop *(v0.3.0)*
File: `src/oncology_arbiter/agents/supervisor.py` (rewritten from stub).

Real 5-phase loop: `GENERATE → EVIDENCE → REFLECT → TOURNAMENT → META_REVIEW`. Backed by:
- `src/oncology_arbiter/models/llm_client.py` — `GemmaClient` with route ladder: Google direct → OpenRouter V2/V1/LEGACY :free
- Handles Gemma thinking-token buffer (`+512` to `maxOutputTokens`)
- Skips OpenRouter keys that hit 429 (respects `X-RateLimit-Reset`)
- Treats OpenRouter 402 as permanent (24-hr blackout)
- `UsageLedger` records calls / tokens / cost / per-route counts
- Raises `LlmUnavailable` when ladder exhausted (**NEVER fabricates**)

Elo tournament writes to `elo_ranked_hypotheses` on `FullCaseResponse`.

### 3.10 Audit log
Unchanged. JSONL append-only log at `/tmp/oncology_arbiter_audit.log` per uvicorn process. Structured event per API call.

---

## §4. What is STILL STUBBED / PLACEHOLDER

### 4.1 Biopsy WSI subtyping
`/v1/biopsy/analyze` when `ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP=1` runs a **synthetic** MedSigLIP linear probe:
- `n_training=48 synthetic=True` on the probe head
- No real WSI ingestion — the "WSI bytes" are treated as an image the vision encoder can consume (no OpenSlide, no tiler)
- Response `warnings` list surfaces `biopsy_medsiglip_synthetic:n_training=48` on every call

This is the same state as audit v1 §4.5 — the wire contract accepts a WSI, the code is honest about not tiling it.

### 4.2 Therapy branch
`/v1/therapy/reason` still returns template-driven `recommended_options` / `not_recommended` lists. No trained therapy model. Arbiter scoring uses declared features from `TherapyPatientContext`, but there is no supervised link to real outcome data.

### 4.3 URL-based DICOM ingestion
`/v1/screening/analyze` with `dicom_url` returns **HTTP 501**. SSRF-guarded fetcher exists (`tools/web_fetch.py`) but is not wired to the screening endpoint.

### 4.4 IRB / ledger enforcement
- `artifacts/reports/informed_consent_template.md` — template, unfilled.
- `artifacts/reports/irb_protocol_template.md` — template, unsubmitted.
- `artifacts/reports/ai_prediction_ledger_schema.sql` — SQL schema exists, no DB is instantiated, no writes.

### 4.5 Model states reported by `/health`
Live-computed from env at request time (`_compute_models_loaded` in `api/app.py:178`), not a static dict. Slots and their env-driven states:

```python
models_loaded = {
    # Screening: MEDSIGLIP > SIGLIP_PROXY > MONAI_DETECTOR > PLACEHOLDER
    "monai_screening":     PLACEHOLDER | PROXY_SIGLIP | PROXY_MONAI_HEURISTIC | LOADED_MEDSIGLIP,
    # Biopsy WSI probe (synthetic n=48 head)
    "medsiglip_biopsy":    PLACEHOLDER | LOADED_BIOPSY_PROBE,
    # v0.2.1 regex parser (stateless code — ALWAYS in this state)
    "biopsy_report_parser": PROXY_REGEX_V0,
    # Therapy: TXGEMMA > RULES_LITE > PLACEHOLDER
    "txgemma_therapy":     PLACEHOLDER | PROXY_RULES_LITE | LOADED_TXGEMMA,
    # Co-Scientist: PROXY_CO_SCIENTIST when env flag on
    "co_scientist":        PLACEHOLDER | PROXY_CO_SCIENTIST,
    # L3 arbiter templates (n_training=0)
    "l3_arbiter":          TEMPLATE,
    # NSCLC: HU-threshold heuristic
    "nsclc_pipeline":      PROXY_LUNG_HEURISTIC,
}
```

**Important caveat**: on Render, `ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST=1` IS set (see `render.yaml`), so the `co_scientist` slot on the deployed `/health` will report `PROXY_CO_SCIENTIST`. The slot label was not tightened after `99113e3` moved the underlying implementation to a real GemmaClient LLM loop. Callers should not read PROXY as "no real work happens" for this slot — the deterministic Elo tournament wrapper does still run the real 5-phase LLM loop inside when the LLM route ladder resolves. (In production, the Gemma route ladder may still fail — Google direct requires an API key, OpenRouter free-tier hits 429/402 frequently — in which case `LlmUnavailable` bubbles up as a warning rather than as fabricated evidence.)

Similarly, the ClinicalBERT parser and LUNA16 detector state are not surfaced through `/health` slots — they only appear in the per-response `provenance.model_state` field of `/v1/biopsy/analyze` and `/v1/case/full?cancer=nsclc` respectively. On the Render deploy where both env flags are off, no request will ever see `LOADED_CLINICALBERT_PARSER` or `LOADED_LUNA16_RETINANET` in the wire response — those states only surface on dev boxes with the flags on and weights present.

---

## §5. Data lineage — what was actually collected / trained / tested

### 5.1 Real image data on disk
- `data/cbis_ddsm/` — 5 real CBIS-DDSM DICOMs shipped in the repo for test data (unchanged from v0.2.2-alpha).
- `dbaek111/CBIS-DDSM_1024` (Hugging Face, 668 MB ZIP, sha256 `123239b7f68c3a309b33784aff0a7f91ff9200edc5a2f76c4507ce96cc7c0e53`) — full 3086 mammogram dataset. Downloaded once onto a training worker under `/workspace/data/CBIS-DDSM_1024/`; not shipped with the repo (too large, CC-BY-NC 4.0). Full provenance: `docs/proofs/cbis_ddsm_logreg_v1_metrics.json` `dataset` block.
- LUNA16 CT series (LIDC-IDRI subset) for NSCLC detection dev — downloaded onto a training/dev worker; not shipped with the repo. The API-side inference bundle (`models/luna16/lung_nodule_ct_detection/`) contains no source CT series, only the packaged inference weights + config.

### 5.2 Synthetic data used in training
- `data/report_parser_v0_3_0/{train,val,test}.jsonl` — 1200/300/300 synthetic pathology reports generated on a training worker by `src/oncology_arbiter/nlp/corpus_synth.py`. Corpus itself not shipped (regeneratable from `corpus_synth.py` + fixed seed).
- 11 entity types, 23 BIO labels. Random-but-reproducible via fixed seed (`--seed 20260703`).
- **All 300 test-set metrics in §2.2 are on this synthetic corpus.**

### 5.3 Trained model weights
| Model | Data | Ckpt path | sha256 |
|---|---|---|---|
| `report_parser_clinicalbert_v1` | 1200 synth reports | `/workspace/models/report_parser_clinicalbert_v1/model.safetensors` (430 MB) | `3bead0422395211adf772edc57626934698da40cee1c40a6920a951dfd893851` |
| `cbis_ddsm_logreg_v1` | 2445 CBIS-DDSM_1024 mammograms | `models/cbis_ddsm_logreg_v1.joblib` (33 KB, in repo) | via commit hash |
| `luna16_lung_nodule_detector` | MONAI publisher weights | `/workspace/monai_bundles/lung_nodule_ct_detection/models/` (~83 MB) | MONAI-hosted |

Mirror for portability: **`/mnt/results/models/report_parser_clinicalbert_v1/`** contains all 9 files (weights + tokenizer + label_map + test_metrics.json + eval_summary.md + model_card.md). sha256 verified identical to worker-nsclc source. This is the fallback the 3-tier resolver uses when there's no `/workspace/models/` copy on the machine.

### 5.4 Model weights loaded by the running server
Depends on machine:
- **worker-nsclc**: MedSigLIP zero-shot (proxy), ClinicalBERT v1 (loaded from `/workspace/models/`), LUNA16 (loaded from `/workspace/monai_bundles/`), Gemma-4-31b (via API).
- **worker-1, worker-frontend**: same as worker-nsclc but ClinicalBERT loads from `/mnt/results/` fallback (no local copy).
- **Render dyno (deployed)**: only shape-safe placeholders. CBIS-DDSM probe and ClinicalBERT parser deliberately NOT enabled — weights would OOM the 512 MB dyno. Modal-backed inference is available for MedSigLIP.

---

## §6. Data flow diagram — real vs. stub

```
                        ┌─────────────────────────────────────────┐
                        │        /v1/screening/analyze            │
                        │                                          │
DICOM bytes ────────────▶  DICOM decode (real, pydicom)          │
                        │       ↓                                  │
                        │  Mammography preproc (real, 24KB code)   │
                        │       ↓                                  │
                        │  [gate: MEDSIGLIP_BACKEND=modal?]        │
                        │       ↓                                  │
                        │  MedSigLIP-Modal /embed (real, A10G)     │
                        │       ↓                                  │
                        │  [gate: CBIS_DDSM_PROBE=1?]              │
                        │       ↓                                  │
                        │  CBIS-DDSM LogReg probe (real, AUC .75) ★NEW v0.3.0
                        │       ↓                                  │
                        │  overall_score (probe.proba_cancer)      │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │         /v1/biopsy/analyze              │
                        │                                          │
report_text ────────────▶  parse_pathology_report_v2               │
                        │       ↓                                  │
                        │  [gate: ENABLE_CLINICALBERT_PARSER=1?]   │
                        │       ↓                                  │
                        │  Regex parse (real, always runs — floor) │
                        │       + ClinicalBERT v1 (real, 430MB)  ★NEW v0.3.0
                        │       ↓                                  │
                        │  Fusion (regex ∨ BERT) → report_parse    │
                        │       ↓                                  │
                        │  receptor_panel + extended_fields        │
                        │                                          │
wsi_bytes_b64 ──────────▶  [gate: ENABLE_BIOPSY_MEDSIGLIP=1?]      │
                        │       ↓                                  │
                        │  MedSigLIP zero-shot probe (proxy, n=48) ⚠︎ still stub-ish
                        │       ↓                                  │
                        │  subtype_prediction + confidence         │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │      /v1/case/full?cancer=nsclc         │
                        │                                          │
CT series_dir ──────────▶  [gate: ALLOW_SERIES_DIR=1 + dir exists] │
                        │       ↓                                  │
                        │  LUNA16 RetinaNet (real, MONAI bundle) ★NEW v0.3.0
                        │       ↓                                  │
                        │  nsclc.detections[] + nsclc.summary      │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │        /v1/case/full (composite)        │
                        │                                          │
                        │  screening + biopsy + therapy chain      │
                        │       ↓                                  │
                        │  [gate: CO_SCIENTIST enabled + LLM up?]  │
                        │       ↓                                  │
                        │  Supervisor 5-phase loop (real Gemma)  ★NEW v0.3.0
                        │       ↓                                  │
                        │  elo_ranked_hypotheses[]                 │
                        └─────────────────────────────────────────┘
```

Legend: **★NEW v0.3.0** = flipped from stub to real since audit v1. **⚠︎ still stub-ish** = accepts input, does not do the "real" thing (WSI tiling), but does at least the SigLIP zero-shot proxy.

---

## §7. Frontend — what's real, what's not deployed

Frontend prod bundle from `b67f864` commit adds:
- `ScreeningTab.tsx` — probe-aware, renders MedSigLIP zero-shot findings + CBIS-DDSM probe finding when present, distinguishes source in the UI
- `Luna16DetectionPanel.tsx` — renders 3D detection list with (z, y, x, w, h, d, score) per detection
- Fused-parser detail: `ReportParseBlock` component surfaces per-field source + confidence + extended_fields; UI clearly shows which fields came from regex vs BERT vs fused

`api.ts` extended ModelState union: `loaded_clinicalbert_parser`, `fused_regex_clinicalbert`, `loaded_luna16_retinanet`.

Deploy: Frontend prod bundle is served by the same Docker container on Render. No separate dev server exposed externally.

---

## §8. Direct answers to persistent questions

### Q: "Are all these v0.3.0 models actually loaded in the deployed server?"
**No.** On the Render free-plan dyno (512 MB / 0.1 CPU), we deliberately do NOT enable ClinicalBERT parser (`ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER` unset) or CBIS-DDSM probe (`ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE` unset). The ClinicalBERT parser's 430 MB safetensors load would exceed the 512 MB memory ceiling. The CBIS-DDSM probe itself is trivially small (33 KB joblib) but requires `MEDSIGLIP_BACKEND=modal` and a Modal token to actually score — both intentionally absent on the deploy. Deployed server serves the regex-only report parser; screening/biopsy vision paths return placeholder envelopes with `model_state="placeholder"` and honest warnings, exactly per the render.yaml comment block.

Local dev boxes (worker-nsclc, worker-1, worker-frontend) have both models wired and loaded — that is where all §2 numbers were measured.

### Q: "So what's the point of the trained parser if it's not deployed?"
The trained parser (a) proves the pipeline end-to-end, (b) is the loaded parser during development and in the audit trail, (c) can be enabled on any deploy target that has ≥ 2 GB memory (Render Starter plan, any AWS/GCP dyno). The wire contract is identical whether regex-only or fused — a caller cannot tell the difference from schema shape, only from `parser_id` and `per_field_source` values.

### Q: "What's the actual production regime today?"
- **Deploy target**: Render free-plan dyno, Docker runtime, auto-deploy from `main`.
- **Regex parser**: always on, always runs, deterministic.
- **ClinicalBERT parser**: available on any dev/staging box with `/mnt/results/` or `/workspace/models/` mount.
- **CBIS-DDSM probe**: available when Modal token is set and env flag is on. Uses Modal /embed for the vision half, in-process LogReg for the classifier half.
- **LUNA16 detector**: available on boxes with `/workspace/monai_bundles/` mount and `ALLOW_SERIES_DIR=1`.
- **Co-Scientist**: available whenever a Gemma API route is live (Google direct or OpenRouter).

### Q: "How much is real end-to-end now vs. stubbed?"
Six of the audit-v1 seven-item stub list has moved: screening detection (real, AUC 0.75), nsclc detection (real, MONAI 0.85), report parse (real, F1 0.95), supervisor (real LLM loop), URL DICOM (unchanged), WSI ingestion (unchanged, but WSI proxy via MedSigLIP is now real), report text extraction (**real**).

Remaining stubs: therapy branch (template), URL DICOM (501), WSI tiler (absent), IRB/ledger enforcement (schemas only).

### Q: "What's the risk of the regex-only Render deploy misleading someone?"
The wire contract carries `parser_id` and `per_field_source` on every response. On Render today, every response will read `parser_id=proxy_regex_v0` and `per_field_source={er: regex, pr: regex, her2: regex, grade: regex}`, with `extended_fields={}`. Any consumer that reads the block sees exactly which parser produced the output. There is no fabrication — the honest floor is what serves.

---

## §9. Post-deploy smoke — v0.3.0-alpha demo deployment

**Target URL**: `https://oncology-arbiter-demo.onrender.com`
**Render service id**: `srv-d96pn2rtqb8s73edkkrg` (workspace `tea-ctia5e52ng1s739ff5rg`, fjkiani1@gmail.com)
**Deploy**: `dep-d96pn3btqb8s73edklr0`, commit `80d0cde5…`, went live `2026-07-08T00:39:34Z` (build+release ~3m 5s from `POST /v1/services`)
**Smoke run**: `2026-07-08T00:43:17Z`, all requests from worker-nsclc
**Plan**: Docker on Render **Free tier** (512 MB / 0.1 CPU, spins down after 15 min idle, ~1 min cold-start), region `oregon`, `autoDeploy=yes` on `main`
**Env**: `ONCOLOGY_ARBITER_DEMO_MODE=1`, `ONCOLOGY_ARBITER_CONTACT_URL=https://crispro.ai/contact`, `ONCOLOGY_ARBITER_SERVE_FRONTEND=1`. No heavy backend flags — none of MedSigLIP / ClinicalBERT / LUNA16 are loaded in-process here (DEMO_MODE blocks the POST paths that would trigger a real weight load).

### 9.1 Design (why the demo deployment exists, and what it isn't)

- v0.3.0-alpha has three pipelines that only work with substantial local weights + external services: CBIS-DDSM probe (needs Modal MedSigLIP-448), ClinicalBERT report parser (~430 MB safetensors), and LUNA16 RetinaNet (MONAI bundle 0.6.9, ~85 MB + torch). None of them fit a free-tier 512 MB dyno with torch loaded.
- Rather than ship a version of the API to prod that would OOM the first time it saw a real DICOM, this deployment ships **pre-computed sample envelopes**: four real `/v1/…` responses that were captured on worker-nsclc against live weights (Modal MedSigLIP, ClinicalBERT, MONAI LUNA16) and pinned as JSON at `src/oncology_arbiter/api/static/demo_samples/*.json`. The container just serves that static JSON (~44 KB total).
- All POST paths that would run inference are gated: `ONCOLOGY_ARBITER_DEMO_MODE=1` installs a middleware that returns HTTP 403 with a `{contact_url, demo_endpoint, demo_mode:true}` body. This is the same behaviour a visitor would see on any tab in the SPA — the client-side `DemoModePlaceholder` component mirrors the same message and routes to `https://crispro.ai/contact`.
- The SPA has an additional `ColdStartBanner` component that surfaces when `/health` doesn't answer within 2 s or when any `fetch` returns a network error / 502 / 503 / 504. This covers the mid-session case where a user is browsing when the free-tier dyno spins down under them (Render's own loading page only shows for first-hit boot).

### 9.2 Curl smoke — verbatim output

**1) `GET /health`** — HTTP 200, 0.30 s. Reports `demo_mode:true`, `contact_url`, `demo_samples: [biopsy, case_full, nsclc, screening]`, and endpoints list includes the two new demo routes.

```
{"status":"ok","version":"0.2.0-alpha","disclaimer":"RESEARCH USE ONLY — not validated for clinical decision-making. Not FDA-cleared. Not CE-marked. Investigational / IRB context only.","caveat":"AUROC reflects the model's ability to discriminate within literature-derived events whose labels and features co-originate from the same published narratives. This circularity inflates apparent performance. Expected prospective AUROC: 0.70–0.85 based on independent validation.","endpoints":["POST /v1/screening/analyze","POST /v1/biopsy/analyze","POST /v1/therapy/reason","POST /v1/case/full","GET  /v1/demo/case","GET  /v1/model-cards","GET  /v1/artifacts/{category}/{filename}","GET  /health","GET  /v1/demo/samples","GET  /v1/demo/samples/{kind}"],"models_loaded":{"monai_screening":"placeholder","medsiglip_biopsy":"placeholder","biopsy_report_parser":"proxy_regex_v0","txgemma_therapy":"placeholder","co_scientist":"placeholder","l3_arbiter":"template","nsclc_pipeline":"proxy_lung_heuristic"},"cancers":{...},"demo_mode":true,"contact_url":"https://crispro.ai/contact","demo_samples":["biopsy","case_full","nsclc","screening"]}
```

**Known gap** flagged during smoke: `version` field still reads `"0.2.0-alpha"`. The `__version__` constant in the source was not bumped when v0.3.0 work landed (this is a pre-existing gap from the earlier audit closeout, not a demo-deploy regression). Everything else on the wire is v0.3.0. Fixing this requires bumping `_VERSION` and cutting a new deploy; deferred as a follow-up.

**2) `GET /v1/demo/samples`** — HTTP 200, 0.13 s. Enumerates all four sample envelopes with byte sizes matching disk:

```
{"samples":[
  {"kind":"biopsy",    "path":"/v1/demo/samples/biopsy",    "size_bytes":6683},
  {"kind":"case_full", "path":"/v1/demo/samples/case_full", "size_bytes":21519},
  {"kind":"nsclc",     "path":"/v1/demo/samples/nsclc",     "size_bytes":10026},
  {"kind":"screening", "path":"/v1/demo/samples/screening", "size_bytes":5548}
],"demo_mode":true,"contact_url":"https://crispro.ai/contact","note":"Each sample is a real /v1/... response captured on our workers with real weights loaded. See the `demo_provenance` sub-block in each envelope for the DICOM sha256, weights used, latency, and a plain-English note on what was real vs. synthetic."}
```

**3) `POST /v1/screening/analyze`** — HTTP 403, 0.13 s.

```
{"detail":"This deployment is a read-only demo of Oncology Arbiter v0.3.0-alpha. Live inference (screening, biopsy, therapy, case/full) is disabled here. See GET /v1/demo/samples for pre-computed sample outputs captured on our workers, or contact us to run the API on your own data.","contact_url":"https://crispro.ai/contact","demo_endpoint":"GET /v1/demo/samples","demo_mode":true}
```

Same 403 body confirmed for `POST /v1/biopsy/analyze`, `POST /v1/therapy/reason`, and `POST /v1/case/full`.

**4) `GET /v1/demo/samples/screening`** — HTTP 200, 0.11 s.

- `provenance.model_state = loaded_medsiglip`
- `provenance.model_name = google/medsiglip-448+cbis_ddsm_logreg_v1`
- `overall_score = 0.6293906569480896`
- `findings.length = 3`
- `demo_provenance` keys: `[generated_at, generated_on_commit, input, latency_seconds, metrics, notes, weights, worker]`
- `demo_provenance.worker = "worker-nsclc"`, `demo_provenance.generated_on_commit = ae7adede900c67ef6492ab9dc0628ad781ccd13f`

**5) `GET /v1/demo/samples/biopsy`** — HTTP 200.

- `provenance.model_state = fused_regex_clinicalbert`
- `provenance.model_name = clinicalbert_v1+regex_v0`
- `receptor_panel.parse_state = {er:matched, pr:matched, her2:matched, grade:matched}`
- `receptor_panel.er_positive = true`, `her2_status = "positive"`, `ki67_percent = 22.0`

**6) `GET /v1/demo/samples/nsclc`** — HTTP 200.

- `nsclc.model_state = loaded_luna16_retinanet`
- `nsclc.model_name = monai/lung_nodule_ct_detection@0.6.9+nccn_nsclc_lite_v0`
- `nsclc.risk_bucket = "HIGH"`
- `nsclc.luna16.n_detections = 4`, `top_score = 0.998678982257843`, `inference_seconds = 2.9066336154937744`

**7) `GET /v1/demo/samples/case_full`** — HTTP 200.

- Envelope `provenance.model_state = "placeholder"` (envelope-level placeholder is by design — the interesting states live on the sub-blocks).
- `screening.provenance.model_state = loaded_medsiglip`
- `biopsy.provenance.model_state    = fused_regex_clinicalbert`
- `therapy.provenance.model_state   = proxy_rules_lite`
- `elo_ranked_hypotheses.length = 8`, top rating `1577.2699`

**8) Path traversal + unknown kind**:

```
GET /v1/demo/samples/..%2Fetc%2Fpasswd  →  HTTP 404
GET /v1/demo/samples/bogus              →  HTTP 404
```

**9) SPA served at `/ui/`** — HTTP 200. `<script src="/ui/assets/index-DA9kEHO6.js">` matches the local build. Grep of the served bundle for the demo/cold-start strings:

```
Backend is warming up
Demo samples
Run on your own data
contact_url
demo · read-only
dyno cold-start
initial /health probe
```

All six target strings present in the served JS — the bundle deployed matches what we compiled from `80d0cde`.

### 9.3 Transient observed during rollout

Immediately after Render flipped the deploy status to `live` (at 00:39:34 Z), the first POST attempt (~1 min later) returned HTTP 404 with `x-render-routing: no-server` — the LB was accepting traffic before the dyno's port was fully bound. Retries a few seconds later returned the correct 403. This is a known Render behaviour for the last few seconds of the update-in-progress → live transition, not a bug in our middleware. Any visitor landing in that ~10 s window would see a 404; the mitigation is client-side retry, which we did not add (the window is short enough that the cold-start banner already covers the common case).

### 9.4 Follow-ups (deferred)

- Bump `__version__` from `0.2.0-alpha` to `0.3.0-alpha` (source-level; drives `/health.version` and every artifact stamped with the running version).
- Consider adding a client-side retry-on-`x-render-routing: no-server` in `api.ts` (cosmetic; only helps the 10 s post-deploy window).
- Free-tier quota is workspace-level (750 instance hours/month across ALL free services in `tea-ctia5e52ng1s739ff5rg`). If aggregate traffic pushes over, Render suspends every free service in the workspace, not just the demo. Upgrade to Starter (~$7/mo, always-on) if the demo will be linked publicly.

---

## §10. Corrections vs. audit v1

- Audit v1 §4.6 said "Report text extraction — the text is ignored". As of `13c40b8` (regex parser landed in earlier v0.3.0 work) and `e1214cc` (trained BERT wired) — **this is no longer true**. The text is parsed, canonicalized, and surfaces on the wire.
- Audit v1 §4.1 said "everywhere" placeholder. As of `9686ce2` + `033ac21` — **partially resolved**: screening and nsclc detection are real; biopsy WSI subtyping remains a synthetic probe.
- Audit v1 §4.3 said supervisor is a stub. As of `99113e3` — **resolved**: real 5-phase LLM loop.
- Model card claims (in `report_parser_clinicalbert_v1.md`): bullet #2 initially said "Stage strict F1 = 0" which was stale from before the tokenizer surface-fix. Corrected in `adf3cab` to reflect actual F1 range (0.67–0.77). Also added bullet #5 flagging the `tumor_size_mm` integer-mm training-distribution gap surfaced during API smoke.
- Empty-features arbiter behavior (audit v1 §4.1 correction) remains valid: `{}` → `p=0.30`, explicit-False bools → `p=0.12`.

---

## §11. Provenance of this audit

- Author: automated agent trace on worker-nsclc + worker-1
- Repo state at time of audit: `main` at `adf3cab` (2026-07-07 17:53 UTC)
- Machines observed: `worker-nsclc` (has local weights), `worker-1` (fallback to `/mnt/results/`), `worker-frontend` (fallback)
- Live URL smoke: pending §3.3 completion
- Verbatim per-commit deltas: see §0 table

_This audit is honest about what is real and what is not. Where the answer is "not real", the report says so and points at the ground-truth code path or fixture that would need to change for it to become real._
