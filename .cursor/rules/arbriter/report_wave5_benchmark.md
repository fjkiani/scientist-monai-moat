# Wave-5B Benchmark Report — oncology-arbiter

**Author:** Biomni for Rahima Nayeem
**Date:** 2026-07-20
**Scope:** Real end-to-end benchmark/eval run across 5 workers, sourced from user-uploaded EGA (BriTROC-1 `EGAD00001011049`) + Synapse credentials.
**Approved plan:** `execution_trace/PLAN.md` (Wave-5B v3).
**Discovery-only, non-clinical.** No FDA/CE claims. No PHI persisted to any durable store (see §7).

---

## 1. Headline numbers

| Task | Metric | Value | 95% CI | n |
|---|---|---:|---|---:|
| HGSOC — dominant CN-signature call | accuracy | **0.939** | [0.891, 0.973] | 147 |
| HGSOC — HRD status (determinate only) | accuracy | **1.000** | [1.000, 1.000] | 61 |
| HGSOC — therapy top-1 (router vs ground-truth category) | top-1 acc | **0.769** | [0.701, 0.837] | 147 |
| HGSOC — therapy top-3 recall | top-3 recall | **0.884** | — | 147 |
| NSCLC — LUNA16 apex detection (biopsy-proven, 18d22267dda20633) | n_detections + risk | **4, HIGH** | — | 1 |
| NSCLC — apex 5× stability | deterministic | **1.000** | — | 5 |
| Breast — CBIS-DDSM zero-shot filename-agreement (medSigLIP-modal) | agreement | **20.0%** | — | 5 |
| LLM co-scientist JSON parse (Modal Qwen2.5-7B gemma-fallback) | parse-ok | **93.3%** | — | 15 |
| EGA BriTROC-1 credential reachability | reaches dataset | **1.000** | — | 679 BAMs listed |

**T1..T9 gate: 9 pass, 1 skipped (deployed-Render smoke opt-in — Render deployment at `oncology-arbiter.onrender.com` is stale at 0.3.0-alpha and does not carry the new HGSOC endpoint until keys are rotated and redeployed).**

Leaderboard: `execution_trace/wave5_benchmark_leaderboard.json` (`schema_version=wave5.v1`, 9 rows, every row carries `input_sha256` + `response_sha256`).
Provenance walk: `execution_trace/wave5_gate/enforce_provenance.py` resolves 11/11 declared sha256 targets to on-disk artifacts.

---

## 2. HGSOC benchmark (primary)

### 2.1 Cohort

- **Source:** BriTROC-1 (Macintyre et al. 2018, *Nat Genet*, [1]) derived CN-Sig1..CN-Sig7 exposures, n=147 samples with dominant-signature ground truth.
- Files: `execution_trace/wave5_hgsoc_metrics.json`, `execution_trace/wave5_hgsoc_predictions.tsv`, and the labeled cohort at `britroc_cohort_labelled.csv` (sha256 `2f992786ff821e00…`).
- Distributions
  - **Dominant CN-signature** (ground truth): CN1=59, CN2=15, CN3=17, CN4=21, CN5=7, CN6=1, **CN7=27** (tandem-duplicator, [1,2]).
  - **HRD status:** HRD-positive n=57, HRD-negative n=4, indeterminate n=86.
  - **Therapy ground truth:** parp_maintenance=23, cdk_pathway_trial=25, platinum_taxane_soc=99 (immunotherapy_trial=0 in this cohort).
- **Germline calls** are labeled `synthetic_germline_prior` on every row. The router's HRD rule chain is the *same* rule that generated the germline prior → HRD ground truth, so the 100% HRD accuracy is **by construction, not an independent validation** (see §5).

### 2.2 Router contract

- Endpoint: `POST /v1/case/hgsoc/analyze` (see `oncology_arbiter/api/schemas.py::HgsocAnalyzeRequest`, `HgsocAnalyzeResponse`).
- Router: `oncology_arbiter/hgsoc/router.py` — deterministic argmax over CN1..CN7 with a 5% tie-band; HRD rule chain over germline `BRCA1 / BRCA2 / CDK12 / other_hrd_gene` + signature-derived tandem-duplicator flag; therapy list drawn from Macintyre 2018 + NCCN Ovarian v2.2025 + SOLO-1 + ARIEL3 with citation_url on every option [1,2,3,4].
- Sig7 = **Macintyre CN-Sig 7 (tandem-duplicator, CDK12/BRCA1-associated), NOT COSMIC SBS7.** This distinction is asserted on every response and in the credential-smoke attestation.

### 2.3 Signature call

- Accuracy = **0.939 (95% CI [0.891, 0.973])** on n=147.
- Confusion (ground truth × predicted; AMBIGUOUS = tie inside 5% band):
  - CN7 → CN7 26, AMBIGUOUS 1
  - CN3 → CN3 17
  - CN2 → CN2 11, AMBIGUOUS 4
  - CN1 → CN1 56, AMBIGUOUS 3
  - CN4 → CN4 20, AMBIGUOUS 1
  - CN5 → CN5 7
  - CN6 → CN6 1
- The 9 non-diagonal rows are **all AMBIGUOUS**, not mis-classifications — the argmax was inside 5% of the runner-up. No CN-signature is confused with another; the router only ever abstains.

Figure: `execution_trace/figures/hgsoc_signature_confusion.png`

### 2.4 HRD call

- Accuracy = **1.000 [1.000, 1.000]** on determinate-only n=61.
- Confusion is clean: HRD_positive/HRD_positive 57, HRD_negative/HRD_negative 4, indeterminate/indeterminate 86.
- **Caveat noted above:** this is a construction check, not an independent HRD validator.

### 2.5 Therapy call

- Router returns `ranked_therapies[]` (each with `name`, `category`, `rationale`, `citation_url`).
- Top-1 accuracy: **0.769 [0.701, 0.837]** (n=147).
- Top-3 recall (ground-truth category appears anywhere in ranked list): **0.884** (n=147).
- McNemar vs random therapy baseline: b=90, c=4, **p₂ = 3.22 × 10⁻²²**.
- Confusion on top-1 (ground truth × predicted category):
  - parp_maintenance → parp_maintenance 23 (perfect)
  - cdk_pathway_trial → cdk_pathway_trial 24, platinum_taxane_soc 1
  - platinum_taxane_soc → platinum_taxane_soc 66, immunotherapy_trial 19, other 14
- Interpretation
  - The 19 platinum-SOC → immunotherapy_trial cases are the router surfacing an immunotherapy trial as *rank 1* on platinum-sensitive cases where NCCN would keep platinum-taxane as SOC; the trial option is a defensible second-line but should not be rank 1.
  - The 14 platinum-SOC → "other" cases are all `MDT tumor board review` (an `other`-category fallback) appearing at rank 1. Same rank-order pathology.
  - Both categories are recovered in top-3 (top-3 recall 0.884 vs top-1 0.769), which is the clinically-honest metric because the router always emits multiple options for MDT review.

Figure: `execution_trace/figures/hgsoc_therapy_confusion.png`

---

## 3. NSCLC — LUNA16 apex regression

- Endpoint: `crispro--luna16-detect.modal.run`, model state `loaded_luna16_retinanet`.
- Apex case (LIDC-IDRI biopsy-proven nodule, case_id `18d22267dda20633`)
  - HTTP 200 in 40.3 s.
  - **n_detections = 4, top diameter 15.77 mm (score 0.031), top score 0.711.**
  - Risk bucket: **HIGH** (Fleischner ≥ 8 mm).
  - All four detection diameters: 12.79, 13.85, 13.08, 15.77 mm; all four scores between 0.031 and 0.711.
- **Determinism:** 5 identical POSTs replayed on the same case produced **1 unique response sha256**.
- Zenodo LUNA16 annotations pulled fresh: 136 986 bytes, sha256 `db9adb75b381f3e9…`.

Figure: `execution_trace/figures/luna16_apex_detections.png`

---

## 4. Breast — CBIS-DDSM zero-shot (negative finding)

- Endpoint: `medsiglip-modal-v0.4.0-alpha` (medSigLIP zero-shot with 5 labels: benign_mass / malignant_mass / benign_calc / malignant_calc / normal).
- 5/5 DICOMs went through as `dicom_b64` → `probs[]` + `top`; all HTTP 200 at ~1.25 s.
- **All prob scores fall in 2 × 10⁻⁵ .. 6 × 10⁻⁵ across all 5 labels, on every case.** The distribution is degenerate; filename-agreement 20 % is essentially chance on unwindowed CBIS-DDSM full-field digital mammograms.
- **Negative finding recorded.** medSigLIP zero-shot in this form is not a useful breast-imaging component; needs preprocessing (window/level, pectoral removal, ROI crop) — none of which the current Modal endpoint applies. This is a real product signal: the current CBIS-DDSM path in production should not be presented as an oncology decision aid.

---

## 5. LLM co-scientist eval (Modal Qwen2.5-7B / gemma-fallback)

- Endpoint used: `crispro--gemma-fallback-chat.modal.run` (Modal Qwen2.5-7B; MedGemma-27b timed out at 604 s cold-start; GEMINI_API_KEY not present in sandbox — so we exercised the **same fallback path production would take on Gemini upstream failure**).
- 15 prompts (5 breast, 5 NSCLC, 5 HGSOC): **15/15 HTTP 200, 14/15 JSON parseable → parse-ok 93.3%.**
- The single parse failure (`ns01`) was completion-length truncation at max_output_tokens = 512 (completion_tokens = 361, hit the cap mid-JSON). Fix: raise cap to ~700 for complex 4-hypothesis prompts.
- Evidence honesty
  - 40 real URLs across the 14 parseable runs, 14 explicit `null`s (`n_evidence_urls_null`).
  - Null rate 25.9 % — the model *does* honestly say "I don't have a citation for hypothesis N" instead of fabricating.
- Latency: median 11.0 s, p95 14.6 s.
- Cost: 6 376 total tokens, **≈$0.00283** at Qwen2.5-7B Modal-hosted scale.

Figure: `execution_trace/figures/llm_eval_summary.png`

---

## 6. EGA BriTROC-1 credential reachability

`execution_trace/wave5_ega_credential_smoke.json`

- Credentials parsed from `EGAD00001011049.env.pdf` (JSON-embedded creds).
- Fingerprints (SHA-256, first 12): username `96ff7f9f4e38…`, password `973ae201fcfe…`.
- pyega3 v5.2.0 CLI authentication succeeded for `fahad@jedilabs.org` and dataset `EGAD00001011049` is reachable.
- **679 BAM entries listed in the dataset (475.72 GB total).**
- `bam_policy = "credential_reachability_only"` — no BAM bytes crossed any sandbox boundary. Not one .bam / .bai was persisted anywhere under `/mnt/shared-workspace` or `/mnt/results` (T9 asserts this).

---

## 7. Governance, credentials, and PHI

### 7.1 The three uploaded credentials must be treated as EXPOSED

All three sets of credentials that Rahima uploaded were shipped as PDFs with plaintext values. Fingerprints (SHA-256, first 12 chars — the raw values are never persisted):

| Credential | SHA-256 (first 12) |
|---|---|
| EGA username | `96ff7f9f4e38…` |
| EGA password | `973ae201fcfe…` |
| Synapse PAT (JWT scope `view download modify`) | `213614f84622…` |
| SAS username | `4c3963d0dd2b…` |
| SAS password | `2070f9590eb5…` |

**Action required from user:** rotate all three credential bundles at their respective providers (EGA, Synapse, SAS) before the next Render redeploy.

### 7.2 Synapse scope

The uploaded Synapse token decoded to scope `view download modify`. Wave-5B did not exercise Synapse — the plan explicitly excluded it. If the Synapse path is re-opened, the guidance is **read-only usage only, no matter the token's granted scope**, until a scoped-down PAT is generated.

### 7.3 EGA controlled-access data

Rahima confirmed DAC approval on record. Wave-5B respected that scope by:

- listing the dataset only (679 BAM entries),
- never pulling BAM bytes across the sandbox boundary,
- never writing any BAM/BAI byte to `/mnt/shared-workspace` or `/mnt/results` (T9 continuously verifies this).

The HGSOC benchmark itself uses **Macintyre 2018 published CN-signature exposures**, not raw reads — a legally and scientifically cleaner surface for the router-accuracy question the endpoint answers.

### 7.4 Leak guard result

`credential_leak_guard.py` scanned `/mnt/shared-workspace/shared/wave5`, `docs`, `scripts`, and `src` for the plaintext of any of the 5 secrets above. **`n_leaks = 0`.**

---

## 8. T1..T9 gate matrix

Formal harness: `execution_trace/wave5_gate/test_wave5_gate.py` (also lives at `tests/unit/test_wave5_gate.py` in the canonical repo).
Run: `PYTHONPATH=src python3 -m pytest tests/unit/test_wave5_gate.py` — **9 pass, 1 skipped**.

| Gate | Meaning | Status |
|---|---|:---:|
| T1 | Leaderboard `schema_version="wave5.v1"` + input+response sha on every row | **PASS** |
| T2 | Every leaderboard sha resolves to a real artifact (11/11 targets) | **PASS** |
| T3a | Local `/health` returns 200 | **PASS** |
| T3b | Deployed Render `/health` returns 200 | **SKIPPED (opt-in)** — Render deploy stale until keys rotated |
| T4 | Legacy `/v1/case/full?cancer=hgsoc` still returns 501 (no silent upgrade) | **PASS** |
| T5 | LUNA16 apex `n_detections==4` + `risk_bucket=="HIGH"` + 5× deterministic | **PASS** |
| T6 | LLM co-scientist ledger has ≥1 HTTP-200 run producing parseable hypotheses | **PASS** |
| T7 | Credential leak guard `n_leaks == 0` | **PASS** |
| T8 | `/v1/case/hgsoc/analyze` accepts canonical payload, returns `ranked_therapies[]`, correctly surfaces PARP for BRCA1+/platinum-sensitive | **PASS** |
| T9 | No BAM/BAI bytes persisted anywhere in the wave-5 tree | **PASS** |

Provenance CLI: `execution_trace/wave5_gate/enforce_provenance.py` — resolves all 11 leaderboard shas to on-disk artifacts.

---

## 9. Files

Durable, user-visible:

- `report_wave5_benchmark.md` — this document
- `execution_trace/PLAN.md` — approved plan (v3)
- `execution_trace/wave5_benchmark_leaderboard.json` — 9-row leaderboard, `schema_version=wave5.v1`
- `execution_trace/wave5_ega_credential_smoke.json` — EGA reachability + leak guard
- `execution_trace/wave5_hgsoc_metrics.json` — signature + HRD + therapy metrics
- `execution_trace/wave5_hgsoc_predictions.tsv` — per-sample predictions (n=147)
- `execution_trace/wave5_luna16_metrics.json` — apex + 5× stability
- `execution_trace/wave5_cbis_metrics.json` — CBIS-DDSM medSigLIP zero-shot (negative finding)
- `execution_trace/wave5_co_scientist_metrics.json` — 15-prompt LLM eval
- `execution_trace/wave5_gate/test_wave5_gate.py` — T1..T9 harness
- `execution_trace/wave5_gate/enforce_provenance.py` — provenance CLI
- `execution_trace/figures/hgsoc_signature_confusion.png` (+ .svg)
- `execution_trace/figures/hgsoc_therapy_confusion.png` (+ .svg)
- `execution_trace/figures/luna16_apex_detections.png` (+ .svg)
- `execution_trace/figures/llm_eval_summary.png` (+ .svg)

Canonical repo (source of truth for the endpoint + tests):

- `oncology_arbiter/api/app.py` — `create_app()` factory, `POST /v1/case/hgsoc/analyze` route
- `oncology_arbiter/api/schemas.py` lines 1340–1427 — `HgsocAnalyzeRequest` / `HgsocAnalyzeResponse`
- `oncology_arbiter/hgsoc/router.py` — deterministic HGSOC router
- `tests/unit/test_wave5_gate.py` — T1..T9 harness
- `scripts/enforce_provenance.py` — provenance CLI
- `scripts/wave5_bench/credential_leak_guard.py` — T7 CLI

---

## 10. Limitations & honesty declarations

1. **Discovery-only.** All subgroups are n < 100 outside the HGSOC signature n=147. No clinical claims.
2. **HRD 100% is a construction check.** The router HRD rule and the ground-truth generator share the same germline rule chain. This is a wiring test, not an independent HRD validator.
3. **CBIS-DDSM negative finding.** The current medSigLIP zero-shot path is not discriminative on unwindowed CBIS-DDSM. Do not present the endpoint as a breast decision aid until it has real preprocessing.
4. **LLM eval is not accuracy.** No adjudicated ground truth exists for hypothesis correctness in a 15-prompt smoke. We measured parse-rate, evidence-URL honesty (null rate 25.9%), latency, and cost — not "is the hypothesis right".
5. **Render deployment is stale at 0.3.0-alpha.** The HGSOC endpoint is only live *locally*; T3b is intentionally skipped until keys are rotated and Render is redeployed.
6. **Credentials are compromised.** The three uploaded PDFs contained plaintext credentials. Rotate before next redeploy.
7. **Sig7 = Macintyre 2018 CN-Sig 7 (tandem-duplicator, CDK12/BRCA1-context)**, not COSMIC SBS 7 (UV mutational signature). Every artifact carries this attestation.
8. **No FDA / CE clearance.** No claim of clinical utility. This is a research prototype.

---

## References

[1] Macintyre G. et al. Copy-number signatures and mutational processes in ovarian carcinoma. *Nat Genet* 2018. doi:10.1038/s41588-018-0179-8
[2] Popova T. et al. Ovarian cancers harboring inactivating mutations in CDK12 display a distinct genomic instability pattern characterized by large tandem duplications. *Cancer Discov* 2016. doi:10.1158/2159-8290.CD-16-0442
[3] Moore K. et al. Maintenance olaparib in patients with newly diagnosed advanced ovarian cancer (SOLO-1). *N Engl J Med* 2018. doi:10.1056/NEJMoa1810858
[4] Coleman R.L. et al. Rucaparib maintenance treatment for recurrent ovarian carcinoma after response to platinum therapy (ARIEL3). *Lancet* 2017. doi:10.1016/S0140-6736(17)32440-6
[5] van Ginneken B. et al. LUNA16 grand-challenge. 2017.
[6] Lee R.S. et al. A curated mammography data set for use in computer-aided detection and diagnosis research (CBIS-DDSM). *Sci Data* 2017. doi:10.1038/sdata.2017.177
