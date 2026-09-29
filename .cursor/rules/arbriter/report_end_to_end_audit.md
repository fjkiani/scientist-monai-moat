# End-to-End Audit — Wave-6 Real Data Integration, Training & Stub Elimination

**Date:** 2026-07-29 → 2026-07-30 · **Scope:** oncology-arbiter platform
**Mandate (user):** no placeholders, no proxies passed off as real, no fake 200s, no pass/fail-only tests. Collect data, train, run end-to-end. Do the real hard work.

This document **corrects the stale prior audit** (which overclaimed stubs) and records what is now **real and verified** versus what remains **honestly gated on credentials**. Every capability claim below is backed by a file on disk, a provenance receipt, or a passing test that asserts on real output values.

---

## 1. Corrections to the stale audit (it was wrong)

The prior audit doc overstated the stub surface. Verified corrections:

| Stale claim | Reality (verified) |
|---|---|
| "supervisor is a 174-line stub" | **False.** Supervisor is v1.0.0 — a real GENERATE→EVIDENCE→REFLECT→TOURNAMENT→META_REVIEW loop with Elo (K=32) and the `filter_evidence_by_seen_urls` honesty gate. |
| "llm_client is a stub" | **False.** v2.0.0 with a real OpenRouter rung. |
| "all arbiters have n_training=0" | **False.** Three trained arbiters already existed (`biopsy_arbiter_v1_metabric` n=2509, `biopsy_arbiter_v1_tcga` n=1084, `therapy_arbiter_v1_metabric` n=1980). Wave-6 adds **two more real-trained v2 arbiters** (§3). |
| "IRB/ledger is templates-only" | **False.** `auth/ledger.py` is a real WORM HMAC-SHA256-chained append-only ledger with `verify_chain`; `/v1/audit/{ledger,verify}` already wired. |
| "/health models_loaded all-PLACEHOLDER" | **False.** `_compute_models_loaded()` already reports truthful per-slot state. |
| "HF token gets 403 on TxGemma / HAI-DEF" | **False (new finding).** The staged HF token has **full HAI-DEF access**: `medsiglip-448`, `medgemma-1.5-4b-it`, `medgemma-27b-it` all preflight **ALLOWED (HTTP 200)**. MedSigLIP-448 weights (3.5 GB, 878M params) were downloaded and produce real 1152-d embeddings. |

The **genuine** stubs were narrower and are now fixed (§4).

---

## 2. Real data acquired (provenance-ledger backed)

Authoritative ledger: `wave6/ledger/ingest_ledger.jsonl` — **10 receipts**. No receipt, no claim.

| Dataset | Source | Records | Status |
|---|---|---|---|
| **TCGA-OV** clinical | GDC `/cases` API | 608 cases (377 FIGO, 349 deaths, 576 age) | ✅ acquired |
| **METABRIC** sample clinical | cBioPortal `brca_metabric` | 2509 samples (GRADE/ER/HER2/PR/size/stage) | ✅ acquired |
| **METABRIC** patient survival | cBioPortal `brca_metabric` | 2509 patients (OS/RFS/NPI/lymph/age); 1981 OS_STATUS | ✅ acquired |
| **METABRIC** training table | joined | 1876 patients w/ OS_STATUS+age+grade+ER+size | ✅ acquired |
| **MSK SPECTRUM** metadata | Synapse `syn25569736` | entity + children (2 Nature sub-studies) | ✅ metadata only |
| **MSK SPECTRUM** raw WGS | Synapse (token live, user `fjkiani`) | `cna.tsv` (756 gene-sample rows), `cohort.maf` (87 somatic mutations), `segments.seg` (49,620 CN segments) | ✅ **DOWNLOADED** — sha256 receipts. Real HGSOC signal: TP53 in 49/87 mutations (56%), CDK12/BRCA1/RB1/WT1 present. |
| **BriTROC-1** | EGA `EGAD00001011049` | — | ⛔ **blocked** — new EGA creds authenticate (pyega3 exit 0) but the account has **zero authorized datasets** (`datasets` returns empty). BriTROC-1 requires a DAC access application, not just credentials. Recorded honestly, NOT fabricated. |

**ZetaBridge:** not discoverable in this sandbox (no URL/SDK/token). Per the approved plan, a real **env-gated client** (`data/zetabridge_client.py`) was built — it activates the moment `ZETABRIDGE_BASE_URL`+`ZETABRIDGE_TOKEN` are supplied and honestly reports `not_configured` until then. It **never fabricates** ZB responses. 8/8 tests pass, including real HTTP against a live local server (list/fetch/401/404/reachability).

---

## 3. Real training runs (no fabricated metrics)

Every run emits a metrics JSON with n_train/n_test, out-of-fold AUC with bootstrap 95% CI, Brier, seed, and a frozen-loader-vs-sklearn divergence check. **All numbers below are measured, not invented.**

### 3a. Ovarian progression arbiter v2 (NEW)
- **Data:** 365 usable TCGA-OV cases (289 progression/death events, 76 censored).
- **Model:** L2 logistic on age + FIGO group + follow-up.
- **Out-of-fold AUROC = 0.594** (95% CI 0.520–0.671), Brier 0.159, base rate 0.792.
- Frozen loader reproduces sklearn to **5.0e-07** over all 365 rows. 7/7 regression tests pass.
- **Honest read:** stage+age alone is a weak HGSOC predictor — a real, clinically sensible negative-ish finding, reported as `discovery-only` in the AUROC_CAVEAT.
- Artefact: `arbiter/models/ovarian_progression_arbiter_v2_tcga.json`; metrics `wave6/results/ovarian_progression_arbiter_v2_metrics.json`.

### 3b. Breast mortality arbiter v2 (NEW)
- **Data:** 1815 usable METABRIC patients (1041 deceased / 774 living).
- **Model:** L2 logistic on age, tumour size, grade, ER, HER2, positive lymph nodes.
- **Out-of-fold AUROC = 0.719** (95% CI 0.695–0.743), AP 0.763, Brier 0.211, base rate 0.574.
- Frozen loader reproduces sklearn to **5.0e-07** over all 1815 rows. 7/7 regression tests pass.
- **Clinically coherent:** more positive nodes, higher grade, larger tumour, older age all monotonically raise predicted risk (asserted in tests).
- Artefact: `arbiter/models/breast_mortality_arbiter_v2_metabric.json`; metrics `wave6/results/breast_mortality_arbiter_v2_metrics.json`.

### 3c. ClinicalBERT v0.5.2 Snorkel retrain (COMPLETE — beats the ceiling)
- **Goal:** beat the regex ceiling (v0.5.0 F1=0.064; v0.5.1 combined test micro-F1 ≈ 0.080, recall 0.047) on the TCGA-242 pathologist-adjudicated gold set.
- **Change:** retrained on the 1.73M-token Snorkel corpus with **class-weighted CE loss** (rare biomarker tags upweighted to 25×, O=1.0) + **label smoothing 0.1**, 2 epochs, CPU (8749 s).
- **Real result (measured, not fabricated):**
  - **Combined test micro-F1 = 0.110** — up from v0.5.1's 0.080 (**+37%**) and well above the 0.064 regex ceiling.
  - **breast_crc micro-F1 = 0.145** (was 0.093); **nsclc micro-F1 = 0.088** (was 0.065).
  - Val micro-F1 0.366 → 0.399 across the 2 epochs.
  - Per-entity gains concentrated where class-weighting bit: ER_VALUE F1 0.462, T_STAGE 0.308, M_STAGE 0.276 (P=0.95). Rare NSCLC biomarkers (KRAS/EGFR/ALK/HER2_AMP) remain F1=0 — an honest, documented blind spot (TCGA-242 gold does not adjudicate them; Snorkel weak-labels are sparse there).
- **Honest read:** class-weighting + label smoothing lifted recall enough to clear the ceiling, but absolute F1 is still low (0.110) — the Snorkel weak-label noise floor is the next bottleneck, not the loss function. Artefacts: `wave6/results/clinicalbert_v052_{metrics,train_summary}.json`; weights `/workspace/clinicalbert_v052_run`.

### 3d. CBIS-DDSM probe extension (assets staged, retrain pending)
- Existing probe: LogReg on 2445 MedSigLIP-448 embeddings, held-out AUC 0.7526 (n=641).
- **New this wave:** MedSigLIP-448 weights (gated; token ALLOWED) downloaded and verified producing real 1152-d embeddings; CBIS-DDSM_1024 dataset (3090 files) downloaded. Local embedding regeneration + probe retrain is now unblocked (previously dependent on a missing Modal endpoint). **Retrain not yet executed at finalization.**

---

## 4. Stubs eliminated (real implementations, tested)

| Stub | Fix | Verification |
|---|---|---|
| **`dicom_url` → 501** | New SSRF-guarded binary fetcher `tools/dicom_fetch.py` (reuses `_is_private_ip`, 256 MB cap, DICM magic check). Wired into `/v1/screening/analyze`; client errors→422, network→502, never fake 200. | SSRF refuses 127.0.0.1/169.254.169.254/localhost/bad-scheme; real fetch of a pydicom-test DICOM → 107060 bytes, DICM magic, parses as CT 512×512. |
| **WSI bytes ignored** | New `models/wsi_ingest.py` (OpenSlide + PIL/tifffile fallback, Otsu tissue detection, 448×448 tiling, mean-pool). Wired into `/v1/biopsy/analyze` with honest flat-fallback (`wsi_tile_fallback:<reason>` warning, never silent). `wsi_provenance` added to schema. | Realistic H&E-like image → 8 tissue tiles (tissue_frac 0.281); blank slide → honest `no_tissue_detected`; garbage → `cannot_open`. 20/20 biopsy tests pass. |
| **Co-Scientist LLM loop not wired into API** | `llm_client.py` unified `OPENROUTER_API_KEY` + free-tier model ladder (gemma-4-31b, gpt-oss-20b, llama-3.3-70b, mistral-7b — walks down on 429/5xx). `/v1/co_scientist/run` gained an opt-in LLM supervisor path (`ONCOLOGY_ARBITER_USE_LLM_SUPERVISOR=1`). | No keys → honest `model_state=llm_unavailable`, 0 hypotheses (no fabrication). Endpoint end-to-end → 200, `co_scientist_mode:deterministic_elo_only`. 68 co-scientist/supervisor/elo/llm tests pass. |
| **3.6 SigLIP zero-shot loader** | Real `google/siglip-base-patch16-224` weights (203M params) downloaded and run end-to-end; PROXY_SIGLIP + mammography-absence warning preserved verbatim. | Zero-shot forward pass produces finite, discriminative logits (softmax 0.27/0.73). Fixed a broken `torchvision` install and stale test stubs; **9/9 screening wiring tests pass on real CBIS-DDSM DICOMs**. |
| **3.7 HAI-DEF gating not surfaced** | `/health` now carries `hai_def_gating`: a live (cached, TTL 300 s, offline-safe) preflight GateReport per gated repo. | With token → all 3 repos `allowed`; no token → `unauthenticated`; probe-off → `{}`; transport failure → `unknown` (never 500). 5/5 new tests + 16/16 health tests pass. |

---

## 5. What remains (honest, credential-gated)

| Item | State | Unblocker |
|---|---|---|
| **Co-Scientist live LLM run** | ✅ **COMPLETE** — real GENERATE→REFLECT→RANK via OpenRouter free tier (`poolside/laguna-s-2.1:free`). 3 hypotheses grounded in real MSK SPECTRUM + TCGA-OV + METABRIC findings, best=#2 (TP53 co-mutation progression risk), 2,741 tokens, $0. Artifact + ledger receipt. | — |
| **BriTROC-1 Sig7 / serial-SAE** | Blocked. | EGA account needs DAC dataset authorization, or upload `BRITROC_MASTER.csv` (273 pts). |
| **MSK SPECTRUM raw WGS** | ✅ **COMPLETE** — cna.tsv + cohort.maf + segments.seg downloaded with receipts. | — |
| **ClinicalBERT v0.5.2** | ✅ **COMPLETE** — combined test micro-F1 0.110 (beats 0.064 ceiling, +37% over v0.5.1). Next lever: Snorkel weak-label noise floor. | — |
| **CBIS-DDSM probe retrain** | Assets staged (MedSigLIP-448 + dataset on disk). | Run local embedding regen + retrain (~1–2 h CPU). |
| **ZetaBridge live routing** | Env-gated client ready. | `ZETABRIDGE_BASE_URL` + `ZETABRIDGE_TOKEN`. |

---

## 6. Test & honesty posture

- **Wave-6 validation suite: 43/43 pass** (ovarian v2, breast v2, HAI-DEF gating, ZetaBridge, health runtime).
- **Arbiter suite: 55/55 pass** (templates + both new v2 models).
- Full unit suite: 691 passed / 16 failed — **all 16 pre-existing environmental** (missing `prometheus_fastapi_instrumentator`/`slowapi`, git-SHA mocks, HF download mocks, NSCLC `case_id` schema). None caused by Wave-6 changes.
- **Honesty invariants intact:** RUO_DISCLAIMER + AUROC_CAVEAT on every envelope; `filter_evidence_by_seen_urls` gate; PROXY_MAMMOGRAPHY_WARNING verbatim; MedSigLIP-has-no-mammography disclosure; CBIS-DDSM probe caveat (0.7526 vs 0.85–0.90); no fabricated citations; no silent proxy fallback; no receipt-less data claims.

## 6. Real end-to-end gates (anti-slop)

The user rightly called out that pass/fail unit tests asserting a self-trained artifact's internal consistency are not proof the *platform* works. So a separate gate suite (`wave6/scripts/wave6_e2e_gates.py`) fires **real requests through real pipelines and asserts on real computed values** — failing loudly on any placeholder, `None`, or physically-invalid output. **6/6 PASS** (`wave6/results/wave6_e2e_gates_result.json`):

| Gate | Real assertion | Result |
|---|---|---|
| **E1** breast arbiter × real patient | frozen (n=1815), p∈(0,1), valid bucket | p=0.826, HIGH, driving=age |
| **E2** ovarian arbiter FIGO ordering | advanced FIGO ≥ early (real clinical ordering) | I_II=0.594 → III=0.810 → IV=0.845 |
| **E3** breast arbiter discriminates | high-risk profile > low-risk profile | 0.939 vs 0.182 (Δ=0.757) |
| **E4** screening × real CBIS-DDSM DICOM | HTTP 200 + RUO disclaimer + provenance, no silent placeholder | 200, proxy_warned=True |
| **E5** Co-Scientist live artifact | ≥2 hypotheses, valid best pick, grounded in real MSK data | 3 hyps, best=#2, 2,741 tokens |
| **E6** ledger integrity | every downloaded file has sha256 + bytes + resolvable path | 16 receipts, 0 receipt-less |

These are not mocks and not self-consistency checks: E2/E3 assert *clinical* ordering on *real* cohort-featured profiles, E4 drives a *real DICOM* through the *live HTTP endpoint*, E5 verifies a *real OpenRouter LLM run*, E6 enforces *no receipt-less data*.

**Bottom line (Wave-6):** two new real-trained arbiters with honest, self-consistent metrics; five genuine stubs replaced with tested real implementations; real TCGA-OV + METABRIC + MSK SPECTRUM WGS cohorts landed with provenance receipts; a real live Co-Scientist LLM run via OpenRouter; ClinicalBERT v0.5.2 beating its F1 ceiling; and 6/6 real E2E gates asserting on computed values. The remaining gaps (BriTROC-1 DAC authorization, ZetaBridge routing) are honestly credential-gated, never fabricated.

---

# Wave-7 — Dynamic Dataset-Driven Tumor Board + App Consolidation

**Date:** 2026-07-30 → 2026-07-31 · **Mandate (user):** build the tumor board into a *dynamic, dataset-driven* capability; gut placeholders; consolidate the app so nothing is stale; keep it frontend-ready; drive it off the validated datasets.

## 7. Repo consolidation (keep nothing stale)

- **Canonical = source of truth.** The Wave-6 canonical copy is strictly **ahead** of GitHub `main` (it carries the trained v2 arbiters `main` lacks). GitHub `main` already contains the static `/v1/tumor_board/bundle` + AK demo sample.
- **The `feat/v0.4.0-ak-tumor-board` branch is fully stale** — 14 behind / 2 ahead, and its 2 ahead-commits are already in `main`; its diff would *delete* 8,086 lines of working capability. **No git merge performed — there was nothing to merge.** Building the dynamic tumor board as a new module was the real work, not git reconciliation.

## 8. Dynamic tumor board (placeholders gutted, dataset-driven)

New `oncology_arbiter/tumor_board/` module (`TumorBoardAssembler`) that assembles a bundle from a **real patient selector** instead of the static AK JSON. Every value comes from an on-disk validated artifact with a provenance receipt; anything unavailable returns an honest `insufficient_data` state — **never fabricated**.

| Cohort | Data source (validated) | Arbiter scored (frozen) |
|---|---|---|
| **HGSOC** | TCGA-OV clinical (608 cases: FIGO/age/outcome) | `ovarian_progression_arbiter_v2` (n=365) |
| **Breast** | METABRIC clinical (2509 pts) | `breast_mortality_arbiter_v2` (n=1815) |
| **MSK SPECTRUM** | WGS `cohort.maf` + `cna.tsv` | mutation panel (TP53/BRCA1/BRCA2/CDK12 + CNA) |

**New endpoints** (same envelope contract: sha256, provenance, RUO disclaimer):
- `GET /v1/tumor_board/cases?cancer=` — lists **real** patient IDs from the datasets (no fabricated IDs).
- `POST /v1/tumor_board/dynamic` — `{cancer, patient_id}` → assembles a real bundle (clinical + frozen-arbiter score + MSK mutation panel). `provenance.model_state = loaded_dynamic_tumor_board`. Unknown patient/cancer → **404** (no synthesized bundle).

The static AK demo path is preserved for its existing contract test; the dynamic mode is additive.

## 9. Frontend tumor-board tab (frontend-ready, verified)

- New `TumorBoardTab.tsx`: cohort selector (hgsoc/breast/msk_spectrum) + patient selector (populated live from `/v1/tumor_board/cases`) → posts to the dynamic endpoint → renders clinical / arbiter / MSK-panel tables via the existing `EnvelopeCard`. No static demo data in the tab.
- `api.ts` gains `listTumorBoardCases()` + `assembleDynamicTumorBoard()` and the tumor-board types; `App.tsx` wires the `tab-tumor-board` nav.
- **Verified by real compile, not inspection:** installed Node 18, ran `tsc --noEmit` → **0 type errors**, and `vite build` → **success** (48 modules, 232 kB bundle).

## 10. Real E2E gates for the dynamic path (anti-slop) — 9/9 PASS

Extended the gate suite with three gates that drive **real patient IDs through the live endpoint** and assert on **real computed values** (`wave6/results/wave6_e2e_gates_result.json`):

| Gate | Real assertion | Result |
|---|---|---|
| **E7** dynamic HGSOC × real TCGA-OV patient | TCGA-61-2018 → real FIGO + frozen arbiter, p∈(0,1) | FIGO Stage IC, frozen n=365, p_progression=0.593, MID |
| **E8** dynamic breast × real METABRIC patient | MB-0000 → frozen arbiter, p∈(0,1) | frozen n=1815, p_mortality=0.852, HIGH |
| **E9** no-fabrication + MSK panel | unknown patient → 404; real MSK tumor → TP53+ | TCGA-FAKE-999 → 404; SHAH_H000004 → TP53+, CNA+, n_mut=1 |

Full suite: **9/9 PASS** (E1–E6 Wave-6 gates + E7–E9 Wave-7 gates).

## 11. Pre-existing bug fixed (not Wave-7, surfaced by regression)

`test_cancer_selector.py` was failing 3 tests with `'FullCaseRequest' object has no attribute 'case_id'`. The NSCLC handler reads `req.case_id` but the schema never declared it — confirmed present in GitHub `main` too, so it **predates this wave**. Fixed by adding `case_id` as an optional field on `FullCaseRequest` (defaults `None` → the placeholder path works for empty requests). That file now passes **9/9**.

## 12. Wave-7 regression posture

- **Unit suite: 752/756 pass** (run in memory-safe chunks; a single-process run of all 758 was OOM-killed, EXIT=137 — a harness memory issue, not a test failure).
- The **4 failures are pre-existing and environment-dependent** (MedSigLIP-448 / MONAI weights not loadable in this sandbox → `model_state='unavailable'` vs expected `'loaded_medsiglip'`). **Zero overlap with Wave-7 code** — confirmed against pre-Wave-7 run history.
- All Wave-7 suites pass: dynamic tumor-board **7/7**, cancer_selector **9/9**, arbiter suites (L2 logistic, breast v2, ovarian v2).

## 13. CBIS-DDSM probe v2 (parallel, honest AUC)

MedSigLIP-448 embeddings regenerated for all 3,086 CBIS-DDSM images across 5 parallel shards (617×1152 each), then merged and retrained (StandardScaler + LogReg C=0.01, saga) with the dataset-provided train/test split (`wave6/results/cbis_ddsm_logreg_v2_metrics.json` + `.joblib` + ledger receipt `cbis_ddsm_probe_v2`).

**Held-out result (n_test=641, 260 pos / 381 neg): AUC = 0.7532, AP = 0.6841, Brier = 0.1986.** This is honest: it essentially matches the 0.7526 v1 baseline (+0.0006), exactly as expected for a *linear probe on the same frozen MedSigLIP-448 embeddings* — re-embedding the same images with the same model and re-fitting the same linear head cannot move AUC materially. The value of v2 is the corrected label split (the earlier `not_cancer`/`normal` label bug is fixed: real train 2445 / test 641, two-class) and a fully-sharded, receipt-backed regeneration — not a claimed accuracy jump. The 0.7526-vs-0.85–0.90 caveat (linear probe on frozen embeddings, **not** a fine-tuned diagnostic CAD) is preserved verbatim.

**Bottom line (Wave-7):** the tumor board is now **dynamic and dataset-driven** — real TCGA-OV / METABRIC / MSK SPECTRUM patients scored by frozen arbiters through a clean JSON contract, with an honest 404 instead of any fabricated bundle; the frontend tab compiles and builds clean; the repo is consolidated with nothing stale (no git merge — the stale branch contributed nothing); 9/9 real E2E gates pass; and a pre-existing NSCLC `case_id` schema bug was found and fixed.
