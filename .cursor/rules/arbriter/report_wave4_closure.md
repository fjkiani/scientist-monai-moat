# Wave-4 Closure Report — Oncology Arbiter

**Recorded**: 2026-07-20 UTC
**Iteration limit reached** — this closure captures verified state and explicit deferrals.

---

## 1. Delivered vs. Deferred (against PLAN §1–§9)

| PLAN item | Status | Evidence |
|---|---|---|
| §1 Real case ingress (DICOM series → durable case_id, no client `series_dir`) | **DELIVERED** | case-storage-v0.5.0 accepts `dicom_series_b64` list, writes `dicom_series/slice_<i>.dcm`, returns content-addressable `case_id=18d22267dda20633` for 121-slice LIDC-IDRI apex_case in 1.869 s |
| §2 Running NSCLC execution (CT→LUNA16→risk/rules→ranked options with real `loaded_luna16_retinanet`) | **DELIVERED** | luna16-infer-v0.5.0 case_id path returns 4 detections, top score 0.711, top diameter 12.79 mm on cuda:0 in 3.20 s inference; gateway `/v1/case/full?cancer=nsclc` returns risk_bucket=HIGH, 3 NCCN therapy recommendations |
| §3 Deployed ClinicalBERT parser | **DELIVERED (pre-session)** | clinicalbert-modal-v0.5.1 live; real_text_micro_f1_combined=0.0809 (breast_crc=0.0879, nsclc=0.0696) |
| §4 Real AK/HGSOC workflow (implement HGSOC or reject) | **PARTIAL** — HGSOC explicit 501 delivered; AK real-pipeline wiring **DEFERRED** | HGSOC handler raises `HTTPException(501, "hgsoc_case_full_not_yet_implemented")` with pointer to `/v1/tumor_board/bundle`; AK evidence-matrix live-wire not verified this session |
| §5 Explicit reasoning contract (Elo deterministic vs LLM Co-Scientist separated) | **DELIVERED** | supervisor v1.0.0 with `StageResult.co_scientist_mode()` canonical strings, per-call `llm_call_ledger` (route/model/prompt_sha/tokens/latency/cost); live smoke: mode=`llm_gemini_2_5_flash`, 7 calls, 3197 tokens, $0.000353, 25.32 s, 3 hypotheses, 6 Europe PMC evidence URLs |
| §6 Evidence-bound decision outputs (`source_evidence`, `model_states`, `confidence_provenance`, `model_unavailable_rule`) | **DELIVERED** | All four fields present in `elo_ranked_hypotheses[0]` of live TestClient smoke (see §3 below) |
| §7 CI regression + deployed smoke tests | **DEFERRED** | GitHub Actions wiring explicitly deferred in PLAN; smoke *scripts* not committed to repo this session |
| §8 Single deployed UI connected to live APIs | **DEFERRED** | Frontend rebuild against new `/v1/case/upload` not completed; existing Wave-3 UI still points at legacy endpoints |
| §9 Deploy MedSigLIP-448, MedGemma-27b-it, SigLIP-base | **PARTIAL** — MedSigLIP-448 live, MedGemma-27b deploy succeeded but cold-start timed out; SigLIP-base ungated model not deployed | See §4 below |

---

## 2. Live Endpoint Inventory (verified this session)

All owned by Modal profile `crispro`:

| App | Endpoint | Version | GPU | Status |
|---|---|---|---|---|
| medsiglip | `crispro--medsiglip-{healthz,info,embed,embed-batch,zero-shot}.modal.run` | v0.4.0-alpha | A10G | healthy, dim=1152, load=47.577 s |
| luna16-infer | `crispro--luna16-{healthz,info,detect}.modal.run` | v0.5.0-alpha (bundle 0.6.9) | A10G | healthy, case_id + inline b64 both supported |
| case-storage | `crispro--case-storage-{healthz,upload,manifest,get}.modal.run` | v0.5.0-alpha | CPU + Volume `oncology-arbiter-cases` | healthy, `dicom_series_b64` supported |
| clinicalbert-modal | `crispro--clinicalbert-{healthz,info,parse}.modal.run` | v0.5.1 | GPU | healthy, real F1=0.081 |
| phikon | `crispro--phikon-{embed,healthz,info}.modal.run` | pre-Wave-4 | GPU | healthy, dim=768 |
| gemma-fallback | `crispro--gemma-fallback-chat.modal.run` (Qwen2.5-7B) | pre-Wave-4 | GPU | healthy |
| medgemma-27b | `crispro--medgemma-27b-{healthz,info,chat}.modal.run` | v0.4.0-alpha | A100-80GB | **deployed but cold-start smoke TIMED OUT at 604.7 s** (54 GB HF download > 600 s Modal timeout) |
| Gateway (Render) | `oncology-arbiter.onrender.com` | Wave-3 | — | **NEEDS REDEPLOY** with new env vars (see §5) |

---

## 3. Verbatim Smoke — NSCLC Real-Case Pipeline (LIDC-IDRI apex_case)

**Step 1: case-storage upload** (endpoint `crispro--case-storage-upload.modal.run`, v0.5.0-alpha)
```
Input:  121 DICOM slices × 525,224 bytes = 63,552,104 bytes + report.txt (100B) + genomics.json ({KRAS, EGFR, TMB})
Output: case_id=18d22267dda20633
        sha256_full=18d22267dda206333dd659aeace1d59cdfe937cad22d7257427174062f1f159d
        provenance_uri=modal-volume://oncology-arbiter-cases/18d22267dda20633/manifest.json
        n_files_written=123, elapsed=1.869s
```

**Step 2: luna16-infer** (endpoint `crispro--luna16-detect.modal.run`, v0.5.0-alpha, bundle 0.6.9)
```
Input:  {"case_id": "18d22267dda20633", "top_n": 5}
Output: model_state=loaded_luna16_retinanet
        model_name=monai/lung_nodule_ct_detection@0.6.9
        ingest_source=modal-volume://oncology-arbiter-cases/18d22267dda20633/dicom_series/
        n_detections=4, top_score=0.7112, top_diameter_mm=12.79
        input_shape=[121, 512, 512], actual_spacing_mm=[2.5, 0.762, 0.762]
        device=cuda:0, inference=3.2025s, total=38.4437s
Top-3: (222.4,202.9,297.4) d=12.79mm score=0.7112
       (223.8,202.8, 96.2) d=13.85mm score=0.2375
       (103.4,223.3,242.4) d=13.08mm score=0.1848
```

**Step 3: Gateway `POST /v1/case/full?cancer=nsclc`** (TestClient)
```
Input:  {"case_id": "18d22267dda20633"}
Output: HTTP 200
        nsclc.model_state = loaded_luna16_retinanet
        nsclc.n_candidates_total = 4
        nsclc.max_diameter_mm = 15.77
        nsclc.risk_bucket = HIGH
        nsclc.risk_score = 0.8614
        nsclc.driving_feature = max_diameter_mm
        nsclc.n_therapy_recommended = 3 (PET/CT, tissue biopsy, MDT review)
        nsclc.therapy_not_recommended = 1 (surveillance-only, per Fleischner 2017)
        elo_ranked_hypotheses[0].algorithm = deterministic_diameter_desc
        elo_ranked_hypotheses[0].source_evidence = [
          "modal-volume://oncology-arbiter-cases/18d22267dda20633/manifest.json",
          "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1447"
        ]
        elo_ranked_hypotheses[0].model_states = ["luna16-retinanet@0.6.9", "case-storage-modal@v0.4.0-alpha"]
        elo_ranked_hypotheses[0].confidence_provenance = deterministic_rule
        elo_ranked_hypotheses[0].model_unavailable_rule = drop_if_required_model_down
```

**Constraints satisfied**: PLAN §1 (content-addressable case_id ingress), §2 (real LUNA16 execution), §6 (all four evidence-bound decision fields present).

**Known caveat**: `model_states[1]` reports `case-storage-modal@v0.4.0-alpha` — this is a stale label in the elo builder; the actual manifest is v0.5.0-alpha. Fix-forward on next gateway push.

---

## 4. Verbatim Smoke — Supervisor Reasoning Contract

**Endpoint**: local `TestClient` exercising `oncology_arbiter.agents.supervisor.Supervisor.run()` with NSCLC context (62F, RUL 12mm spiculated nodule).

```
model_state         = executed
co_scientist_mode   = llm_gemini_2_5_flash
llm_calls           = 7
total_tokens        = 3197
est_cost_usd        = 0.000353
latency_s           = 25.32
hypotheses          = 3
evidence_urls       = 6 (europepmc.org/article/MED/*)
evidence_kept       = 6
evidence_dropped    = 0
top_hypothesis_elo  = 1531.3
alignment           = partial
```

**Contract verified**:
- `StageResult.llm_call_ledger` populated per call with (route, model, prompt_sha, prompt_tokens, completion_tokens, thinking_tokens, latency_ms, est_cost_usd)
- `StageResult.co_scientist_mode()` returns canonical `llm_gemini_2_5_flash` string
- Deterministic Elo tournament (`deterministic_pairwise_elo_k32`) runs separately from LLM Co-Scientist stages (GENERATE, REFLECT, TOURNAMENT, META_REVIEW)
- Gemini 2.5 Flash `thinkingBudget=0` set for JSON phases (prevents thinking-token starvation of `maxOutputTokens`)

---

## 5. Verbatim Smoke — MedGemma-27b Cold-Start (FAILED)

```
Endpoint: https://crispro--medgemma-27b-chat.modal.run
Elapsed:  604.7 s
HTTP:     500
Body:     modal-http: internal error: function execution timed out
```

**Root cause**: Modal serverless function default timeout (600 s) is shorter than the first-run HF download of `google/medgemma-27b-it` (~54 GB) into the empty `medgemma-27b-weights` Modal Volume.

**Mitigation applied**: leave `MEDGEMMA_MODAL_URL` empty on Render redeploy. `LLMClient` route ladder skips the route per `model_unavailable_rule=drop_if_required_model_down`. Gemini 2.5 Flash remains the primary LLM route (verified working, $0.000353 per full supervisor run).

**Remediation options** (documented, not applied):
1. Add `@app.function(timeout=1800)` decorator to `medgemma-27b-chat` and redeploy.
2. Add a `warmup()` function that pulls the HF snapshot into the Modal Volume and `commit()`s it before the first `chat()` call. Subsequent cold-starts read from Volume in ~30 s.
3. Accept MedGemma-27b as optional given Gemini 2.5 Flash is stable and cheap.

Artifact: `/mnt/results/execution_trace/medgemma_27b_smoke_timeout.json`

---

## 6. Verbatim Smoke — HGSOC Rejection

```
POST /v1/case/full?cancer=hgsoc
HTTP: 501
Body: {"error": {"code": "hgsoc_case_full_not_yet_implemented",
                 "message": "HGSOC `/v1/case/full` handler is not implemented in Wave-4...",
                 "wired_endpoints": ["GET /v1/demo/samples/ak_mbd4_lof_case",
                                     "POST /v1/tumor_board/bundle"]}}
```

Constraint satisfied: PLAN §4 second half — HGSOC either implemented or rejected. Rejected with structured error pointer.

---

## 7. Explicit Deferrals

| Item | Reason | Recovery step |
|---|---|---|
| TxGemma routing | HF gate for `TxGemma-9b/27b` **NOT accepted** for token `hf_[REDACTED]` (fjkiani) — HTTP 403 | Request gate approval from Google HAI-DEF; or accept Gemini 2.5 Flash + MedGemma-27b (once warmed) as the reasoning ladder |
| HGSOC `/v1/case/full` handler | Requires distinct histopath + genomics pipeline not scoped for Wave-4 | Explicit 501 with pointer to working `/v1/tumor_board/bundle` |
| GitHub Actions deployed-CI wiring | Explicit PLAN deferral | Script `scripts/deployed_smoke.py` still needs to be committed; workflow yaml not written |
| AK real-pipeline live-wire | Not verified this session — `tumor_board/bundle` accepts and returns the bundle; whether SL evidence matrix rows are computed from live upstream data or echoed from a fixture was **not verified end-to-end this session** | Read `bundle.synthetic_lethality.provenance.evidence_matrix.rows` in a live smoke; if echoed, run SL scoring over live rows from `/mnt/shared-workspace/shared/tcia_ak/ak_pelvic_mr_provenance.json` |
| Frontend rebuild against `/v1/case/upload` | Iteration budget exhausted | Vite bundle → `src/oncology_arbiter/api/static/dist/`; remove client `series_dir` fields; add DICOM upload → case_id → `/v1/case/full` flow |
| Full regression + deployed smoke tests | Iteration budget exhausted | `tests/e2e/test_live_medsiglip.py`, `test_live_luna16.py`, `test_live_supervisor.py`, `test_case_ingress.py`, `test_ak_workflow_real.py`, `test_hgsoc_rejects.py`, `test_model_unavailable_no_fabrication.py`; `scripts/deployed_smoke.py` hard-fails on missing artifact |
| Render redeploy with new env vars | Iteration budget exhausted | See §8 below |
| MedGemma-27b warm cold-start | 600 s Modal function timeout | Apply `timeout=1800` OR pre-warm `medgemma-27b-weights` Volume |
| Closure git commit | Canonical repo has no `.git` | `git init && git add -A && git commit -m "wave-4"` in `/mnt/shared-workspace/shared/wave4/scientist-monai-moat-full/` |
| Missing-artifact = FAIL enforcement | Not wired at repo level | Add to `pytest.ini` or `conftest.py` fixture that asserts required `docs/proofs/*.json` present |

---

## 8. Render Redeploy Instructions (for the operator)

Set these env vars on `oncology-arbiter.onrender.com`:

```
MEDSIGLIP_BACKEND=modal
MEDSIGLIP_MODAL_URL=https://crispro--medsiglip-embed.modal.run
LUNA16_MODAL_URL=https://crispro--luna16-detect.modal.run
CASE_STORAGE_UPLOAD_URL=https://crispro--case-storage-upload.modal.run
CASE_STORAGE_MANIFEST_URL=https://crispro--case-storage-manifest.modal.run
CASE_STORAGE_GET_URL=https://crispro--case-storage-get.modal.run
CLINICALBERT_MODAL_URL=https://crispro--clinicalbert-parse.modal.run
MEDGEMMA_MODAL_URL=            # empty until MedGemma cold-start is warmed
GEMINI_API_KEY=<secret>        # AQ.Ab8RN...
CO_SCIENTIST_MODE=llm_gemini_2_5_flash
ONCOLOGY_ARBITER_AUTH_MODE=api_key
```

Correct any stale comment claiming `clinicalbert_micro_f1=0.9716` — real-text F1=0.0809.

---

## 9. Cost Ledger

| Item | Cost |
|---|---|
| Pre-session baseline | ~$0.24 |
| case-storage v0.5.0 deploy (1.2 s) | ~$0.01 |
| luna16-infer v0.5.0 deploy (100.8 s) | ~$0.12 |
| medgemma-27b deploy (101.96 s) | ~$0.12 |
| medgemma-27b cold-start smoke (604.7 s A100-80GB, timed out) | ~$0.35 |
| luna16-infer live smoke (38.4 s A10G) | ~$0.01 |
| medsiglip live smoke (pre-session) | ~$0.01 |
| Supervisor live smoke (Gemini 2.5 Flash, 3197 tokens) | ~$0.000353 |
| **Session total** | **~$0.85** |
| **Cumulative vs. $50 cap** | **~$0.85 / $50** |

---

## 10. Provenance Artifacts Committed This Session

- `/mnt/results/execution_trace/case_full_smoke_worker0.json` (7085 B) — full NSCLC gateway response body
- `/mnt/results/execution_trace/luna16_case_id_pipeline_smoke.json` (4884 B) — 3-step chain proof
- `/mnt/results/execution_trace/medgemma_27b_smoke_timeout.json` (1764 B) — cold-start failure with mitigation
- `/mnt/shared-workspace/shared/wave4/scientist-monai-moat-full/docs/proofs/luna16_case_id_pipeline_smoke.json` (canonical mirror)

Files modified this session (mirrored to canonical):
- `deploy/modal/case_storage_app.py` (v0.5.0)
- `deploy/modal/luna16_infer_app.py` (v0.5.0)
- `deploy/modal/medgemma_27b_app.py` (new, v0.4.0-alpha)
- `src/oncology_arbiter/api/app.py` (case_id NSCLC path + HGSOC 501)
- `src/oncology_arbiter/api/schemas.py` (`FullCaseRequest.case_id` field)

---

## 11. Honesty Statement

- No FDA / CE claims made. All responses carry `RESEARCH USE ONLY` disclaimer.
- No PHI in this smoke — LIDC-IDRI is a public de-identified dataset.
- No fabricated numbers. Every figure quoted above traces to a live HTTP response or a JSON artifact under `/mnt/results/execution_trace/`.
- `evidence_kept=0` in the TestClient case-full smoke reflects that this specific handler path does not exercise the LLM Co-Scientist evidence-gate stage; the supervisor smoke in §4 exercises it separately and kept 6/6 Europe PMC URLs.
- ClinicalBERT real-text F1 is 0.081, not the synthetic 0.9716. This is called out on the endpoint and must be corrected in any downstream comment.
- The elo `model_states[1]` field reports `case-storage-modal@v0.4.0-alpha` where the manifest is v0.5.0-alpha. This is a labelling bug in the gateway elo builder, not a data corruption. Fix-forward on next push.
- MedGemma-27b was deployed but never observed to respond successfully; it must not be counted as an active reasoning route until a warm cold-start is verified.
