# Production-Acceptance Audit — Interim Status Report
**Repo**: `fjkiani/scientist-monai-moat` (`oncology-arbiter`) · branch `audit/manski-gate-and-test-migration`
**Base commit**: `bf5e54f8d8801286ba4b0ad7ca03ce0c7fc75b10` (unchanged, no commits/pushes)
**Classification**: `RED_NOT_MERGEABLE` (unchanged — remains RED until all gates pass; nothing in this session lifts that status)
**Report timestamp**: this session, prior to full completion of 3 in-flight background jobs (see "Still Running" below)

This is an **interim handoff report**, not a final acceptance report. Three substantive background computations were still running when this session's iteration budget was exhausted. No result below is fabricated or projected — everything stated as "confirmed" was produced by a completed real run against live data/endpoints; everything still pending is labeled as pending, with the exact command/location needed to retrieve it once it lands.

---

## 1. What this session was asked to do

Two directives from the user, binding, no further scoping questions permitted:
1. **Gut fake pass/fail tests, build real tests** across 7 in-scope Modal endpoints: medsiglip-448, phikon-pathology, luna16-infer, case-storage, clinicalbert, gemma-fallback, medgemma-27b.
2. **Begin gathering real datasets** in a locked Wave-1 order: CBIS-DDSM → LUNA16 → LIDC-IDRI → TCGA/CPTAC pathology reports → NLST → Camelyon16/17 (conditional). Wave-2 (EMBED, VinDr, NLST-CDAS) deferred to drafted-application status, not attempted.

Locked gate law (verbatim, not renegotiable): any MedSigLIP artifact with embedding dimension ≠ 1152 is `CONTAMINATED_PROXY`; ClinicalBERT `SYNTHETIC-v0.3.x` F1 headlines are banned; EMBED is `BLOCKED_PENDING_EMORY_DUA`; NCCN gets no scrape/no AI ingestion ever; ≥0.85 AUC is a hard ship-gate for the word "screening" — if measured lower, publish `BELOW_SHIP_GATE_SCREENING`, never invent a path to 0.85.

---

## 2. Confirmed findings this session (written to canonical audit file)

All four items below are now recorded as Findings **G, H, I, J** in `gate_violation_bf5e54f8.json`, kept byte-identical in two locations: the canonical copy and the in-repo copy (`artifacts/audit/gate_violation_bf5e54f8.json`). 10 total evidence keys (A–J), 27,638 bytes.

### Finding G — LUNA16 large-volume input causes a deterministic server-side HTTP 500
- Full 89-series subset0 FROC run crashed on series 4 (`...111172165674661221381920536987`): `shape_dhw=(538,512,512)`, `spacing_xyz=(0.742,0.742,0.625)mm`, 282MB — 3–4.5× larger than the 3 preceding series that succeeded (119–161 slices, 62–84MB).
- Root-caused as **deterministic, not transient**: a corrected retry (env var fixed, same uploaded `case_id=0326b58ee90697ca`, no re-upload) still failed after 185.7s with a byte-identical `HTTP 500: Internal Server Error` from `call_detect`. The ~186s delay (vs 26–33s for successful series) indicates real server-side compute before an unhandled exception — consistent with a resource or time-budget limit being exceeded on this input size, not a flaky network blip.
- **Population risk-sizing** (metadata-only scan, all 89 subset0 series): slice-count D stats — min=109, max=733, mean=257.0, median=209. **20/89 series have D>300** (same range as the confirmed failure), 35 have D>250, 46 have D>200.
- **Mechanistic link to pre-existing Finding F** (the endpoint never resamples to its declared target spacing `[1.25, 0.703125, 0.703125]mm` — confirmed via the bundle's own `configs/inference.json`): Finding F is the *mild* consequence (silent spatial miscalibration on coarser-than-target native spacing). Finding G is the *severe* consequence — much finer native z-spacing (0.625mm here) means far more native slices for the same physical extent, and feeding that unresampled volume directly to the RetinaNet appears to exceed a resource limit and crash outright.
- **Remediation applied**: `scripts/run_luna16_froc_subset0.py` patched so a per-series failure is logged (with `failed_stage`, `shape_dhw`, `spacing_xyz`) and the batch **continues** rather than crashing entirely — this is pipeline resilience, not mocking; every real failure remains permanently recorded verbatim. Full 89-series run relaunched (`luna16_froc_subset0_run_v2`, resumable from series 5). **Status: still running at report time** — population-scale confirmation of the D>300 failure threshold is the open empirical question this run will answer.

### Finding H — MedSigLIP deploy-drift root cause traced via git history
Git history trace shows the live container has been frozen pre-commit `76ba65b`; model/weights identical since `v0.3.0`. This confirms no confound was introduced into the CBIS-DDSM cross-cohort AUROC measurement by version drift.

### Finding I — CBIS-DDSM training-script reproducibility gap
The committed training script hardcodes `C=1.0`, but the shipped `models/cbis_ddsm_logreg_v1.joblib` was fit with `C=0.01` — the script as committed cannot reproduce the shipped probe. Flagged, not silently patched.

### Finding J — CBIS-DDSM 645-vs-641 series count reconciled
Mathematically reconciled: 4 dual-pathology series were dropped by the HF mirror between the 645-series NBIA manifest and the 641-series original test set. Status: `RECONCILED_NOT_A_DEFECT`.

---

## 3. Real datasets acquired/verified this session

| Dataset | Status | Location | Notes |
|---|---|---|---|
| CBIS-DDSM (NBIA subset) | 645 series downloaded prior window; cross-cohort MedSigLIP-1152 AUROC job **launched, still running** | `/mnt/shared-workspace/cbis_ddsm/nbia_test_subset/` | Original frozen probe: test AUC=0.752584 (n=641) |
| LUNA16 subset0 | 89 series present; vendored official evaluator + annotations in-repo | `/workspace/luna16/subset0/subset0/`, `vendor/luna16_evaluation/` | v1 run crashed on series 4 (Finding G); v2 resilient run in flight |
| NCT-CRC-HE-100K + CRC-VAL-HE-7K (Phikon population validation, explicitly **not** a product screening claim — approved scope addendum) | 107,180 images confirmed on disk; full-population embedding job **launched, still running** | `/workspace/pathology_crc/` (machine `w2`) | Batch cap of 64 discovered empirically (server returns HTTP 200 with `{"error":...}`, not an HTTP error code — now explicitly handled) |
| NLST (chest CT, images-only, Wave-1 scope) | **20/20 series downloaded successfully**, verified slice-count match, zero truncation | `/mnt/shared-workspace/nlst/subset/` | Discovered NLST lives on a separate token-gated NBIA server (`nlst.cancerimagingarchive.net`), opposite auth pattern from the main CBIS-DDSM server. Confirmed via live query: 26,254 NLST patients, 100% `Phantom="NO"` — directly answers "beyond planted phantoms." Images-only; ground-truth labels require CDAS (Wave-2, ungathered). |
| LIDC-IDRI pixel re-fetch | Not started | — | Queued next |
| TCGA/CPTAC pathology reports (ClinicalBERT real-text) | Not started beyond existing 8-report fixture | — | Queued next |
| Camelyon16/17 | Not started (conditional on histopath claims remaining) | — | Queued next |
| EMBED / VinDr / NLST-CDAS (Wave-2) | Not drafted | — | Deferred per lock; needs application-text drafting, not "can't obtain" |

---

## 4. Still running — no results yet (do not treat as failed; callbacks pending)

| Job | Machine | What it produces | Where to read once complete |
|---|---|---|---|
| `cbis_ddsm_nbia_embed_eval` | worker-0 | MedSigLIP-1152 cross-cohort AUROC on 645 real CBIS-DDSM NBIA DICOMs | `/workspace/smm/artifacts/cbis_ddsm_nbia_dicom/{run.log,cbis_ddsm_nbia_dicom_v1_metrics.json}` |
| `luna16_froc_subset0_run_v2` | worker-0 | Full 89-series resilient FROC run; empirical test of Finding-G's D>300 failure threshold | `/workspace/smm/artifacts/.../{results.csv,processed_seriesuids.json}` |
| `phikon_nct_crc_embed_full` | w2 | Full 107,180-image Phikon embeddings (train+val), feeding a 9-class linear-probe validation | `/workspace/pathology_crc/phikon_embeddings/{embeddings.npy,done_mask.npy,manifest.json}` |

**Apply the ship-gate rule the moment CBIS-DDSM AUROC lands**: if measured AUROC ≥ 0.85, it may support the word "screening"; if below, it must be published as `BELOW_SHIP_GATE_SCREENING` — no exceptions, no re-fitting to chase the threshold.

A background diagnostic sub-job (`luna16_series4_retry_debug`) also completed during this session; its output matches the already-documented first diagnostic attempt in Finding G (upload succeeded at `case_id=0326b58ee90697ca`, 31.1s; detect call failed on a since-fixed env-var bug) — it is superseded by the corrected retry already folded into Finding G and requires no further action.

---

## 5. Exact next steps (in order)

1. On callback for each of the 3 running jobs: read the output files listed above; apply the ≥0.85 ship-gate rule to CBIS-DDSM; check the D>300 hypothesis against the full LUNA16 v2 results and run the vendored FROC evaluator + plot; fit/evaluate the Phikon 9-class linear probe (train on NCT-CRC-HE-100K, held out on CRC-VAL-HE-7K) and write `phikon_nct_crc_population_validation.json` with the explicit "not a screening claim" disclaimer.
2. Feed the 20 downloaded NLST series through the same real `upload_dicom_series()`/`call_detect()` pipeline as LUNA16 — qualitative cross-cohort comparison only (no FROC, no ground truth without CDAS).
3. Fold final numbers into a new Finding (K) or G/H update, written to both canonical and in-repo copies together.
4. LIDC-IDRI pixel re-fetch; TCGA/CPTAC pathology-report ClinicalBERT reassessment; Camelyon16/17 if still needed; draft (not attempt) Wave-2 EMBED/VinDr/NLST-CDAS applications.
5. Triage the 6 pre-existing real test failures (laterality field, `body["biopsy"]` None, MONAI zero findings, medsiglip-regression-cascade) surfaced by the new real-I/O tests.
6. Vendor all new scripts into git (`scripts/luna16_froc.py`, `scripts/download_cbis_ddsm_nbia_test_subset.py`, `scripts/eval_cbis_ddsm_nbia_dicom.py`, `scripts/run_luna16_froc_subset0.py`, `scripts/embed_phikon_nct_crc.py`, `scripts/download_nlst_ct_subset.py`, `vendor/luna16_evaluation/`); re-run `git status`; commit — **no push, no merge**. Branch remains `RED_NOT_MERGEABLE` until every gate passes for real.

---

## 6. Machines in use (session cap = 3)

- **worker-0**: repo checkout, CBIS-DDSM eval job, LUNA16 FROC v2 job — both running.
- **w2**: NCT-CRC-HE-100K/CRC-VAL-HE-7K data + Phikon full-embedding job — running.
- **w3**: completed the NLST download this session (20/20 series, 736s wall time); currently free for next task (LIDC-IDRI or TCGA/CPTAC).

No credentials printed. No force-push, merge, or deploy performed or attempted.
