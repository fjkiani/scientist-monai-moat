# Audit Execution Report — Items 1–5 (LUNA16 Root Cause, Weight-Floor Gates, Data Inventory, Checkpoint, Subset0 Verification)

**Repo**: `fjkiani/scientist-monai-moat` · **Branch**: `audit/manski-gate-and-test-migration` · **Locked base commit**: `bf5e54f8d8801286ba4b0ad7ca03ce0c7fc75b10`
**Local HEAD** (uncommitted new work sits on top, not yet staged): `a81e9a9` → `85a7ce5` → `bf5e54f` (base). **Never pushed** — push remains structurally blocked (MCP token has read-only/403 scope) and is not being attempted with the exposed PAT.

**Overall classification: unchanged — `RED_NOT_MERGEABLE`.** This window did not change ship-readiness; it hardened the evidence base (root-cause analysis, honest CI-gate values, ground-truth inventory, data-integrity verification) that the eventual merge decision rests on.

**Standing law, reaffirmed, not re-litigated**: ≥0.85 AUC required to use the word "screening" (else report `BELOW_SHIP_GATE_SCREENING`); `EMBED` = `BLOCKED_PENDING_EMORY_DUA`; NCCN = no scrape / no AI ingest, ever; no invented path to 0.85; no PAT use (the GitHub PAT pasted earlier in this session must be rotated — it has been exposed in the conversation and must not be reused; its value is intentionally not repeated here).

---

## 1. LUNA16 root-cause dig (Execution Item 1)

The v2 population analysis (89 subset0 series, 55 ok / 34 failed HTTP 500s at `call_detect`, 38.2% failure rate) had already established that `D` (slice count ÷ `DimSize_z`) is the dominant predictor: `D>300` → 90% failure vs `D≤300` → 23.2%; `D`-alone AUC = 0.811.

**New hypothesis tested — and rejected.** A separately-discovered fact (the endpoint's `/info` declares a `target_spacing_mm` it never applies — no `Spacingd`/`Orientationd` step in `detect()`) motivated testing whether **physical z-extent** (`D × spacing_z`) is the true driver, since native `dz=2.5mm` is exactly 2× the undeclared target spacing. Result:

| Feature | AUC | p-value | AIC |
|---|---|---|---|
| `D` alone | 0.811 | — | 86.03 |
| `z_extent_mm` (`D × spacing_z`) | **0.502** (chance) | 0.939 | 122.37 |

`z_extent_mm` fails because `D` and `spacing_z` have **opposite-sign** associations with failure that cancel in their product — a mathematical explanation, not a coincidence (nested likelihood-ratio tests: `D` adds significant information beyond `z_extent` at p=1.6×10⁻⁹; `z_extent` adds nothing beyond `D`, p=0.745). **This is a genuine negative result**: it argues against the unapplied-resample-target being the dominant HTTP-500 driver, and should be read as evidence, not a dead end.

A data-driven clean-cutoff search (Youden thresholds + shallow decision trees) found no feature or shallow rule achieves zero misclassification: depth-2 tree reaches 82.0% accuracy (spacing_z≤0.88); depth-3 reaches 85.4% by adding an `offset_z` split — **flagged here as likely overfitting** given n=89 and an arbitrary continuous cutpoint, not a real discriminator. An overlap zone (`D∈[127,325]`, n=61) still shows an unresolved 26.2% failure rate not explained by any header-derived feature. Categorical fields (`CompressedData`, `ElementType`, image dimensions) are constant across all 89 series and were ruled out.

**Honest conclusion**: the root cause remains an unhandled server-side exception correlated with, but not deterministically reducible to, `D`. Server-side logs/code (out of scope, no access) would be needed to fully resolve the remaining 26.2% overlap-zone ambiguity. Full detail: `docs/proofs/luna16_froc_subset0_v3_root_cause_addendum.json`.

---

## 2. `weights_meet_floor` CI-gate formalization (Execution Item 2)

The `WeightsProvenance` schema's `weights_meet_floor` block was missing from every real-data metrics sidecar. Formalized against `docs/regulatory/WEIGHT_REDISTRIBUTION.md`'s 3 named floors, using sha256-verified checkpoint identities:

| Checkpoint | Achieved | Floor | Result |
|---|---|---|---|
| CBIS-DDSM logreg v1 (HF-sourced) | 0.752584 | 0.85 AUROC | **FAIL** |
| CBIS-DDSM NBIA DICOM v1 (frozen, cross-check) | 0.729992 | 0.85 AUROC | **FAIL** |
| ClinicalBERT v0.5.1 (real corpus) | 0.079315 (micro-F1) | 0.85 | **FAIL** |
| ClinicalBERT v0.5.2 sliding-window | 0.253466 (micro-F1) | 0.85 | **FAIL** (also explicitly in-corpus, not held-out) |
| LUNA16 refine v0.4.0 | — | ΔFROC@2 ≥ +5% | **NOT_TRAINED** (Modal-blocked, baseline file absent) |
| Mammo "MONAI" RetinaNet v0.4.0 | — | 0.85 AUROC | **NOT_TRAINED** (Modal-blocked; also a naming mismatch — code is torchvision RetinaNet, not MONAI) |

No metric was fabricated for the 2 never-trained checkpoints; they are explicitly recorded as `NOT_TRAINED_BLOCKED_PENDING_MODAL_CREDENTIALS` with `weights_meet_floor: null`. ClinicalBERT is not one of the policy doc's 3 named floors — a documentation gap, flagged not silently patched; its own model-card floor (0.85) was applied for honest comparison anyway. Files: `docs/proofs/cbis_ddsm_logreg_v1_metrics.json`, `cbis_ddsm_nbia_dicom_v1_metrics.json`, `clinicalbert_v51_metrics.json`, `clinicalbert_v51_sliding_window_metrics.json`, `not_trained_checkpoints_status.json`.

---

## 3. Ground-truth data inventory (Execution Item 3)

`docs/proofs/data_receipts/must_inventory.json` — 17 assets, each independently verified on-disk (not from prior claims):

| Category | Example finding |
|---|---|
| More complete than claimed | LUNA16 subset0: 89/89 series present (claim was a stale `.partial` download) |
| Massively understated | CBIS NBIA DICOM: **645 series** on disk vs. 18 claimed |
| Different format/count | Pathology text: 2,789-row parquet on disk, not the claimed 9,523-row CSV |
| Already done, already honest-fail | CBIS probe, ClinicalBERT — real runs exist, both fail their floors (see §2) |
| Absent | v0.6.9 LUNA16 baseline, RIDER/NSCLC imaging, Modal CLI credentials |
| Not a constraint | Disk space (948GB free) |

---

## 4. LUNA16 checkpoint to shared storage (Execution Item 4)

`/workspace/luna16` (machine-local, 12GB, previously a single point of failure) copied to `/mnt/shared-workspace/luna16_froc/extracted_subset0/`. **Verified via full per-file SHA-256 comparison** (not just size/count): **178/178 files byte-identical**, 0 mismatches. The original `subset0.zip` is separately already checkpointed (see §5) — this data now has two independently verified copies outside the at-risk machine-local disk. Detail: `docs/proofs/luna16_checkpoint_verification.json`.

---

## 5. Subset0 completeness verification (Execution Item 5) — includes a self-correction

Initial plan was a count-level verification only, since no local zip was believed to exist. **While executing Item 4, a genuine `subset0.zip` was discovered already sitting in shared storage** (from an earlier download this session). This upgraded the verification substantially, and the record was corrected rather than left at the weaker claim:

| Check | Result |
|---|---|
| Zip size | 6,811,924,508 bytes — **exact match** to official |
| Zip md5 (computed fresh) | `1065b0f42b8c25cf29260fd924a3c3a2` — **exact match** to Zenodo record 3723295 |
| Zip's internal 89 UIDs vs. our on-disk 89 | **Set-identical** (not just count-equal) |
| Independent third party (github.com/eyast/LUNA16, actually downloaded the dataset) | 89 series/178 files for subset0 — matches; aggregate 888 total series across 10 subsets matches official docs |
| Our vendored `annotations.csv` vs. official + third party | **1186 nodules / 601 distinct series** — exact match to both |

**A genuine self-correction, disclosed**: an earlier pass of this same investigation flagged 1 of 12 spot-checked UIDs from a low-reliability informal source as "discordant" (labeled `subset2` there but present in our subset0). Re-fetching that source **in full** (rather than a truncated search snippet) showed the complete list actually has 14 entries, and this specific UID is labeled `subset0` there too — the "subset2" label belonged to a different UID entirely. The original finding was an artifact of an incomplete fetch, not a real data problem. Corrected result: **14/14 consistent** (now moot in any case given the md5 proof above). Full detail: `docs/proofs/luna16_subset0_completeness_verification.json`.

**Verdict: `BYTE_LEVEL_MD5_CONFIRMED_COMPLETE_AND_AUTHENTIC`.**

---

## Open items (unchanged / carried forward)

| Item | Status |
|---|---|
| git push (Step 9) | Blocked — MCP token read-only/403; not attempted with the exposed PAT |
| New docs/proofs files this window | Untracked/modified in working tree; **not staged or committed** — no decision made on folding into a commit |
| Wave-1/Wave-2 data status + draft application files (Step 12) | Not started |
| 6 pre-existing real test failures (laterality, `body["biopsy"]`, MONAI-zero-findings, MedSigLIP-cascade) | Not addressed this window |
| LUNA16 overlap zone (D∈[127,325], 26.2% unresolved failure rate) | Root cause still not fully explained; needs server-side logs (no access) |
| PAT rotation | Still needed — two distinct GitHub PATs have been pasted into this conversation (one earlier redacted, one fully exposed); both must be rotated. Values intentionally not repeated in this report. |

No new claim in this report changes the `RED_NOT_MERGEABLE` classification or any locked ship-gate number.
