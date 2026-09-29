# Index & consolidation of the `report_*.md` development log

**Purpose of this document.** Ten `report_*.md` files had accumulated in this deliverable
set across the life of this audit. This index (1) prunes the two that were fully stale, (2)
resolves a naming/date anomaly between the two `end_to_end_audit` files, (3) fixes a live
credential exposure found in one of them, and (4) gives a single chronological entry point
into the remaining eight, which are kept intact rather than lossily flattened — each documents
a distinct, real, load-bearing experiment or deploy event that the authoritative technical
finding record (`artifacts/audit/gate_violation_bf5e54f8.json` and, for the 7-endpoint live I/O
tests, `docs/proofs/seven_endpoint_live_io_test_evidence.json`) does not restate in full.

**If you only read one file for the current ship/no-ship determination, read
`gate_violation_bf5e54f8.json` (Findings A–L) plus the 7-endpoint evidence receipt, not this
index.** This index is a map of the historical development record, not a restatement of the
live gate verdict, which remains **RED_NOT_MERGEABLE**.

---

## 1. What changed in this pass

| Action | File(s) | Reason |
|---|---|---|
| **Archived** (moved to `archive_superseded_reports/`, not deleted) | `report_audit_session_status.md` | Interim handoff report describing 3 background jobs as "still running." All three have since completed and their results are folded into `gate_violation_bf5e54f8.json` Findings G/K/L. Fully superseded, zero unique remaining content. |
| **Archived** (moved to `archive_superseded_reports/`, not deleted) | `report_wave3_assessment.md` | Pre-Phase-2 snapshot describing the three arbiters as untrained templates (`n_training=0`). `report_wave3_closure.md` (same phase, 18 days later) supersedes it with the actual fitted-and-gated results. Its two non-duplicated facts (zero-mock test audit; 337 KB frontend bundle) are preserved in §3 below. |
| **Redacted in place** | `report_v51_final.md` | §8 of the original contained a **live GitHub PAT in plaintext**. The token has been replaced with a `***REDACTED_ROTATE_IMMEDIATELY***` placeholder. The rest of the file's real content (5-seed ClinicalBERT v0.5.1 training results, Modal deploy history) is unchanged. **Action still required from the user: rotate that PAT at github.com/settings/tokens — redacting the document does not invalidate the credential itself.** |
| **Kept as-is** | `report_wave3_closure.md`, `report_wave4_closure.md`, `report_wave5_benchmark.md`, `report_wave5c_benchmark.md`, `report_wave5e_clinical_validation.md`, `report_end_to_end_audit.md`, `report_end_to_end_audit_v2.md` | Each documents a distinct real experiment, gate result, or deploy event not restated elsewhere. Flattening these into one document would drop caveats (e.g. wave3_closure's "construction check, not independent validation" note on HRD=1.00; wave5e's "suspicious-looking small-denominator" self-caveat on TCGA-OV=1.00) that materially change how a reader should weigh the headline numbers. |

Net: **10 → 8 active files** (2 archived, 0 deleted, 1 secret remediated), plus this index.

---

## 2. The two named files that do not exist

The original consolidation request named 12 files; only 10 ever existed in this deliverable set.
Confirmed absent, not merely misplaced:

- `report_parser_regex_v2.md` — no file of this name was ever produced.
- `report_parser_clinicalbert_v1_model_card.md` — no file of this name was ever produced.

**Do not conflate either with** `docs/model_cards/report_parser_clinicalbert_v1.md`, which does
exist but lives **inside the audited repo** (`src`-adjacent docs, not a `/mnt/results` deliverable)
and is a different artifact: a model card, not a "v1 model card report." That in-repo file is
also one of the eleven files the quarantine receipt lists as still containing the banned
`SYNTHETIC-v0.3.1` provenance tag string (see §4).

---

## 3. Resolving the `end_to_end_audit` vs `end_to_end_audit_v2` naming anomaly

Both files claim to supersede something, and the naming reads backwards at first glance:
`_v2.md` (mtime **2026-07-07**) is dated and scoped *earlier* than the no-suffix file (mtime
**2026-07-31**), which covers materially more ground (Wave-6/7) and explicitly says it
"corrects the stale prior audit."

**Resolution (confirmed via mtime + content scope, not assumed):** this is ordinary ad hoc
naming drift across independent sessions, not a data contradiction. `report_end_to_end_audit_v2.md`
was named "v2" relative to some earlier, no-longer-present document at the time it was written
(2026-07-07, v0.3.0-alpha vintage) — it is chronologically the **older** of the two surviving
files. Three weeks later, a fresh end-to-end audit was written covering Wave-6/7 and was simply
named without a version suffix, rather than being incremented to "v3." The chronologically
correct reading order is:

1. `report_end_to_end_audit_v2.md` (2026-07-07, v0.3.0-alpha) — **older**
2. `report_end_to_end_audit.md` (2026-07-31) — **newer, supersedes (1) on every claim it re-touches**

No content in either file contradicts the other once read in this order; each documents the
state of a different, later point in the project. No further action needed beyond this note.

---

## 4. Cross-check against the quarantine receipt's banned-tag list

`artifacts/audit/quarantine_receipt.json` bans any product-copy headline built on
`SYNTHETIC-v0.3.0` / `SYNTHETIC-v0.3.1`-provenance ClinicalBERT F1 numbers. A direct grep of
all 8 kept `report_*.md` files for the literal tag strings returns **zero matches** — none of
them use the banned tag text. (Files that *do* still contain the literal tag are all inside the
repo proper: `deploy/modal/clinicalbert_app.py`, `docs/PROD_FLIP_CHECKLIST_v0.4.1.md`,
`docs/audit/apex_prod_v1.md`, `docs/model_cards/report_parser_clinicalbert_v1.md`,
`docs/model_cards/report_parser_clinicalbert_v3_real_v51.md`, and five source files under
`src/oncology_arbiter/` — tracked separately, not part of this `/mnt/results` consolidation.)

**One file needs a forward-pointer caveat rather than a tag match, and gets one here:**
`report_end_to_end_audit_v2.md` §2.2 reports ClinicalBERT v1 field-level F1 = **0.9546**
(relaxed) / **0.8733** (strict) on "300 synthetic reports" — plain-English self-disclosure of
the synthetic corpus, not the banned tag string, and written before the quarantine policy
existed (2026-07-07 vs. the receipt's 2026-09-27 issuance). It is not a policy violation, but a
reader encountering it out of context could mistake it for current parser performance. It is
not: the current, real-corpus numbers, all measured later and all far below any usable ship
threshold, are:

| ClinicalBERT version | Corpus | Micro-F1 | Source |
|---|---|---|---|
| v1 (2026-07-07) | 300 **synthetic** reports | 0.9546 relaxed / 0.8733 strict | `report_end_to_end_audit_v2.md` — **do not headline** |
| v0.5.1 (live today) | real, 296-report held-out | **0.0809** | live `/info`, matches `report_v51_final.md` exactly |
| v0.5.2-sliding-window (real, exists in repo, never deployed) | real, 296-report held-out | 0.2535 (in-corpus, **not** held-out) | `quarantine_receipt.json` |
| v0.5.2 Snorkel retrain (combined) | real | 0.110 | `report_end_to_end_audit.md` |

The four numbers are not in tension once the corpus and evaluation-protocol differences are
accounted for — they trace a real (if still commercially unusable) improvement path on
genuine data, clearly separable from the withdrawn synthetic-corpus number.

---

## 5. Chronological index of the 8 kept reports (+ 2 archived)

| Order | File | Date (mtime) | One-line summary | Headline number(s) |
|---|---|---|---|---|
| 1 | *(archived)* `report_wave3_assessment.md` | 2026-07-02 | Pre-fit Wave-3 snapshot: 339/353 tests pass, zero `unittest.mock` in the tree (grep-verified), arbiters still template placeholders. | 353 tests collected; 337 KB frontend prod bundle |
| 2 | `report_end_to_end_audit_v2.md` | 2026-07-07 | v0.3.0-alpha wiring audit: screening probe + LUNA16 + ClinicalBERT v1 flip from stub to wired; explains why Render's free dyno never enables ClinicalBERT/CBIS-probe (512 MB ceiling) — corroborates this session's own live Render `/health` probe. | CBIS-DDSM probe v1 AUC=0.7526 (n=641); LUNA16 mAP=0.852 (**publisher-reported, not rerun**); ClinicalBERT v1 F1=0.9546/0.8733 (**synthetic corpus — see §4**) |
| 3 | `report_v51_final.md` *(secret redacted)* | 2026-07-20 | ClinicalBERT v0.5.1 real 5-seed training + Modal deploy (2 profiles blocked on spend limit before `crispro` succeeded) + live prod verification. | Combined test F1=0.0809±0.0018 (seed 456 best); +33.82% vs v0.5.0, rollback (−5pt threshold) not triggered |
| 4 | `report_wave3_closure.md` | 2026-07-20 | Real arbiter coefficient fits ship (all 3 PASS gate); Phikon WSI infra ships; biopsy receptor probe and Co-Scientist LLM promotion both honestly FAIL their gates. | Screening AUROC=0.854 [.840,.863] n=3568; Biopsy AUROC=0.920 [.908,.931] n=2509; Therapy AUROC=0.924 [.909,.937] n=1980 |
| 5 | `report_wave4_closure.md` | 2026-07-21 | Real case-storage + LUNA16-infer wiring on a real LIDC-IDRI case; MedGemma-27b cold-start times out; TxGemma HF-gate access rejected. | LUNA16 n_detections=4, top_score=0.711; MedGemma timeout=604.7s (**this session independently reproduced 603.18s, ~2 months later, via a different call path**) |
| 6 | `report_wave5_benchmark.md` | 2026-07-21 | HGSOC BriTROC-1 CN-signature + therapy benchmark; CBIS-DDSM medSigLIP zero-shot **negative finding**; credentials properly fingerprint-only (no raw exposure). | Signature acc=0.939 (n=147); HRD acc=1.00 (**self-flagged construction check**); therapy top-1=0.769/top-3=0.884 |
| 7 | `report_wave5c_benchmark.md` | 2026-07-21 | T10–T16 gates (7/7 PASS); BriTROC-1 controlled-access metadata probe; SigLIP natural-image proxy **negative finding**; identifies a concrete HGSOC therapy-catalogue gap (no CDK-pathway option). | BriTROC-1: 679 files/475.72 GB listed, 0 bytes downloaded; arbiter concordance 5/5 lenient, 3/5 strict |
| 8 | `report_wave5e_clinical_validation.md` | 2026-07-21 | Closes the Wave-5C gap: adds prostate + NSCLC-clinical routers; validates against real TCGA-OV/COAD-READ cohorts; head-to-head lift over the Wave-5C-legacy platinum-demotion behavior. | TCGA-OV top-1=1.00 (n=294, **self-flagged "suspicious-looking, small denominator"**); TCGA-COAD/READ top-1=0.821 (n=609); platinum-demotion lift=+100pp |
| 9 | *(archived)* `report_audit_session_status.md` | today (this session) | Interim status snapshot, superseded within the same session. | — |
| 10 | `report_end_to_end_audit.md` | 2026-07-31 | Most complete synthesis: two new real trained arbiters, ClinicalBERT Snorkel retrain, CBIS-DDSM probe v2, dynamic tumor-board module. Explicitly corrects the prior stale audit. | ovarian_progression_arbiter_v2 AUROC=0.594 [.520,.671] n=365; breast_mortality_arbiter_v2 AUROC=0.719 [.695,.743] n=1815; ClinicalBERT v0.5.2 combined F1=0.110; CBIS-DDSM v2 AUC=0.7532 (n=641, **cross-validates v1's 0.7526**) |

---

## 6. What this index deliberately does not do

It does not re-litigate any AUROC, re-run any experiment, or change the ship-gate verdict.
It does not merge the 8 kept reports into a single narrative — each remains the authoritative
source for its own numbers, several of which carry caveats (construction-check HRD accuracy,
small-denominator TCGA-OV concordance, publisher-reported-not-rerun LUNA16 mAP) that are easy
to lose in a flattening pass. The current binding technical verdict for this repo/branch/commit
remains **RED_NOT_MERGEABLE**, per `artifacts/audit/gate_violation_bf5e54f8.json`.
