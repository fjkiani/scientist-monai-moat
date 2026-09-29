---
name: oncology-moat-clinicalbert
description: >-
  Forces ClinicalBERT tonight: deploy v0.5.2-sliding-window or run real corpus
  train. Aligns arbriter index F1 table (0.08 live vs 0.25 undeployed vs synthetic
  0.95 banned). Bans 8-report fixture cosplay. Use when ClinicalBERT, report parser.
---

# Oncology Moat — ClinicalBERT (TONIGHT)

**RUO.** From `arbriter/report_wave_reports_index.md` §4:

| Version | Micro-F1 | Status |
|---------|----------|--------|
| Synthetic v1 | 0.95 | **BANNED** |
| Live v0.5.1 | **0.0809** | deployed |
| v0.5.2 sliding-window | 0.2535 in-corpus | **exists, not deployed** |
| Snorkel retrain | 0.110 | still unusable |

## Tonight bar — new train **and** deploy

1. Verify/acquire `data/pathology_text/TCGA_Reports.csv` and labels; missing
   corpus triggers immediate acquisition and continuation, never a status exit.
2. Create patient-disjoint train/validation/test manifests with SHA256.
3. Run `clinicalbert_train.py` (or repaired successor) on real reports and
   persist new weights, metrics, seed, wall time, and per-entity held-out F1.
4. Deploy those exact weights with a new app version and matching model SHA in
   `/info`; merely deploying old v0.5.2 does not satisfy training.
5. Re-run live contract tests; product must parse real long reports through
   sliding windows and report the trained SHA.
6. Commit and push all train/deploy wiring.

## Forbidden

- 8 fixtures as train
- Headlining synthetic 0.95
- Deploy-drift essay without completing train and deployment
- "Rotate PAT" as substitute for deploy
- Deploying old weights without a new real-corpus train
- Pending/blocked/deferred labels, fixture-only work, or capability removal

## DONE

| Gate | Evidence |
|------|----------|
| New real-corpus weight sha + corpus/split sha | |
| Held-out per-entity and micro-F1 | |
| Live `/info` version + same weight sha | curl/pytest |
| `src/` or `deploy/` changed | `git show` |
| tip moved | |
