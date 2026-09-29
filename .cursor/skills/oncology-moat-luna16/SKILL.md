---
name: oncology-moat-luna16
description: >-
  Forces LUNA16 resample/crash code fix tonight on scientist-monai-moat. Aligns
  Findings F/G and arbriter execution items — bans further root-cause JSON as DONE.
  Use when LUNA16, FROC, CPM, subset0, or luna16-infer.
---

# Oncology Moat — LUNA16 (TONIGHT)

**RUO.** Measurement done (38.2% fail, CPM low). **Code + train tonight.**

## Canon

- Findings F/G in `gate_violation_bf5e54f8.json`
- `arbriter/report_audit_execution_items_1_5.md` §1 — more root-cause is **not** remediation
- `actual_spacing_mm=[2.5,…]` vs declared 1.25 in wave4 smokes

## Forbidden

- `luna16_*root_cause*.json` as DONE
- Subset0 md5 inventory as DONE
- FROC CSV-only commits
- Claiming NOT_TRAINED refine floor PASS

## Tonight bar

1. `Grep`/`Read` luna16 detect path under `deploy/modal/` + `src/`.
2. Implement and apply resampling so native geometry matches the detector
   contract. A clearer error is not a substitute for working inference.
3. Add regression test that fails if Spacingd/resample absent.
4. Build the LUNA training manifest from official annotations/candidates;
   acquire/extract missing subset files immediately.
5. Run the refinement training required by
   `docs/regulatory/WEIGHT_REDISTRIBUTION.md`; persist checkpoint, SHA256,
   seed, command, wall time, and held-out FROC@2.
6. Wire the trained checkpoint into `luna16-infer`.
7. Run a real-series smoke and population regression showing the new loaded
   SHA, geometry, crash rate, and FROC.
8. Commit `src/`/`deploy/` + weights/manifest + push.

Forbidden exits: hard-fail-only remediation, resample-only completion,
`NOT_TRAINED`, pending/blocked/deferred, dataset inventory as completion, or
capability removal.

## DONE

| Gate | Evidence |
|------|----------|
| detect/resample code diff | `git show` |
| test nodeid | |
| trained checkpoint + SHA256 | |
| train manifest SHA + held-out FROC@2 | |
| smoke/population result loads trained SHA | |
| tip moved | |

Root-cause addendum alone = **FAIL**.
