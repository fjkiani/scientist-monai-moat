---
name: oncology-moat-weight-floors
description: >-
  Forces actual training for every named checkpoint in WEIGHT_REDISTRIBUTION.md
  tonight. Bans null/pending/blocked sidecars and inventory theater. Use for
  weight floors, LUNA16 refine, Mammo RetinaNet, CBIS, biopsy, or regulatory gates.
---

# Oncology Moat — Weight Floors (TONIGHT)

**RUO.** Policy: `docs/regulatory/WEIGHT_REDISTRIBUTION.md`.

## Named floors (from policy + Alpha `report_audit_execution_items_1_5.md` §2)

| Checkpoint | Floor | Branch status to re-verify |
|------------|-------|----------------------------|
| CBIS-DDSM logreg | AUROC ≥ 0.85 | FAIL (~0.75) — frozen Jul-10 |
| Biopsy MedSigLIP probe | AUROC ≥ 0.85 | synthetic / wrong dim — FAIL |
| LUNA16 refine v0.4.0 | ΔFROC@2 ≥ +5% | **NOT_TRAINED** |
| Mammo "MONAI" RetinaNet v0.4.0 | AUROC ≥ 0.85 | **NOT_TRAINED** (+ naming lie: torchvision) |

## Forbidden theater

- Writing `weights_meet_floor: true` without achieved ≥ floor
- Inventing metrics for untrained checkpoints
- Calling data inventory / md5 subset0 verification a floor pass
- Any `BLOCKED`, `PENDING`, `DEFERRED`, `NOT_TRAINED`, null metric, kill,
  capability removal, or claim deletion as terminal state

## Tonight procedure

1. `Read` `docs/regulatory/WEIGHT_REDISTRIBUTION.md`.
2. Acquire every required dataset/credential and run training for **all four**
   named checkpoints. Missing inputs are resolved in-line; do not stop.
3. For **each** named checkpoint, produce a weight artifact and sidecar JSON
   under `docs/proofs/` with:
   - numeric `achieved_metric`
   - `floor_metric`
   - honest `weights_meet_floor: true|false`
   - `status`: `PASS` or `TRAINED_BELOW_FLOOR`
   - non-null `weight_sha256`
   - train manifest SHA, seed, command, wall time
4. Wire every artifact into its product path and prove the loaded SHA.
5. Push all weight/code commits to origin.

## DONE tonight

| Gate | Evidence |
|------|----------|
| Sidecars for all 4 named checkpoints | paths |
| Zero false `weights_meet_floor: true` | python scan |
| Four non-null weight SHA256 values | weight files |
| Four product paths load those SHAs | integration tests |
| Commit on origin | `git ls-remote` |

Inventory, download PID, null sidecar, or fewer than four trained artifacts =
**FAIL**. Falling below a metric floor is permitted only after real training;
parking the capability is not.
