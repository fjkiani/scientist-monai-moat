---
name: oncology-moat-stage-arbiters
description: >-
  Forces replacement of n_training=0 screening/biopsy/therapy arbiter templates
  on scientist-monai-moat with real fitted weights tonight. Bans citing Wave-3/6
  report AUROCs without files on HEAD. Use when stage arbiter, logistic arbiter,
  METABRIC, TCGA-OV, screening_arbiter, biopsy_arbiter, therapy_arbiter.
---

# Oncology Moat — Stage Arbiters (TONIGHT)

**RUO.** The three stage JSON templates are **not trained models**.

## Canon alignment

- Alpha reports: `.cursor/rules/arbriter/report_end_to_end_audit.md` §3 (v2 arbiters), `report_wave_reports_index.md` (Wave-3 closure headlines).
- **Live HEAD check beats reports:** if file missing on branch, report claim is **not delivered**.

## Current defects (re-verify)

```bash
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('src/oncology_arbiter/arbiter/models')
for name in ['screening_arbiter_template_v0.json','biopsy_arbiter_template_v0.json','therapy_arbiter_template_v0.json']:
    d=json.loads((root/name).read_text())
    assert d.get('n_training')==0, name
    print('TEMPLATE_OK', name)
PY
```

Hand-written `coefficients` with `n_training=0` = **forbidden in product path after tonight**.

## Forbidden theater

- Citing Wave-3 AUROC 0.854 / 0.920 / 0.924 as shipped while templates remain
- Citing Wave-6 `ovarian_progression_arbiter_v2` / `breast_mortality_arbiter_v2` without restoring those JSON files onto **this** branch
- Tweaking template coefficients by hand and setting `n_training` fake
- Metrics-only markdown under `arbriter/` or `docs/proofs/`

## Tonight procedure

1. `Read` `src/oncology_arbiter/arbiter/logistic.py` — loader contract.
2. `Shell`: locate data — METABRIC / TCGA-OV / CBIS features per arbiter
   design. If absent, acquire it immediately and continue through fitting;
   a download PID is progress, never a terminal status.
3. Fit real L2 logistic (or restore verified Wave-6 artifacts from git history / object store onto branch).
4. Write **non-template** files, e.g.:
   - `src/oncology_arbiter/arbiter/models/screening_arbiter_v1.json`
   - `…/biopsy_arbiter_v1.json`
   - `…/therapy_arbiter_v1.json`
5. Each file MUST include: `n_training>0`, `n_positive`, `n_negative`, seed, feature list, OOF AUROC + CI, `trained_on_real_patient_data: true`.
6. Wire `logistic.py` / API to load v1; templates load → hard error in prod path.
7. Commit + push (`oncology-moat-anti-fortress`).

## DONE tonight — all three, no partial credit

| Gate | Evidence |
|------|----------|
| Screening v1 on HEAD with `n_training>0` | `git show` + python assert |
| Biopsy v1 on HEAD with `n_training>0` | `git show` + python assert |
| Therapy v1 on HEAD with `n_training>0` | `git show` + python assert |
| sha256 for all three weight JSON files | `shasum -a 256` |
| OOF AUROC + CI + n for all three | `docs/proofs/<name>_metrics.json` |
| Product loads all three v1 SHAs | integration test |
| Tip moved on origin | `git ls-remote` |

Forbidden exits: `BLOCKED`, `PENDING`, `DEFERRED`, one-of-three delivery,
template rejection without replacements, report citations without artifacts,
or capability removal. Hand-coeff templates still in product path = **FAIL**.
