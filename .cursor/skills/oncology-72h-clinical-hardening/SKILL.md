---
name: oncology-72h-clinical-hardening
description: >-
  72-hour clinical hardening for scientist-monai-moat: cannibalize weak models
  (ClinicalBERT→MedGemma JSON, biopsy-probe→Phikon), stratum guardrails, RSNA/VinDr
  + LUNA 1:1 expansion, 500-case ghost trial, Docker pack. Use when Alpha says
  72h, clinical-grade product, tumor board pilot, MedGemma pathology extraction,
  Ghost Trial, or Mars hardening clock.
---

# Oncology 72h Clinical Hardening (Mars)

**Canonical personal twin:** `~/.cursor/skills/oncology-72h-clinical-hardening/SKILL.md`  
**Always-on pin:** `~/.cursor/rules/oncology-72h-clinical-hardening.mdc`

Copy the operating matrix from the personal skill. Repo-local bind for sobs working inside this tree.

## Immediate first strike

```bash
cd /Users/fahadkiani/Desktop/development/scientist-monai-moat
PYTHONPATH=src python3 scripts/demo/run_medgemma_pathology_extract.py \
  --report fixtures/patient_sample_01/pathology_note.txt
```

## Companions

`anti-sandbagging` · `zeta-enforcer` · `oncology-moat-platform` · `oncology-moat-anti-fortress` · `moat-agent-discipline`

## Band DONE gates

| Band | Evidence |
|------|----------|
| 00–24 NLP | `artifacts/medgemma_pathology/*` + API primary route ≠ ClinicalBERT NER |
| 00–24 Histo | Phikon product route + test |
| 00–24 API | calc-only → 422 unit test |
| 24–48 Mammo | RSNA/VinDr embedding/train receipt |
| 24–48 LUNA | FROC@2 path vs `b5e79231…` |
| 48–72 Ghost | `artifacts/ghost_trial/ledger_*.json` |
| 48–72 Docker | container smoke log |
