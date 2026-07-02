# Model card errata — `oncology-arbiter`

This file tracks corrections and updates to the model cards under
`docs/model_cards/`. Every entry MUST include (a) the affected card,
(b) the change, (c) the source URL and fetch date, and (d) the commit
that introduced the correction.

## Schema

```
### YYYY-MM-DD — <card slug>

- **Change**: <what was fixed / added>
- **Source**: <URL, fetch date>
- **Commit**: <short SHA>
- **Notes**: <optional context>
```

## Entries

### 2026-07-01 — medsiglip_448

- **Change**: Initial card creation.
- **Source**: <https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card>, fetched 2026-07-01.
- **Commit**: (to be filled at merge time)
- **Notes**: Card explicitly clarifies that the "Invasive Breast Cancer"
  0.933/0.930 AUCs are on **pathology patches**, not mammography.
  Any downstream copy that treats these as mammography numbers is wrong
  and must be caught in review.

### 2026-07-01 — medgemma_1_5_4b

- **Change**: Initial card creation. Documents Jan 13, 2026 version 1.5.0
  release, May 20, 2025 initial 4B release, and July 9, 2025 EOI-token
  bug fix.
- **Source**: <https://developers.google.com/health-ai-developer-foundations/medgemma/model-card>, fetched 2026-07-01.
- **Commit**: (to be filled at merge time)
- **Notes**: MedGemma 1 4B (fine-tuned for CXR) scores higher on
  MIMIC-CXR RadGraph F1 (30.3) than MedGemma 1.5 4B (27.2); 1.5 is not
  strictly a superset improvement.

### 2026-07-01 — medgemma_1_27b

- **Change**: Initial card creation. Uses the MedGemma 1.5 model card as
  source (which reports MedGemma 1 27B numbers as comparison baseline).
- **Source**: <https://developers.google.com/health-ai-developer-foundations/medgemma/model-card>, fetched 2026-07-01.
- **Commit**: (to be filled at merge time)
- **Notes**: MedGemma 1 27B is still the strongest MedGemma variant on
  text-only medical Q&A (MedQA 4-op 85.3 vs 1.5 4B 69.1).

### 2026-07-01 — siglip_base_patch16_224

- **Change**: Initial card creation.
- **Source**: <https://huggingface.co/google/siglip-base-patch16-224>
  (identifier confirmed by prior code references; parameter count is
  common knowledge for SigLIP base).
- **Commit**: (to be filled at merge time)
- **Notes**: Ungated public proxy for CI-time MedSigLIP substitution.
  MUST NEVER be reported as MedSigLIP output.
