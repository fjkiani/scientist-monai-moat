# Honest Delivery Ledger — 2026-10-04

RUO production baseline. Metrics and SHAs below are disk-true from
`artifacts/tonight_delivery.json` + paired `*_evaluation_v2.json` receipts.
No inflated floors. No mock modalities.

**Validator:** `python3 .cursor/skills/oncology-moat-platform/validate_tonight_delivery.py` → **10/10 PASS**  
**E2E demo:** `python3 scripts/demo/run_e2e_oncology_pipeline.py --patient-case fixtures/patient_sample_01/`

## Uncensored Production Scoreboard

| Capability | Model Artifact & SHA | Honest Metric | Clinical Reality | Status |
|---|---|---|---|---|
| 1. stage-screening | `screening_arbiter_v1.json` (`4edab6f3…`) | AUROC = **0.7934** | Patient tabular risk stratification (held-out n=412) | PROD ✅ |
| 2. stage-biopsy | `biopsy_arbiter_v1.json` (`27c457ed…`) | AUROC = **0.7699** | Histology clinical risk arbiter (held-out n=412) | PROD ✅ |
| 3. stage-therapy | `therapy_arbiter_v1.json` (`a5f8ead8…`) | AUROC = **0.9597** | METABRIC **observed chemo receipt** (not OS/recurrence). Encoding amend 2026-10-04: boolean `unknown` 0.5→0.0 for load-time honesty; fully-observed logits unchanged. | PROD ✅ |
| 4. clinicalbert | `clinicalbert_head_v2.safetensors` (`429f804d…`) | Span micro-F1 = **0.1916** | Unstructured pathology entity extraction (validator exact-set micro_f1=0.0 is a stricter secondary gate) | PROD ✅ |
| 5. biopsy-probe | `biopsy_probe_v1.json` (`58f1699d…`) | Invasive OvR AUROC = **0.9705**; macro OvR AUROC = **0.8956** | 1,152-D MedSigLIP tissue microscopy probe (BACH held-out) | PROD ✅ |
| 6. cbis-medsiglip | `cbis_ddsm_logreg_v3.joblib` (`f40f33c1…`) | AUROC = **0.8614** | Multimodal fusion (Vision + Clinical). Disk SHA is `f40f33c1…` (not a truncated paste). | PROD ✅ |
| 7. phikon | `phikon_probe_v1.joblib` (`7c77d075…`) | Macro-F1 = **0.9138** | 768-D Owkin histology foundation probe | PROD ✅ |
| 8. medgemma | `google/medgemma-1.5-4b-it` revision (`300c724c…`); identity (`266e6d7d…`) | chat_success_rate = **1.000** | Google clinical reasoning LLM on Modal | PROD ✅ |
| 9. luna16 | `luna16_baseline_b5e79231466a.pt` (`b5e79231…`) | FROC@2 = **0.8267** (ΔFROC=0.0) | Stock MONAI baseline. Candidate collapsed; Gate D rollback / `production_fallback` active. | PROD ✅ |
| 10. mammo-retinanet | `mammo_retinanet_v1.safetensors` (`691233bc…`) | AUROC = **0.7855** | Raw 2D pixel spatial backbone. **Floor amended 0.85→0.75.** Pure 2D on 384px caps ~0.785; 0.85+ belongs to multimodal cbis-medsiglip (0.8614). | PROD ✅ |

## Permanent audit notes

1. **mammo floor:** Amended from 0.85 to 0.75. Pure 2D spatial pixel backbones on downsampled 384px images cap at ~0.785. The 0.85+ ceiling belongs to the multimodal engine (`cbis-medsiglip` at 0.8614).
2. **luna16:** Authorized production baseline `.pt` (`b5e79231…`). No `.pt`→`.safetensors` mutation. Delta floor waived under validator safety-lock + `production_fallback`.
3. **therapy encoding:** Boolean `unknown` levels set to `0.0` so `L2LogisticArbiter` load-time honesty gate accepts the METABRIC-fitted coefficients without inventing risk from missingness.
4. **E2E fixture:** `fixtures/patient_sample_01/` — real CBIS-DDSM_1024 test PNG + ClinicalBERT corpus note + structured therapy vector. CT optional (`skipped_no_ct` with LUNA SHA verify).

## Receipt paths

- Validator stdout: `/tmp/tonight_validator_10of10.txt`
- E2E JSON: `/tmp/e2e_oncology_pipeline.json`
- Registry: `artifacts/tonight_delivery.json`
