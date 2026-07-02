# Model card: `google/medgemma-27b-it` (MedGemma 1 27B multimodal)

Source-of-truth: same MedGemma model card as MedGemma 1.5 4B, which reports
MedGemma 1 27B numbers as a comparison baseline for the 1.5 4B release.

## Identifiers

| Field | Value |
|---|---|
| HuggingFace repo | `google/medgemma-27b-it` |
| Model version | 1.0.x (MedGemma 1 line; not yet updated to 1.5) |
| Key publication | MedGemma Technical Report, arXiv:2507.05201 |
| License / terms | Health AI Developer Foundations (HAI-DEF) Terms of Use — **gated on HuggingFace** |

## Architecture

- Based on Gemma 3 27B (decoder-only Transformer, GQA, ≥128K context)
- Multimodal variant: SigLIP image encoder + LLM
- Text-only variant also released as `google/medgemma-27b-text-it`
- Instruction tuned

## Published performance — same source as 1.5 4B card

### Text-only medical (higher is better)

| Dataset | **MedGemma 1 27B** | MedGemma 1.5 4B | Gemma 3 4B |
|---|---:|---:|---:|
| MedQA (4-op) | **85.3** | 69.1 | 50.7 |
| MedMCQA | **70.2** | 59.8 | 45.4 |
| PubMedQA | **77.2** | 68.2 | 68.4 |
| MMLU Med | **86.2** | 69.6 | 67.2 |
| MedXpertQA (text) | **23.7** | 16.4 | 11.6 |
| AfriMed-QA (n=25) | **72.0** | 56.0 | 48.0 |

MedGemma 1 27B remains the strongest published MedGemma variant on
text-only medical Q&A.

### Multimodal 2D imaging (Macro F1 unless noted)

| Task | **MedGemma 1 27B** | MedGemma 1.5 4B |
|---|---:|---:|
| MIMIC-CXR (top 5) | **90.0** | 89.5 |
| CheXpert (top 5) | **49.9** | 48.2 |
| CXR14 (3 conditions) | 45.3 | **48.4** |
| PathMCQA accuracy | **71.6** | 70.0 |
| US-DermMCQA accuracy | 71.7 | **73.5** |
| EyePACS accuracy | 75.3 | **76.8** |

**MedGemma 1 27B and MedGemma 1.5 4B trade wins across tasks.** 1.5 4B is
NOT strictly better; use 1.5 4B when you need efficiency and the 4B numbers
are equivalent or superior for your task, use 1 27B when the 27B number is
higher for your task.

### 3D imaging

| Task | Metric | **MedGemma 1 27B** | MedGemma 1.5 4B |
|---|---|---:|---:|
| CT Dataset 1 (7 conditions) | Macro accuracy | 57.8 | **61.1** |
| MRI Dataset 1 (10 conditions) | Macro accuracy | 57.4 | **64.7** |

## Training data

Same modality coverage as MedGemma 1.5 4B (see `medgemma_1_5_4b.md`), plus:

- FHIR-based EHR data (27B multimodal variant only)

**Mammography imagery is NOT listed in the training data.**

## What is NOT in the card

- No mammography performance number.
- No DBT number.
- No breast MRI number.
- No CBIS-DDSM or EMBED benchmark.

## Intended use — verbatim

> "MedGemma is an open multimodal generative AI model intended to be used
> as a starting point that enables more efficient development of downstream
> healthcare applications."

## Recommended use in `oncology-arbiter`

- MedGemma 1 27B is a text-heavy reasoner. Use when the pipeline needs
  strong medical text QA and can afford 27B weights.
- For pure image encoding, prefer `google/medsiglip-448` — same encoder
  family, lighter, purpose-built for embedding tasks.
- All governance disclaimers (`RUO_DISCLAIMER`, `AUROC_CAVEAT`,
  `ModelState.GATED` fallback) apply identically.

## Errata

*(See `errata.md`.)*
