# Model card: `google/medgemma-1.5-4b-it`

Source-of-truth model card:
<https://developers.google.com/health-ai-developer-foundations/medgemma/model-card>
(fetched into this repo's `execution_trace/transcript.jsonl`; all numeric
values below come from that source unless flagged as our own measurement).

## Identifiers

| Field | Value |
|---|---|
| HuggingFace repo | `google/medgemma-1.5-4b-it` |
| Model version | `1.5.0` |
| Model created (v1.5) | Jan 13, 2026 |
| Model created (initial 4B) | May 20, 2025 |
| Bug-fix release | July 9, 2025 (missing EOI token, restored multimodal perf) |
| Key publication | MedGemma Technical Report, arXiv:2507.05201 (Sellergren et al., 2025) |
| License / terms | Health AI Developer Foundations (HAI-DEF) Terms of Use — **gated on HuggingFace** |

## Architecture

Based on Gemma 3 (decoder-only Transformer):

- Input modalities: text + vision (multimodal)
- Output modality: text only
- Attention: Grouped-query attention (GQA)
- Context length: at least 128K tokens
- Image encoder: SigLIP variant pre-trained on de-identified medical images
- Instruction tuned (`-it` suffix)

MedGemma 1.5 is currently only released as the 4B multimodal instruction-tuned
variant. The prior MedGemma 1 27B multimodal variant remains available
separately.

## Published performance — verbatim from model card

### Text-only medical benchmarks

| Dataset | Gemma 3 4B | MedGemma 1 4B | **MedGemma 1.5 4B** | MedGemma 1 27B |
|---|---:|---:|---:|---:|
| MedQA (4-op) | 50.7 | 64.4 | **69.1** | 85.3 |
| MedMCQA | 45.4 | 55.7 | **59.8** | 70.2 |
| PubMedQA | 68.4 | 73.4 | **68.2** | 77.2 |
| MMLU Med | 67.2 | 70.0 | **69.6** | 86.2 |
| MedXpertQA (text only) | 11.6 | 14.2 | **16.4** | 23.7 |
| AfriMed-QA (n=25 test) | 48.0 | 52.0 | **56.0** | 72.0 |

### Multimodal 2D image classification (Macro F1)

| Task | Gemma 3 4B | MedGemma 1 4B | **MedGemma 1.5 4B** | MedGemma 1 27B |
|---|---:|---:|---:|---:|
| MIMIC-CXR (top 5 conditions) | 81.2 | 88.9 | **89.5** | 90.0 |
| CheXpert CXR (top 5) | 32.6 | 48.1 | **48.2** | 49.9 |
| CXR14 (3 conditions) | 32.0 | 50.1 | **48.4** | 45.3 |
| PathMCQA histopathology accuracy | 37.1 | 69.8 | **70.0** | 71.6 |
| WSI-Path ROUGE | 2.3 | 2.2 | **49.4** | 4.1 |
| US-DermMCQA accuracy | 52.5 | 71.8 | **73.5** | 71.7 |
| EyePACS fundus accuracy | 14.4 | 64.9 | **76.8** | 75.3 |

### 3D imaging

| Task | Metric | MedGemma 1 4B | **MedGemma 1.5 4B** |
|---|---|---:|---:|
| CT Dataset 1 (7 conditions) | Macro accuracy | 58.2 | **61.1** |
| CT-RATE validation (18 conditions) | Macro F1 | 23.5 | **27.0** |
| MRI Dataset 1 (10 conditions) | Macro accuracy | 51.3 | **64.7** |

### Chest X-ray report generation (RadGraph F1 on MIMIC-CXR)

| Model | RadGraph F1 |
|---|---:|
| MedGemma 1 4B (tuned for CXR) | 30.3 |
| **MedGemma 1.5 4B** | **27.2** |
| MedGemma 1 27B | 27.0 |

Note: the fine-tuned MedGemma 1 4B variant scores higher than 1.5 4B on
this specific task; the 1.5 improvement is broad, not universal.

## Training data — verbatim

Multimodal variants use a SigLIP image encoder pre-trained on de-identified
medical images from these modalities:

- Radiology (chest X-rays via MIMIC-CXR; ChestImaGenome bounding boxes; CT and MR)
- Histopathology (TCGA, CAMELYON)
- Ophthalmology (fundus / EyePACS)
- Dermatology (SCIN, PAD-UFES-20)
- PMC-OA (biomedical literature figures)
- Mendeley Digital Knee X-Ray
- Proprietary radiology / dermatology / pathology sets

The LLM component is trained on medical text, medical Q&A pairs, FHIR-based
EHR data (27B multimodal only), radiology images, histopathology patches,
ophthalmology images, and dermatology images.

**Mammography imagery is NOT listed in the training data.**

## What is NOT in the card

- No mammography (2D screening / diagnostic) performance number.
- No DBT number.
- No breast MRI number.
- No CBIS-DDSM or EMBED benchmark.
- No probability calibration curves.

If `oncology-arbiter` reports a mammography number involving MedGemma, that
number MUST come from our own evaluation and MUST NOT be attributed to the card.

## Intended use — verbatim from card

> "MedGemma is an open multimodal generative AI model intended to be used
> as a starting point that enables more efficient development of downstream
> healthcare applications involving medical text and images."

## Recommended use in `oncology-arbiter`

- MedGemma 1.5 4B is the **explanation/report generation** side of the
  arbiter, not the primary classifier. Use it to produce structured
  free-text rationale over findings surfaced by other models.
- The `AUROC_CAVEAT` symbol MUST attach to any discrimination number
  we surface that is derived from MedGemma output.
- Never expose raw MedGemma text as clinical recommendation. Wrap in
  `RUO_DISCLAIMER`.

## Access model and gating

- HAI-DEF terms of use apply; users must accept the terms on HuggingFace
  before the weights are downloadable.
- Runtime handler in `src/oncology_arbiter/models/hai_def.py` (Phase 2,
  worker-3 scaffolding) returns `ModelState.GATED` on 401 without a
  fabricated response.

## Errata

*(See `errata.md`.)*
