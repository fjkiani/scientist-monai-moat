# Model card: `google/medsiglip-448`

Source-of-truth model card:
<https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card>
(fetched into this repo's `execution_trace/transcript.jsonl`; all numeric
values below come from that source unless flagged as our own measurement).

## Identifiers

| Field | Value |
|---|---|
| HuggingFace repo | `google/medsiglip-448` |
| Model version | `1.0.0` |
| Model created | July 9, 2025 |
| Key publication | MedGemma Technical Report, arXiv:2507.05201 (Sellergren et al., 2025) |
| License / terms | Health AI Developer Foundations (HAI-DEF) Terms of Use — **gated on HuggingFace** |

## Architecture

Two-tower encoder based on SigLIP-400M (Zhai et al., 2023):

- Vision encoder: 400M-parameter Vision Transformer
- Text encoder: 400M-parameter Text Transformer
- Image resolution: **448 × 448**
- Context length: **64 tokens** for text
- Input normalization: images in **(-1, 1)** range at 448 × 448
- Outputs: image embedding, text embedding, and similarity logits

The vision encoder is the same one that powers image interpretation in the
generative MedGemma 1.5 4B model.

## Training data

De-identified image-text pairs from:

- Chest X-rays (MIMIC-CXR)
- Dermatology images (SCIN, PAD-UFES-20, plus proprietary sets)
- Ophthalmology images (EyePACS fundus)
- Histopathology slides (TCGA, CAMELYON, plus 4 proprietary pathology sets)
- Slices of CT and MRI volumes (proprietary US radiology dataset)
- Non-medical natural image-text pairs (to retain general image understanding)

**Mammography imagery is NOT listed in the training data.** This is the
single most important limitation for `oncology-arbiter`: MedSigLIP has NOT
been shown mammograms during pre-training. Downstream mammography use is
out-of-distribution transfer.

## Published performance — verbatim from model card

### Chest X-ray zero-shot AUC vs ELIXR (n=518, 2-class)

| Finding | Med-SigLIP zero-shot | ELIXR zero-shot |
|---|---:|---:|
| Enlarged cardiomediastinum | 0.858 | 0.800 |
| Cardiomegaly | 0.904 | 0.891 |
| Lung opacity | 0.931 | 0.888 |
| Lung lesion | 0.822 | 0.747 |
| Consolidation | 0.880 | 0.875 |
| Edema | 0.891 | 0.880 |
| Pneumonia | 0.864 | 0.881 |
| Atelectasis | 0.836 | 0.754 |
| Pneumothorax | 0.862 | 0.800 |
| Pleural effusion | 0.914 | 0.930 |
| Pleural other | 0.650 | 0.729 |
| Fracture | 0.708 | 0.637 |
| Support devices | 0.852 | 0.894 |
| **Average** | **0.844** | **0.824** |

Note: MedSigLIP sees 448×448 inputs; ELIXR sees 1280×1280.

### Multi-domain AUC (Med-SigLIP vs HAI-DEF)

| Domain | Finding | n | Classes | Zero-shot | Linear probe | HAI-DEF linear probe |
|---|---|---:|---:|---:|---:|---:|
| Dermatology | Skin conditions | 1,612 | 79 | 0.851 | 0.881 | 0.843 |
| Ophthalmology | Diabetic retinopathy | 3,161 | 5 | 0.759 | 0.857 | N/A |
| **Pathology** | **Invasive breast cancer** | **5,000** | **3** | **0.933** | **0.930** | **0.943** |
| Pathology | Breast NP | 5,000 | 3 | 0.721 | 0.727 | 0.758 |
| Pathology | Breast TF | 5,000 | 3 | 0.780 | 0.790 | 0.832 |
| Pathology | Cervical dysplasia | 5,000 | 3 | 0.889 | 0.864 | 0.898 |
| Pathology | Prostate cancer needle-core biopsy | 5,000 | 4 | 0.892 | 0.886 | 0.915 |
| Pathology | Radical prostatectomy | 5,000 | 4 | 0.896 | 0.887 | 0.921 |
| Pathology | TCGA study types | 5,000 | 10 | 0.922 | 0.970 | 0.964 |
| Pathology | Tissue types | 5,000 | 16 | 0.930 | 0.972 | 0.947 |
| **Average** | | | | **0.870** | **0.878** | **0.897** |

**Read the breast-cancer row carefully.** The 0.933 zero-shot / 0.930
linear-probe AUC for "Invasive Breast Cancer" is on **histopathology
patches (n=5,000, 3 classes)**, NOT mammograms. Any project narrative
that conflates this with mammographic detection performance is wrong.

### What is NOT in the card

- No mammography (2D screening/diagnostic mammogram) performance number.
- No DBT (digital breast tomosynthesis) number.
- No breast-MRI number.
- No CBIS-DDSM or EMBED benchmark.
- No calibration curves (Brier, ECE) — the card reports discrimination (AUC) only.

If `oncology-arbiter` reports a mammography AUC with MedSigLIP, that number
MUST come from our own evaluation on a specified dataset, and it MUST NOT
be attributed to the model card.

## Access model and gating

- HAI-DEF terms of use apply; users must accept the terms on HuggingFace
  before the weights become downloadable.
- The `google/medsiglip-448` repo returns 401 without accepted terms
  even with a valid HF token. See `src/oncology_arbiter/models/hai_def.py`
  (Phase 2, worker-3 scaffolding) for the runtime handler.
- License permits research and non-clinical use consistent with HAI-DEF
  terms. **This is not an FDA-cleared device.** The `RUO_DISCLAIMER`
  symbol in `oncology_arbiter/__init__.py` MUST accompany every user-facing
  report that references MedSigLIP output.

## Intended use — verbatim from card

> "MedSigLIP is a machine learning-based software development tool that
> generates numerical representations from input images and associated text.
> These representations are referred to as embeddings. [...] MedSigLIP
> itself does not provide any medical functionality, nor is it intended to
> process or interpret medical data for a medical purpose."

## Recommended use in `oncology-arbiter`

- Use MedSigLIP as an **image encoder** whose output enters our L2
  logistic arbiter alongside other model probabilities. Do NOT expose
  its raw zero-shot logits as a screening decision.
- When the ungated `google/siglip-base-patch16-224` proxy is used instead
  (worker-4 smoke test), report the swap explicitly in the audit envelope
  via `ModelState.PROXY_SIGLIP` — never claim proxy output as MedSigLIP output.
- Always co-report the `AUROC_CAVEAT` symbol contents when discrimination
  metrics are shown.

## Errata

*(Errata for this card are tracked in `errata.md`. Add entries when Google
publishes a new model version or corrects a metric.)*
