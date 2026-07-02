# Model card: `google/siglip-base-patch16-224` (ungated public SigLIP proxy)

This is the **public, ungated** SigLIP checkpoint used by `oncology-arbiter`
as a smoke-test proxy when the gated `google/medsiglip-448` cannot be
accessed. It is a general-domain SigLIP, NOT a medical model.

## Identifiers

| Field | Value |
|---|---|
| HuggingFace repo | `google/siglip-base-patch16-224` |
| Model family | SigLIP base (Zhai et al., 2023) |
| Key publication | Zhai et al., "Sigmoid Loss for Language Image Pre-Training," arXiv:2303.15343 |
| License | Apache 2.0 (per public HuggingFace listing) |
| Access model | **Ungated** — downloadable without accepting terms |

## Architecture

- Vision transformer, patch size 16, input resolution 224 × 224
- Base configuration (~200M parameters vision + text)
- Contrastive image-text pre-training with the sigmoid loss variant of CLIP

## Training data

WebLI, a large web-crawled image-text dataset. **General-domain images and
captions; NO medical imagery specifically curated.**

## Position in `oncology-arbiter`

This model is used ONLY for:

1. **Smoke tests** in `tests/models/test_siglip_baseline.py` (worker-4
   deliverable) that confirm the SigLIP inference path in
   `src/oncology_arbiter/models/siglip_baseline.py` actually runs end-to-end
   on a real CBIS-DDSM mammogram. This checks the plumbing (tensor shapes,
   preprocessing, embedding extraction) — NOT the clinical validity.
2. **Fallback classification** when `google/medsiglip-448` is gated
   (`ModelState.PROXY_SIGLIP` in the audit envelope). Every output derived
   from the proxy MUST be tagged `ModelState.PROXY_SIGLIP` so that
   downstream consumers can distinguish it from the medical-domain model.

## Explicit non-uses

- **Do not** report proxy zero-shot AUCs on mammography as evidence of
  MedSigLIP's mammography performance. They are two different models with
  different pre-training corpora and different input resolutions.
- **Do not** silently substitute the proxy for MedSigLIP in production.
  The audit envelope must mark the swap.
- **Do not** produce a screening recommendation from the proxy alone.
  `RUO_DISCLAIMER` applies.

## Why we need this proxy

The gated MedSigLIP repo returns 401 to any HuggingFace client that has
not accepted the HAI-DEF terms of use on-account. Continuous integration
runs on ephemeral sandboxes that cannot maintain that acceptance state,
which means the CI pipeline needs an ungated fallback to exercise
`siglip_baseline.py` code paths. The proxy fills exactly that role and
nothing more.

## Errata

*(See `errata.md`.)*
