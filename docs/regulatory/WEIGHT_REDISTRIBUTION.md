# Weight Redistribution Policy (v0.4.0-alpha)

This document defines the rules the `weights_meet_floor` CI gate enforces
before any weight artifact can be shipped in a release tag.

## 1. Motivation

Version 0.3.0-alpha exposed one clear failure mode: shipping a fine-tuned
checkpoint whose achieved metric was below a documented public baseline
silently regresses the surface without breaking any test. The v0.4.0-alpha
release adds a mandatory CI gate that refuses to greenlight such a
regression — every checkpoint must attach a `WeightsProvenance` record
with `weights_meet_floor: true`.

## 2. Contract

Every fine-tuned checkpoint ships with a JSON sidecar at
`docs/proofs/<checkpoint_id>_metrics.json` that includes a
`weights_meet_floor` block:

```json
{
  "weights_meet_floor": {
    "weights_meet_floor": true,
    "achieved_metric": 0.883,
    "floor_metric": 0.85,
    "floor_source": "docs/proofs/cbis_ddsm_logreg_v1_metrics.json (typical fine-tuned CNN baseline 0.85–0.90)",
    "metric_name": "AUROC"
  }
}
```

The pydantic model is `oncology_arbiter.api.schemas.WeightsProvenance`.

## 3. Per-checkpoint floors

| Checkpoint                        | Metric                       | Floor  | Floor source                                                                                          |
|-----------------------------------|------------------------------|--------|-------------------------------------------------------------------------------------------------------|
| Mammo MONAI RetinaNet (v0.4.0)    | AUROC                        | 0.85   | Typical fine-tuned CNN baseline on CBIS-DDSM (0.85–0.90 range)                                        |
| Biopsy MedSigLIP probe (v0.3.0)   | AUROC                        | 0.85   | `docs/proofs/cbis_ddsm_logreg_v1_metrics.json`                                                         |
| LUNA16 refine (v0.4.0)            | ΔFROC@2 FPs/scan             | +5%    | v0.6.9 baseline documented in `docs/proofs/luna16_v069_metrics.json`                                   |

The floor for the Mammo checkpoint is documented as a *range* — the CI
gate enforces the lower bound (0.85). If a future model card wants to
raise the floor above 0.85 for its own regression protection it may set
`floor_metric` higher; the gate only refuses `achieved_metric < floor_metric`.

## 4. Gate behaviour

`weights_meet_floor` runs on every PR touching:

- `docs/proofs/*.json`
- `src/oncology_arbiter/api/schemas.py`
- Any weight artifact under `weights/` (rare in this repo — most weights
  live on Modal buckets referenced by `Provenance.model_endpoint_url`)

If any sidecar has `weights_meet_floor.weights_meet_floor == false` OR
`achieved_metric < floor_metric`, the job exits nonzero and blocks the
merge.

## 5. What is NOT a redistribution

The gate ONLY guards the `weights_meet_floor` boolean and the numeric
comparison. It does NOT re-run the training loop, and it does NOT verify
that `achieved_metric` was produced by the process described in
`training_provenance` — that is the responsibility of the model-card
reviewer at PR review time.

## 6. HIPAA posture

Weight sidecars must not contain any patient identifier — the
`no_hf_token_grep` job additionally checks that no sidecar under
`docs/proofs/` contains a value that looks like an MRN, SSN, or a real
patient name.
