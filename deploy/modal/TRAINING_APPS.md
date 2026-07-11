# Modal apps (v0.4.0-alpha)

This directory hosts every Modal deployment for OncologyArbiter.

| App | Modal name | Purpose | GPU |
| --- | --- | --- | --- |
| MedSigLIP-448 | `medsiglip-448` | Vision embedding + zero-shot (inference) | A10G |
| TxGemma-9B chat | `txgemma-9b` | Therapy-reasoning LLM (inference) | A10G |
| LUNA16 refine | `luna16-refine` | Fine-tune shipped bundle (training) | A10G |
| CBIS-DDSM RetinaNet | `cbis-ddsm-detect` | Mass/calc detector (training) | A10G |

## Prod flip for MedSigLIP

The default deploy runs `min_containers=0` (cold start on each request,
$0/h idle). Setting `MEDSIGLIP_MODAL_MODE=prod` at deploy time bumps to
`min_containers=1` and `scaledown_window=900`s, keeping one warm A10G
replica during prod hours.

```bash
MEDSIGLIP_MODAL_MODE=prod modal deploy deploy/modal/medsiglip_app.py
```

Flip back with `MEDSIGLIP_MODAL_MODE=staging` (or unset) + redeploy.

## TxGemma deploy

```bash
# 1. Create the HF token secret if not present:
modal secret create txgemma-hf-token HF_TOKEN=hf_...

# 2. Deploy the endpoint
modal deploy deploy/modal/txgemma_app.py

# 3. Point oncology-arbiter at the deployment
export TXGEMMA_BACKEND=modal
export TXGEMMA_MODAL_URL="https://<workspace>--txgemma"
```

The API layer picks up `TXGEMMA_BACKEND=modal` and swaps
`oncology_arbiter.models.txgemma_client.TxGemmaClient` for
`oncology_arbiter.models.txgemma_modal_client.TxGemmaModalClient` at
request time. No downstream code changes needed.

`/reason` responses always include the `honesty_warning` string. If the
underlying HF gate returns 403, the client raises
`GatedAccessError(access_level=FORBIDDEN)` — mirrors the local client.

---

# Training apps (v0.4.0-alpha)

Two GPU training apps for OncologyArbiter's imaging modules. Both are
_cost-optimized_ (`min_containers=0`), meaning the container only spins up
when a job is submitted.

| App | Modal name | GPU | Data volume | Output volume |
| --- | --- | --- | --- | --- |
| LUNA16 RetinaNet fine-tune | `luna16-refine` | A10G | `luna16-data` + `luna16-baseline-weights` | `luna16-training-runs` |
| CBIS-DDSM RetinaNet detector | `cbis-ddsm-detect` | A10G | `cbis-ddsm-data` + `oa-repo-code` | `cbis-ddsm-training-runs` |

## One-time setup

1. **Modal token**: `modal token new` (or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` env).
2. **Upload data volumes** — done on a machine that already has the raw data:

   ```bash
   # LUNA16 subsets + bundle
   modal run scripts/upload_luna16_to_modal.py::upload \
     --luna16-dir /workspace/data/luna16 \
     --bundle-dir /workspace/monai_bundles/lung_nodule_ct_detection

   # CBIS-DDSM DICOM + oa src
   modal run scripts/upload_cbis_ddsm_to_modal.py::upload \
     --cbis-dir /workspace/data/CBIS-DDSM_full \
     --oa-src   /workspace/oa-repo/src
   ```

3. **Deploy the healthz + trigger endpoints**:

   ```bash
   modal deploy deploy/modal/luna16_finetune_app.py
   modal deploy deploy/modal/cbis_ddsm_detection_app.py
   ```

## Kicking off training

### LUNA16 refine-v1

```bash
modal run deploy/modal/luna16_finetune_app.py --fold 0 --epochs 20 --learning-rate 1e-3
```

Or via HTTP:

```bash
curl -X POST https://<user>--luna16-refine-trigger.modal.run \
  -H 'content-type: application/json' \
  -d '{"fold":0,"epochs":20,"learning_rate":0.001}'
```

Container behavior:

1. Loads the shipped bundle baseline weights from `luna16-baseline-weights` Volume.
2. On first run, unpacks any un-unpacked subset zips and resamples fold-0
   series to `(0.703125, 0.703125, 1.25) mm` NIfTI under
   `/vol/luna16/nifti/`. This is cached across runs.
3. Invokes `python -m monai.bundle run training` with overridden `epochs` /
   `learning_rate` / `dataset_dir` / `data_list_file_path`.
4. Runs `validate` twice (baseline vs refined weights) and emits
   `refine_metrics.json` in `luna16-training-runs/<run_id>/` with schema:

   ```json
   {
     "schema_version": "v0.4.0-alpha",
     "fold_index": 0,
     "n_train_series": 534, "n_val_series": 67,
     "target_spacing_mm": [0.703125, 0.703125, 1.25],
     "baseline": {"froc_at_2fps": ..., "map_iou0.1": ...},
     "refined":  {"froc_at_2fps": ..., "map_iou0.1": ...},
     "delta":    {"froc_at_2fps": ..., "map_iou0.1": ...},
     "plan_target": {"froc_at_2fps_delta_min": 0.05, "met": true}
   }
   ```

### CBIS-DDSM detector

```bash
modal run deploy/modal/cbis_ddsm_detection_app.py --epochs 20 --learning-rate 1e-4
```

Or:

```bash
curl -X POST https://<user>--cbis-ddsm-detect-trigger.modal.run \
  -H 'content-type: application/json' \
  -d '{"epochs":20,"learning_rate":0.0001}'
```

Container behavior:

1. Reads `series_manifest.json` from `cbis-ddsm-data`.
2. Builds case manifest from DICOM `PatientID` regex (defined in
   `oncology_arbiter.mammography.cbis_ddsm_detection`).
3. Converts full mammograms → percentile-windowed uint8 PNG, derives lesion
   bounding boxes from paired ROI masks via connected-components.
4. Fine-tunes `retinanet_resnet50_fpn_v2` (COCO-pretrained backbone,
   `num_classes=2` background + lesion).
5. Evaluates with `pycocotools.COCOeval` on the Test split; emits
   `detection_metrics.json` in `cbis-ddsm-training-runs/<run_id>/` with
   `map_at_iou_0.5` and `map_at_iou_0.5_0.95`.

## Dry runs (no GPU cost)

Both apps accept `--dry-run`. LUNA16's dry-run stops after resample; CBIS-DDSM's
stops after preparing 5 cases + writing COCO manifests. Use these to verify
Volume mounts before starting a full run.
