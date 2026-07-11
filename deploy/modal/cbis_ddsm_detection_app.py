"""Modal app: CBIS-DDSM RetinaNet mass/calc detection (2D mammography).

Trains a 2D RetinaNet detector on the CBIS-DDSM full-mammogram + ROI-mask
corpus, using the ``build_case_manifest`` / ``bbox_from_roi_mask`` scaffold
from :mod:`oncology_arbiter.mammography.cbis_ddsm_detection`.

Inputs (Modal Volume ``cbis-ddsm-data``)
----------------------------------------
- ``CBIS-DDSM_full/`` — raw TCIA DICOM series (unmodified download).
- ``series_manifest.json`` — cache of ``nbia.getSeries("CBIS-DDSM")`` output.

Steps
-----
1. Build case manifest from DICOM ``PatientID`` values (regex from
   :mod:`oncology_arbiter.mammography.cbis_ddsm_detection`).
2. Convert full mammograms to normalized ``uint8`` PNGs; derive bboxes from
   paired ROI masks via connected-components.
3. Train a torchvision RetinaNet with ResNet-50 FPN backbone on the
   Training split; evaluate on the Test split with COCO mAP.
4. Emit ``detection_metrics.json`` for the audit ledger.

Design notes
------------
- Torchvision RetinaNet is used (not MONAI) because CBIS-DDSM is 2D and
  torchvision's detection API is a smaller footprint than MONAI's 3D
  ``RetinaNetDetector``.
- We do NOT re-implement the manifest builder — we import from
  ``oncology_arbiter.mammography.cbis_ddsm_detection`` which lives on main.
- Cost: ``min_containers=0``. The CBIS-DDSM DICOM corpus is 163 GB.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_VERSION = "cbis-ddsm-detect-v0.4.0-alpha"

CBIS_VOL = modal.Volume.from_name("cbis-ddsm-data", create_if_missing=True)
OA_CODE_VOL = modal.Volume.from_name("oa-repo-code", create_if_missing=True)
OUTPUT_VOL = modal.Volume.from_name("cbis-ddsm-training-runs", create_if_missing=True)

CBIS_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "libgl1")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-image==0.24.0",
        "pillow==10.4.0",
        "pydicom==2.4.4",
        "pycocotools==2.0.8",
        "tqdm==4.66.5",
        "fastapi==0.115.0",
        "pydantic==2.9.2",
    )
)

app = modal.App("cbis-ddsm-detect")


HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="cbis-ddsm-detect-healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": "cbis-ddsm-detect", "version": APP_VERSION}


def _normalize_full_mammogram(pixel_array, bits_stored: int):
    """Windowed uint8 from a MG DICOM, using percentile [1,99]."""
    import numpy as np

    arr = pixel_array.astype("float32")
    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        hi = float(arr.max()) or 1.0
        lo = float(arr.min())
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255.0).astype("uint8")


def _prepare_case(
    case_id: str,
    case,
    dicom_root: Path,
    out_root: Path,
) -> Optional[Dict[str, Any]]:
    """Convert one CBIS-DDSM case to normalized PNG + bboxes.

    Returns
    -------
    dict | None
        ``{image_path, boxes, labels, split, lesion_type}`` or None if either
        the full mammogram or all ROI masks are unreadable.
    """
    import numpy as np
    import pydicom
    from PIL import Image

    from oncology_arbiter.mammography.cbis_ddsm_detection import bbox_from_roi_mask

    if case.full_mammogram_uid is None:
        return None

    # Full mammogram: pick single dcm in that series folder
    full_dir = dicom_root / case.full_mammogram_uid
    dcm_paths = list(full_dir.rglob("*.dcm"))
    if not dcm_paths:
        return None
    full = pydicom.dcmread(dcm_paths[0])
    full_arr = full.pixel_array
    img_u8 = _normalize_full_mammogram(full_arr, int(getattr(full, "BitsStored", 16)))
    H, W = img_u8.shape

    # ROI masks (may be several)
    boxes: List[List[float]] = []
    labels: List[int] = []
    for roi_uid in case.roi_mask_uids:
        roi_dir = dicom_root / roi_uid
        roi_paths = list(roi_dir.rglob("*.dcm"))
        if not roi_paths:
            continue
        roi = pydicom.dcmread(roi_paths[0])
        roi_arr = roi.pixel_array
        # Rescale mask to full mammogram shape if different (CBIS masks match)
        if roi_arr.shape != (H, W):
            roi_img = Image.fromarray(roi_arr).resize((W, H), Image.NEAREST)
            roi_arr = np.array(roi_img)
        bb = bbox_from_roi_mask(roi_arr > 0, min_area_frac=1e-3)
        for b in bb:
            boxes.append([b.x_min, b.y_min, b.x_max, b.y_max])
            labels.append(0)  # single-class lesion detection

    if not boxes:
        return None

    out_dir = out_root / case.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.png"
    Image.fromarray(img_u8).save(out_path)
    return {
        "image_path": str(out_path),
        "boxes": boxes,
        "labels": labels,
        "split": case.split,
        "lesion_type": case.lesion_type,
    }


@app.function(
    image=CBIS_IMAGE,
    gpu="A10G",
    volumes={
        "/vol/cbis": CBIS_VOL,
        "/vol/code": OA_CODE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
    timeout=8 * 60 * 60,
    memory=32 * 1024,
    cpu=8.0,
)
def train_detector(
    epochs: int = 20,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    input_size: int = 800,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fine-tune a torchvision RetinaNet on CBIS-DDSM.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Adam LR.
    batch_size : int
        Batch size per step.
    input_size : int
        Longer-side resize target in pixels.
    dry_run : bool
        If True, only build the case manifest + first-case PNG and return.
    """
    import shutil
    import sys

    sys.path.insert(0, "/vol/code/src")
    from oncology_arbiter.mammography.cbis_ddsm_detection import (
        build_case_manifest,
        coco_style_annotations,
    )

    run_id = f"cbis-{int(time.time())}"
    run_dir = Path("/vol/output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_json = Path("/vol/cbis/series_manifest.json")
    dicom_root = Path("/vol/cbis/CBIS-DDSM_full")
    if not manifest_json.exists():
        raise RuntimeError(f"series_manifest.json missing at {manifest_json}")
    with open(manifest_json) as f:
        series_records = json.load(f)

    print(f"[{run_id}] Building case manifest from {len(series_records)} series...", flush=True)
    # scaffold expects (patient_id, description, series_uid) tuples; adapt
    case_manifest = build_case_manifest(series_records)
    print(f"[{run_id}] {len(case_manifest)} cases in manifest.", flush=True)

    # Prepare PNGs + bboxes
    png_root = Path("/vol/cbis/prepared_pngs")
    coco_records: List[Dict[str, Any]] = []
    n_prepared = 0
    for case_id, case in case_manifest.items():
        rec = _prepare_case(case_id, case, dicom_root, png_root)
        if rec is None:
            continue
        rec["case_id"] = case_id
        coco_records.append(rec)
        n_prepared += 1
        if dry_run and n_prepared >= 5:
            break
        if n_prepared % 200 == 0:
            print(f"[{run_id}] prepared {n_prepared} cases...", flush=True)

    print(f"[{run_id}] Prepared {n_prepared} cases with ≥1 lesion box.", flush=True)

    # Emit COCO manifests
    train_recs = [r for r in coco_records if r["split"] == "Training"]
    test_recs  = [r for r in coco_records if r["split"] == "Test"]
    train_coco = _to_coco(train_recs)
    test_coco  = _to_coco(test_recs)
    (run_dir / "train_coco.json").write_text(json.dumps(train_coco, indent=2))
    (run_dir / "test_coco.json").write_text(json.dumps(test_coco, indent=2))
    print(f"[{run_id}] train={len(train_recs)} test={len(test_recs)}", flush=True)

    if dry_run:
        return {"status": "dry_run", "n_prepared": n_prepared, "run_dir": str(run_dir)}

    # Train
    metrics = _train_retinanet(
        train_records=train_recs,
        val_records=test_recs,
        run_dir=run_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        input_size=input_size,
    )
    metrics_out = {
        "schema_version": "v0.4.0-alpha",
        "run_id": run_id,
        "app_version": APP_VERSION,
        "n_train": len(train_recs),
        "n_test": len(test_recs),
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_size": input_size,
        },
        "metrics": metrics,
    }
    (run_dir / "detection_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    OUTPUT_VOL.commit()
    return {"status": "ok", "metrics": metrics_out, "run_dir": str(run_dir)}


def _to_coco(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize per-case detection records into a minimal COCO JSON."""
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    ann_id = 1
    for i, r in enumerate(records):
        images.append({
            "id": i,
            "file_name": Path(r["image_path"]).name,
            "path": r["image_path"],
        })
        for box, label in zip(r["boxes"], r["labels"]):
            x1, y1, x2, y2 = box
            anns.append({
                "id": ann_id,
                "image_id": i,
                "category_id": label,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO xywh
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 0, "name": "lesion"}],
    }


def _train_retinanet(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    run_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    input_size: int,
) -> Dict[str, float]:
    """Minimal RetinaNet fine-tune loop over CBIS-DDSM PNG + bboxes."""
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights

    class CbisDataset(Dataset):
        def __init__(self, records: List[Dict[str, Any]], size: int):
            self.records = records
            self.size = size

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            img = Image.open(r["image_path"]).convert("RGB")
            W0, H0 = img.size
            img = img.resize((self.size, self.size), Image.BILINEAR)
            sx, sy = self.size / W0, self.size / H0
            boxes = []
            labels = []
            for b, l in zip(r["boxes"], r["labels"]):
                x1, y1, x2, y2 = b
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                labels.append(l + 1)  # RetinaNet expects labels ≥ 1
            tensor = TF.to_tensor(img)
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            }
            return tensor, target

    def collate(batch):
        return list(zip(*batch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        num_classes=2,  # background + lesion
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(
        CbisDataset(train_records, input_size),
        batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate,
    )

    losses_hist: List[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses_hist.append(avg)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={avg:.4f}", flush=True)

    # Val: compute mAP@0.5 with pycocotools if available
    metrics = {"final_train_loss": losses_hist[-1] if losses_hist else 0.0}
    try:
        metrics.update(_coco_eval(model, val_records, input_size, device))
    except Exception as e:
        print(f"[warn] COCO eval failed: {e}", flush=True)
        metrics["coco_eval_error"] = str(e)

    # Save checkpoint
    torch.save(model.state_dict(), run_dir / "model.pt")
    return metrics


def _coco_eval(model, val_records, input_size, device) -> Dict[str, float]:
    """Run inference on val set and score with pycocotools."""
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    gt = _to_coco(val_records)
    # pycocotools needs paths on disk — dump temp GT
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix="_gt.json", delete=False) as f:
        json.dump(gt, f)
        gt_path = f.name
    coco_gt = COCO(gt_path)

    predictions: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i, r in enumerate(val_records):
            img = Image.open(r["image_path"]).convert("RGB")
            W0, H0 = img.size
            resized = img.resize((input_size, input_size), Image.BILINEAR)
            tensor = TF.to_tensor(resized).to(device)
            out = model([tensor])[0]
            sx, sy = W0 / input_size, H0 / input_size
            for box, score, label in zip(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": i,
                    "category_id": int(label) - 1,  # back to 0-indexed
                    "bbox": [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy],
                    "score": float(score),
                })

    if not predictions:
        return {"map_at_iou_0.5": 0.0, "map_at_iou_0.5_0.95": 0.0}

    with tempfile.NamedTemporaryFile("w", suffix="_pred.json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name
    coco_dt = coco_gt.loadRes(pred_path)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "map_at_iou_0.5_0.95": float(ev.stats[0]),
        "map_at_iou_0.5": float(ev.stats[1]),
        "map_at_iou_0.75": float(ev.stats[2]),
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "fastapi==0.115.0", "pydantic==2.9.2"
    ),
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", label="cbis-ddsm-detect-trigger")
def trigger(payload: Dict[str, Any]) -> Dict[str, Any]:
    epochs = int(payload.get("epochs", 20))
    lr = float(payload.get("learning_rate", 1e-4))
    dry_run = bool(payload.get("dry_run", False))
    call = train_detector.spawn(epochs=epochs, learning_rate=lr, dry_run=dry_run)
    return {"call_id": call.object_id, "params": {"epochs": epochs, "lr": lr, "dry_run": dry_run}}


@app.local_entrypoint()
def main(epochs: int = 20, learning_rate: float = 1e-4, dry_run: bool = False) -> None:
    result = train_detector.remote(
        epochs=epochs, learning_rate=learning_rate, dry_run=dry_run
    )
    print(json.dumps(result, indent=2))
