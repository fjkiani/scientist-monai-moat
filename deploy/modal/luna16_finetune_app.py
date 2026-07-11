"""Modal app: LUNA16 RetinaNet fine-tune (refine-v1).

Runs the MONAI ``lung_nodule_ct_detection`` bundle's training loop on a Modal
A10G / A100 GPU using the LUNA16 dataset stored in a Modal ``Volume``.

Design
------
- **Volume-backed inputs**: raw LUNA16 ``.mhd/.raw`` from the 10 subset zips,
  the ``LUNA16_datasplit`` fold JSONs, and the MONAI bundle live inside the
  ``luna16-data`` Modal Volume. The bundle's shipped ``model.pt`` (LUNA16
  fold-0 baseline) is used as the fine-tune starting point.
- **In-container resample**: on first run, subsets are unpacked and every
  fold-0 series is resampled to ``(0.703125, 0.703125, 1.25) mm`` NIfTI under
  ``/vol/nifti/``. The result is cached in the Volume so subsequent fine-tune
  runs skip this step.
- **Bundle-driven training**: we invoke ``monai.bundle run`` for the bundle's
  ``training`` and ``validate`` config keys. Nothing bundle-specific is
  re-implemented here — the app is a thin wrapper that (a) prepares data, (b)
  overrides ``epochs`` / ``learning_rate`` / ``dataset_dir`` at CLI level, (c)
  emits a ``refine_metrics.json`` for the audit ledger.
- **Cost**: ``min_containers=0``, GPU only during training. Volume storage is
  persistent so we don't re-upload 63 GB on every trigger.

Endpoints
---------
- ``GET  /healthz``   → liveness
- ``POST /finetune``  → JSON ``{fold, epochs, lr, dry_run}`` → JobHandle
- ``GET  /status/{handle}`` → progress + metrics

CLI
---
``python -m deploy.modal.luna16_finetune_app finetune --fold 0 --epochs 20``
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

APP_VERSION = "luna16-refine-v0.4.0-alpha"

# ── Volume ──────────────────────────────────────────────────────────
# NB: this Volume is populated OUT-OF-BAND (see scripts/upload_luna16_to_modal.py).
# The training container mounts it read-write so it can cache resampled NIfTI.
LUNA16_VOL = modal.Volume.from_name("luna16-data", create_if_missing=True)
BASELINE_VOL = modal.Volume.from_name(
    "luna16-baseline-weights", create_if_missing=True
)
OUTPUT_VOL = modal.Volume.from_name("luna16-training-runs", create_if_missing=True)

# ── Image ───────────────────────────────────────────────────────────
LUNA16_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "libgl1", "unzip", "git")
    .pip_install(
        # MONAI + torch stack matched to bundle metadata.json
        "torch==2.3.1",
        "torchvision==0.18.1",
        "monai==1.3.2",
        "SimpleITK==2.4.0",
        "nibabel==5.2.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-image==0.24.0",
        "pytorch-ignite==0.5.0.post2",
        "tensorboard==2.17.1",
        "fire==0.6.0",
        "pandas==2.2.2",
        "fastapi==0.115.0",
        "pydantic==2.9.2",
    )
)

app = modal.App("luna16-refine")


# ── Health probe ────────────────────────────────────────────────────
HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="luna16-refine-healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": "luna16-refine", "version": APP_VERSION}


# ── Resample helper (CPU, runs inside Modal container) ──────────────
TARGET_SPACING_MM = (0.703125, 0.703125, 1.25)


def _resample_one(mhd_path: Path, out_path: Path) -> None:
    """Resample a LUNA16 CT to target voxel spacing, save as NIfTI."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(mhd_path))
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / TARGET_SPACING_MM[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING_MM)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # air HU outside FOV
    resampled = resampler.Execute(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(out_path))


def _unpack_and_resample(
    zip_dir: Path,
    unpack_to: Path,
    nifti_out: Path,
    needed_uids: set,
) -> Dict[str, str]:
    """Unpack any subsets not yet on disk, then resample any UIDs not yet in nifti_out.

    Returns
    -------
    dict
        UID → NIfTI absolute path.
    """
    import zipfile

    unpack_to.mkdir(parents=True, exist_ok=True)
    nifti_out.mkdir(parents=True, exist_ok=True)

    # Unpack all subsets that don't have their series on disk yet
    uid_to_mhd: Dict[str, Path] = {}
    for i in range(10):
        zip_path = zip_dir / f"subset{i}.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".mhd"):
                    continue
                uid = Path(name).stem
                target = unpack_to / name
                if not target.exists():
                    z.extract(name, unpack_to)
                    raw_name = name.replace(".mhd", ".raw")
                    z.extract(raw_name, unpack_to)
                uid_to_mhd[uid] = target

    # Resample only what we need
    uid_to_nifti: Dict[str, str] = {}
    for uid in needed_uids:
        nifti_path = nifti_out / f"{uid}.nii.gz"
        if not nifti_path.exists():
            mhd = uid_to_mhd.get(uid)
            if mhd is None:
                raise RuntimeError(f"UID {uid} not found in any subset zip")
            _resample_one(mhd, nifti_path)
        uid_to_nifti[uid] = str(nifti_path)
    return uid_to_nifti


# ── Fine-tune function ──────────────────────────────────────────────
@app.function(
    image=LUNA16_IMAGE,
    gpu="A10G",
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
    timeout=6 * 60 * 60,  # 6h max wall
    memory=32 * 1024,
    cpu=8.0,
)
def finetune(
    fold: int = 0,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    val_interval: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run LUNA16 RetinaNet fine-tune on a given fold.

    Parameters
    ----------
    fold : int
        LUNA16 fold index (0-9).
    epochs : int
        Fine-tune epochs. Bundle default is 300 (from-scratch); we fine-tune
        from shipped weights so 20 is the default.
    learning_rate : float
        Adam LR. Bundle default is 1e-2; we drop by 10x for fine-tune.
    batch_size : int
        Bundle default 4 patches per step.
    val_interval : int
        Validate every N epochs.
    dry_run : bool
        If True, do not actually train — return the resolved config and paths.
    """
    import json
    import shutil
    import subprocess

    run_id = f"fold{fold}-e{epochs}-lr{learning_rate}-{int(time.time())}"
    run_dir = Path("/vol/output") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Discover paths
    bundle_dir = Path("/vol/baseline/lung_nodule_ct_detection")
    if not (bundle_dir / "models" / "model.pt").exists():
        raise RuntimeError(
            f"Baseline bundle missing at {bundle_dir}. Upload via "
            f"scripts/upload_luna16_to_modal.py first."
        )
    fold_json = Path(f"/vol/luna16/LUNA16_datasplit/dataset_fold{fold}.json")
    if not fold_json.exists():
        raise RuntimeError(f"Datasplit missing: {fold_json}")

    # Which UIDs do we need?
    with open(fold_json) as f:
        fold_data = json.load(f)

    def _uids_from(items):
        out = set()
        for it in items:
            img = it["image"] if isinstance(it, dict) else it
            stem = Path(img).stem
            if stem.endswith(".nii"):
                stem = stem[:-4]
            out.add(stem)
        return out

    needed = _uids_from(fold_data["training"]) | _uids_from(fold_data["validation"])

    print(f"[{run_id}] Preparing {len(needed)} series (unpack + resample)...", flush=True)
    uid_to_nifti = _unpack_and_resample(
        zip_dir=Path("/vol/luna16"),
        unpack_to=Path("/vol/luna16/unpacked"),
        nifti_out=Path("/vol/luna16/nifti"),
        needed_uids=needed,
    )
    print(f"[{run_id}] Resampled {len(uid_to_nifti)} series.", flush=True)

    dataset_dir = "/vol/luna16/nifti"

    resolved = {
        "run_id": run_id,
        "fold": fold,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "val_interval": val_interval,
        "n_train_series": len(fold_data["training"]),
        "n_val_series": len(fold_data["validation"]),
        "target_spacing_mm": list(TARGET_SPACING_MM),
        "dataset_dir": dataset_dir,
        "fold_json": str(fold_json),
        "bundle_dir": str(bundle_dir),
        "run_dir": str(run_dir),
    }
    print(json.dumps({"resolved": resolved}, indent=2), flush=True)

    if dry_run:
        (run_dir / "resolved.json").write_text(json.dumps(resolved, indent=2))
        return {"status": "dry_run", **resolved}

    # Prepare output ckpt dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Seed with baseline weights so bundle finds them
    shutil.copy(bundle_dir / "models" / "model.pt", ckpt_dir / "model.pt")

    # Invoke bundle training loop
    cmd = [
        "python", "-m", "monai.bundle", "run", "training",
        "--config_file", str(bundle_dir / "configs" / "train.json"),
        "--bundle_root", str(bundle_dir),
        "--dataset_dir", dataset_dir,
        "--data_list_file_path", str(fold_json),
        "--ckpt_dir", str(ckpt_dir),
        "--output_dir", str(run_dir),
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--val_interval", str(val_interval),
    ]
    print(f"[{run_id}] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(bundle_dir), check=False)
    if proc.returncode != 0:
        return {"status": "failed", "returncode": proc.returncode, **resolved}

    # Run validation with baseline weights → baseline FROC
    val_baseline = _run_validate(bundle_dir, dataset_dir, fold_json, bundle_dir / "models" / "model.pt")
    # Run validation with refined weights
    val_refined = _run_validate(bundle_dir, dataset_dir, fold_json, ckpt_dir / "model.pt")

    delta = {
        "froc_at_2fps": val_refined.get("froc_at_2fps", 0.0) - val_baseline.get("froc_at_2fps", 0.0),
        "map_iou0.1": val_refined.get("map_iou0.1", 0.0) - val_baseline.get("map_iou0.1", 0.0),
    }
    metrics = {
        "schema_version": "v0.4.0-alpha",
        "run_id": run_id,
        "fold_index": fold,
        "n_train_series": len(fold_data["training"]),
        "n_val_series": len(fold_data["validation"]),
        "target_spacing_mm": list(TARGET_SPACING_MM),
        "baseline": val_baseline,
        "refined": val_refined,
        "delta": delta,
        "plan_target": {
            "froc_at_2fps_delta_min": 0.05,
            "met": delta["froc_at_2fps"] >= 0.05,
        },
    }
    metrics_path = run_dir / "refine_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    OUTPUT_VOL.commit()
    return {"status": "ok", "metrics": metrics, "run_dir": str(run_dir)}


def _run_validate(
    bundle_dir: Path,
    dataset_dir: str,
    fold_json: Path,
    ckpt: Path,
) -> Dict[str, float]:
    """Run bundle's validate config, parse FROC + mAP from stdout."""
    import re
    import subprocess

    cmd = [
        "python", "-m", "monai.bundle", "run", "validate",
        "--config_file", str(bundle_dir / "configs" / "evaluate.json"),
        "--bundle_root", str(bundle_dir),
        "--dataset_dir", dataset_dir,
        "--data_list_file_path", str(fold_json),
        "--ckpt_path", str(ckpt),
    ]
    print(f"[validate] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(bundle_dir), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out, flush=True)

    # Bundle prints metric lines like "FROC score at 2.0 FP/scan: 0.867"
    metrics: Dict[str, float] = {}
    m = re.search(r"FROC.*2\.?0? ?FP.*?:\s*([0-9.]+)", out, re.IGNORECASE)
    if m:
        metrics["froc_at_2fps"] = float(m.group(1))
    m = re.search(r"mAP.*IoU\s*0?\.?1.*?:\s*([0-9.]+)", out, re.IGNORECASE)
    if m:
        metrics["map_iou0.1"] = float(m.group(1))
    return metrics


# ── FastAPI trigger ─────────────────────────────────────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "fastapi==0.115.0", "pydantic==2.9.2"
    ),
    timeout=30,
)
@modal.fastapi_endpoint(method="POST", label="luna16-refine-trigger")
def trigger(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Kick off a fine-tune run asynchronously; return function call id."""
    fold = int(payload.get("fold", 0))
    epochs = int(payload.get("epochs", 20))
    lr = float(payload.get("learning_rate", 1e-3))
    dry_run = bool(payload.get("dry_run", False))
    call = finetune.spawn(fold=fold, epochs=epochs, learning_rate=lr, dry_run=dry_run)
    return {"call_id": call.object_id, "params": {"fold": fold, "epochs": epochs, "lr": lr, "dry_run": dry_run}}


# ── Local entry point ──────────────────────────────────────────────
@app.local_entrypoint()
def main(
    fold: int = 0,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    dry_run: bool = False,
) -> None:
    """Local trigger: ``modal run deploy/modal/luna16_finetune_app.py --dry-run``."""
    result = finetune.remote(
        fold=fold, epochs=epochs, learning_rate=learning_rate, dry_run=dry_run
    )
    print(json.dumps(result, indent=2))
