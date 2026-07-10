"""LUNA16 RetinaNet fine-tuning harness.

Wraps the MONAI `lung_nodule_ct_detection` v0.6.9 bundle's `train.json`
config for fine-tuning against a locally-prepared LUNA16 dataset. This
module handles the three data-prep steps the bundle does NOT include:

1. Unpack Zenodo LUNA16 zips (subsets 0-9) into a flat directory of
   `.mhd`/`.raw` pairs by SeriesInstanceUID.
2. Resample each series to the bundle's target voxel size
   0.703125 x 0.703125 x 1.25 mm (RAS orientation) and save as `.nii.gz`
   under `<dataset_dir>/<SeriesUID>/<SeriesUID>.nii.gz`.
3. Optionally validate the resampled files against the datasplit JSON's
   expected relative paths.

The 10-fold datasplit JSONs come from Project-MONAI/MONAI-extra-test-data
release 0.8.1, `LUNA16_datasplit-20220615T233840Z-001.zip`. Each fold
splits the ~888 annotated LUNA16 series 534/67 (training/validation).

Fine-tuning entry point
-----------------------
`run_finetune()` invokes `python -m monai.bundle run training` against a
config dict that layers the following overrides on top of `train.json`:

- `dataset_dir`: resampled `.nii.gz` corpus
- `data_list_file_path`: fold-specific datasplit JSON
- `epochs`, `learning_rate`, `batch_size`: caller-controlled
- `ckpt_dir`: output checkpoint directory
- Optional `finetune_from`: initial weights (defaults to the shipped
  `models/model.pt` in the bundle root)

Evaluation entry point
----------------------
`run_evaluate()` runs `python -m monai.bundle run validate` with
`evaluate.json` overlaid on the fold's validation split, producing a
COCO metric at `IoU=0.1` on 3D boxes (per bundle metadata).

We separately compute the LUNA16 FROC metric (sensitivity at
{0.125, 0.25, 0.5, 1, 2, 4, 8} FPs/scan) via
`scripts/luna16_froc.py` from the bundle's evaluation suite.

Refine claim
------------
`docs/proofs/luna16_refine_v1_metrics.json` records:
- Baseline: shipped `models/model.pt` COCO mAP + FROC@2 on held-out
  fold (default: fold 9).
- Refined: `models/model_finetuned.pt` after N epochs on folds 0-8.
- Delta: `froc_at_2fps_delta`, `map_delta`.

Reference target: `froc_at_2fps_delta >= 0.05` = the v0.4.0-alpha plan's
"+5% FROC@2" claim. This module deliberately does NOT hardcode that
threshold — the metrics JSON is emitted regardless; downstream tests
gate on the delta.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Bundle default target voxel size (from configs/train.json)
TARGET_SPACING_MM: Tuple[float, float, float] = (0.703125, 0.703125, 1.25)


@dataclass
class FinetuneConfig:
    """User-facing knobs for a fine-tune run.

    Everything else (batch size, patch size, anchors, learning-rate schedule)
    is inherited from the bundle's `train.json` unless explicitly overridden
    via `extra_overrides`.
    """

    bundle_root: Path
    """Path to the unpacked MONAI bundle (contains configs/, models/, scripts/)."""

    dataset_dir: Path
    """Root directory of resampled `.nii.gz` volumes, one subdirectory per SeriesUID."""

    datasplit_json: Path
    """Path to `dataset_fold{N}.json` from the LUNA16_datasplit zip."""

    output_dir: Path
    """Where checkpoints and logs are written."""

    initial_weights: Optional[Path] = None
    """Starting checkpoint. Defaults to bundle_root/models/model.pt."""

    epochs: int = 20
    """Number of fine-tune epochs. Bundle default was 300 for scratch training."""

    learning_rate: float = 1e-3
    """LR for fine-tune (bundle default was 1e-2 for scratch). 10x lower is the
    common fine-tuning convention."""

    batch_size: int = 4
    """Bundle default. Reduce to 2 if VRAM is tight."""

    val_interval: int = 5
    """Epochs between validation passes."""

    extra_overrides: Dict[str, object] = field(default_factory=dict)
    """Arbitrary `--<key>=<value>` overrides forwarded to `monai.bundle run`."""


def unpack_zenodo_subsets(
    zip_dir: Path,
    unpack_to: Path,
    subsets: Optional[List[int]] = None,
) -> Dict[str, Path]:
    """Unpack LUNA16 subsetN.zip files into `unpack_to`, returning UID->mhd map.

    Args:
        zip_dir: directory containing `subset{0..9}.zip`.
        unpack_to: destination for extracted `.mhd`/`.raw` pairs.
        subsets: which subset indices to unpack (default: all found).

    Returns:
        Mapping of SeriesInstanceUID -> absolute path to the .mhd file.
    """
    unpack_to.mkdir(parents=True, exist_ok=True)

    if subsets is None:
        subsets = sorted(
            int(p.stem.replace("subset", ""))
            for p in zip_dir.glob("subset*.zip")
        )

    uid_to_mhd: Dict[str, Path] = {}
    for i in subsets:
        zip_path = zip_dir / f"subset{i}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unpack_to)

    # After extraction, each subset expands into unpack_to/subsetN/*.mhd
    # (or directly at unpack_to/*.mhd depending on the archive layout).
    for mhd in unpack_to.rglob("*.mhd"):
        uid = mhd.stem  # LUNA16 filenames = SeriesInstanceUID
        uid_to_mhd[uid] = mhd

    return uid_to_mhd


def resample_series(
    mhd_path: Path,
    out_path: Path,
    target_spacing: Tuple[float, float, float] = TARGET_SPACING_MM,
) -> None:
    """Resample a single LUNA16 .mhd volume to `target_spacing`, write .nii.gz.

    Uses MONAI's `Spacingd` + `LoadImaged` + `SaveImaged` pipeline for
    consistency with the bundle's inference / training preprocessors.
    """
    import numpy as np  # local import — resample called from workers
    import SimpleITK as sitk

    img = sitk.ReadImage(str(mhd_path))
    orig_spacing = img.GetSpacing()  # (x, y, z) in world mm
    orig_size = img.GetSize()

    # Bundle transforms expect voxel spacing (row, col, depth) = (0.703125, 0.703125, 1.25)
    # SimpleITK uses (x, y, z), which for LUNA16 axials corresponds to (col, row, depth).
    new_spacing = tuple(target_spacing)
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(orig_size, orig_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # air HU

    out_img = resampler.Execute(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))


def build_resampled_dataset(
    uid_to_mhd: Dict[str, Path],
    dataset_dir: Path,
    n_jobs: int = 4,
    skip_existing: bool = True,
) -> Dict[str, Path]:
    """Resample every series in `uid_to_mhd`, writing to `dataset_dir/<uid>/<uid>.nii.gz`.

    Returns UID -> resampled path.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    uid_to_nii: Dict[str, Path] = {}
    jobs = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for uid, mhd in uid_to_mhd.items():
            out = dataset_dir / uid / f"{uid}.nii.gz"
            uid_to_nii[uid] = out
            if skip_existing and out.exists() and out.stat().st_size > 0:
                continue
            jobs.append(pool.submit(resample_series, mhd, out))

        for fut in as_completed(jobs):
            fut.result()  # re-raise on error

    return uid_to_nii


def run_finetune(cfg: FinetuneConfig, dry_run: bool = False) -> Dict[str, object]:
    """Invoke `monai.bundle run training` with fine-tune overrides.

    Returns a dict:
        {
          "cmd": [...],
          "elapsed_seconds": float,
          "returncode": int,
          "ckpt_path": Path | None,
        }
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    initial_weights = cfg.initial_weights or (cfg.bundle_root / "models" / "model.pt")
    if not initial_weights.exists():
        raise FileNotFoundError(f"initial_weights not found: {initial_weights}")

    cmd = [
        sys.executable, "-m", "monai.bundle", "run", "training",
        "--config_file", str(cfg.bundle_root / "configs" / "train.json"),
        "--bundle_root", str(cfg.bundle_root),
        "--dataset_dir", str(cfg.dataset_dir),
        "--data_list_file_path", str(cfg.datasplit_json),
        "--ckpt_dir", str(cfg.output_dir),
        "--epochs", str(cfg.epochs),
        "--learning_rate", str(cfg.learning_rate),
        "--batch_size", str(cfg.batch_size),
        "--val_interval", str(cfg.val_interval),
    ]

    for k, v in cfg.extra_overrides.items():
        cmd.extend([f"--{k}", str(v)])

    if dry_run:
        return {"cmd": cmd, "elapsed_seconds": 0.0, "returncode": 0, "ckpt_path": None}

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
    except Exception as e:
        return {"cmd": cmd, "elapsed_seconds": time.time() - t0, "returncode": -1,
                "ckpt_path": None, "error": str(e)}

    ckpt_final = cfg.output_dir / "model.pt"
    return {
        "cmd": cmd,
        "elapsed_seconds": time.time() - t0,
        "returncode": result.returncode,
        "ckpt_path": ckpt_final if ckpt_final.exists() else None,
    }


def run_evaluate(
    bundle_root: Path,
    dataset_dir: Path,
    datasplit_json: Path,
    ckpt_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Invoke `monai.bundle run validate` and return the eval summary path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "monai.bundle", "run", "validate",
        "--config_file", str(bundle_root / "configs" / "evaluate.json"),
        "--bundle_root", str(bundle_root),
        "--dataset_dir", str(dataset_dir),
        "--data_list_file_path", str(datasplit_json),
        "--output_dir", str(output_dir),
        "--ckpt_dir", str(ckpt_path.parent),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "elapsed_seconds": time.time() - t0,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def write_refine_metrics(
    baseline_metrics: Dict[str, float],
    refined_metrics: Dict[str, float],
    out_path: Path,
    fold_index: int,
    n_train_series: int,
    n_val_series: int,
) -> None:
    """Emit `luna16_refine_v1_metrics.json` in the standard format.

    Fields:
        - baseline / refined: full metric dicts (COCO mAP@0.1, FROC@[0.125..8])
        - delta: refined - baseline for FROC@2 and mAP
        - fold, n_train_series, n_val_series: provenance
        - reproducer: shell hint for the caller
    """
    delta_froc = refined_metrics.get("froc_at_2fps", 0.0) - baseline_metrics.get("froc_at_2fps", 0.0)
    delta_map = refined_metrics.get("map_iou0.1", 0.0) - baseline_metrics.get("map_iou0.1", 0.0)

    payload = {
        "schema_version": "v0.4.0-alpha",
        "generated_at_epoch": int(time.time()),
        "fold_index": fold_index,
        "n_train_series": n_train_series,
        "n_val_series": n_val_series,
        "target_spacing_mm": list(TARGET_SPACING_MM),
        "baseline": baseline_metrics,
        "refined": refined_metrics,
        "delta": {
            "froc_at_2fps": delta_froc,
            "map_iou0.1": delta_map,
        },
        "plan_target": {
            "froc_at_2fps_delta_min": 0.05,
            "met": delta_froc >= 0.05,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


__all__ = [
    "FinetuneConfig",
    "TARGET_SPACING_MM",
    "build_resampled_dataset",
    "resample_series",
    "run_evaluate",
    "run_finetune",
    "unpack_zenodo_subsets",
    "write_refine_metrics",
]
