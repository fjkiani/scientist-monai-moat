#!/usr/bin/env python3
"""Official LUNA16 FROC@2 on Gate D held-out (88) — baseline vs refined.

RUO. Clinical meter = interpolated sensitivity at 2 FP/scan from vendor
``noduleCADEvaluationLUNA16.py``. COCO AP/AR is NOT clinical.

Pipeline:
  1. Load Gate D ckpt + baseline model.pt
  2. Infer on 88 nifti_v3 test series → results.csv (world mm)
  3. vendor noduleCADEvaluationLUNA16.py → FROC curve → FROC@2

Usage::
  MODAL_PROFILE=fjkiani modal run deploy/modal/luna16_heldout_froc_app.py \\
    --ckpt-request-id gateD-fjkiani-20261003T0435Z-zo \\
    --request-id froc-gateE-$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# Never resolve repo via Path(__file__).parents[N] — Modal mounts this file at
# /root/<name>.py (only parents[0]=/root, parents[1]=/), which raises IndexError(2).
# Bake annotations + vendor eval into the image under /opt/; use with_name() for
# client-side add_local only (skipped on the worker when siblings are absent).
APP_VERSION = "luna16-heldout-froc-v0.2.2-score-thresh"
EXPECTED_BASELINE = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
EXPECTED_DATALIST = "c5f6adf3fe79b787e8b038862220c8aed16964f36cd8b642dbaa4ccfc10a73c3"
NIFTI_PREFIX = "/vol/luna16/nifti_v3"
OFFICIAL_FPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TOP_N = 100

LUNA16_VOL = modal.Volume.from_name("luna16-data")
BASELINE_VOL = modal.Volume.from_name("luna16-baseline-weights")
OUTPUT_VOL = modal.Volume.from_name("luna16-training-runs", create_if_missing=True)

# Assets live beside this file on the CLIENT. Worker import sees /root/<py>
# without the sibling dir — skip add_local there (image already baked).
_BUNDLE = Path(__file__).with_name("_luna16_froc_bundle")

_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "libgl1")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "monai==1.4.0",
        "nibabel==5.2.1",
        "numpy==1.24.4",
        "pytorch-ignite==0.4.11",
        "scipy==1.13.1",
        "scikit-image==0.24.0",
        "pandas==2.2.2",
        "matplotlib==3.9.2",
        "scikit-learn==1.5.1",
        "SimpleITK==2.4.0",
        "fire==0.6.0",
    )
)
_datalist = Path(__file__).with_name("luna16_full_482_60_datalist.json")
if _datalist.is_file():
    _IMAGE = _IMAGE.add_local_file(
        str(_datalist), remote_path="/opt/luna16_full_482_60_datalist.json"
    )
if (_BUNDLE / "py3_port").is_dir():
    _IMAGE = _IMAGE.add_local_dir(str(_BUNDLE / "py3_port"), remote_path="/opt/luna16_eval")
if (_BUNDLE / "ann" / "annotations.csv").is_file():
    _IMAGE = _IMAGE.add_local_file(
        str(_BUNDLE / "ann" / "annotations.csv"),
        remote_path="/opt/luna16_ann/annotations.csv",
    )
if (_BUNDLE / "ann" / "annotations_excluded.csv").is_file():
    _IMAGE = _IMAGE.add_local_file(
        str(_BUNDLE / "ann" / "annotations_excluded.csv"),
        remote_path="/opt/luna16_ann/annotations_excluded.csv",
    )
IMAGE = _IMAGE

app = modal.App("luna16-heldout-froc")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_detector(ckpt_path: Path, device: str, score_thresh: float = 0.02):
    import torch
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import (
        RetinaNet,
        resnet_fpn_feature_extractor,
    )
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets import resnet

    backbone = resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )
    fe = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    net = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=fe,
        size_divisible=[16, 16, 8],
        use_list_output=False,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "feature_extractor.body.conv1.weight" in ckpt:
        net.load_state_dict(ckpt)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt)

    ag = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    det = RetinaNetDetector(
        network=net,
        anchor_generator=ag,
        spatial_dims=3,
        num_classes=1,
        size_divisible=[16, 16, 8],
    )
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=300,
    )
    sw_device = device if device.startswith("cuda") else "cpu"
    det.set_sliding_window_inferer(
        roi_size=[192, 192, 80],
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device=sw_device,
    )
    det.to(device).eval()
    return det


def _infer_series(det, nifti_path: Path, series_uid: str, device: str) -> List[Dict[str, Any]]:
    import numpy as np
    import SimpleITK as sitk
    import torch

    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)=(z,y,x)
    v = np.clip(arr, -1024.0, 300.0)
    v = (v + 1024.0) / (1024.0 + 300.0)
    x = [torch.from_numpy(v[None, ...]).to(device)]
    with torch.no_grad():
        out = det(x)
    raw = out[0] if out else {}
    boxes = raw.get("box")
    scores = raw.get("label_scores")
    if boxes is None or scores is None:
        return []
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    rows: List[Dict[str, Any]] = []
    for i, box in enumerate(boxes_np):
        z0, y0, x0, z1, y1, x1 = [float(v) for v in box]
        cz = (z0 + z1) / 2.0
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        # ITK continuous index is (x, y, z)
        wx, wy, wz = img.TransformContinuousIndexToPhysicalPoint((cx, cy, cz))
        rows.append(
            {
                "seriesuid": series_uid,
                "coordX": float(wx),
                "coordY": float(wy),
                "coordZ": float(wz),
                "probability": float(scores_np[i]),
            }
        )
    rows.sort(key=lambda r: r["probability"], reverse=True)
    return rows[:TOP_N]


def _write_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        for r in rows:
            w.writerow([r["seriesuid"], r["coordX"], r["coordY"], r["coordZ"], r["probability"]])


def _parse_froc_at(froc_txt: Path, fp_target: float = 2.0) -> Optional[float]:
    import numpy as np

    if not froc_txt.is_file():
        return None
    fps: List[float] = []
    sens: List[float] = []
    for line in froc_txt.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            fps.append(float(parts[0]))
            sens.append(float(parts[1]))
        except ValueError:
            continue
    if len(fps) < 2:
        return None
    # Official curve may be unsorted / have plateaus — interp needs nondecreasing x
    order = np.argsort(fps)
    fps_a = np.asarray(fps, dtype=float)[order]
    sens_a = np.asarray(sens, dtype=float)[order]
    # dedupe fps keeping last sens
    uniq_fps: List[float] = []
    uniq_sens: List[float] = []
    for f, s in zip(fps_a, sens_a):
        if uniq_fps and abs(f - uniq_fps[-1]) < 1e-12:
            uniq_sens[-1] = s
        else:
            uniq_fps.append(float(f))
            uniq_sens.append(float(s))
    return float(np.interp(fp_target, uniq_fps, uniq_sens))


def _parse_official_points(froc_txt: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fp in OFFICIAL_FPS:
        v = _parse_froc_at(froc_txt, fp)
        if v is not None:
            out[str(fp)] = v
    return out


def _run_cad(results_csv: Path, series_file: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = Path("/opt/luna16_ann/annotations.csv")
    ann_excl = Path("/opt/luna16_ann/annotations_excluded.csv")
    cmd = [
        sys.executable,
        "/opt/luna16_eval/noduleCADEvaluationLUNA16.py",
        str(ann),
        str(ann_excl),
        str(series_file),
        str(results_csv),
        str(out_dir),
    ]
    log = out_dir / "cad.log"
    with log.open("w") as lf:
        proc = subprocess.run(cmd, cwd="/opt/luna16_eval", stdout=lf, stderr=subprocess.STDOUT, text=True)
    # CAD writes froc_<stem>.txt where stem = results filename without ext
    stem = results_csv.stem
    froc_txt = out_dir / f"froc_{stem}.txt"
    if not froc_txt.is_file():
        # fallback: any froc_*.txt
        cands = list(out_dir.glob("froc_*.txt"))
        froc_txt = cands[0] if cands else froc_txt
    points = _parse_official_points(froc_txt) if froc_txt.is_file() else {}
    return {
        "returncode": proc.returncode,
        "froc_txt": str(froc_txt) if froc_txt.is_file() else None,
        "froc_at_2": points.get("2.0"),
        "froc_points": points,
        "log_tail": log.read_text(errors="replace")[-3000:] if log.is_file() else "",
    }


def _arm(
    label: str,
    ckpt: Path,
    series_uids: List[str],
    out_root: Path,
    device: str,
    score_thresh: float = 0.02,
) -> Dict[str, Any]:
    arm_dir = out_root / label
    arm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    det = _build_detector(ckpt, device, score_thresh=score_thresh)
    all_rows: List[Dict[str, Any]] = []
    per_series: Dict[str, Any] = {}
    for i, uid in enumerate(series_uids):
        nii = Path(NIFTI_PREFIX) / uid / f"{uid}.nii.gz"
        if not nii.is_file():
            per_series[uid] = {"error": f"missing {nii}"}
            continue
        try:
            rows = _infer_series(det, nii, uid, device)
            all_rows.extend(rows)
            per_series[uid] = {"n_detections": len(rows)}
        except Exception as e:  # noqa: BLE001 — record and continue
            per_series[uid] = {"error": f"{type(e).__name__}: {e}"}
        if (i + 1) % 10 == 0:
            print(f"[{label}] {i+1}/{len(series_uids)} series", flush=True)
    results_csv = arm_dir / "results.csv"
    _write_results_csv(all_rows, results_csv)
    series_file = arm_dir / "seriesuids_heldout.csv"
    # one uid per line, no header — vendor collect() accepts with/without; use headerless for safety
    # Official seriesuids.csv has no header in some dumps; our prior scaffold used header.
    # py3_port collect reads lines as UIDs — check
    series_file.write_text("\n".join(series_uids) + "\n")
    (arm_dir / "per_series.json").write_text(json.dumps(per_series, indent=2))
    cad = _run_cad(results_csv, series_file, arm_dir / "cad")
    n_ok = sum(1 for v in per_series.values() if "error" not in v)
    n_err = sum(1 for v in per_series.values() if "error" in v)
    return {
        "label": label,
        "ckpt_sha256": _sha256_file(ckpt),
        "n_series_ok": n_ok,
        "n_series_err": n_err,
        "n_candidates": len(all_rows),
        "results_csv": str(results_csv),
        "elapsed_seconds": round(time.time() - t0, 3),
        **cad,
    }


@app.function(
    image=IMAGE,
    gpu="A100",
    cpu=8.0,
    memory=65536,
    timeout=21600,
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
)
def run_paired_froc(
    request_id: str = "",
    ckpt_request_id: str = "",
    baseline_score_thresh: float = 0.02,
    refined_score_thresh: float = 0.02,
) -> Dict[str, Any]:
    """Infer 88 held-out with baseline + refined; score official FROC@2."""
    import torch

    rid = request_id or f"froc-{time.strftime('%Y%m%dT%H%M%SZ')}"
    if not ckpt_request_id:
        return {"status": "failed", "error": "ckpt_request_id required", "verdict": "NOT AUTHORIZED"}

    out_root = Path("/vol/output") / "fullfit-v0.1" / rid / "froc"
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = Path("/opt/luna16_full_482_60_datalist.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_DATALIST:
        return {"status": "failed", "error": "datalist sha mismatch", "verdict": "FAILED"}
    full = json.loads(raw)
    series_uids = [r["series_uid"] for r in full["test"]]

    refined_ckpt = (
        Path("/vol/output")
        / "fullfit-v0.1"
        / ckpt_request_id
        / "train"
        / "checkpoints"
        / "model.pt"
    )
    baseline_ckpt = Path("/vol/baseline/lung_nodule_ct_detection/models/model.pt")
    if not refined_ckpt.is_file():
        return {"status": "failed", "error": f"missing refined {refined_ckpt}", "verdict": "FAILED"}
    if not baseline_ckpt.is_file() or _sha256_file(baseline_ckpt) != EXPECTED_BASELINE:
        return {"status": "failed", "error": "baseline missing/hash mismatch", "verdict": "FAILED"}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} n_test={len(series_uids)} "
        f"baseline_thresh={baseline_score_thresh} refined_thresh={refined_score_thresh}",
        flush=True,
    )

    baseline = _arm(
        "baseline",
        baseline_ckpt,
        series_uids,
        out_root,
        device,
        score_thresh=baseline_score_thresh,
    )
    refined = _arm(
        "refined",
        refined_ckpt,
        series_uids,
        out_root,
        device,
        score_thresh=refined_score_thresh,
    )

    b2 = baseline.get("froc_at_2")
    r2 = refined.get("froc_at_2")
    delta = None
    if b2 is not None and r2 is not None:
        delta = {
            "froc_at_2": r2 - b2,
            "floor_ref_0.05": 0.05,
            "meets_floor_ref": (r2 - b2) >= 0.05,
        }

    proven = (
        baseline.get("returncode") == 0
        and refined.get("returncode") == 0
        and b2 is not None
        and r2 is not None
    )
    receipt: Dict[str, Any] = {
        "app_version": APP_VERSION,
        "request_id": rid,
        "ckpt_request_id": ckpt_request_id,
        "n_test": len(series_uids),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "baseline_score_thresh": baseline_score_thresh,
        "refined_score_thresh": refined_score_thresh,
        "baseline": baseline,
        "refined": refined,
        "froc_at_2_baseline": b2,
        "froc_at_2_refined": r2,
        "delta": delta,
        "status": "ok" if proven else "failed",
        "verdict": "PROVEN" if proven else "UNPROVEN",
        "verdict_note": (
            "Paired official CAD FROC@2 parsed. +0.05 floor is reference only — "
            "do not auto-promote; do not overwrite luna16-infer."
            if proven
            else "Missing FROC@2 parse and/or CAD failure — clinical UNPROVEN."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (out_root / "result.json").write_text(json.dumps(receipt, indent=2))
    OUTPUT_VOL.commit()
    return receipt


@app.local_entrypoint()
def main(
    ckpt_request_id: str = "gateD-fjkiani-20261003T0435Z-zo",
    request_id: str = "",
    baseline_score_thresh: float = 0.02,
    refined_score_thresh: float = 0.02,
):
    rid = request_id or f"froc-fjkiani-{time.strftime('%Y%m%dT%H%M%SZ')}-zo"
    r = run_paired_froc.remote(
        request_id=rid,
        ckpt_request_id=ckpt_request_id,
        baseline_score_thresh=baseline_score_thresh,
        refined_score_thresh=refined_score_thresh,
    )
    local = Path("/tmp/luna-xfer") / "froc_gateE"
    local.mkdir(parents=True, exist_ok=True)
    (local / "froc_attempt.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
