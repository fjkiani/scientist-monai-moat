"""Modal app: LUNA16 RetinaNet **inference** endpoint.

Distinct from ``luna16_finetune_app.py`` which drives training. This app
exposes the shipped MONAI ``lung_nodule_ct_detection@0.6.9`` bundle as
FastAPI endpoints for the oncology-arbiter gateway to call from Render.

Endpoints
---------
- ``GET  /luna16-healthz``  → liveness + weight status
- ``GET  /luna16-info``     → bundle version, target spacing, HU range
- ``POST /luna16-detect``   → JSON body ``{ "volume_hu_b64": "...",
                                            "shape": [D, H, W],
                                            "spacing_mm": [dz, dy, dx],
                                            "top_n": 5 }``
                              returns detected boxes + provenance.

Design
------
- **Volume-backed weights** in ``luna16-baseline-weights``: the bundle
  is unpacked once (by ``scripts/upload_luna16_bundle.py``) and stays
  there. Container mounts it read-only.
- **CPU**-compatible but we ship on **A10G** so a 200-slice CT still
  runs in ~30-90 seconds.
- **min_containers=0** for cost; ~90s cold start (weight load).
- ``model_state="loaded_luna16_retinanet"`` on every response.

Deploy
------
    modal deploy deploy/modal/luna16_infer_app.py

RESEARCH USE ONLY — trained on LIDC-IDRI/LUNA16 subset only.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

APP_VERSION = "luna16-infer-v0.5.2-alpha-resample"
BUNDLE_VERSION = "0.6.9"
# MONAI lung_nodule_ct_detection@0.6.9 training geometry (dz, dy, dx) mm.
LUNA16_TARGET_SPACING_MM = (1.25, 0.703125, 0.703125)
DZ_PASSTHROUGH_TOL_MM = 0.05
AIR_HU = -1024.0

DISCLAIMER = (
    "detector=monai/lung_nodule_ct_detection@0.6.9; trained on LUNA16 fold 0 "
    "(LIDC-IDRI subset, single-site chest CT). Not validated on screening CT, "
    "contrast CT, or pediatric CT. Nodule size range 3-30 mm. Detections are "
    "RESEARCH USE ONLY and require radiologist review."
)

app = modal.App("luna16-infer")

BASELINE_VOL = modal.Volume.from_name(
    "luna16-baseline-weights", create_if_missing=True
)
# v0.5.0: read-side mount of the case-storage bytes so the detector can
# process a case by its ``case_id`` rather than an inline ``volume_hu_b64``
# blob (which was breaking Modal body-size limits for 200-slice CTs).
CASES_VOL = modal.Volume.from_name(
    "oncology-arbiter-cases", create_if_missing=True
)

IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "libgl1", "unzip", "git", "wget")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "monai==1.3.2",
        "SimpleITK==2.4.0",
        "nibabel==5.2.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-image==0.24.0",
        "pytorch-ignite==0.5.0.post2",
        "fastapi==0.115.0",
        "pydantic==2.9.2",
        "requests==2.32.3",
        # v0.5.0: DICOM series reader for case_id-backed CT input.
        "pydicom==2.4.4",
    )
)


# ---------------------------------------------------------------------------
# Weights loader: idempotently fetches the bundle into the Volume on cold start.
# ---------------------------------------------------------------------------

def _ensure_bundle_in_volume(mount_dir: str = "/vol/baseline") -> Path:
    """Require the MONAI bundle already present on ``luna16-baseline-weights``.

    GitHub ``hosting_storage_v1`` no longer ships ``v0.6.9`` (404). Seed the
    volume from an existing workspace (crispro/fjkiani) via
    ``modal volume get`` / ``modal volume put`` — do not auto-download.

    Returns the local Path to the extracted bundle directory.
    """
    bundle_root = Path(mount_dir) / "lung_nodule_ct_detection"
    weight_path = bundle_root / "models" / "model.pt"
    if weight_path.exists():
        return bundle_root

    raise RuntimeError(
        f"LUNA16 baseline bundle missing at {weight_path}. "
        "Seed volume luna16-baseline-weights with lung_nodule_ct_detection/ "
        "(copy from crispro). Auto-download of v0.6.9 is unavailable (GitHub 404)."
    )


# ---------------------------------------------------------------------------
# Detector singleton, cached inside the container process.
# ---------------------------------------------------------------------------

_DETECTOR = None
_BUNDLE_DIR: Optional[Path] = None


def _load_detector():
    global _DETECTOR, _BUNDLE_DIR
    if _DETECTOR is not None:
        return _DETECTOR
    import torch
    from monai.networks.nets import resnet
    from monai.apps.detection.networks.retinanet_network import (
        RetinaNet, resnet_fpn_feature_extractor,
    )
    from monai.apps.detection.utils.anchor_utils import (
        AnchorGeneratorWithAnchorShape,
    )
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector

    _BUNDLE_DIR = _ensure_bundle_in_volume()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resnet.resnet50(
        spatial_dims=3, n_input_channels=1,
        conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7],
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
    ckpt_path = _BUNDLE_DIR / "models" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(
        ckpt if not isinstance(ckpt, dict)
        or "feature_extractor.body.conv1.weight" in ckpt
        else ckpt["model"]
    )

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
        score_thresh=0.02,
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=300,
    )
    # Full-volume CUDA tensor OOM'd A10G; ROI tiles run on GPU via sw_device.
    # MONAI: sw_device = predictor device; device = stitch buffer (typically CPU).
    det.set_sliding_window_inferer(
        roi_size=[192, 192, 80],
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        sw_device=device,
        device=torch.device("cpu"),
    )
    det.to(device).eval()
    _DETECTOR = det
    return _DETECTOR


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------

HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE, timeout=30)
@modal.fastapi_endpoint(method="GET", label="luna16-healthz")
def healthz() -> Dict[str, str]:
    return {
        "status": "ok",
        "app": "luna16-infer",
        "version": APP_VERSION,
        "bundle_version": BUNDLE_VERSION,
    }


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

@app.function(
    image=IMAGE,
    volumes={"/vol/baseline": BASELINE_VOL},
    timeout=600,
    memory=8 * 1024,
)
@modal.fastapi_endpoint(method="GET", label="luna16-info")
def info() -> Dict[str, Any]:
    """Return bundle metadata + confirmed weight presence."""
    import torch

    bundle_dir = _ensure_bundle_in_volume()
    metadata_path = bundle_dir / "configs" / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}

    import hashlib

    weight_path = bundle_dir / "models" / "model.pt"
    h = hashlib.sha256()
    with weight_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    weight_sha256 = h.hexdigest()

    return {
        "app": "luna16-infer",
        "version": APP_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "model_state": "loaded_luna16_retinanet",
        "model_name": f"monai/lung_nodule_ct_detection@{BUNDLE_VERSION}",
        "weight_sha256": weight_sha256,
        "clinical_meter": "FROC@2 held-out n=88",
        "clinical_meter_value": 0.8267399271674979,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "preprocessing": {
            "hu_range": [-1024.0, 300.0],
            "target_spacing_mm": list(LUNA16_TARGET_SPACING_MM),
            "runtime_resample": True,
            "resample_helper": "_resample_volume_to_luna16",
            "roi_size": [192, 192, 80],
            "sw_overlap": 0.25,
            "score_thresh": 0.02,
            "nms_thresh": 0.22,
        },
        "disclaimer": DISCLAIMER,
        "metadata": {k: metadata.get(k) for k in ("name", "task", "version", "description") if k in metadata},
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _load_case_id_series(case_id: str) -> Tuple[Any, List[float], Tuple[float, float]]:
    """v0.5.0: read the case-storage bundle for ``case_id`` and reconstruct
    an HU volume from ``/vol/cases/<case_id>/dicom_series/*.dcm``.

    Returns
    -------
    (volume_hu_f32, z_positions_mm_ascending, (row_mm, col_mm))
    """
    import numpy as np
    import pydicom

    case_dir = Path("/vol/cases") / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"case_id {case_id!r} not found in cases volume")

    series_dir = case_dir / "dicom_series"
    if not series_dir.exists():
        raise FileNotFoundError(
            f"case_id {case_id!r} has no dicom_series/ subdirectory "
            f"(single-file dicom.bin uploads are not supported by "
            f"luna16-detect; upload as dicom_series_b64 list)"
        )

    dcm_files = sorted(x for x in series_dir.iterdir() if x.is_file())
    if not dcm_files:
        raise FileNotFoundError(f"empty dicom_series/ for case_id {case_id!r}")

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        except Exception:
            continue
        if "PixelData" not in ds:
            continue
        datasets.append(ds)
    if not datasets:
        raise FileNotFoundError(f"no readable DICOM slices for case_id {case_id!r}")

    def _slice_key(ds) -> float:
        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                return float(ipp[2])
            except (TypeError, ValueError):
                pass
        inst = ds.get("InstanceNumber", None)
        if inst is not None:
            try:
                return float(inst)
            except (TypeError, ValueError):
                pass
        return float(abs(hash(getattr(ds, "filename", "") or "")) % (10 ** 9))

    datasets.sort(key=_slice_key)

    first = datasets[0]
    rows = int(first.Rows)
    cols = int(first.Columns)
    ps = first.get("PixelSpacing", [1.0, 1.0])
    row_mm = float(ps[0])
    col_mm = float(ps[1])

    zs: List[float] = []
    stacked = np.empty((len(datasets), rows, cols), dtype=np.float32)
    for i, ds in enumerate(datasets):
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError(
                f"series has slice with mismatched Rows/Columns: "
                f"expected {rows}x{cols}, saw {int(ds.Rows)}x{int(ds.Columns)}"
            )
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        stacked[i] = arr * slope + intercept
        ipp = ds.get("ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else float(i)
        zs.append(z)

    return stacked, zs, (row_mm, col_mm)


def _resample_volume_to_luna16(
    volume_hu: Any,
    spacing_mm: Tuple[float, float, float],
    *,
    z_only: bool = True,
) -> Tuple[Any, Tuple[float, float, float], Tuple[float, float, float], bool, str]:
    """Runtime Spacingd-equivalent: resample HU volume to LUNA16 geometry.

    Mirrors ``oncology_arbiter.lung.resample.resample_for_luna16`` so the
    Modal image does not need the full package. Returns
    ``(volume, out_spacing, source_spacing, was_resampled, method)``.
    """
    import numpy as np
    import SimpleITK as sitk

    v = np.asarray(volume_hu, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"expected 3D HU volume, got shape {v.shape}")

    src_dz, src_dy, src_dx = (float(s) for s in spacing_mm)
    if src_dz <= 0 or src_dy <= 0 or src_dx <= 0:
        raise ValueError(f"spacing entries must be positive; got {spacing_mm}")

    tgt_dz, tgt_dy, tgt_dx = LUNA16_TARGET_SPACING_MM
    if z_only:
        tgt_dy, tgt_dx = src_dy, src_dx

    dz_ok = abs(src_dz - tgt_dz) <= DZ_PASSTHROUGH_TOL_MM
    inplane_ok = (
        abs(src_dy - tgt_dy) <= DZ_PASSTHROUGH_TOL_MM
        and abs(src_dx - tgt_dx) <= DZ_PASSTHROUGH_TOL_MM
    )
    if dz_ok and inplane_ok:
        return v, (src_dz, src_dy, src_dx), (src_dz, src_dy, src_dx), False, "passthrough"

    img = sitk.GetImageFromArray(v)
    img.SetSpacing((src_dx, src_dy, src_dz))
    img.SetOrigin((0.0, 0.0, 0.0))
    old_size = img.GetSize()
    new_spacing = (tgt_dx, tgt_dy, tgt_dz)
    new_size = [
        int(round(old_size[i] * (src_dx, src_dy, src_dz)[i] / new_spacing[i]))
        for i in range(3)
    ]
    if new_size[2] < 1:
        raise ValueError(
            f"z-resample would produce {new_size[2]} slices; "
            f"source={src_dz}mm target={tgt_dz}mm"
        )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(AIR_HU)
    out_np = sitk.GetArrayFromImage(resampler.Execute(img)).astype(np.float32)
    return out_np, (tgt_dz, tgt_dy, tgt_dx), (src_dz, src_dy, src_dx), True, "sitk_linear"


@app.function(
    image=IMAGE,
    gpu="A10G",
    volumes={"/vol/baseline": BASELINE_VOL, "/vol/cases": CASES_VOL},
    timeout=600,
    memory=16 * 1024,
    cpu=4.0,
    min_containers=0,
    scaledown_window=180,
)
@modal.fastapi_endpoint(method="POST", label="luna16-detect")
def detect(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run LUNA16 RetinaNet on a HU volume.

    Body variants
    -------------
    A) case_id path (v0.5.0, production):
       ``{"case_id": "<16-hex>", "top_n": 20}`` — reads the DICOM series
       from the ``oncology-arbiter-cases`` Modal Volume, builds the HU
       volume with pydicom (slope/intercept respected), then runs the
       detector. Payload never has to carry raw CT bytes.

    B) inline volume_hu_b64 path (legacy dev/test):
       ``{"volume_hu_b64": "...", "shape": [D,H,W], "spacing_mm": [dz,dy,dx],
          "top_n": 20}`` — for local unit tests.
    """
    import numpy as np
    import torch

    t0 = time.time()
    if not isinstance(payload, dict):
        return {"error": "payload must be a JSON object"}

    top_n = int(payload.get("top_n", 20))
    ingest_source = None

    # ---- (A) case_id path -------------------------------------------------
    case_id = payload.get("case_id")
    if case_id:
        try:
            # Reload volume before reading (another container may have
            # just committed the case).
            CASES_VOL.reload()
            volume, z_positions, (row_mm, col_mm) = _load_case_id_series(str(case_id))
        except (FileNotFoundError, ValueError) as exc:
            return {"error": f"case_id read failed: {type(exc).__name__}: {exc}"}
        # Estimate axial spacing from adjacent slice z-positions (median).
        if len(z_positions) >= 2:
            zdiffs = np.diff(np.asarray(sorted(z_positions), dtype=np.float64))
            dz = float(np.median(np.abs(zdiffs)))
            if not (dz > 0):
                dz = 1.25
        else:
            dz = 1.25
        shape = [int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])]
        spacing = [dz, row_mm, col_mm]
        ingest_source = f"modal-volume://oncology-arbiter-cases/{case_id}/dicom_series/"
    else:
        # ---- (B) inline volume_hu_b64 path ----------------------------------
        v_b64 = payload.get("volume_hu_b64")
        if not v_b64:
            return {"error": "either case_id or volume_hu_b64 required"}
        shape = payload.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or not all(isinstance(x, int) and x > 0 for x in shape)
        ):
            return {"error": "shape must be [D, H, W] positive ints"}
        spacing = payload.get("spacing_mm", [1.25, 0.703125, 0.703125])
        if len(spacing) != 3:
            return {"error": "spacing_mm must be a 3-tuple"}

        try:
            raw = base64.b64decode(v_b64)
        except Exception as exc:
            return {"error": f"base64 decode failed: {exc}"}

        expected_bytes = int(shape[0] * shape[1] * shape[2] * 4)
        if len(raw) != expected_bytes:
            return {
                "error": (
                    f"byte count mismatch: shape={shape} expects {expected_bytes} bytes, "
                    f"got {len(raw)}"
                )
            }

        volume = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        ingest_source = "inline_volume_hu_b64"

    # Spacingd-equivalent: match MONAI bundle training geometry before detect.
    # Prefer full 3-axis when request asks; default z_only keeps memory sane.
    z_only = bool(payload.get("z_only_resample", True))
    source_spacing = (
        float(spacing[0]), float(spacing[1]), float(spacing[2])
    )
    try:
        volume, spacing_t, source_spacing, was_resampled, resample_method = (
            _resample_volume_to_luna16(volume, source_spacing, z_only=z_only)
        )
    except ValueError as exc:
        return {"error": f"resample failed: {exc}"}
    spacing = list(spacing_t)
    shape = [int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])]

    v = np.clip(volume, -1024.0, 300.0)
    v = (v + 1024.0) / (1024.0 + 300.0)
    v = v[None, ...]  # add channel

    det = _load_detector()
    device = next(det.network.parameters()).device
    # Keep full volume on CPU; MONAI sliding-window copies ROI tiles to GPU.
    x = [torch.from_numpy(np.ascontiguousarray(v))]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    t_pre = time.time()
    with torch.no_grad():
        out = det(x, use_inferer=True)
    t_infer = time.time()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_out = out[0] if out else {
        "box": torch.zeros(0, 6),
        "label": torch.zeros(0),
        "label_scores": torch.zeros(0),
    }
    raw_boxes = raw_out["box"].detach().cpu().numpy()
    raw_scores = raw_out["label_scores"].detach().cpu().numpy()

    dz, dy, dx = float(spacing[0]), float(spacing[1]), float(spacing[2])
    boxes: List[Dict[str, float]] = []
    for i, box in enumerate(raw_boxes):
        z0, y0, x0, z1, y1, x1 = box.tolist()
        cz = (z0 + z1) / 2.0
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        wz = (z1 - z0) * dz
        wy = (y1 - y0) * dy
        wx = (x1 - x0) * dx
        boxes.append({
            "center_z_mm": float(cz * dz),
            "center_y_mm": float(cy * dy),
            "center_x_mm": float(cx * dx),
            "width_mm": float(wx),
            "height_mm": float(wy),
            "depth_mm": float(wz),
            "diameter_mm": float(max(wx, wy, wz)),
            "score": float(raw_scores[i]),
        })
    boxes.sort(key=lambda b: b["score"], reverse=True)
    top_score = boxes[0]["score"] if boxes else 0.0

    t_end = time.time()
    return {
        "model_state": "loaded_luna16_retinanet",
        "model_name": f"monai/lung_nodule_ct_detection@{BUNDLE_VERSION}",
        "bundle_version": BUNDLE_VERSION,
        "app_version": APP_VERSION,
        # v0.5.0: emit the case_id + ingest_source for provenance chains.
        "case_id": case_id if case_id else None,
        "ingest_source": ingest_source,
        "n_detections": len(boxes),
        "top_score": top_score,
        "detections": boxes[:top_n],
        "inference_seconds": round(t_infer - t_pre, 4),
        "total_seconds": round(t_end - t0, 4),
        "input_shape": list(shape),
        "preprocessing_summary": {
            "hu_range": [-1024.0, 300.0],
            "target_spacing_mm": list(LUNA16_TARGET_SPACING_MM),
            "source_spacing_mm": list(source_spacing),
            "actual_spacing_mm": [dz, dy, dx],
            "was_resampled": was_resampled,
            "resample_method": resample_method,
            "z_only_resample": z_only,
            "roi_size": [192, 192, 80],
            "sw_overlap": 0.25,
            "score_thresh": 0.02,
            "nms_thresh": 0.22,
        },
        "device": str(device),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Local CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Local test: ``modal run deploy/modal/luna16_infer_app.py`` — prints info()."""
    result = info.remote()
    print(json.dumps(result, indent=2))
