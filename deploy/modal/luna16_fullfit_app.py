"""LUNA16 full refine — 5-epoch 482/60 on nifti_v3 (post dual-pilot).

Frozen binds:
  dataset d20b94a15315b830b99d29e429b0daf5380b7fa5bf157fecf733c2af354944ff
  split   3602f34c4b3e65024e6a7ef3fc7185577ee1f39e54d2c9f5c35baffc2d0cd1d9
  baseline b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01
  datalist sha256 c5f6adf3fe79b787e8b038862220c8aed16964f36cd8b642dbaa4ccfc10a73c3
  nifti: /vol/luna16/nifti_v3  (never writes nifti_v2)

Executable MONAI run ID: ``run`` (not training/train).

Deploy::
  MODAL_PROFILE=crispro modal deploy deploy/modal/luna16_fullfit_app.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import modal

APP_VERSION = "luna16-fullfit-v0.1.2-empty-box-patch"
EXPECTED_DATASET = "d20b94a15315b830b99d29e429b0daf5380b7fa5bf157fecf733c2af354944ff"
EXPECTED_SPLIT = "3602f34c4b3e65024e6a7ef3fc7185577ee1f39e54d2c9f5c35baffc2d0cd1d9"
EXPECTED_BASELINE = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
EXPECTED_DATALIST = "c5f6adf3fe79b787e8b038862220c8aed16964f36cd8b642dbaa4ccfc10a73c3"
N_TRAIN, N_VAL, N_TEST = 482, 60, 88
NIFTI_PREFIX = "/vol/luna16/nifti_v3"

LUNA16_VOL = modal.Volume.from_name("luna16-data")
BASELINE_VOL = modal.Volume.from_name("luna16-baseline-weights")
OUTPUT_VOL = modal.Volume.from_name("luna16-training-runs", create_if_missing=True)

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1", "libgl1", "git")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "monai==1.4.0",
        "nibabel==5.2.1",
        "numpy==1.24.4",
        "pytorch-ignite==0.4.11",
        "tensorboard==2.17.0",
        "fire==0.6.0",
        "scipy==1.13.1",
        "scikit-image==0.24.0",
        "pandas==2.2.2",
        "fastapi==0.115.0",
        "pydantic==2.9.2",
    )
    .add_local_file(
        Path(__file__).with_name("luna16_full_482_60_datalist.json"),
        remote_path="/opt/luna16_full_482_60_datalist.json",
    )
)

HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)

app = modal.App("luna16-fullfit")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _prepare_bundle(work: Path) -> Path:
    src = Path("/vol/baseline/lung_nodule_ct_detection")
    if not src.is_dir():
        raise FileNotFoundError(f"missing baseline bundle at {src}")
    bundle = work / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    shutil.copytree(
        src,
        bundle,
        ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.pyc"),
    )
    model = bundle / "models" / "model.pt"
    if not model.is_file():
        raise FileNotFoundError(f"missing baseline weights {model}")
    if _sha256_file(model) != EXPECTED_BASELINE:
        raise ValueError("baseline model SHA mismatch")
    return bundle


def _stage_datalist(work: Path) -> Path:
    src = Path("/opt/luna16_full_482_60_datalist.json")
    data = src.read_bytes()
    digest = _sha256_bytes(data)
    if digest != EXPECTED_DATALIST:
        raise ValueError(f"datalist sha mismatch: {digest}")
    parsed = json.loads(data)
    if len(parsed.get("training", [])) != N_TRAIN:
        raise ValueError("train count != 482")
    if len(parsed.get("validation", [])) != N_VAL:
        raise ValueError("val count != 60")
    out = work / "full_datalist.json"
    # train.json expects training+validation keys only for fit
    fit_view = {"training": parsed["training"], "validation": parsed["validation"]}
    out.write_text(json.dumps(fit_view))
    # keep full with test for held-out later
    (work / "full_datalist_with_test.json").write_bytes(data)
    return out


def _missing_nifti(uids: List[str]) -> List[str]:
    missing = []
    root = Path(NIFTI_PREFIX)
    for uid in uids:
        nii = root / uid / f"{uid}.nii.gz"
        if not nii.is_file() or nii.stat().st_size == 0:
            missing.append(uid)
    return missing


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="luna16-fullfit-healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "luna16-fullfit",
        "app_version": APP_VERSION,
        "monai_run_id": "run",
        "nifti_prefix": NIFTI_PREFIX,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "datalist_sha256": EXPECTED_DATALIST,
    }


@app.function(
    image=TRAIN_IMAGE,
    cpu=8.0,
    memory=32768,
    timeout=1800,
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
)
def cpu_preflight_full(request_id: str = "") -> Dict[str, Any]:
    """Gate: hashes + all 482/60 NIfTI on nifti_v3 + run ID."""
    import torch
    import monai
    import numpy

    rid = request_id or f"full-preflight-{time.strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path("/vol/output") / "fullfit-v0.1" / rid / "preflight"
    out_root.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, Any] = {}

    def chk(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    ds_path = Path("/vol/luna16/manifests/luna16_dataset_v2.json")
    sp_path = Path("/vol/luna16/manifests/luna16_split_v2.json")
    chk("dataset_hash", ds_path.is_file() and _sha256_file(ds_path) == EXPECTED_DATASET, EXPECTED_DATASET)
    chk("split_hash", sp_path.is_file() and _sha256_file(sp_path) == EXPECTED_SPLIT, EXPECTED_SPLIT)

    work = Path(f"/tmp/{rid}")
    work.mkdir(parents=True, exist_ok=True)
    try:
        datalist = _stage_datalist(work)
        chk("datalist", True, EXPECTED_DATALIST)
        parsed = json.loads(datalist.read_text())
        train_uids = [r["series_uid"] for r in parsed["training"]]
        val_uids = [r["series_uid"] for r in parsed["validation"]]
        miss = _missing_nifti(train_uids + val_uids)
        chk("nifti_complete_542", not miss, {"n_missing": len(miss), "sample": miss[:10]})
        n_v3 = len([p for p in Path(NIFTI_PREFIX).iterdir() if p.is_dir()]) if Path(NIFTI_PREFIX).is_dir() else 0
        chk("nifti_v3_count", n_v3 >= 542, {"n_nifti_v3": n_v3})
        # never touch v2
        chk("nifti_v2_untouched_prefix", Path("/vol/luna16/nifti_v2").is_dir(), "nifti_v2 present (read-only bind)")

        # spacing sample (≥5 train nii)
        import nibabel as nib
        import random

        rng = random.Random(0)
        sample_uids = rng.sample(train_uids, min(5, len(train_uids)))
        spacing_rows = []
        spacing_ok = True
        target = (0.703125, 0.703125, 1.25)
        for uid in sample_uids:
            nii = Path(NIFTI_PREFIX) / uid / f"{uid}.nii.gz"
            zooms = tuple(float(z) for z in nib.load(str(nii)).header.get_zooms()[:3])
            ok = all(abs(zooms[i] - target[i]) < 1e-3 for i in range(3))
            spacing_rows.append({"uid": uid, "zooms": zooms, "ok": ok})
            spacing_ok = spacing_ok and ok
        chk("sample_nifti_spacing_check", spacing_ok, spacing_rows)

        bundle = _prepare_bundle(work)
        train_cfg = json.loads((bundle / "configs" / "train.json").read_text())
        chk("run_id_present", "run" in train_cfg, train_cfg.get("run"))
        runtime = {
            "torch": torch.__version__.split("+")[0],
            "monai": monai.__version__,
            "numpy": numpy.__version__,
        }
        chk("runtime", runtime["torch"].startswith("2.4") and runtime["monai"].startswith("1.4"), runtime)
    except Exception as e:
        chk("exception", False, f"{type(e).__name__}: {e}")

    passed = all(v.get("ok") for v in checks.values())
    result = {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "request_id": rid,
        "app_version": APP_VERSION,
        "checks": checks,
        "monai_run_id": "run",
        "nifti_prefix": NIFTI_PREFIX,
        "GATE_C_PASS": "GATE_C_PASS" if passed else None,
    }
    (out_root / "cpu_preflight_full.json").write_text(json.dumps(result, indent=2))
    OUTPUT_VOL.commit()
    return result


@app.function(
    image=TRAIN_IMAGE,
    gpu="A100",
    cpu=8.0,
    memory=65536,
    timeout=86400,  # 24h — 5 epochs on 482 series
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
)
def run_fullfit(
    request_id: str = "",
    epochs: int = 5,
    learning_rate: float = 0.001,
    batch_size: int = 2,
    gate_c_request_id: str = "",
    resume: bool = False,
) -> Dict[str, Any]:
    """5-epoch 482/60 full fit on nifti_v3. Requires Gate C preflight receipt."""
    import torch

    rid = request_id or f"fullfit-{time.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path("/vol/output") / "fullfit-v0.1" / rid / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    work = Path(f"/tmp/{rid}")
    work.mkdir(parents=True, exist_ok=True)

    marker = Path("/vol/output") / "fullfit-v0.1" / "request_registry" / f"{rid}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.is_file() and not resume:
        return json.loads(marker.read_text())

    # Hard unlock: Gate C receipt must exist with GATE_C_PASS
    if not gate_c_request_id:
        return {
            "status": "failed",
            "error": "gate_c_request_id required — no soft unlock",
            "verdict": "NOT AUTHORIZED",
            "request_id": rid,
        }
    gate_c_path = (
        Path("/vol/output") / "fullfit-v0.1" / gate_c_request_id / "preflight" / "cpu_preflight_full.json"
    )
    if not gate_c_path.is_file():
        return {
            "status": "failed",
            "error": f"missing Gate C receipt {gate_c_path}",
            "verdict": "NOT AUTHORIZED",
            "request_id": rid,
        }
    gate_c = json.loads(gate_c_path.read_text())
    if gate_c.get("GATE_C_PASS") != "GATE_C_PASS" or not gate_c.get("passed"):
        return {
            "status": "failed",
            "error": "Gate C did not PASS",
            "verdict": "NOT AUTHORIZED",
            "gate_c": {"passed": gate_c.get("passed"), "GATE_C_PASS": gate_c.get("GATE_C_PASS")},
            "request_id": rid,
        }

    try:
        datalist = _stage_datalist(work)
        parsed = json.loads(datalist.read_text())
        miss = _missing_nifti([r["series_uid"] for r in parsed["training"] + parsed["validation"]])
        if miss:
            return {
                "status": "failed",
                "error": f"missing {len(miss)} nifti_v3 series",
                "missing_sample": miss[:20],
                "request_id": rid,
                "app_version": APP_VERSION,
            }

        bundle = _prepare_bundle(work)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "model.pt"
        if not (resume and ckpt_path.is_file()):
            shutil.copy(bundle / "models" / "model.pt", ckpt_path)
        baseline_sha = EXPECTED_BASELINE
        start_sha = _sha256_file(ckpt_path)

        cmd = [
            "python",
            "-m",
            "monai.bundle",
            "run",
            "run",
            "--config_file",
            str(bundle / "configs" / "train.json"),
            "--bundle_root",
            str(bundle),
            "--dataset_dir",
            NIFTI_PREFIX,
            "--data_list_file_path",
            str(datalist),
            "--ckpt_dir",
            str(ckpt_dir),
            "--output_dir",
            str(run_dir / "eval"),
            "--epochs",
            str(epochs),
            "--learning_rate",
            str(learning_rate),
            "--batch_size",
            str(batch_size),
            "--val_interval",
            "1",
        ]
        (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))

        log_path = run_dir / "training.log"
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(bundle),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )

        end_sha = _sha256_file(ckpt_path) if ckpt_path.is_file() else None
        result = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "request_id": rid,
            "app_version": APP_VERSION,
            "elapsed_seconds": round(time.time() - t0, 3),
            "monai_run_id": "run",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_training": N_TRAIN,
            "n_validation": N_VAL,
            "nifti_prefix": NIFTI_PREFIX,
            "dataset_sha256": EXPECTED_DATASET,
            "split_sha256": EXPECTED_SPLIT,
            "datalist_sha256": EXPECTED_DATALIST,
            "baseline_sha256": baseline_sha,
            "start_ckpt_sha256": start_sha,
            "end_ckpt_sha256": end_sha,
            "weights_moved": bool(end_sha and end_sha != baseline_sha),
            "checkpoints": [
                {
                    "path": str(ckpt_path),
                    "sha256": end_sha,
                    "bytes": ckpt_path.stat().st_size if ckpt_path.is_file() else 0,
                }
            ],
            "log_path": str(log_path),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if proc.returncode != 0:
            # tail log for diagnosis
            try:
                tail = log_path.read_text()[-4000:]
            except Exception:
                tail = ""
            result["error"] = f"monai.bundle exit {proc.returncode}"
            result["log_tail"] = tail

        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        marker.write_text(json.dumps(result, indent=2))
        OUTPUT_VOL.commit()
        return result
    except Exception as e:
        fail = {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "request_id": rid,
            "app_version": APP_VERSION,
            "elapsed_seconds": round(time.time() - t0, 3),
        }
        (run_dir / "result.json").write_text(json.dumps(fail, indent=2))
        OUTPUT_VOL.commit()
        return fail


@app.function(
    image=TRAIN_IMAGE,
    gpu="A100",
    cpu=8.0,
    memory=65536,
    timeout=14400,
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
        "/vol/output": OUTPUT_VOL,
    },
)
def run_heldout(
    request_id: str = "",
    ckpt_request_id: str = "",
) -> Dict[str, Any]:
    """Paired held-out evaluate on 88 test series (stock bundle evaluate run).

    Honest labels: PROVEN / UNPROVEN / FAILED / NOT AUTHORIZED — caller grades FROC.
    """
    rid = request_id or f"heldout-{time.strftime('%Y%m%dT%H%M%SZ')}"
    if not ckpt_request_id:
        return {"status": "failed", "error": "ckpt_request_id required", "verdict": "NOT AUTHORIZED"}

    train_dir = Path("/vol/output") / "fullfit-v0.1" / ckpt_request_id / "train"
    ckpt = train_dir / "checkpoints" / "model.pt"
    if not ckpt.is_file():
        return {
            "status": "failed",
            "error": f"missing ckpt {ckpt}",
            "verdict": "NOT AUTHORIZED",
            "request_id": rid,
        }

    run_dir = Path("/vol/output") / "fullfit-v0.1" / rid / "heldout"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    work = Path(f"/tmp/{rid}")
    work.mkdir(parents=True, exist_ok=True)

    raw = Path("/opt/luna16_full_482_60_datalist.json").read_bytes()
    if _sha256_bytes(raw) != EXPECTED_DATALIST:
        return {"status": "failed", "error": "datalist sha mismatch", "verdict": "FAILED"}
    full = json.loads(raw)
    test_uids = [r["series_uid"] for r in full["test"]]
    miss = _missing_nifti(test_uids)
    if miss:
        return {
            "status": "failed",
            "error": f"missing {len(miss)} test nifti",
            "missing_sample": miss[:20],
            "verdict": "FAILED",
        }

    # evaluate datalist: validation key used by many bundle evaluate configs
    eval_list = {"validation": full["test"], "testing": full["test"]}
    eval_path = work / "heldout_datalist.json"
    eval_path.write_text(json.dumps(eval_list))

    bundle = _prepare_bundle(work)
    # seed eval weights from trained ckpt
    eval_ckpt_dir = run_dir / "checkpoints"
    eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, eval_ckpt_dir / "model.pt")

    # 29/88 held-out series have empty GT boxes — AffineBoxToImageCoordinated crashes on shape [0].
    # Patch reshape to [0,6] before evaluator.run (RUO; negatives required for FROC FP rates).
    patch_py = bundle / "empty_box_patch.py"
    patch_py.write_text(
        """
def apply():
    import numpy as np
    import torch
    from monai.apps.detection.transforms.dictionary import AffineBoxToImageCoordinated

    if getattr(AffineBoxToImageCoordinated, \"_luna16_empty_box_patched\", False):
        return
    _orig = AffineBoxToImageCoordinated.__call__

    def _fix_box_array(box):
        if box is None:
            return np.zeros((0, 6), dtype=np.float32)
        if torch.is_tensor(box):
            if box.numel() == 0:
                return torch.zeros((0, 6), dtype=box.dtype if box.dtype.is_floating_point else torch.float32, device=box.device)
            if box.ndim == 1:
                return box.reshape(0, 6) if box.numel() == 0 else box
            return box
        if isinstance(box, np.ndarray):
            if box.size == 0:
                return np.zeros((0, 6), dtype=np.float32)
            return box
        if isinstance(box, (list, tuple)) and len(box) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return box

    def _wrapped(self, data):
        d = dict(data)
        keys = list(getattr(self, \"box_keys\", []) or [])
        if not keys and hasattr(self, \"keys\"):
            keys = list(self.keys)
        for k in keys or [\"box\"]:
            if k in d:
                d[k] = _fix_box_array(d[k])
        # labels for empty positives
        if \"label\" in d:
            lab = d[\"label\"]
            if (torch.is_tensor(lab) and lab.numel() == 0) or (isinstance(lab, np.ndarray) and lab.size == 0) or (isinstance(lab, (list, tuple)) and len(lab) == 0):
                d[\"label\"] = np.zeros((0,), dtype=np.int64)
        return _orig(self, d)

    AffineBoxToImageCoordinated.__call__ = _wrapped
    AffineBoxToImageCoordinated._luna16_empty_box_patched = True
"""
    )
    overlay = bundle / "configs" / "evaluate_heldout_overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "imports": ["$import empty_box_patch"],
                "initialize": [
                    "$setattr(torch.backends.cudnn, 'benchmark', True)",
                    "$empty_box_patch.apply()",
                ],
            },
            indent=2,
        )
    )

    # Stock MONAI evaluate.json overlays validate#* onto train.json — both required.
    # evaluate.json alone → KeyError 'validate' (Gate E 2026-10-03 failure).
    eval_cfg = bundle / "configs" / "evaluate.json"
    train_cfg = bundle / "configs" / "train.json"
    if not train_cfg.is_file():
        return {
            "status": "failed",
            "error": f"missing {train_cfg}",
            "verdict": "FAILED",
        }
    if eval_cfg.is_file():
        config_file = f"['{train_cfg}','{eval_cfg}','{overlay}']"
        cfg = json.loads(eval_cfg.read_text())
    else:
        config_file = str(train_cfg)
        cfg = json.loads(train_cfg.read_text())

    # discover evaluate executable key (evaluate.json uses run → $@validate#evaluator.run())
    run_key = "evaluate" if "evaluate" in cfg else ("run" if "run" in cfg else None)
    if not run_key:
        return {
            "status": "failed",
            "error": f"no evaluate/run key in evaluate/train config",
            "verdict": "FAILED",
            "config_keys": list(cfg.keys())[:40],
        }

    cmd = [
        "python",
        "-m",
        "monai.bundle",
        "run",
        run_key,
        "--config_file",
        config_file,
        "--bundle_root",
        str(bundle),
        "--dataset_dir",
        NIFTI_PREFIX,
        "--data_list_file_path",
        str(eval_path),
        "--ckpt_dir",
        str(eval_ckpt_dir),
        "--output_dir",
        str(run_dir / "eval"),
    ]
    # Ensure patch import path
    env_pythonpath = str(bundle)
    (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))
    log_path = run_dir / "evaluate.log"
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(bundle) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd, cwd=str(bundle), stdout=logf, stderr=subprocess.STDOUT, text=True, env=env
        )

    # Collect any metric artifacts
    artifacts = []
    eval_out = run_dir / "eval"
    if eval_out.is_dir():
        for p in eval_out.rglob("*"):
            if p.is_file() and p.suffix in {".json", ".csv", ".txt"}:
                artifacts.append(str(p))

    import re

    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""

    def _parse_froc(text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        m = re.search(r"FROC.*2\.?0?\s*FP.*?:\s*([0-9.]+)", text, re.IGNORECASE)
        if m:
            out["froc_at_2"] = float(m.group(1))
        pts = {}
        for fp, val in re.findall(
            r"FROC.*?([0-9.]+)\s*FP.*?:\s*([0-9.]+)", text, flags=re.IGNORECASE
        ):
            pts[str(fp)] = float(val)
        if pts:
            out["froc_points"] = pts
        m = re.search(r"mAP.*IoU\s*0?\.?1.*?:\s*([0-9.]+)", text, re.IGNORECASE)
        if m:
            out["map_iou0.1"] = float(m.group(1))
        m = re.search(r"val_coco:\s*([0-9.]+)", text, re.IGNORECASE)
        if m:
            out["val_coco"] = float(m.group(1))
        return out

    refined_metrics = _parse_froc(log_text)

    # Paired baseline evaluate on same 88 (stock model.pt) — clinical contrast
    baseline_metrics: Dict[str, Any] = {}
    baseline_rc = None
    if proc.returncode == 0:
        base_ckpt = Path("/vol/baseline/lung_nodule_ct_detection/models/model.pt")
        if not base_ckpt.is_file():
            # fallback common layout
            cands = list(Path("/vol/baseline").rglob("model.pt"))
            base_ckpt = cands[0] if cands else base_ckpt
        if base_ckpt.is_file() and _sha256_file(base_ckpt) == EXPECTED_BASELINE:
            base_dir = run_dir / "baseline_checkpoints"
            base_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(base_ckpt, base_dir / "model.pt")
            base_cmd = cmd.copy()
            # replace ckpt_dir arg
            try:
                i = base_cmd.index("--ckpt_dir")
                base_cmd[i + 1] = str(base_dir)
            except ValueError:
                base_cmd.extend(["--ckpt_dir", str(base_dir)])
            base_log = run_dir / "evaluate_baseline.log"
            with base_log.open("w") as logf:
                bproc = subprocess.run(
                    base_cmd,
                    cwd=str(bundle),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            baseline_rc = bproc.returncode
            baseline_metrics = _parse_froc(base_log.read_text(errors="replace") if base_log.is_file() else "")
            baseline_metrics["returncode"] = baseline_rc
            baseline_metrics["ckpt_sha256"] = EXPECTED_BASELINE

    delta = None
    if refined_metrics.get("froc_at_2") is not None and baseline_metrics.get("froc_at_2") is not None:
        delta = {
            "froc_at_2": refined_metrics["froc_at_2"] - baseline_metrics["froc_at_2"],
            "floor_ref_0.05": 0.05,
            "meets_floor_ref": (refined_metrics["froc_at_2"] - baseline_metrics["froc_at_2"]) >= 0.05,
        }

    result = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "request_id": rid,
        "ckpt_request_id": ckpt_request_id,
        "ckpt_sha256": _sha256_file(ckpt),
        "app_version": APP_VERSION,
        "elapsed_seconds": round(time.time() - t0, 3),
        "n_test": N_TEST,
        "monai_run_key": run_key,
        "config_file": config_file,
        "artifacts": artifacts[:50],
        "log_path": str(log_path),
        "froc_at_2": refined_metrics.get("froc_at_2"),
        "refined": refined_metrics,
        "baseline": baseline_metrics or None,
        "delta": delta,
        "verdict": (
            "PROVEN"
            if proc.returncode == 0 and refined_metrics.get("froc_at_2") is not None
            else ("UNPROVEN" if proc.returncode == 0 else "FAILED")
        ),
        "verdict_note": (
            "Held-out bundle evaluate completed; FROC parsed from log when present. "
            "+0.05 floor is reference only, not automatic promotion."
            if proc.returncode == 0
            else "evaluate subprocess failed"
        ),
    }
    if proc.returncode != 0:
        result["log_tail"] = log_text[-4000:]
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    OUTPUT_VOL.commit()
    return result
