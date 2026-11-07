"""One-GPU Modal runner for the ClinicalBERT v2 head trainer."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import modal

REMOTE_DATA_DIR = Path("/clinicalbert-data")
REMOTE_TRAINER = Path("/opt/train_clinicalbert_head_v2.py")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.1",
        "safetensors==0.5.3",
        "numpy==2.1.0",
    )
)

# Local source paths are needed only while Modal constructs the image.  A Modal
# Function imports this module again as /root/clinicalbert_train_v2.py, where
# repository-relative parents and the session mounts do not exist.
if modal.is_local():
    local_data_dir = Path(os.environ.get(
        "CLINICALBERT_DATA_DIR",
        "/mnt/shared-workspace/shared/ten-capability-delivery/clinicalbert/data_v2",
    )).resolve()
    local_trainer = (
        Path(__file__).resolve().parents[2] / "scripts" / "train_clinicalbert_head_v2.py"
    ).resolve()
    if not local_data_dir.is_dir():
        raise RuntimeError(f"ClinicalBERT data directory does not exist: {local_data_dir}")
    if not local_trainer.is_file():
        raise RuntimeError(f"ClinicalBERT trainer does not exist: {local_trainer}")
    image = image.add_local_file(str(local_trainer), str(REMOTE_TRAINER))
    image = image.add_local_dir(str(local_data_dir), str(REMOTE_DATA_DIR))

app = modal.App("clinicalbert-head-v2-training")


@app.function(image=image, gpu="L4", cpu=8.0, memory=32768, timeout=6 * 60 * 60)
def train_on_gpu(config: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    out_dir = Path("/tmp/clinicalbert-output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    command = [
        "python", str(REMOTE_TRAINER),
        "--data-dir", str(REMOTE_DATA_DIR),
        "--out-dir", str(out_dir),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--inference-batch-size", str(config["inference_batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--weight-decay", str(config["weight_decay"]),
        "--max-length", str(config["max_length"]),
        "--stride", str(config["stride"]),
        "--seed", str(config["seed"]),
        "--base-model", str(config["base_model"]),
        "--base-revision", str(config["base_revision"]),
        "--dataset-sha256", str(config["dataset_sha256"]),
        "--split-sha256", str(config["split_sha256"]),
        "--source-commit", str(config["source_commit"]),
    ]
    for key, flag in (
        ("limit_train", "--limit-train"),
        ("limit_validation", "--limit-validation"),
        ("limit_test", "--limit-test"),
        ("limit_external", "--limit-external"),
    ):
        value = config.get(key)
        if value is not None:
            command.extend([flag, str(value)])

    log_path = out_dir / "clinicalbert_training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process_env = os.environ.copy()
        process_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ClinicalBERT trainer failed with exit code {return_code}")

    summary = json.loads((out_dir / "clinicalbert_training_summary_v2.json").read_text())
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
    return archive.getvalue(), summary


def _safe_extract(payload: bytes, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (out_dir / member.name).resolve()
            if out_dir.resolve() not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(out_dir)


@app.local_entrypoint()
def main(
    mode: str = "pilot",
    out_dir: str = "/workspace/clinicalbert_modal_output",
    source_commit: str = "3c1cac6157051a1aebccc9933edc9d7c5085aa8f",
    pilot_epochs: int = 3,
    pilot_train: int = 512,
    pilot_validation: int = 64,
    pilot_test: int = 16,
    pilot_external: int = 16,
) -> None:
    if mode not in {"pilot", "broad"}:
        raise ValueError("mode must be pilot or broad")
    if mode == "pilot" and min(
        pilot_epochs, pilot_train, pilot_validation, pilot_test, pilot_external
    ) <= 0:
        raise ValueError("all pilot sizes and pilot_epochs must be positive")
    config: dict[str, Any] = {
        "epochs": pilot_epochs if mode == "pilot" else 3,
        "batch_size": 16 if mode == "pilot" else 24,
        "inference_batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "max_length": 512,
        "stride": 128,
        "seed": 20260929,
        "base_model": "emilyalsentzer/Bio_ClinicalBERT",
        "base_revision": "d5892b39a4adaed74b92212a44081509db72f87b",
        "dataset_sha256": "13fd452d103da8c97634ba089ceabfa72e905a9365524699f7aefefac5ca62c2",
        "split_sha256": "0d7f3f754adf9305dadbcc628786217a329b5050e802b7968bc4a5c2ccf5b888",
        "source_commit": source_commit,
    }
    if mode == "pilot":
        config.update({
            "limit_train": pilot_train,
            "limit_validation": pilot_validation,
            "limit_test": pilot_test,
            "limit_external": pilot_external,
        })
    payload, summary = train_on_gpu.remote(config)
    destination = Path(out_dir).resolve()
    _safe_extract(payload, destination)
    print(json.dumps({"mode": mode, "out_dir": str(destination), "summary": summary}, indent=2, sort_keys=True))
