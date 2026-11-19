"""Modal deployment for the real-report ClinicalBERT v2 sliding-window parser."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import modal

APP_VERSION = "clinicalbert-modal-v2.0.0-real-tcga"
ARTIFACT_FILENAME = "clinicalbert_head_v2.safetensors"
ARTIFACT_SHA256 = "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50"
BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BASE_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
REMOTE_BUNDLE_DIR = Path("/model")
REMOTE_BASE_DIR = Path("/clinicalbert-base")
REMOTE_RUNTIME = Path("/opt/clinicalbert_runtime_v2.py")
DISCLAIMER = (
    "Research Use Only. Not FDA-cleared. Not CE-marked. "
    "Not intended for clinical use."
)


def _download_base_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=BASE_MODEL,
        revision=BASE_REVISION,
        local_dir=str(REMOTE_BASE_DIR),
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.1",
        "safetensors==0.5.3",
        "huggingface_hub==0.34.4",
        "fastapi==0.115.0",
        "numpy==2.1.0",
    )
    .run_function(_download_base_model)
)

# Source mounts are resolved only by the local deploy process. Modal imports
# this module again in the container, where repository-relative paths do not
# exist and the already-built image contains the files below.
if modal.is_local():
    repo_root = Path(__file__).resolve().parents[2]
    local_bundle = Path(os.environ.get(
        "CLINICALBERT_WEIGHT_DIR",
        str(repo_root / "artifacts" / "clinicalbert"),
    )).resolve()
    local_runtime = repo_root / "src" / "oncology_arbiter" / "nlp" / "clinicalbert_runtime_v2.py"
    required = [
        local_bundle / ARTIFACT_FILENAME,
        local_bundle / "clinicalbert_head_v2_config.json",
        local_bundle / "clinicalbert_metrics_v2.json",
        local_runtime,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"ClinicalBERT deployment inputs missing: {missing}")
    image = image.add_local_file(str(local_runtime), str(REMOTE_RUNTIME))
    for path in required[:3]:
        image = image.add_local_file(str(path), str(REMOTE_BUNDLE_DIR / path.name))

app = modal.App("clinicalbert")
health_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.115.0")


@app.function(image=health_image)
@modal.fastapi_endpoint(method="GET", label="clinicalbert-healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "clinicalbert",
        "app_version": APP_VERSION,
        "expected_artifact_sha256": ARTIFACT_SHA256,
        "disclaimer": DISCLAIMER,
    }


@app.cls(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=300,
    scaledown_window=900,
    min_containers=0,
)
class ClinicalBertModal:
    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, str(REMOTE_RUNTIME.parent))
        from clinicalbert_runtime_v2 import ClinicalBertV2Runtime

        started = time.perf_counter()
        self.runtime = ClinicalBertV2Runtime(
            REMOTE_BUNDLE_DIR,
            base_model_dir=REMOTE_BASE_DIR,
            device="cpu",
        )
        self.container_load_seconds = time.perf_counter() - started
        identity = self.runtime.identity()
        if identity["artifact_sha256"] != ARTIFACT_SHA256:
            raise RuntimeError("deployed ClinicalBERT head identity mismatch")
        if identity["base_revision"] != BASE_REVISION:
            raise RuntimeError("deployed ClinicalBERT base revision mismatch")

    @modal.fastapi_endpoint(method="GET", label="clinicalbert-info")
    def info(self) -> Dict[str, Any]:
        return {
            "app": "clinicalbert",
            "app_version": APP_VERSION,
            **self.runtime.identity(),
            "container_load_seconds": round(self.container_load_seconds, 3),
            "disclaimer": DISCLAIMER,
        }

    @modal.fastapi_endpoint(method="POST", label="clinicalbert-parse")
    def parse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        report_text = payload.get("report_text")
        try:
            result = self.runtime.parse(report_text)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"app": "clinicalbert", "app_version": APP_VERSION, **result}
