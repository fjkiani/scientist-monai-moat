"""Modal app: owkin/phikon 768-d CLS embedder for the 62k tissue probe.

The production probe ``models/phikon_probe_v1.joblib`` was fit on 768-d
CLS tokens from ``AutoImageProcessor`` + ``AutoModel`` (``owkin/phikon``),
the same forward pass as ``deploy/modal/phikon_probe_train_app.py``.

Wire contract (matches ``PhikonClient`` and ``scripts/embed_phikon_nct_crc.py``):

- ``POST`` JSON ``{"inputs": [{"image_b64": "..."}]}``
- response ``embedding`` for one image, ``embeddings`` for the batch
- batch cap 64. Over-cap returns HTTP 200 ``{"error": "batch size N exceeds cap of 64"}``

Deploy::

    MODAL_PROFILE=testing1235 modal deploy deploy/modal/phikon_embed_app.py

Labeled URL: ``https://testing1235--phikon-embed.modal.run``
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import modal

APP_VERSION = "phikon-embed-v1-testing1235"
MODEL_REPO = "owkin/phikon"
EMBED_DIM = 768
BATCH_CAP = 64

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "safetensors==0.4.5",
        "pillow==10.4.0",
        "huggingface_hub==0.24.7",
        "fastapi==0.115.0",
    )
)
HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)

app = modal.App("phikon-pathology")
HF_SECRET = modal.Secret.from_name("hf_token")


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="phikon-healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "app": "phikon-pathology", "version": APP_VERSION}


@app.cls(
    image=IMAGE,
    gpu="A10G",
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=900,
    min_containers=0,
)
class PhikonEmbed:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        started = time.time()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {"token": token} if token else {}
        self.processor = AutoImageProcessor.from_pretrained(MODEL_REPO, **kwargs)
        self.model = AutoModel.from_pretrained(MODEL_REPO, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.model_revision = str(getattr(self.model.config, "_commit_hash", "") or "")
        self.load_seconds = round(time.time() - started, 3)

    def _vectors(self, images: list[Any]) -> list[list[float]]:
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state[:, 0, :]
        rows = hidden.detach().float().cpu().tolist()
        if any(len(row) != EMBED_DIM for row in rows):
            raise RuntimeError(f"expected {EMBED_DIM}-d CLS, got {[len(r) for r in rows]}")
        return rows

    def _embed_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        inputs = payload.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            return {"error": "inputs must be a non-empty list of {image_b64}"}
        if len(inputs) > BATCH_CAP:
            return {"error": f"batch size {len(inputs)} exceeds cap of {BATCH_CAP}"}
        started = time.time()
        try:
            images = []
            for item in inputs:
                if not isinstance(item, dict) or not item.get("image_b64"):
                    return {"error": "each input requires image_b64"}
                raw = base64.b64decode(item["image_b64"])
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            vectors = self._vectors(images)
        except Exception as exc:  # noqa: BLE001 — return the documented error body
            return {"error": f"{type(exc).__name__}: {exc}"}
        body: dict[str, Any] = {
            "embeddings": vectors,
            "embedding_dim": EMBED_DIM,
            "n": len(vectors),
            "model": MODEL_REPO,
            "model_revision": self.model_revision,
            "app_version": APP_VERSION,
            "seconds": round(time.time() - started, 3),
        }
        if len(vectors) == 1:
            body["embedding"] = vectors[0]
        return body

    @modal.fastapi_endpoint(method="GET", label="phikon-info")
    def info(self) -> dict[str, Any]:
        return {
            "app": "phikon-pathology",
            "app_version": APP_VERSION,
            "model": MODEL_REPO,
            "model_revision": self.model_revision,
            "embedding_dim": EMBED_DIM,
            "pooling": "cls",
            "batch_cap": BATCH_CAP,
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    @modal.fastapi_endpoint(method="POST", label="phikon-embed")
    def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._embed_payload(payload)
