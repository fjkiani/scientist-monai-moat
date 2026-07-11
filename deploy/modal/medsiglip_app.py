"""Modal app: `google/medsiglip-448` GPU inference for OncologyArbiter.

Endpoints
---------
- `GET  /healthz`    → liveness (no model touch)
- `GET  /info`       → warm the model, return metadata (dim, device, resolution)
- `POST /embed`      → single DICOM (raw octet-stream) → 1152-dim vision embedding
- `POST /embed_batch`→ JSON `{dicoms_b64: [...]}` → list of embeddings
- `POST /zero_shot`  → JSON `{dicom_b64, prompts: [...]}` → SigLIP softmax scores

Design notes
------------
- Vision-tower only for /embed and /embed_batch — no text tokens, faster and
  avoids SiglipTokenizer entirely if we ever downgrade sentencepiece.
- DICOM decoding is embedded in-app (percentile [1%, 99%] window, MONOCHROME1
  invert, Modality LUT); we do NOT depend on any repo module to keep the Modal
  image self-contained.
- HF token is pulled from Modal secret `medsiglip-hf-token` (env: HF_TOKEN).
- Cold start ~90s (weight pull from HF CAS ≈3.3 GB). `min_containers=0` for cost.
- Vision embedding dim = 1152 (SigLIP-So400m/14 vision tower hidden_size).

Deploy: `modal deploy deploy/modal/medsiglip_app.py`
"""
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import modal

APP_VERSION = "medsiglip-modal-v0.4.0-alpha"

# ── Prod flip knob ───────────────────────────────────────────────────
# ``MEDSIGLIP_MODAL_MODE=prod`` deploys with min_containers=1 (keeps one
# warm replica during prod hours; ~$0.10/h idle A10G). Default remains
# min_containers=0 for zero-cost staging/dev deploys. Read at deploy
# time, so switching modes needs a re-deploy.
_MODAL_MODE = (os.environ.get("MEDSIGLIP_MODAL_MODE") or "staging").lower()
_MIN_CONTAINERS = 1 if _MODAL_MODE == "prod" else 0
_SCALEDOWN_S = 900 if _MODAL_MODE == "prod" else 300

# ── Image ────────────────────────────────────────────────────────────
# NB: sentencepiece + protobuf REQUIRED for SiglipTokenizer to import.
# Even if we only use the vision tower, `AutoModel.from_pretrained` still
# constructs the text tower unless we explicitly load `SiglipVisionModel`.
MEDSIGLIP_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "safetensors==0.4.5",
        "pillow==10.4.0",
        "numpy==1.26.4",
        "pydicom==2.4.4",
        "huggingface_hub==0.24.7",
        "fastapi==0.115.0",
        "sentencepiece==0.2.0",
        "protobuf==5.28.2",
    )
)

app = modal.App("medsiglip-448")
HF_SECRET = modal.Secret.from_name("medsiglip-hf-token")


# ── Standalone healthz (no GPU, no model) ─────────────────────────────
HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="medsiglip-healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": "medsiglip-448", "version": "v0.3.0"}


# ── GPU-backed class ─────────────────────────────────────────────────
@app.cls(
    image=MEDSIGLIP_IMAGE,
    gpu="A10G",
    secrets=[HF_SECRET],
    scaledown_window=_SCALEDOWN_S,
    timeout=180,
    min_containers=_MIN_CONTAINERS,
)
class MedSigLipModal:
    @modal.enter()
    def load(self) -> None:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModel, AutoProcessor

        t0 = time.time()

        # Fail-fast HEAD probe on the gated repo.
        token = os.environ.get("HF_TOKEN")
        assert token, "HF_TOKEN missing from Modal secret medsiglip-hf-token"
        api = HfApi(token=token)
        info = api.model_info("google/medsiglip-448")
        assert info is not None, "model_info returned None for google/medsiglip-448"

        # Load processor + model. AutoModel loads the full SigLIP two-tower
        # model; the text tower is only touched by /zero_shot.
        self.processor = AutoProcessor.from_pretrained(
            "google/medsiglip-448", token=token
        )
        self.model = AutoModel.from_pretrained(
            "google/medsiglip-448",
            token=token,
            torch_dtype=torch.float32,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Cache the vision-tower hidden size (embedding dim).
        vc = self.model.config.vision_config
        self.embedding_dim = int(getattr(vc, "hidden_size", 1152))
        self.input_resolution = int(getattr(vc, "image_size", 448))

        self.load_seconds = round(time.time() - t0, 3)
        self.warmed_at = time.time()

    # ─── DICOM decoding ──────────────────────────────────────────────
    def _dicom_bytes_to_pil(self, dcm_bytes: bytes):
        """Decode raw DICOM bytes → PIL.Image RGB 448x448-ready."""
        import numpy as np
        import pydicom
        from PIL import Image

        ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
        arr = ds.pixel_array.astype("float32")

        # Modality LUT
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        # MONOCHROME1 → invert so bright = high value
        photometric = str(
            getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        ).upper()
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr

        # Percentile [1%, 99%] windowing → [0, 1]
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        # Grayscale [0,1] → 8-bit → PIL RGB (replicate)
        arr8 = (arr * 255.0).astype("uint8")
        pil = Image.fromarray(arr8, mode="L").convert("RGB")
        return pil

    def _pixels_bytes_to_pil(self, img_bytes: bytes):
        """Decode PNG/JPEG bytes → PIL RGB. Used for already-processed images
        (e.g., the CBIS-DDSM_1024 PNG cohort) so callers do not need to
        re-encode as DICOM."""
        from PIL import Image

        pil = Image.open(io.BytesIO(img_bytes))
        # SigLIP wants 3-channel RGB. Grayscale mammograms become RGB by
        # channel replication (same as the DICOM path).
        return pil.convert("RGB")

    # ─── Core embed logic ────────────────────────────────────────────
    def _embed_pils(self, pils: List[Any]) -> List[List[float]]:
        import torch

        pixel_values = self.processor(
            images=pils, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)

        pooled = getattr(vision_out, "pooler_output", None)
        if pooled is None:
            # Fallback to CLS-equivalent from last hidden state
            pooled = vision_out.last_hidden_state[:, 0, :]

        return pooled.detach().cpu().float().tolist()

    # ─── /info ───────────────────────────────────────────────────────
    @modal.fastapi_endpoint(method="GET", label="medsiglip-info")
    def info(self) -> Dict[str, Any]:
        return {
            "model_repo": "google/medsiglip-448",
            "input_resolution": self.input_resolution,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "load_seconds": self.load_seconds,
            "warmed_at": self.warmed_at,
            "app_version": APP_VERSION,
        }

    # ─── /embed  (JSON body: `{dicom_b64: "..."}` ────────────────────
    #
    # Uniform JSON payload with the other endpoints — Modal's
    # @fastapi_endpoint doesn't reliably inject raw fastapi.Request on
    # class methods (it treats unknown types as query params). Base64
    # overhead (~33%) is negligible against total per-call latency.
    @modal.fastapi_endpoint(method="POST", label="medsiglip-embed")
    def embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON: `{"dicom_b64": "..."}` OR `{"pixels_b64": "..."}`.

        ``dicom_b64`` decodes as a DICOM (Modality LUT + windowing).
        ``pixels_b64`` decodes as a PNG/JPEG (RGB conversion only).
        """
        t0 = time.time()
        dicom_b64 = payload.get("dicom_b64")
        pixels_b64 = payload.get("pixels_b64")
        if dicom_b64 and pixels_b64:
            return {"error": "supply only one of dicom_b64 or pixels_b64"}
        if not dicom_b64 and not pixels_b64:
            return {"error": "dicom_b64 or pixels_b64 required"}
        try:
            if dicom_b64:
                raw = base64.b64decode(dicom_b64)
                pil = self._dicom_bytes_to_pil(raw)
                input_format = "dicom"
            else:
                raw = base64.b64decode(pixels_b64)
                pil = self._pixels_bytes_to_pil(raw)
                input_format = "pixels"
            embedding = self._embed_pils([pil])[0]
        except Exception as e:  # pragma: no cover - runtime shape only
            return {"error": f"{type(e).__name__}: {e}"}
        return {
            "embedding": embedding,
            "dim": len(embedding),
            "input_format": input_format,
            "seconds": round(time.time() - t0, 3),
            "app_version": APP_VERSION,
        }

    # ─── /embed_batch  (JSON body: base64 list) ─────────────────────
    @modal.fastapi_endpoint(method="POST", label="medsiglip-embed-batch")
    def embed_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON: `{"dicoms_b64": ["...", "..."]}`. Max 32 per call.

        For PNG/JPEG inputs (already-processed pixel arrays), use
        ``pixels_b64`` instead — this is honest about the format expected
        and lets Track V embed the CBIS-DDSM_1024 PNG cohort directly.
        """
        t0 = time.time()
        dicom_b64s = payload.get("dicoms_b64")
        pixel_b64s = payload.get("pixels_b64")
        if dicom_b64s and pixel_b64s:
            return {"error": "supply only one of dicoms_b64 or pixels_b64"}
        b64s = dicom_b64s or pixel_b64s or []
        if not isinstance(b64s, list) or not b64s:
            return {"error": "dicoms_b64 or pixels_b64 must be non-empty list"}
        if len(b64s) > 32:
            return {"error": "max batch size is 32"}
        try:
            if dicom_b64s is not None:
                pils = [self._dicom_bytes_to_pil(base64.b64decode(b)) for b in b64s]
            else:
                pils = [self._pixels_bytes_to_pil(base64.b64decode(b)) for b in b64s]
            embeddings = self._embed_pils(pils)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {
            "embeddings": embeddings,
            "n": len(embeddings),
            "dim": len(embeddings[0]) if embeddings else 0,
            "input_format": "dicom" if dicom_b64s is not None else "pixels",
            "seconds": round(time.time() - t0, 3),
            "app_version": APP_VERSION,
        }

    # ─── /zero_shot  (JSON body: dicom_b64 OR pixels_b64 + prompts) ──
    @modal.fastapi_endpoint(method="POST", label="medsiglip-zero-shot")
    def zero_shot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON:
        ``{"dicom_b64": "...", "prompts": [...]}`` OR
        ``{"pixels_b64": "...", "prompts": [...]}``.
        Returns sigmoid probabilities per prompt (SigLIP semantics).
        """
        import torch

        t0 = time.time()
        dicom_b64 = payload.get("dicom_b64")
        pixels_b64 = payload.get("pixels_b64")
        prompts: List[str] = payload.get("prompts", [])
        if dicom_b64 and pixels_b64:
            return {"error": "supply only one of dicom_b64 or pixels_b64"}
        if not dicom_b64 and not pixels_b64:
            return {"error": "dicom_b64 or pixels_b64 required"}
        if not isinstance(prompts, list) or not prompts:
            return {"error": "prompts must be non-empty list"}
        if len(prompts) > 32:
            return {"error": "max 32 prompts"}
        try:
            if dicom_b64:
                pil = self._dicom_bytes_to_pil(base64.b64decode(dicom_b64))
                input_format = "dicom"
            else:
                pil = self._pixels_bytes_to_pil(base64.b64decode(pixels_b64))
                input_format = "pixels"
        except Exception as e:
            return {"error": f"decode: {type(e).__name__}: {e}"}

        try:
            inputs = self.processor(
                text=prompts,
                images=[pil],
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # [1, n_prompts]
            # SigLIP uses SIGMOID (independent binary classifiers), not softmax.
            probs = torch.sigmoid(logits).detach().cpu().float().tolist()[0]
        except Exception as e:
            return {"error": f"forward: {type(e).__name__}: {e}"}

        return {
            "prompts": prompts,
            "probs": probs,
            "top": prompts[int(max(range(len(probs)), key=lambda i: probs[i]))],
            "input_format": input_format,
            "seconds": round(time.time() - t0, 3),
            "app_version": APP_VERSION,
        }
