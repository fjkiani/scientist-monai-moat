"""Modal app: TxGemma-9B chat inference for therapy reasoning.

Google publishes TxGemma at ``google/txgemma-9b-chat`` (HAI-DEF gated). This
Modal deployment hosts a chat-only server; the local client
``oncology_arbiter.models.txgemma_client`` can then swap between the local
``transformers`` load and this remote endpoint via
``TXGEMMA_BACKEND=modal``.

Contract mirror of ``medsiglip_app.py``
---------------------------------------
- ``GET  /healthz``  → liveness, no model touch
- ``GET  /info``     → warm model, return {model_repo, dim, device}
- ``POST /reason``   → JSON ``{prompt, max_new_tokens, temperature}`` →
                       ``{text, model_repo, honesty_warning, app_version}``

Design notes
------------
- We load ``google/txgemma-9b-chat`` with ``AutoModelForCausalLM`` +
  ``AutoTokenizer`` and run bf16 on A10G. Cold start ≈120s including
  weight pull from HF CAS (≈18 GB).
- HF token is pulled from Modal secret ``txgemma-hf-token`` (env ``HF_TOKEN``).
- ``min_containers=0`` for cost. Bump to ``1`` post-flip for prod hours.
- Honesty warning is always attached to responses.

Deploy
------
    modal deploy deploy/modal/txgemma_app.py

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import modal


APP_VERSION = "txgemma-modal-v0.4.0-alpha"

TXGEMMA_HONESTY_WARNING = (
    "TxGemma is a Google research LLM (HAI-DEF gated). Its outputs are "
    "recommendations from a generative language model, NOT verified "
    "clinical decisions. It MUST NOT be used to make treatment choices. "
    "Real clinical use requires a certified breast oncologist and a full "
    "guideline consultation. RESEARCH USE ONLY."
)


# ── Image ────────────────────────────────────────────────────────────
TXGEMMA_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "protobuf==5.28.2",
        "huggingface_hub==0.24.7",
        "fastapi==0.115.0",
        "pydantic==2.9.2",
    )
)

app = modal.App("txgemma-9b")
HF_SECRET = modal.Secret.from_name("txgemma-hf-token")


# ── Healthz (no GPU, no model) ───────────────────────────────────────
HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="txgemma-healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": "txgemma-9b", "version": APP_VERSION}


# ── GPU class ────────────────────────────────────────────────────────
@app.cls(
    image=TXGEMMA_IMAGE,
    gpu="A10G",
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=180,
    min_containers=0,
    memory=32 * 1024,
)
class TxGemma:
    """Chat-only TxGemma-9B loaded from HF CAS on first request."""

    model_repo: str = "google/txgemma-9b-chat"

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN not present in txgemma-hf-token secret")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo, token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        self.device = "cuda"

    @modal.fastapi_endpoint(method="GET", label="txgemma-info")
    def info(self) -> Dict[str, Any]:
        import torch

        return {
            "app": "txgemma-9b",
            "app_version": APP_VERSION,
            "model_repo": self.model_repo,
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "cuda_available": bool(torch.cuda.is_available()),
            "honesty_warning": TXGEMMA_HONESTY_WARNING,
        }

    @modal.fastapi_endpoint(method="POST", label="txgemma-reason")
    def reason(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run a therapy-planning prompt.

        Payload
        -------
        prompt : str
            Conversational prompt including patient context.
        max_new_tokens : int, default 512
        temperature : float, default 0.0 (deterministic)
        """
        import torch

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        max_new = int(payload.get("max_new_tokens", 512))
        temperature = float(payload.get("temperature", 0.0))

        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-6),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # strip prompt from generated
        prompt_len = inputs["input_ids"].shape[1]
        gen = out_ids[0, prompt_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "text": text,
            "elapsed_ms": elapsed_ms,
            "model_repo": self.model_repo,
            "app_version": APP_VERSION,
            "honesty_warning": TXGEMMA_HONESTY_WARNING,
        }


@app.local_entrypoint()
def main(prompt: str = "Recommend adjuvant therapy for a 62F ER+/HER2- pT2N1 breast cancer patient.") -> None:
    tx = TxGemma()
    tx.load.remote()
    result = tx.reason.remote({"prompt": prompt, "max_new_tokens": 200})
    import json
    print(json.dumps(result, indent=2))
