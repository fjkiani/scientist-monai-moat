"""Modal app: MedGemma-27b-it chat inference for medical reasoning.

Google publishes MedGemma at ``google/medgemma-27b-it`` (HAI-DEF gated,
accepted for the shipping HF token). This Modal deployment hosts a chat-only
server that the local :class:`oncology_arbiter.models.llm_client.LLMClient`
can call when ``MEDGEMMA_MODAL_URL`` is set.

Contract
--------
- ``GET  /healthz``        → liveness, no model touch
- ``GET  /info``           → {model_repo, num_params, device, dtype}
- ``POST /medgemma-chat``  → JSON body:
    ``{"messages": [{"role": "user|system|assistant", "content": str}],
       "max_tokens": int (default 512),
       "temperature": float (default 0.2)}``
   →
    ``{"text": str,
       "model": "google/medgemma-27b-it",
       "prompt_tokens": int,
       "completion_tokens": int,
       "honesty_warning": str,
       "app_version": str}``

Design notes
------------
- 27B params in bf16 = ~54 GB → needs A100-80GB (Modal ``a100-80gb``).
- HF weights pulled from ``google/medgemma-27b-it`` via CAS on cold start;
  ~2-3 min including weight load.
- Uses Modal secret ``huggingface-secret`` (env ``HF_TOKEN``).
- ``min_containers=0`` → idle cost $0. Cold start ~180 s.
- Honesty warning is ALWAYS attached to every response; the client MUST
  surface it verbatim in the user-facing response.

Deploy
------
    modal deploy deploy/modal/medgemma_27b_app.py

Endpoint URLs after deploy (with ``fjkiani`` account under ``crispro`` profile):
    https://crispro--medgemma-27b-healthz.modal.run
    https://crispro--medgemma-27b-info.modal.run
    https://crispro--medgemma-27b-chat.modal.run

RESEARCH USE ONLY — Google MedGemma card:
https://huggingface.co/google/medgemma-27b-it

MedGemma is a research model. It has NOT been prospectively validated for
patient care. Any output must be treated as a suggestion for a licensed
clinician's review, never as an autonomous clinical decision.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import modal


APP_VERSION = "medgemma-27b-modal-v0.5.0-production-ready"
MODEL_REPO = "google/medgemma-27b-it"

MEDGEMMA_HONESTY_WARNING = (
    "MedGemma is a Google research LLM (google/medgemma-27b-it, HAI-DEF "
    "gated). Its outputs are recommendations from a generative language "
    "model, NOT verified clinical decisions. It has not been prospectively "
    "validated for patient care. Any use requires a certified clinician's "
    "review. RESEARCH USE ONLY."
)


# ── Image ────────────────────────────────────────────────────────────
MEDGEMMA_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "git")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "protobuf==5.28.1",
        "hf-transfer==0.1.8",
        "fastapi[standard]==0.115.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


app = modal.App("medgemma-27b")
model_volume = modal.Volume.from_name("medgemma-27b-weights", create_if_missing=True)


@app.cls(
    image=MEDGEMMA_IMAGE,
    gpu="a100-80gb",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=1800,
    scaledown_window=900,
    min_containers=1,
)
class MedGemma:
    """Wraps a MedGemma-27b-it inference server."""

    @modal.enter()
    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        t0 = time.time()
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set in Modal secret")

        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, token=token, use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            token=token,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.load_seconds = round(time.time() - t0, 3)

    @modal.method()
    def info(self) -> Dict[str, Any]:
        import torch
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "app": "medgemma-27b",
            "app_version": APP_VERSION,
            "model": MODEL_REPO,
            "num_params": num_params,
            "device": self.device,
            "dtype": str(self.dtype),
            "load_seconds": getattr(self, "load_seconds", None),
            "torch_version": torch.__version__,
            "honesty_warning": MEDGEMMA_HONESTY_WARNING,
        }

    @modal.method()
    def ready(self) -> Dict[str, Any]:
        """Touch a loaded model parameter; this is readiness, not CPU liveness."""
        import torch

        probe = self.model.get_input_embeddings().weight[0, 0].detach()
        return {
            "status": "ready" if bool(torch.isfinite(probe).item()) else "not_ready",
            "model_loaded": True,
            "model": MODEL_REPO,
            "app_version": APP_VERSION,
            "device": str(probe.device),
            "load_seconds": getattr(self, "load_seconds", None),
            "honesty_warning": MEDGEMMA_HONESTY_WARNING,
        }

    @modal.method()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        import torch

        # Build the chat prompt with the model's chat template.
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt",
        ).input_ids.to(self.device)

        prompt_tokens = int(input_ids.shape[1])
        t0 = time.time()
        with torch.inference_mode():
            gen = self.model.generate(
                input_ids,
                max_new_tokens=int(max_tokens),
                do_sample=(temperature > 0.0),
                temperature=float(temperature),
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_s = round(time.time() - t0, 3)

        completion_ids = gen[0, prompt_tokens:]
        completion_text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True,
        ).strip()
        completion_tokens = int(completion_ids.shape[0])

        return {
            "text": completion_text,
            "request_id": str(uuid.uuid4()),
            "model": MODEL_REPO,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_s": latency_s,
            "honesty_warning": MEDGEMMA_HONESTY_WARNING,
            "app_version": APP_VERSION,
        }


# ── HTTP endpoints ──────────────────────────────────────────────────
@app.function(image=MEDGEMMA_IMAGE, min_containers=0)
@modal.fastapi_endpoint(method="GET", label="medgemma-27b-healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "medgemma-27b",
        "app_version": APP_VERSION,
        "disclaimer": "Research Use Only. Not FDA-cleared. Not CE-marked.",
    }


@app.function(image=MEDGEMMA_IMAGE, min_containers=0, timeout=1800)
@modal.fastapi_endpoint(method="GET", label="medgemma-27b-info")
def info() -> Dict[str, Any]:
    return MedGemma().info.remote()


@app.function(image=MEDGEMMA_IMAGE, min_containers=0, timeout=1800)
@modal.fastapi_endpoint(method="GET", label="medgemma-27b-ready")
def ready() -> Dict[str, Any]:
    return MedGemma().ready.remote()


@app.function(image=MEDGEMMA_IMAGE, min_containers=0, timeout=1800)
@modal.fastapi_endpoint(method="POST", label="medgemma-27b-chat")
def chat(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return {"error": "messages must be a non-empty list"}
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.2))
    return MedGemma().chat.remote(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
