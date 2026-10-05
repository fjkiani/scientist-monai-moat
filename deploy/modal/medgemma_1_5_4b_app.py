"""Modal app: genuine MedGemma 1.5 4B-IT chat (RUO).

Cloned from the working ``medgemma_27b_app.py`` load/chat path:
``AutoTokenizer`` + ``AutoModelForCausalLM`` + string-content chat template.
Bans Qwen / TxGemma / AutoProcessor multimodal path (that path crashes on
string ``content`` with TypeError in processing_utils).

Model: ``google/medgemma-1.5-4b-it`` — bf16 on A10G (fits; no 8-bit needed).

Contract
--------
- GET  medgemma-1-5-4b-healthz
- GET  medgemma-1-5-4b-info
- GET  medgemma-1-5-4b-ready
- POST medgemma-1-5-4b-chat
    IN:  {"messages":[{"role","content"}], "max_tokens", "temperature"}
    OUT: {"text","model","prompt_tokens","completion_tokens",
          "honesty_warning","app_version","quantization","revision_sha256"}

Deploy::
    MODAL_PROFILE=fjkiani modal deploy deploy/modal/medgemma_1_5_4b_app.py
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Dict

import modal

APP_VERSION = "medgemma-1-5-4b-modal-v1.1.0-causal-clone"
MODEL_REPO = "google/medgemma-1.5-4b-it"
QUANTIZATION = "bf16"
HF_SECRET_NAME = os.environ.get("MEDGEMMA_HF_SECRET", "hf_token")

MEDGEMMA_HONESTY_WARNING = (
    "MedGemma 1.5 4B-IT is a Google research LLM (google/medgemma-1.5-4b-it, "
    "HAI-DEF gated, bf16 on A10G). Outputs are generative suggestions, NOT "
    "verified clinical decisions. RESEARCH USE ONLY."
)

MEDGEMMA_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "git")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "tokenizers==0.21.2",
        "accelerate==1.5.2",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "protobuf==5.29.4",
        "jinja2==3.1.6",
        "hf-transfer==0.1.9",
        "fastapi[standard]==0.115.4",
        "requests==2.32.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("medgemma-1-5-4b")
model_volume = modal.Volume.from_name("medgemma-1-5-4b-weights", create_if_missing=True)


def _revision_sha256_from_config(token: str) -> tuple[str, str]:
    """Return (hf_commit_40, content_sha256_of_config.json) for identity bind."""
    from huggingface_hub import HfApi
    import requests

    info = HfApi().model_info(MODEL_REPO, token=token)
    commit = info.sha
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{commit}/config.json"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    return commit, hashlib.sha256(r.content).hexdigest()


@app.function(image=MEDGEMMA_IMAGE, min_containers=0)
@modal.fastapi_endpoint(method="GET", label="medgemma-1-5-4b-healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "medgemma-1-5-4b",
        "app_version": APP_VERSION,
        "model_repo": MODEL_REPO,
        "quantization": QUANTIZATION,
        "disclaimer": "Research Use Only. Not FDA-cleared. Not CE-marked.",
    }


@app.cls(
    image=MEDGEMMA_IMAGE,
    gpu="A10G",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=1800,
    scaledown_window=900,
    min_containers=1,
)
class MedGemma:
    """Same class/layout as working medgemma-27b MedGemma cls."""

    @modal.enter()
    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(f"HF_TOKEN not set in Modal secret {HF_SECRET_NAME}")

        self.hf_commit, self.revision_sha256 = _revision_sha256_from_config(token)
        self.device = "cuda:0"

        # 27b-clone path: tokenizer + CausalLM only (NO AutoProcessor).
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=token)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_REPO,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.model_class = "AutoModelForCausalLM"
        except Exception as causal_exc:
            # Multimodal Gemma3 checkpoint: load generative class, keep tokenizer.
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_REPO,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.model_class = f"AutoModelForImageTextToText(fallback:{type(causal_exc).__name__})"

        self.model.eval()

        # Decoder-only: never feed config pad_id=0 into generate (Gemma3 then
        # emits <pad> forever under broken/edge configs). Use eos as pad.
        eos_id = int(self.tokenizer.eos_token_id)
        self.tokenizer.pad_token_id = eos_id
        self._gen_pad_id = eos_id
        self._gen_eos_id = eos_id
        gc = self.model.generation_config
        gc.pad_token_id = eos_id
        gc.eos_token_id = eos_id
        gc.do_sample = False
        for attr in ("top_p", "top_k", "temperature"):
            if hasattr(gc, attr):
                setattr(gc, attr, None)

        smoke_msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
        enc = self.tokenizer.apply_chat_template(
            smoke_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self._gen_pad_id,
                eos_token_id=self._gen_eos_id,
            )
        new = out[0, enc["input_ids"].shape[1] :]
        smoke_text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        if not smoke_text:
            raw = self.tokenizer.decode(new, skip_special_tokens=False)
            raise RuntimeError(
                f"enter-smoke produced empty text; raw={raw!r} ids={new.tolist()}"
            )
        with torch.inference_mode():
            logits = self.model(**enc).logits[0, -1].float()
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("enter-smoke: non-finite logits on prompt end")
        self.smoke_text = smoke_text
        self.load_seconds = round(time.time() - t0, 3)

    @modal.fastapi_endpoint(method="GET", label="medgemma-1-5-4b-info")
    def info(self) -> Dict[str, Any]:
        import torch

        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "app": "medgemma-1-5-4b",
            "app_version": APP_VERSION,
            "model": MODEL_REPO,
            "model_id": MODEL_REPO,
            "model_class": self.model_class,
            "hf_commit": self.hf_commit,
            "revision_sha256": self.revision_sha256,
            "quantization": QUANTIZATION,
            "num_params": num_params,
            "device": self.device,
            "load_seconds": getattr(self, "load_seconds", None),
            "enter_smoke_text": getattr(self, "smoke_text", None),
            "torch_version": torch.__version__,
            "honesty_warning": MEDGEMMA_HONESTY_WARNING,
        }

    @modal.fastapi_endpoint(method="GET", label="medgemma-1-5-4b-ready")
    def ready(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "model_loaded": True,
            "model": MODEL_REPO,
            "model_id": MODEL_REPO,
            "revision_sha256": self.revision_sha256,
            "hf_commit": self.hf_commit,
            "app_version": APP_VERSION,
            "quantization": QUANTIZATION,
            "device": self.device,
            "load_seconds": getattr(self, "load_seconds", None),
            "enter_smoke_text": getattr(self, "smoke_text", None),
            "honesty_warning": MEDGEMMA_HONESTY_WARNING,
        }

    @modal.fastapi_endpoint(method="POST", label="medgemma-1-5-4b-chat")
    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        try:
            if not isinstance(payload, dict):
                return {"error": "expected JSON body object"}
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                return {"error": "messages must be a non-empty list"}
            # Normalize: string content only (27b contract). Reject list parts.
            norm = []
            for m in messages:
                if not isinstance(m, dict) or "role" not in m:
                    return {"error": "each message needs role+content"}
                content = m.get("content")
                if isinstance(content, list):
                    # flatten multimodal parts → text only
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(str(part.get("text", "")))
                        elif isinstance(part, str):
                            texts.append(part)
                    content = "\n".join(t for t in texts if t)
                if not isinstance(content, str):
                    return {"error": "content must be a string (text-only chat)"}
                norm.append({"role": m["role"], "content": content})

            max_tokens = int(payload.get("max_tokens", 512))
            temperature = float(payload.get("temperature", 0.2))

            encoded = self.tokenizer.apply_chat_template(
                norm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda:0") for k, v in encoded.items()}
            prompt_tokens = int(inputs["input_ids"].shape[1])

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": self._gen_pad_id,
                "eos_token_id": self._gen_eos_id,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
                gen_kwargs["top_k"] = 64
            else:
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None
                gen_kwargs["temperature"] = None

            t0 = time.time()
            with torch.inference_mode():
                gen = self.model.generate(**inputs, **gen_kwargs)
            latency_s = round(time.time() - t0, 3)

            completion_ids = gen[0, prompt_tokens:]
            completion_text = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()

            return {
                "text": completion_text,
                "request_id": str(uuid.uuid4()),
                "model": MODEL_REPO,
                "model_id": MODEL_REPO,
                "revision_sha256": self.revision_sha256,
                "hf_commit": self.hf_commit,
                "quantization": QUANTIZATION,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(completion_ids.shape[0]),
                "latency_s": latency_s,
                "honesty_warning": MEDGEMMA_HONESTY_WARNING,
                "app_version": APP_VERSION,
            }
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "app_version": APP_VERSION,
                "model": MODEL_REPO,
                "quantization": QUANTIZATION,
            }


@app.function(image=MEDGEMMA_IMAGE, timeout=300)
def deployment_fingerprint() -> Dict[str, Any]:
    """Stable sha256 digest of the deployed Modal image software surface."""
    import pathlib
    import subprocess

    os_release = pathlib.Path("/etc/os-release").read_text()
    freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    material = (
        f"app={APP_VERSION}\nmodel={MODEL_REPO}\nquant={QUANTIZATION}\n"
        f"{os_release}\n{freeze}"
    )
    digest = "sha256:" + hashlib.sha256(material.encode()).hexdigest()
    return {
        "deployment_image_digest": digest,
        "app_version": APP_VERSION,
        "model_id": MODEL_REPO,
        "pip_freeze_packages": len([ln for ln in freeze.splitlines() if ln.strip()]),
    }


@app.local_entrypoint()
def main() -> None:
    """Identity handshake entrypoint for tonight_delivery packaging."""
    import json

    import requests

    fp = deployment_fingerprint.remote()
    info_url = "https://crispro--medgemma-1-5-4b-info.modal.run"
    chat_url = "https://crispro--medgemma-1-5-4b-chat.modal.run"
    info = requests.get(info_url, timeout=180).json()
    chat = requests.post(
        chat_url,
        json={
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 32,
            "temperature": 0.0,
        },
        timeout=180,
    ).json()
    payload = {
        "status": "ok",
        "model_id": info.get("model_id") or MODEL_REPO,
        "revision_sha256": info.get("revision_sha256"),
        "hf_commit": info.get("hf_commit"),
        "deployment_image_digest": fp["deployment_image_digest"],
        "info_url": info_url,
        "chat_url": chat_url,
        "chat_sla_seconds": 120,
        "app_version": APP_VERSION,
        "quantization": QUANTIZATION,
        "num_params": info.get("num_params"),
        "chat_smoke": {
            "latency_s": chat.get("latency_s"),
            "has_text": bool(chat.get("text")),
            "error": chat.get("error"),
            "revision_sha256": chat.get("revision_sha256"),
        },
        "fingerprint": fp,
        "info": info,
    }
    print(json.dumps(payload, indent=2))
