"""Local MedGemma 1.5 4B chat when Modal workspace execution is disabled.

Loads the identity-locked HF snapshot from ``HF_HOME`` / ``MEDGEMMA_LOCAL_DIR``
(same revision as ``models/medgemma_1_5_4b_identity.json``). Used as a last-
resort transport so L4c can stay live during Modal HTTP/SDK workspace locks.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
    EXPECTED_MODEL_ID,
    EXPECTED_REVISION_SHA256,
    verify_artifact_identity,
)

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"tokenizer": None, "model": None, "device": None}


def _resolve_model_dir() -> Path:
    override = (os.environ.get("MEDGEMMA_LOCAL_DIR") or "").strip()
    if override:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"MEDGEMMA_LOCAL_DIR missing: {path}")
        return path
    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    snap = (
        hf_home
        / "hub"
        / f"models--{EXPECTED_MODEL_ID.replace('/', '--')}"
        / "snapshots"
    )
    if not snap.is_dir():
        raise FileNotFoundError(f"local MedGemma snapshot root missing: {snap}")
    # Prefer identity hf_commit when present.
    identity = verify_artifact_identity()
    commit = str(identity.get("hf_commit") or "").strip()
    if commit and (snap / commit).is_dir():
        return snap / commit
    kids = [p for p in snap.iterdir() if p.is_dir()]
    if not kids:
        raise FileNotFoundError(f"no snapshots under {snap}")
    return kids[0]


def _ensure_loaded() -> tuple[Any, Any, str]:
    with _LOCK:
        if _STATE["model"] is not None:
            return _STATE["tokenizer"], _STATE["model"], _STATE["device"]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = _resolve_model_dir()
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        except Exception:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        model.to(device)
        model.eval()
        _STATE.update({"tokenizer": tokenizer, "model": model, "device": device})
        return tokenizer, model, device


def local_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Generate chat text from the local identity-locked MedGemma snapshot."""
    identity = verify_artifact_identity()
    tokenizer, model, device = _ensure_loaded()
    import torch

    encoded = None
    for kwargs in ({"enable_thinking": False}, {"thinking": False}, {}):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            break
        except TypeError:
            continue
    if encoded is None:
        raise RuntimeError("local MedGemma apply_chat_template failed")
    inputs = {k: v.to(device) for k, v in encoded.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "do_sample": float(temperature) > 0.0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if float(temperature) > 0.0:
        gen_kwargs["temperature"] = float(temperature)
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, **gen_kwargs)
    latency_s = round(time.time() - t0, 3)
    completion_ids = gen[0, prompt_tokens:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    if "{" not in text:
        text = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    return {
        "text": text,
        "request_id": str(uuid.uuid4()),
        "model": EXPECTED_MODEL_ID,
        "model_id": EXPECTED_MODEL_ID,
        "revision_sha256": EXPECTED_REVISION_SHA256,
        "hf_commit": identity.get("hf_commit"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion_ids.shape[0]),
        "latency_s": latency_s,
        "app_version": identity.get("app_version"),
        "honesty_warning": identity.get("honesty_warning"),
        "transport": f"local_{device}",
        "quantization": str(model.dtype),
    }
