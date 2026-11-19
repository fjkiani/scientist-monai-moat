"""MedGemma 1.5 4B chat transport: HTTP first, Modal SDK `.remote()` fallback.

Modal public HTTP can return ``workspace … is disabled`` while the same app
remains callable via ``modal.Cls.from_name(...).generate.remote(...)``.
Therapy + pathology share this path so L4c does not depend on TxGemma.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class MedGemmaChatError(RuntimeError):
    pass


def _http_chat(
    chat_url: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    request = urllib.request.Request(
        chat_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def _sdk_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    app_name: str = "medgemma-1-5-4b",
    class_name: str = "MedGemma",
) -> dict[str, Any]:
    import modal

    profile = (
        os.environ.get("MODAL_PROFILE")
        or os.environ.get("MEDGEMMA_MODAL_PROFILE")
        or "testing1235"
    )
    # Ensure SDK uses the intended workspace even if process default differs.
    os.environ["MODAL_PROFILE"] = profile
    Cls = modal.Cls.from_name(app_name, class_name)
    out = Cls().generate.remote(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not isinstance(out, dict):
        raise MedGemmaChatError(f"medgemma sdk returned non-dict: {type(out)}")
    out = {**out, "transport": out.get("transport") or "modal_sdk_generate"}
    return out


def medgemma_chat(
    messages: list[dict[str, str]],
    *,
    chat_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
    prefer_sdk: bool | None = None,
) -> dict[str, Any]:
    """Call MedGemma chat; fall back to Modal SDK when HTTP is disabled."""
    if not messages:
        raise MedGemmaChatError("medgemma_chat: empty messages")

    force_sdk = prefer_sdk
    if force_sdk is None:
        force_sdk = (
            os.environ.get("MEDGEMMA_FORCE_SDK", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    last_err: Exception | None = None
    if not force_sdk:
        try:
            data = _http_chat(
                chat_url,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            if data.get("error"):
                raise MedGemmaChatError(f"medgemma HTTP error: {data['error']}")
            return {**data, "transport": data.get("transport") or "http"}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            # Workspace-disabled / 404 / network → SDK path
            detail = str(exc).lower()
            if not (
                "disabled" in detail
                or "404" in detail
                or "httperror" in type(exc).__name__.lower()
                or isinstance(exc, (urllib.error.URLError, urllib.error.HTTPError, TimeoutError))
                or isinstance(exc, MedGemmaChatError)
            ):
                # Still try SDK for any transport failure
                pass

    sdk_err: Exception | None = None
    try:
        return _sdk_chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as exc:  # noqa: BLE001
        sdk_err = exc

    # Modal workspace execution can be disabled while the weight volume remains
    # readable — fall back to local identity-locked snapshot (MPS/CUDA/CPU).
    try:
        from oncology_arbiter.nlp.medgemma_local_runtime import local_chat

        return local_chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as local_exc:  # noqa: BLE001
        raise MedGemmaChatError(
            "medgemma chat failed HTTP+SDK+local: "
            f"http={last_err!r}; sdk={sdk_err!r}; "
            f"local={type(local_exc).__name__}: {local_exc}"
        ) from local_exc
