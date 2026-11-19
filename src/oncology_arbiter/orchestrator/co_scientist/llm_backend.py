"""Agnostic L5 LLM backend factory.

Backends share the GemmaClient-shaped surface: ``.chat(messages, max_tokens,
temperature) -> LlmResponse``. Swap with:

    CO_SCIENTIST_LLM_BACKEND=medgemma|gemini_argon

Live mode engages when the selected backend is configured (MedGemma URL or
``GEMINI_API_KEY``). Engine id is ``{backend}_clinical_reasoning_swarm``.
"""
from __future__ import annotations

import os
from typing import Any, Literal

from .const import (
    ENGINE_OFFLINE,
    LLM_BACKEND_GEMINI_ARGON,
    LLM_BACKEND_MEDGEMMA,
    engine_id_for_backend,
)

BackendName = Literal["medgemma", "gemini_argon"]


def resolve_backend(explicit: str | None = None) -> BackendName:
    raw = (explicit or os.environ.get("CO_SCIENTIST_LLM_BACKEND") or "").strip().lower()
    if raw in {LLM_BACKEND_MEDGEMMA, LLM_BACKEND_GEMINI_ARGON}:
        return raw  # type: ignore[return-value]
    # Auto: prefer explicit Gemini switch when key present and MedGemma unset.
    if (os.environ.get("GEMINI_API_KEY") or "").strip() and not (
        os.environ.get("MEDGEMMA_MODAL_URL") or ""
    ).strip():
        return LLM_BACKEND_GEMINI_ARGON  # type: ignore[return-value]
    return LLM_BACKEND_MEDGEMMA  # type: ignore[return-value]


def backend_is_configured(backend: BackendName) -> bool:
    if backend == LLM_BACKEND_GEMINI_ARGON:
        return bool((os.environ.get("GEMINI_API_KEY") or "").strip())
    return bool((os.environ.get("MEDGEMMA_MODAL_URL") or "").strip())


def resolve_live_available() -> bool:
    return backend_is_configured(resolve_backend())


def build_live_llm(
    *,
    backend: BackendName | None = None,
    timeout_s: float = 180.0,
) -> tuple[Any, str, str]:
    """Return ``(client, engine_id, backend_name)`` for the live swarm."""
    chosen = resolve_backend(backend)
    if chosen == LLM_BACKEND_GEMINI_ARGON:
        from oncology_arbiter.models.gemini_argon_client import GeminiArgonClient

        client = GeminiArgonClient(timeout_s=timeout_s)
        return client, engine_id_for_backend(chosen), chosen

    from oncology_arbiter.models.llm_client import GemmaClient
    from oncology_arbiter.models.medgemma_1_5_4b_wiring import assert_pinned_chat_url

    assert_pinned_chat_url((os.environ.get("MEDGEMMA_MODAL_URL") or "").strip())
    client = GemmaClient(timeout_s=timeout_s)
    return client, engine_id_for_backend(LLM_BACKEND_MEDGEMMA), LLM_BACKEND_MEDGEMMA


def is_live_swarm_engine(engine: str | None) -> bool:
    if not engine or engine == ENGINE_OFFLINE:
        return False
    return str(engine).endswith("_clinical_reasoning_swarm")
