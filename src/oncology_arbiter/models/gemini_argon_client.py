"""Gemini Argon client — GemmaClient-shaped surface for L5 Co-Scientist.

Transport (``CO_SCIENTIST_GEMINI_TRANSPORT``):

  * ``model`` (default) — ``interactions.create(model=…)`` sync JSON judge calls
  * ``agent`` — Antigravity agent interaction with Zeta system instruction
  * ``generate`` — classic ``models.generate_content`` fallback

Requires ``GEMINI_API_KEY``. Never log or persist the key.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from oncology_arbiter.models.llm_client import LlmResponse, LlmUnavailable, UsageLedger
from oncology_arbiter.orchestrator.co_scientist.const import (
    GEMINI_ARGON_DEFAULT_AGENT,
    GEMINI_ARGON_DEFAULT_MODEL,
    GEMINI_ARGON_MAX_POLL,
    GEMINI_ARGON_POLL_SECONDS,
    ZETA_SYSTEM_INSTRUCTION,
)

__all__ = ["GeminiArgonClient"]


def _normalize_api_key(raw: str) -> str:
    key = (raw or "").strip()
    # Common paste typo: leading extra "A" → "AAIza…" instead of "AIza…"
    if key.startswith("AAIza"):
        key = key[1:]
    return key


class GeminiArgonClient:
    """Drop-in live LLM for pairwise judge / generative evolve."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        transport: Optional[str] = None,
        timeout_s: float = 180.0,
        system_instruction: Optional[str] = None,
        deterministic: bool = True,
    ) -> None:
        key = _normalize_api_key(api_key or os.environ.get("GEMINI_API_KEY") or "")
        if not key:
            raise LlmUnavailable("GEMINI_API_KEY is required for gemini_argon backend")
        try:
            from google import genai
        except ImportError as exc:
            raise LlmUnavailable(
                "google-genai package missing; install google-genai"
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = (
            model
            or os.environ.get("GEMINI_ARGON_MODEL")
            or GEMINI_ARGON_DEFAULT_MODEL
        ).strip()
        self.agent = (
            agent
            or os.environ.get("GEMINI_ARGON_AGENT")
            or GEMINI_ARGON_DEFAULT_AGENT
        ).strip()
        self.transport = (
            transport
            or os.environ.get("CO_SCIENTIST_GEMINI_TRANSPORT")
            or "model"
        ).strip().lower()
        self.timeout_s = float(timeout_s)
        self.system_instruction = system_instruction or ZETA_SYSTEM_INSTRUCTION
        self.deterministic = bool(deterministic)
        self.usage = UsageLedger()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LlmResponse:
        eff_temperature = 0.0 if self.deterministic else float(temperature)
        system_parts: list[str] = [self.system_instruction]
        user_parts: list[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").lower()
            content = str(msg.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        system = "\n\n".join(p for p in system_parts if p.strip())
        user = "\n\n".join(user_parts) if user_parts else ""

        started = time.time()
        try:
            if self.transport == "agent":
                text, request_id = self._chat_agent(system=system, user=user)
                route = "gemini_argon_agent"
            elif self.transport == "generate":
                text, request_id = self._chat_generate(
                    system=system, user=user, max_tokens=max_tokens, temperature=eff_temperature,
                )
                route = "gemini_argon_generate"
            else:
                text, request_id = self._chat_interaction_model(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=eff_temperature,
                )
                route = "gemini_argon_model"
        except LlmUnavailable:
            self.usage.record_failure("gemini_argon")
            raise
        except Exception as exc:
            self.usage.record_failure("gemini_argon")
            raise LlmUnavailable(f"gemini_argon failed: {type(exc).__name__}: {exc}") from exc

        latency = time.time() - started
        if not (text or "").strip():
            raise LlmUnavailable("gemini_argon returned empty text")
        resp = LlmResponse(
            text=text.strip(),
            route=route,
            model=self.model if self.transport != "agent" else f"agent:{self.agent}",
            prompt_tokens=0,
            completion_tokens=0,
            thinking_tokens=0,
            est_cost_usd=0.0,
            latency_s=latency,
            app_version="gemini-argon-l5-v1",
            request_id=request_id,
            honesty_warning=(
                "Gemini Argon output is generative RUO — not FDA-cleared; "
                "not for clinical decision-making."
            ),
        )
        self.usage.record(resp)
        return resp

    def _chat_generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str | None]:
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": system,
        }
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        text = getattr(resp, "text", None) or ""
        req_id = getattr(resp, "response_id", None) or getattr(resp, "id", None)
        return str(text), str(req_id) if req_id else None

    def _chat_interaction_model(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str | None]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": user,
            "system_instruction": system,
            "background": False,
        }
        # generation_config is optional; tolerate SDK drift
        try:
            kwargs["generation_config"] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        except Exception:
            pass
        interaction = self._client.interactions.create(**kwargs)
        interaction = self._await_interaction(interaction)
        text = getattr(interaction, "output_text", None) or ""
        return str(text), getattr(interaction, "id", None)

    def _chat_agent(self, *, system: str, user: str) -> tuple[str, str | None]:
        tools = [
            {"type": "code_execution"},
            {"type": "url_context"},
        ]
        interaction = self._client.interactions.create(
            agent=self.agent,
            input=user,
            system_instruction=system,
            background=True,
            tools=tools,
            agent_config={"type": "antigravity", "model": self.model},
            environment={"type": "remote", "network": "disabled"},
        )
        interaction = self._await_interaction(interaction)
        text = getattr(interaction, "output_text", None) or ""
        return str(text), getattr(interaction, "id", None)

    def _await_interaction(self, interaction: Any) -> Any:
        status = getattr(interaction, "status", None)
        if status in (None, "completed"):
            if status == "completed" or getattr(interaction, "output_text", None):
                return interaction
        deadline = time.time() + self.timeout_s
        polls = 0
        inter_id = getattr(interaction, "id", None)
        if not inter_id:
            return interaction
        while time.time() < deadline and polls < GEMINI_ARGON_MAX_POLL:
            time.sleep(GEMINI_ARGON_POLL_SECONDS)
            polls += 1
            interaction = self._client.interactions.get(inter_id)
            status = getattr(interaction, "status", None)
            if status == "completed":
                return interaction
            if status == "failed":
                err = getattr(interaction, "error", None)
                raise LlmUnavailable(f"gemini_argon interaction failed: {err}")
        raise LlmUnavailable(
            f"gemini_argon interaction timed out after {self.timeout_s}s "
            f"(id={inter_id}, status={getattr(interaction, 'status', None)})"
        )
