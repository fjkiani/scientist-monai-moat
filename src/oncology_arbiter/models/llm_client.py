"""Gemma LLM client with a routing ladder.

Route order (first success wins):

    1. **Google direct** (``generativelanguage.googleapis.com``) — Gemma-4-31b-it,
       primary route, verified live 2026-07-06.
    2. **OpenRouter V2** ``:free`` tier.
    3. **OpenRouter V1** ``:free`` tier.
    4. **OpenRouter legacy** key (also ``:free``).

Every route target is Gemma-4-31b-it. The client is stateless and thread-safe;
callers should reuse a single instance to keep the requests session pool hot.

Environment variables (all optional; missing keys are skipped):

    GEMMA_GOOGLE_KEY   — Google direct API key (starts with ``AQ.``)
    OPENROUTER_V2      — OpenRouter key #2
    OPENROUTER_V1      — OpenRouter key #1
    OPENROUTER_LEGACY  — OpenRouter legacy key
    GEMMA_MODEL        — override default model (``gemma-4-31b-it``)

Failure mode: if every route is unavailable (rate-limited, no key, upstream
outage), ``chat()`` raises :class:`LlmUnavailable` — never returns a fake
response, never invents content. Callers must catch and degrade gracefully.

Cost accounting: every call records tokens used + estimated USD spend in
:attr:`GemmaClient.usage`. Live smoke and the ``/health`` endpoint can
introspect this without needing an external ledger.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

__all__ = ["GemmaClient", "LlmUnavailable", "LlmResponse", "LLM_CLIENT_VERSION"]

LLM_CLIENT_VERSION = "1.0.0"

# Default per-model token pricing (USD per 1M tokens). OpenRouter :free is $0.
# Google direct Gemma is free within the daily quota. Numbers exist so the
# session-total ledger in the SPA can render a real number even if it's $0.
_PRICING_USD_PER_MTOK = {
    "google/gemma-4-31b-it": {"input": 0.0, "output": 0.0},  # Google direct, free tier
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},
    "google/gemma-4-26b-a4b-it": {"input": 0.0, "output": 0.0},
    "google/gemma-4-26b-a4b-it:free": {"input": 0.0, "output": 0.0},
}


class LlmUnavailable(RuntimeError):
    """Raised when every route in the ladder is exhausted or rate-limited.

    Callers MUST NOT swallow this — the honest response is to surface the
    fact that the LLM couldn't answer, not to fabricate a response.
    """


@dataclass
class LlmResponse:
    """One completed LLM call."""
    text: str
    route: str        # e.g. "google_direct" | "openrouter_v2" | "openrouter_v1"
    model: str        # canonical model id, e.g. "google/gemma-4-31b-it"
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int  # Gemma "thinking" tokens (Google direct only)
    est_cost_usd: float
    latency_s: float


@dataclass
class UsageLedger:
    """Session-wide accounting."""
    calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_thinking_tokens: int = 0
    total_cost_usd: float = 0.0
    per_route_calls: Dict[str, int] = field(default_factory=dict)
    per_route_failures: Dict[str, int] = field(default_factory=dict)

    def record(self, resp: LlmResponse) -> None:
        self.calls += 1
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        self.total_thinking_tokens += resp.thinking_tokens
        self.total_cost_usd += resp.est_cost_usd
        self.per_route_calls[resp.route] = self.per_route_calls.get(resp.route, 0) + 1

    def record_failure(self, route: str) -> None:
        self.per_route_failures[route] = self.per_route_failures.get(route, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "per_route_calls": dict(self.per_route_calls),
            "per_route_failures": dict(self.per_route_failures),
        }


def _messages_to_google_contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Adapt OpenAI-style messages to Google generateContent shape.

    Google Gemma doesn't accept ``role: system`` — we prefix any system
    message onto the first user turn instead.
    """
    contents: List[Dict[str, Any]] = []
    system_prefix = ""
    for m in messages:
        if m["role"] == "system":
            system_prefix += m["content"].strip() + "\n\n"
            continue
        role = "user" if m["role"] == "user" else "model"
        text = m["content"]
        if role == "user" and system_prefix:
            text = system_prefix + text
            system_prefix = ""
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


class GemmaClient:
    """Route Gemma calls through the ladder.

    Usage::

        client = GemmaClient()
        resp = client.chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=200,
        )
        print(resp.text, resp.route, resp.est_cost_usd)

    """

    def __init__(
        self,
        model: Optional[str] = None,
        google_key: Optional[str] = None,
        openrouter_keys: Optional[List[str]] = None,
        timeout_s: float = 45.0,
    ) -> None:
        self.model_bare = model or os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
        self.model = f"google/{self.model_bare}"
        self.google_key = google_key or os.environ.get("GEMMA_GOOGLE_KEY")
        if openrouter_keys is None:
            openrouter_keys = [
                os.environ.get("OPENROUTER_V2"),
                os.environ.get("OPENROUTER_V1"),
                os.environ.get("OPENROUTER_LEGACY"),
            ]
        self.openrouter_keys = [k for k in openrouter_keys if k]
        self.timeout_s = timeout_s
        self.usage = UsageLedger()
        self._session = requests.Session()
        # Track exhausted OpenRouter keys within the current UTC day (429s).
        # Cleared when Reset epoch passes.
        self._exhausted_openrouter: Dict[str, float] = {}

    # --------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LlmResponse:
        """Send messages to the best available route.

        :raises LlmUnavailable: if every route fails or is exhausted.
        """
        errors: List[str] = []

        # ---- Route 1: Google direct ----
        if self.google_key:
            try:
                return self._call_google(messages, max_tokens, temperature)
            except _RouteFailure as e:
                self.usage.record_failure("google_direct")
                errors.append(f"google_direct: {e}")

        # ---- Route 2-4: OpenRouter ladder ----
        for i, key in enumerate(self.openrouter_keys):
            if not key:
                continue
            route_name = f"openrouter_{['v2','v1','legacy'][i] if i < 3 else str(i)}"
            # Skip if we already saw a 429 for this key today
            reset_ts = self._exhausted_openrouter.get(key)
            if reset_ts and time.time() < reset_ts:
                errors.append(f"{route_name}: rate-limited until {int(reset_ts)}")
                continue
            try:
                return self._call_openrouter(key, route_name, messages, max_tokens, temperature)
            except _RouteFailure as e:
                self.usage.record_failure(route_name)
                errors.append(f"{route_name}: {e}")

        raise LlmUnavailable(
            f"All routes exhausted. Model={self.model_bare}. Errors: {' | '.join(errors)}"
        )

    # --------------------------------------------------------------------

    def _call_google(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LlmResponse:
        """Google generativelanguage direct call.

        Gemma emits "thinking" tokens internally that count against
        ``maxOutputTokens`` — pad by 512 so short responses don't get
        cut off mid-thought.
        """
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_bare}:generateContent?key={self.google_key}"
        )
        payload = {
            "contents": _messages_to_google_contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens + 512,
            },
        }
        t0 = time.time()
        try:
            r = self._session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise _RouteFailure(f"network: {e}")
        elapsed = time.time() - t0

        if r.status_code == 429:
            raise _RouteFailure(f"rate_limited (HTTP 429)")
        if r.status_code >= 400:
            raise _RouteFailure(f"HTTP {r.status_code}: {r.text[:200]}")

        try:
            data = r.json()
        except ValueError:
            raise _RouteFailure(f"non-JSON response ({r.text[:200]!r})")

        candidates = data.get("candidates", [])
        if not candidates:
            raise _RouteFailure(f"no candidates: {json.dumps(data)[:300]}")

        parts = candidates[0].get("content", {}).get("parts", [])
        # Skip "thought" parts, join the rest
        text_pieces = [p["text"] for p in parts if p.get("text") and not p.get("thought")]
        text = "".join(text_pieces).strip()

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        thinking_tokens = usage.get("thoughtsTokenCount", 0)

        # If the model spent all its budget thinking, the response may be empty.
        # Surface that honestly instead of returning "".
        if not text:
            finish = candidates[0].get("finishReason", "?")
            raise _RouteFailure(
                f"empty response (finish={finish}, thinking_tokens={thinking_tokens}). "
                f"Try increasing max_tokens."
            )

        pricing = _PRICING_USD_PER_MTOK.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            (prompt_tokens / 1e6) * pricing["input"]
            + (completion_tokens / 1e6) * pricing["output"]
        )

        resp = LlmResponse(
            text=text,
            route="google_direct",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            est_cost_usd=cost,
            latency_s=elapsed,
        )
        self.usage.record(resp)
        return resp

    # --------------------------------------------------------------------

    def _call_openrouter(
        self,
        api_key: str,
        route_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LlmResponse:
        """OpenRouter chat completion. Tries ``:free`` first, then plain."""
        for model_variant in (f"{self.model}:free", self.model):
            payload = {
                "model": model_variant,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            t0 = time.time()
            try:
                r = self._session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout_s,
                )
            except requests.RequestException as e:
                raise _RouteFailure(f"network: {e}")
            elapsed = time.time() - t0

            if r.status_code == 429:
                # Parse reset epoch, remember it so we don't hammer.
                try:
                    reset_ms = int(
                        r.json().get("error", {}).get("metadata", {}).get("headers", {}).get(
                            "X-RateLimit-Reset", 0
                        )
                    )
                    if reset_ms > 0:
                        self._exhausted_openrouter[api_key] = reset_ms / 1000.0
                except Exception:
                    pass
                # Try the paid variant next
                continue
            if r.status_code == 402:
                # Insufficient credits — permanent for this key, don't retry
                self._exhausted_openrouter[api_key] = time.time() + 24 * 3600
                raise _RouteFailure(f"insufficient_credits (HTTP 402)")
            if r.status_code >= 400:
                raise _RouteFailure(f"HTTP {r.status_code}: {r.text[:200]}")

            try:
                data = r.json()
            except ValueError:
                raise _RouteFailure(f"non-JSON response")

            choices = data.get("choices", [])
            if not choices:
                raise _RouteFailure(f"no choices: {json.dumps(data)[:300]}")
            text = choices[0].get("message", {}).get("content", "").strip()
            if not text:
                raise _RouteFailure(f"empty response")

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            pricing = _PRICING_USD_PER_MTOK.get(model_variant, {"input": 0.0, "output": 0.0})
            cost = (
                (prompt_tokens / 1e6) * pricing["input"]
                + (completion_tokens / 1e6) * pricing["output"]
            )

            resp = LlmResponse(
                text=text,
                route=route_name,
                model=model_variant,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=0,
                est_cost_usd=cost,
                latency_s=elapsed,
            )
            self.usage.record(resp)
            return resp

        raise _RouteFailure("all variants exhausted")


class _RouteFailure(Exception):
    """Internal marker so we fall through to the next route."""


# ── Module-level singleton for API-side use ─────────────────────────
_default_client: Optional[GemmaClient] = None


def get_default_client() -> GemmaClient:
    """Return the process-wide default client, lazily created."""
    global _default_client
    if _default_client is None:
        _default_client = GemmaClient()
    return _default_client
