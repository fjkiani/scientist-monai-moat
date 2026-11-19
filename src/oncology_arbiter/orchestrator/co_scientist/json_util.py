"""MedGemma JSON extraction + chat-with-retry helpers."""
from __future__ import annotations

import json
import re
from typing import Any

from .const import JSON_RETRY_SUFFIX, MEDGEMMA_RETRY_TEMPERATURE

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Prefer the supervisor MedGemma parser (thought/final + last valid object)."""
    try:
        from oncology_arbiter.agents.supervisor import _extract_json as _sup_extract

        parsed = _sup_extract(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    text = (text or "").strip()
    text = re.sub(
        r"<unused\d+>thought[\s\S]*?(?:</?unused\d+>|</thought>|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def chat_json_with_retry(
    llm: Any,
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    """Call MedGemma; retry once if thought-budget ate the JSON."""
    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed = extract_json_object(resp.text) or {}
    usable = bool(parsed.get("winner")) or isinstance(parsed.get("variants"), list)
    if not usable:
        resp = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user + JSON_RETRY_SUFFIX},
            ],
            max_tokens=max_tokens,
            temperature=MEDGEMMA_RETRY_TEMPERATURE,
        )
        parsed = extract_json_object(resp.text) or {}
    return resp, parsed
