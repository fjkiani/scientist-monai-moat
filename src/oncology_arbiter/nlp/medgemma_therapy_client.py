"""MedGemma Modal client for structured therapy recommendations (RUO).

Replaces TxGemma on ``/v1/therapy/reason``. Uses the same identity-locked
MedGemma 1.5 4B chat host as pathology extraction.
"""
from __future__ import annotations

import json
import re
from typing import Any

from oncology_arbiter.models.medgemma_1_5_4b_wiring import verify_artifact_identity
from oncology_arbiter.nlp.medgemma_chat_transport import (
    MedGemmaChatError,
    medgemma_chat,
)

_JSON_RE = re.compile(r"\{[\s\S]*\}")
THERAPY_CHAT_URL = "https://testing1235--medgemma-1-5-4b-chat.modal.run"

THERAPY_SYSTEM_PROMPT = (
    "You are a research-use-only oncology therapy reasoner. "
    "Given breast-cancer patient context and biopsy findings, return ONLY one "
    "JSON object (first character '{') with keys:\n"
    '  "primary_drug": string (canonical regimen or drug name),\n'
    '  "recommended": [{"regimen": str, "line_of_therapy": int, "rationale": str}],\n'
    '  "not_recommended": [{"regimen": str, "line_of_therapy": int, "rationale": str}],\n'
    '  "reasoning": string\n'
    "No markdown. No prose outside JSON. RESEARCH USE ONLY — not clinical advice."
)


class MedGemmaTherapyError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(
        r"<unused\d+>thought[\s\S]*?(?:</?unused\d+>|</thought>|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    match = _JSON_RE.search(text)
    if not match:
        raise MedGemmaTherapyError("medgemma_therapy: no JSON object in model text")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise MedGemmaTherapyError("medgemma_therapy: JSON payload is not an object")
    return payload


def _normalize_options(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        regimen = str(item.get("regimen") or item.get("drug") or item.get("drug_name") or "").strip()
        if not regimen:
            continue
        try:
            lot = int(item.get("line_of_therapy") or 1)
        except (TypeError, ValueError):
            lot = 1
        if lot < 1:
            lot = 1
        rationale = str(item.get("rationale") or item.get("reason") or "MedGemma therapy suggestion").strip()
        out.append(
            {
                "regimen": regimen,
                "line_of_therapy": lot,
                "rationale": rationale,
            }
        )
    return out


def reason_therapy(
    *,
    patient_context: dict[str, Any],
    biopsy_summary: dict[str, Any],
    disease: str = "breast cancer",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Call identity-locked MedGemma and return structured therapy options."""
    identity = verify_artifact_identity()
    chat_url = str(identity["chat_url"]).rstrip("/")
    if chat_url != THERAPY_CHAT_URL:
        raise MedGemmaTherapyError(
            "medgemma_therapy: identity chat_url is not the testing1235 MedGemma chat host"
        )

    user_blob = {
        "disease": disease,
        "patient_context": patient_context,
        "biopsy_summary": biopsy_summary,
        "instruction": "Recommend first-line research options grounded in the biopsy receptors.",
    }
    messages = [
        {
            "role": "user",
            "content": THERAPY_SYSTEM_PROMPT
            + "\n\nPATIENT_CASE_JSON:\n"
            + json.dumps(user_blob, ensure_ascii=False)
            + "\n\nJSON:",
        }
    ]

    try:
        data = medgemma_chat(
            messages,
            chat_url=chat_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except MedGemmaChatError as exc:
        raise MedGemmaTherapyError(str(exc)) from exc

    if data.get("error"):
        raise MedGemmaTherapyError(f"medgemma_therapy chat error: {data['error']}")
    if data.get("revision_sha256") and data.get("revision_sha256") != identity["revision_sha256"]:
        raise MedGemmaTherapyError("medgemma_therapy: revision_sha256 drift vs identity")

    raw_text = str(data.get("text") or "")
    parsed = _extract_json_object(raw_text)
    recommended = _normalize_options(parsed.get("recommended"))
    not_recommended = _normalize_options(parsed.get("not_recommended"))
    primary = str(parsed.get("primary_drug") or "").strip()
    if not primary and recommended:
        primary = recommended[0]["regimen"]
    if not recommended and primary:
        recommended = [
            {
                "regimen": primary,
                "line_of_therapy": 1,
                "rationale": str(parsed.get("reasoning") or "MedGemma primary regimen"),
            }
        ]
    if not recommended:
        raise MedGemmaTherapyError("medgemma_therapy: empty recommended list")

    return {
        "primary_drug": primary or recommended[0]["regimen"],
        "recommended": recommended,
        "not_recommended": not_recommended,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "raw_text": raw_text[:4000],
        "chat_url": chat_url,
        "transport": data.get("transport"),
        "request_id": data.get("request_id"),
        "latency_s": data.get("latency_s"),
        "model_id": data.get("model_id") or identity["model_id"],
        "revision_sha256": data.get("revision_sha256") or identity["revision_sha256"],
        "app_version": data.get("app_version") or identity.get("app_version"),
        "honesty_warning": data.get("honesty_warning") or identity.get("honesty_warning"),
    }
