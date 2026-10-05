"""MedGemma Modal client for structured pathology biomarker extraction (RUO)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from oncology_arbiter.models.medgemma_1_5_4b_wiring import verify_artifact_identity
from oncology_arbiter.nlp.pathology_extraction_schema import (
    EXTRACTION_SYSTEM_PROMPT,
    PathologyExtraction,
)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


class MedGemmaPathologyError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if not match:
        raise MedGemmaPathologyError("medgemma_pathology: no JSON object in model text")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise MedGemmaPathologyError("medgemma_pathology: JSON payload is not an object")
    return payload


def extract_pathology(
    report_text: str,
    *,
    max_tokens: int = 384,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Call identity-locked MedGemma chat and return validated PathologyExtraction."""
    if not report_text or not str(report_text).strip():
        raise MedGemmaPathologyError("medgemma_pathology: empty report_text")

    identity = verify_artifact_identity()
    chat_url = str(identity["chat_url"])
    # Keep prompt bounded — long TCGA reports; head+tail preserves receptor blocks.
    body_text = str(report_text)
    if len(body_text) > 12000:
        body_text = body_text[:8000] + "\n...\n" + body_text[-4000:]

    user_content = (
        EXTRACTION_SYSTEM_PROMPT
        + "\n\nPATHOLOGY REPORT:\n"
        + body_text
        + "\n\nJSON:"
    )
    req_body = json.dumps(
        {
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    request = urllib.request.Request(
        chat_url,
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise MedGemmaPathologyError(
            f"medgemma_pathology HTTP {exc.code}: {detail}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise MedGemmaPathologyError(
            f"medgemma_pathology request failed: {type(exc).__name__}: {exc}"
        ) from exc

    if data.get("error"):
        raise MedGemmaPathologyError(f"medgemma_pathology chat error: {data['error']}")
    if data.get("revision_sha256") != identity["revision_sha256"]:
        raise MedGemmaPathologyError("medgemma_pathology: revision_sha256 drift vs identity")

    raw_text = str(data.get("text") or "")
    parsed = _extract_json_object(raw_text)
    # Drop unknown keys before pydantic; keep honesty on disclaimer.
    allowed = {
        "er_status",
        "er_percent",
        "pr_status",
        "her2_status",
        "nottingham_grade",
        "tumor_size_mm",
    }
    slim = {k: parsed.get(k) for k in allowed if k in parsed}

    def _norm_status(value: Any, *, her2: bool = False) -> str:
        if value is None:
            return "UNKNOWN"
        text = str(value).strip().upper().replace(" ", "_")
        aliases = {
            "POS": "POSITIVE",
            "NEG": "NEGATIVE",
            "NOT_AMPLIFIED": "NON_AMPLIFIED",
            "NONAMPLIFIED": "NON_AMPLIFIED",
            "HER2_POSITIVE": "POSITIVE",
            "HER2_NEGATIVE": "NEGATIVE",
        }
        text = aliases.get(text, text)
        if her2 and text in {"NOT_AMPLIFIED", "NON-AMPLIFIED"}:
            text = "NON_AMPLIFIED"
        return text

    if "er_status" in slim:
        slim["er_status"] = _norm_status(slim.get("er_status"))
    if "pr_status" in slim:
        slim["pr_status"] = _norm_status(slim.get("pr_status"))
    if "her2_status" in slim:
        slim["her2_status"] = _norm_status(slim.get("her2_status"), her2=True)
    # Models often emit tumor size in cm despite the mm contract. Values in
    # (0, 15] are almost always centimeters for breast pathology; convert.
    size = slim.get("tumor_size_mm")
    if isinstance(size, (int, float)) and 0 < float(size) <= 15.0:
        slim["tumor_size_mm"] = float(size) * 10.0
    model = PathologyExtraction.model_validate(slim)
    return {
        "status": "ok",
        "source": "medgemma_pathology",
        "model_id": data.get("model_id") or identity.get("model_id"),
        "revision_sha256": data.get("revision_sha256"),
        "latency_s": data.get("latency_s"),
        "extraction": model.model_dump(),
        "raw_text": raw_text,
        "disclaimer": model.disclaimer,
    }
