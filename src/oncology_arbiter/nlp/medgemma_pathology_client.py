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
    # Strip MedGemma/Gemma3 thinking channel wrappers when present.
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
    # Prefer first JSON object (tolerates trailing prose / second blobs).
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
        raise MedGemmaPathologyError("medgemma_pathology: no JSON object in model text")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MedGemmaPathologyError(
            f"medgemma_pathology: JSON parse failed: {exc}"
        ) from exc
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
        raw = str(value).strip().upper()
        compact = (
            raw.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        aliases = {
            "POS": "POSITIVE",
            "NEG": "NEGATIVE",
            "NOT_AMPLIFIED": "NON_AMPLIFIED",
            "NONAMPLIFIED": "NON_AMPLIFIED",
            "NON_AMPLIFIED": "NON_AMPLIFIED",
            "HER2_POSITIVE": "POSITIVE",
            "HER2_NEGATIVE": "NEGATIVE",
            "NEGATIVE_1+": "1+",
            "NEGATIVE_(1+)": "1+",
            "NEGATIVE__1+_": "1+",
            "NEGATIVE__1+": "1+",
            "POSITIVE_3+": "3+",
            "POSITIVE_(3+)": "3+",
            "POSITIVE__3+_": "3+",
            "EQUIVOCAL_2+": "2+",
            "EQUIVOCAL_(2+)": "2+",
            "EQUIVOCAL__2+_": "2+",
        }
        # collapse repeated underscores from paren/punct scrub
        while "__" in compact:
            compact = compact.replace("__", "_")
        compact = compact.strip("_")
        if compact in aliases:
            return aliases[compact]
        if her2:
            if "3+" in raw:
                return "3+"
            if "2+" in raw:
                return "2+"
            if "1+" in raw:
                return "1+"
            if "NON" in raw and "AMPLIF" in raw:
                return "NON_AMPLIFIED"
            if "NOT" in raw and "AMPLIF" in raw:
                return "NON_AMPLIFIED"
            if "AMPLIF" in raw and "NON" not in raw and "NOT" not in raw:
                return "AMPLIFIED"
            if "POSITIVE" in raw:
                return "POSITIVE"
            if "NEGATIVE" in raw:
                return "NEGATIVE"
            if "EQUIVOCAL" in raw:
                return "EQUIVOCAL"
            if raw in {"0", "1+", "2+", "3+"}:
                return raw
            return "UNKNOWN"
        if compact in {"POSITIVE", "NEGATIVE", "EQUIVOCAL", "UNKNOWN"}:
            return compact
        if "POSITIVE" in raw:
            return "POSITIVE"
        if "NEGATIVE" in raw:
            return "NEGATIVE"
        if "EQUIVOCAL" in raw:
            return "EQUIVOCAL"
        return "UNKNOWN"

    def _norm_grade(value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, int) and value in (1, 2, 3):
            return value
        if isinstance(value, float) and int(value) in (1, 2, 3):
            return int(value)
        text = str(value).strip().upper()
        import re

        m = re.search(r"\b([123])\b", text)
        if m:
            return int(m.group(1))
        m = re.search(r"G([123])\b", text)
        if m:
            return int(m.group(1))
        if "POOR" in text or "HIGH" in text:
            return 3
        if "MODERATE" in text or "INTERMEDIATE" in text:
            return 2
        if "WELL" in text or "LOW" in text:
            return 1
        return None

    def _norm_float(value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            import re

            m = re.search(r"(\d+(?:\.\d+)?)", text)
            return float(m.group(1)) if m else None

    if "er_status" in slim:
        slim["er_status"] = _norm_status(slim.get("er_status"))
    if "pr_status" in slim:
        slim["pr_status"] = _norm_status(slim.get("pr_status"))
    if "her2_status" in slim:
        slim["her2_status"] = _norm_status(slim.get("her2_status"), her2=True)
    if "nottingham_grade" in slim:
        slim["nottingham_grade"] = _norm_grade(slim.get("nottingham_grade"))
    if "er_percent" in slim:
        slim["er_percent"] = _norm_float(slim.get("er_percent"))
    # Models often emit tumor size in cm despite the mm contract. Values in
    # (0, 15] are almost always centimeters for breast pathology; convert.
    size = _norm_float(slim.get("tumor_size_mm")) if "tumor_size_mm" in slim else None
    if size is not None:
        if 0 < float(size) <= 15.0:
            size = float(size) * 10.0
        slim["tumor_size_mm"] = size
    try:
        model = PathologyExtraction.model_validate(slim)
    except Exception:
        # Last-ditch: drop fields that still fail validation.
        for key in list(slim.keys()):
            try:
                PathologyExtraction.model_validate({key: slim[key]})
            except Exception:
                slim.pop(key, None)
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
