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
THERAPY_CHAT_URL = "https://crispro--medgemma-1-5-4b-chat.modal.run"

THERAPY_SYSTEM_PROMPT = (
    "You are a research-use-only oncology therapy reasoner. "
    "Given breast-cancer patient context and biopsy findings, return ONLY one "
    "JSON object (first character '{') with keys:\n"
    '  "primary_drug": string (canonical regimen or drug name),\n'
    '  "recommended": [{"regimen": str, "line_of_therapy": int, "rationale": str}],\n'
    '  "not_recommended": [{"regimen": str, "line_of_therapy": int, "rationale": str}],\n'
    '  "reasoning": string\n'
    "recommended must contain 1 or 2 objects. "
    "Obey only the CLINICAL GUIDELINE in this message. "
    "Do not apply a guideline for a different receptor subtype.\n"
    "No markdown. No prose outside JSON. First character '{'. "
    "RESEARCH USE ONLY — not clinical advice."
)

_AI = ("letrozole", "anastrozole", "exemestane")
_ENDOCRINE = _AI + ("tamoxifen", "fulvestrant")
_CHEMO = (
    "paclitaxel",
    "doxorubicin",
    "carboplatin",
    "cyclophosphamide",
    "capecitabine",
    "gemcitabine",
    "eribulin",
)
_IO = ("pembrolizumab", "atezolizumab", "nivolumab")
_HER2_DRUGS = ("trastuzumab", "pertuzumab", "herceptin", "trastuzumab deruxtecan", "t-dxd")


def get_subtype_prompt_constraint(
    er_pos: bool,
    pr_pos: bool,
    her2_pos: bool,
    postmenopausal: bool,
) -> str:
    """One subtype guideline. Python picks the branch; the 4B does not route."""
    if her2_pos:
        return (
            "CLINICAL GUIDELINE: Patient is HER2-positive. Recommended first-line therapy "
            "MUST include HER2-targeted therapy (Trastuzumab / Pertuzumab + chemotherapy). "
            "Do NOT select endocrine monotherapy."
        )
    elif er_pos or pr_pos:
        if postmenopausal:
            return (
                "CLINICAL GUIDELINE: Patient is HR-positive / HER2-negative postmenopausal. "
                "First-line therapy MUST be an Aromatase Inhibitor (Letrozole, Anastrozole, or Exemestane). "
                "Cytotoxic chemotherapy is contraindicated as first-line monotherapy."
            )
        else:
            return (
                "CLINICAL GUIDELINE: Patient is HR-positive / HER2-negative premenopausal. "
                "First-line therapy MUST be Tamoxifen or ovarian suppression + Aromatase Inhibitor."
            )
    else:
        return (
            "CLINICAL GUIDELINE: Patient is Triple-Negative Breast Cancer (ER-negative, PR-negative, HER2-negative). "
            "Endocrine therapy (Tamoxifen, Letrozole) is CONTRAINDICATED and completely ineffective. "
            "First-line therapy MUST be systemic cytotoxic chemotherapy (e.g., Carboplatin, Paclitaxel, Doxorubicin) or Immunotherapy."
        )


def subtype_inputs_from_case(
    patient_context: dict[str, Any],
    biopsy_summary: dict[str, Any],
) -> tuple[bool, bool, bool, bool]:
    """Map a validated receptor panel to router bools. Missing fields refuse."""
    panel = biopsy_summary.get("receptor_panel") if isinstance(biopsy_summary, dict) else None
    if not isinstance(panel, dict):
        raise MedGemmaTherapyError(
            "medgemma_therapy: receptor panel incomplete; subtype router refused"
        )
    er = panel.get("er_positive")
    pr = panel.get("pr_positive")
    her2_raw = panel.get("her2_status")
    if not isinstance(er, bool) or not isinstance(pr, bool) or not isinstance(her2_raw, str):
        raise MedGemmaTherapyError(
            "medgemma_therapy: receptor panel incomplete; subtype router refused"
        )
    her2_norm = her2_raw.strip().lower()
    if her2_norm not in {"positive", "negative", "equivocal"}:
        raise MedGemmaTherapyError(
            "medgemma_therapy: her2_status is not positive, negative, or equivocal"
        )
    meno = ""
    if isinstance(patient_context, dict) and patient_context.get("menopausal_status") is not None:
        meno = str(patient_context.get("menopausal_status")).strip().lower()
    her2_pos = her2_norm == "positive"
    if not her2_pos and (er or pr) and meno not in {"pre", "peri", "post"}:
        raise MedGemmaTherapyError(
            "medgemma_therapy: menopausal_status required for HR-positive subtype routing"
        )
    if her2_norm == "equivocal" and not (er or pr):
        raise MedGemmaTherapyError(
            "medgemma_therapy: HER2-equivocal HR-negative has no subtype prompt"
        )
    return er, pr, her2_pos, meno == "post"


def subtype_id(er_pos: bool, pr_pos: bool, her2_pos: bool, postmenopausal: bool) -> str:
    if her2_pos:
        return "her2"
    if er_pos or pr_pos:
        return "hr_post" if postmenopausal else "hr_pre"
    return "tnbc"


def _regimen_text(primary: str, recommended: list[dict[str, Any]]) -> str:
    return (primary + " " + " ".join(str(item.get("regimen") or "") for item in recommended)).lower()


def guideline_violated(kind: str, primary: str, recommended: list[dict[str, Any]]) -> bool:
    text = _regimen_text(primary, recommended)
    if kind == "her2":
        return not any(drug in text for drug in _HER2_DRUGS)
    if kind == "hr_post":
        has_ai = any(drug in text for drug in _AI)
        bleed = any(drug in text for drug in (*_CHEMO, "tamoxifen", "fulvestrant"))
        return (not has_ai) or bleed
    if kind == "hr_pre":
        return not any(drug in text for drug in ("tamoxifen", *_AI))
    if kind == "tnbc":
        endocrine = any(drug in text for drug in _ENDOCRINE)
        cytotoxic = any(drug in text for drug in (*_CHEMO, *_IO))
        return endocrine or not cytotoxic
    raise MedGemmaTherapyError(f"medgemma_therapy: unknown subtype {kind}")


def build_therapy_user_message(
    *,
    patient_context: dict[str, Any],
    biopsy_summary: dict[str, Any],
    disease: str,
    repair: bool = False,
) -> tuple[str, str]:
    er_pos, pr_pos, her2_pos, postmenopausal = subtype_inputs_from_case(
        patient_context, biopsy_summary
    )
    kind = subtype_id(er_pos, pr_pos, her2_pos, postmenopausal)
    constraint = get_subtype_prompt_constraint(er_pos, pr_pos, her2_pos, postmenopausal)
    user_blob = {
        "disease": disease,
        "patient_context": patient_context,
        "biopsy_summary": biopsy_summary,
        "instruction": "Apply only the CLINICAL GUIDELINE above. Recommend 1 or 2 first-line research options.",
    }
    repair_line = ""
    if repair:
        repair_line = (
            "\nThe previous answer violated the CLINICAL GUIDELINE. "
            "Obey only that guideline.\n"
        )
    content = (
        THERAPY_SYSTEM_PROMPT
        + "\n\n"
        + constraint
        + "\n\nPATIENT_CASE_JSON:\n"
        + json.dumps(user_blob, ensure_ascii=False)
        + repair_line
        + "\n\nJSON:"
    )
    return kind, content


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
            "medgemma_therapy: identity chat_url is not the pinned MedGemma chat host"
        )

    kind, content = build_therapy_user_message(
        patient_context=patient_context,
        biopsy_summary=biopsy_summary,
        disease=disease,
    )
    messages = [{"role": "user", "content": content}]

    def _call(payload_messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            data = medgemma_chat(
                payload_messages,
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
        return data

    data = _call(messages)
    raw_text = str(data.get("text") or "")
    parsed = _extract_json_object(raw_text)
    recommended = _normalize_options(parsed.get("recommended"))
    not_recommended = _normalize_options(parsed.get("not_recommended"))
    primary = str(parsed.get("primary_drug") or "").strip()
    if not primary and recommended:
        primary = recommended[0]["regimen"]
    if guideline_violated(kind, primary, recommended):
        _kind, repair_content = build_therapy_user_message(
            patient_context=patient_context,
            biopsy_summary=biopsy_summary,
            disease=disease,
            repair=True,
        )
        data = _call([{"role": "user", "content": repair_content}])
        raw_text = str(data.get("text") or "")
        parsed = _extract_json_object(raw_text)
        recommended = _normalize_options(parsed.get("recommended"))
        not_recommended = _normalize_options(parsed.get("not_recommended"))
        primary = str(parsed.get("primary_drug") or "").strip()
        if not primary and recommended:
            primary = recommended[0]["regimen"]
        if guideline_violated(kind, primary, recommended):
            raise MedGemmaTherapyError(
                "medgemma_therapy: subtype guideline violated after repair; "
                f"subtype={kind}; raw=" + raw_text[:500]
            )
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
        raise MedGemmaTherapyError(
            "medgemma_therapy: empty recommended list; raw=" + raw_text[:500]
        )

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
        "subtype": kind,
    }
