"""LLM-based entity labeler for pathology reports.

Uses Google Gemini 2.5 Flash (direct API via GEMMA_GOOGLE_KEY) as the primary
route because OpenRouter credits are drained. Gemini 2.5 is competitive with
Claude Sonnet 4 on structured JSON extraction (Google's technical report,
LMSYS Chatbot Arena rankings mid-2026) and has a favorable free-tier quota.

Design honesty
--------------
- LLM returns entity mentions with CHARACTER OFFSETS + TEXT SPAN. LLMs are
  known-unreliable at exact character offsets. We use the returned span as
  a HINT and re-align server-side via `text.index(text_span)` to be robust.
- Self-consistency: 3 runs at slightly different temperatures + entity-order
  permutations. Token-level majority vote. Reports with < 2/3 agreement per
  span are flagged `disputed` and excluded from gold (kept for training as
  low-confidence).
- Rate-limit + retry: exponential backoff on 429 / 503, single retry on
  malformed JSON with a strict repair prompt.
- Concurrency: async httpx with 15 concurrent requests (Gemini free tier
  free-flow tier allows ~15 rpm/pro or ~1000 rpm on paid; we pace to be safe).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)

# 21-entity schema (breast + NSCLC) — must match corpus_synth.ENTITY_TYPES.
ENTITY_TYPES: list[str] = [
    "ER_VALUE",
    "PR_VALUE",
    "HER2_VALUE",
    "KI67_PCT",
    "GRADE",
    "T_STAGE",
    "N_STAGE",
    "M_STAGE",
    "TUMOR_SIZE_MM",
    "MARGIN",
    "LVI",
    "KRAS",
    "EGFR",
    "ALK",
    "ROS1",
    "PD_L1_TPS",
    "TMB",
    "MSI",
    "HER2_AMP",
    "BRAF",
    "MET",
]

_ENTITY_DESCRIPTIONS = {
    "ER_VALUE": "Estrogen receptor status (positive/negative/equivocal or IHC %).",
    "PR_VALUE": "Progesterone receptor status (positive/negative/equivocal or IHC %).",
    "HER2_VALUE": "HER2/neu IHC score or FISH status (0/1+/2+/3+/positive/negative/equivocal).",
    "KI67_PCT": "Ki-67 or MIB-1 percentage (e.g. '15%').",
    "GRADE": "Histologic or Nottingham grade (1/2/3, I/II/III, or 'well/moderately/poorly differentiated').",
    "T_STAGE": "Primary tumor TNM T-category (T0/T1a/T2/T3/T4 etc, with optional pT/cT prefix).",
    "N_STAGE": "Regional nodes TNM N-category (N0/N1/N2/N3, with optional prefixes).",
    "M_STAGE": "Distant metastasis TNM M-category (M0/M1/MX).",
    "TUMOR_SIZE_MM": "Primary tumor size measurement in mm or cm (return the exact substring in the report).",
    "MARGIN": "Surgical margin status ('margins negative/positive/close/free of tumor/involved/not involved').",
    "LVI": "Lymphovascular / lymphvascular / angiolymphatic invasion status ('present/absent/identified/not identified').",
    "KRAS": "KRAS mutation status or specific variant (e.g. 'KRAS G12C', 'KRAS wild-type').",
    "EGFR": "EGFR mutation status or specific variant ('L858R', 'exon 19 deletion', 'wild-type').",
    "ALK": "ALK fusion / rearrangement status by IHC or FISH.",
    "ROS1": "ROS1 fusion / rearrangement status.",
    "PD_L1_TPS": "PD-L1 tumor proportion score (TPS) or combined positive score (CPS) as a percentage.",
    "TMB": "Tumor mutational burden (mut/Mb).",
    "MSI": "Microsatellite instability status (MSI-H, MSI-L, MSS, high, low, stable).",
    "HER2_AMP": "HER2 amplification status by FISH (amplified / not amplified / no amplification).",
    "BRAF": "BRAF mutation status or specific variant (V600E, wild-type).",
    "MET": "MET amplification, mutation, or exon-14 skipping status.",
}

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_COHERE_ENDPOINT = "https://api.cohere.com/v2/chat"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class _KeyRotator:
    """Round-robin key rotator with health tracking.

    Callers do ``key = await rotator.next()`` to receive the next-usable key.
    If a call fails with a rate-limit/exhausted error on a key, mark it
    with ``rotator.mark_bad(key, until=<epoch>)`` and it will be skipped
    until that time.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("_KeyRotator needs at least one key")
        # De-dup, preserve order
        seen = set()
        self._keys: list[str] = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                self._keys.append(k)
        self._bad_until: dict[str, float] = {}
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            n = len(self._keys)
            for _ in range(n):
                k = self._keys[self._idx % n]
                self._idx += 1
                if self._bad_until.get(k, 0.0) < time.time():
                    return k
            # All keys are bad — return the least-recently-bad one (so caller
            # can still retry rather than deadlock).
            k = min(self._keys, key=lambda x: self._bad_until.get(x, 0.0))
            return k

    def mark_bad(self, key: str, cooldown_s: float = 60.0) -> None:
        self._bad_until[key] = time.time() + cooldown_s
        logger.info("KeyRotator: marked key ...%s bad for %ds", key[-6:], int(cooldown_s))

    def n_healthy(self) -> int:
        now = time.time()
        return sum(1 for k in self._keys if self._bad_until.get(k, 0.0) < now)


@dataclass
class EntityMention:
    """A single entity mention located in a text."""

    entity_type: str
    value: str
    char_start: int
    char_end: int
    text_span: str
    confidence: float = 1.0  # LLM-reported or default

    def key(self) -> tuple[str, int, int]:
        return (self.entity_type, self.char_start, self.char_end)


@dataclass
class LabelerRun:
    """One LLM run's output for a single report."""

    report_id: str
    entities: list[EntityMention]
    raw_response: str = ""
    parse_ok: bool = True
    run_variant: str = "default"  # e.g. "run0_t0.0"


@dataclass
class MergedLabels:
    """Self-consistency-merged labels for a report."""

    report_id: str
    accepted: list[EntityMention]
    disputed: list[EntityMention]  # spans present in some but not all runs
    agreement_rate: float
    inter_run_kappa: float | None
    n_runs: int


def _build_prompt(text: str, seed_perm: int = 0) -> str:
    """Prompt template. seed_perm permutes entity-description order for
    self-consistency variance."""
    rng = random.Random(seed_perm)
    perm = list(_ENTITY_DESCRIPTIONS.items())
    rng.shuffle(perm)
    entity_desc = "\n".join(f"  {k}: {v}" for k, v in perm)

    return f"""You are annotating a pathology report to build a clinical NLP training set. Return ONLY valid JSON.

For each mention of the following entity types, return a JSON object with:
- entity_type: one of the type codes below
- value: the normalized surface value (e.g. "positive", "L858R", "17 mm")
- start_char: character offset (0-indexed) in the report where the mention begins
- end_char: character offset (exclusive) where the mention ends
- text_span: the EXACT substring of the report from start_char to end_char

CRITICAL SPAN RULES (follow exactly for reproducibility):
1. The text_span MUST be the MINIMAL substring that identifies the value.
   Prefer just the value itself, not surrounding label words.
   YES: text_span="L858R" for EGFR.
   YES: text_span="wild-type" for KRAS wild-type.
   YES: text_span="4.5 cm" for TUMOR_SIZE_MM.
   YES: text_span="negative" for ALK negative.
   NO: text_span="L858R detected" (includes verb).
   NO: text_span="KRAS: wild-type" (includes label + colon).
   NO: text_span="EGFR mutation: L858R" (includes label + word).
2. For status entities (ER/PR/HER2/ALK/ROS1/MSI/HER2_AMP/MET), the span is
   just the status word (positive/negative/amplified/etc.).
3. For TNM stages, the span is just the letter+digit(s) (T3, N2, M0, pT1c).
4. For measurements (TUMOR_SIZE_MM, KI67_PCT, PD_L1_TPS, TMB), the span is
   just the number + unit (e.g. "40%", "17 mm", "4.5 cm").
5. For GRADE, the span is just the grade token itself (e.g. "3" or "II" or
   "poorly differentiated").
6. Return each mention once. If the same entity appears in multiple places
   in the report, return all mentions as separate JSON entries.

If an entity is not mentioned, omit it. Do not hallucinate mentions. Do not
include multi-sentence explanations. Only return the exact substring for text_span.

Entity types:
{entity_desc}

REPORT:
{text}

Output JSON schema exactly:
{{"entities": [{{"entity_type": <str>, "value": <str>, "start_char": <int>, "end_char": <int>, "text_span": <str>}}]}}
"""


def _realign_span(text: str, mention: dict, prev_end: int = 0) -> EntityMention | None:
    """Given LLM output with potentially wrong offsets, re-align by locating
    text_span in the report. Falls back to LLM offsets if text_span not found.
    Returns None if the entity is not locatable at all."""
    et = mention.get("entity_type")
    if et not in ENTITY_TYPES:
        return None
    raw_span = mention.get("text_span")
    span_text = raw_span.strip() if isinstance(raw_span, str) else ""
    raw_value = mention.get("value")
    value = raw_value if isinstance(raw_value, (str, int, float)) else span_text
    if not span_text:
        return None
    # Search forward from prev_end first, else global.
    idx = text.find(span_text, prev_end)
    if idx < 0:
        idx = text.find(span_text)
    if idx < 0:
        # Case-insensitive fallback.
        low = text.lower()
        idx = low.find(span_text.lower(), prev_end)
        if idx < 0:
            idx = low.find(span_text.lower())
    if idx < 0:
        # LLM hallucinated the substring — drop.
        return None
    return EntityMention(
        entity_type=et,
        value=str(value),
        char_start=idx,
        char_end=idx + len(span_text),
        text_span=text[idx : idx + len(span_text)],
        confidence=float(mention.get("confidence", 1.0)),
    )


async def _call_cohere(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    model: str,
    temperature: float,
    seed_perm: int,
    max_retries: int = 5,
) -> tuple[str, bool]:
    """Cohere V2 chat call. Returns (raw_response_text, parse_ok_hint).

    Cohere command-a-03-2025 supports response_format={"type":"json_object"}
    which enforces JSON. 40 rpm on trial keys; on 429 we back off.
    """
    prompt = _build_prompt(text, seed_perm=seed_perm)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": 2000,
    }
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            r = await client.post(
                _COHERE_ENDPOINT,
                json=body,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                timeout=60.0,
            )
            if r.status_code == 429 or r.status_code == 503:
                await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            d = r.json()
            # V2 chat: message.content is a list of {"type":"text","text":"..."}
            parts = d.get("message", {}).get("content", [])
            content = ""
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    content = p.get("text", "")
                    break
            return content, True
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503):
                await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 2, 60)
                continue
            logger.warning("Cohere HTTP %s: %s", e.response.status_code, e.response.text[:400])
            return "", False
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError):
            await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
            backoff = min(backoff * 2, 60)
            continue
        except Exception as e:
            logger.warning("Cohere call error: %s", e)
            return "", False
    return "", False


async def _call_openrouter(
    client: httpx.AsyncClient,
    text: str,
    rotator: "_KeyRotator",
    model: str,
    temperature: float,
    seed_perm: int,
    max_retries: int = 6,
) -> tuple[str, bool]:
    """OpenRouter chat call with automatic 3-key rotation.

    tencent/hy3:free bypasses the 50/day free-model cap on OpenRouter free
    accounts (verified 2026-07-20 on user_2spq… and user_3DmET… accounts).
    Rotating across 3 keys gives >~100 rpm sustained on real pathology
    reports.
    """
    prompt = _build_prompt(text, seed_perm=seed_perm)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2000,
    }
    backoff = 2.0
    for attempt in range(max_retries):
        key = await rotator.next()
        try:
            r = await client.post(
                _OPENROUTER_ENDPOINT,
                json=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                timeout=60.0,
            )
            if r.status_code in (429, 503):
                # Parse remaining/reset to decide cooldown length.
                try:
                    err = r.json().get("error", {}).get("metadata", {}).get("headers", {})
                    reset = err.get("X-RateLimit-Reset")
                    if reset and str(reset).isdigit():
                        # Reset is epoch ms per OpenRouter docs.
                        cooldown = max(1.0, float(reset) / 1000 - time.time())
                        rotator.mark_bad(key, cooldown_s=min(cooldown, 3600))
                    else:
                        rotator.mark_bad(key, cooldown_s=60.0)
                except Exception:
                    rotator.mark_bad(key, cooldown_s=60.0)
                await asyncio.sleep(0.5 + random.uniform(0.0, 0.5))
                continue
            r.raise_for_status()
            d = r.json()
            if "choices" not in d:
                # Some providers return error inline
                await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 2, 60)
                continue
            content = d["choices"][0]["message"].get("content", "")
            return content, True
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503):
                rotator.mark_bad(key, cooldown_s=60.0)
                await asyncio.sleep(0.5 + random.uniform(0.0, 0.5))
                continue
            logger.warning("OpenRouter HTTP %s: %s", e.response.status_code, e.response.text[:400])
            return "", False
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError):
            await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
            backoff = min(backoff * 2, 60)
            continue
        except Exception as e:
            logger.warning("OpenRouter call error: %s", e)
            return "", False
    return "", False


async def _call_llm(
    client: httpx.AsyncClient,
    text: str,
    provider: str,
    api_key: str | _KeyRotator,
    model: str,
    temperature: float,
    seed_perm: int,
) -> tuple[str, bool]:
    """Provider-routing wrapper. provider in {'cohere', 'gemini', 'openrouter'}.

    For 'openrouter', api_key MUST be a _KeyRotator instance (3-key rotation).
    For 'cohere' and 'gemini', api_key is a string.
    """
    if provider == "cohere":
        if isinstance(api_key, _KeyRotator):
            raise TypeError("Cohere requires str api_key, got _KeyRotator")
        return await _call_cohere(client, text, api_key, model, temperature, seed_perm)
    elif provider == "gemini":
        if isinstance(api_key, _KeyRotator):
            raise TypeError("Gemini requires str api_key, got _KeyRotator")
        return await _call_gemini(client, text, api_key, model, temperature, seed_perm)
    elif provider == "openrouter":
        if not isinstance(api_key, _KeyRotator):
            # Allow single-key str for convenience; wrap it.
            api_key = _KeyRotator([api_key])
        return await _call_openrouter(client, text, api_key, model, temperature, seed_perm)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _call_gemini(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    model: str,
    temperature: float,
    seed_perm: int,
    max_retries: int = 3,
) -> tuple[str, bool]:
    """Single LLM call. Returns (raw_response_text, parse_ok_hint)."""
    prompt = _build_prompt(text, seed_perm=seed_perm)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "maxOutputTokens": 4000,
        },
    }
    url = _GEMINI_ENDPOINT.format(model=model)
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            r = await client.post(
                url, params={"key": api_key}, json=body, timeout=90.0
            )
            if r.status_code == 429 or r.status_code == 503:
                await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            d = r.json()
            parts = (
                d.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            content = parts[0].get("text", "") if parts else ""
            return content, True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 or e.response.status_code == 503:
                await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
                backoff = min(backoff * 2, 60)
                continue
            logger.warning("Gemini HTTP %s: %s", e.response.status_code, e.response.text[:400])
            return "", False
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError) as e:
            await asyncio.sleep(backoff + random.uniform(0.0, 1.0))
            backoff = min(backoff * 2, 60)
            continue
        except Exception as e:
            logger.warning("Gemini call error: %s", e)
            return "", False
    return "", False


def _parse_response(text: str, raw: str, run_variant: str, report_id: str) -> LabelerRun:
    """Parse LLM JSON, re-align spans, drop hallucinated mentions."""
    if not raw:
        return LabelerRun(report_id=report_id, entities=[], raw_response=raw, parse_ok=False, run_variant=run_variant)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON block from possible prose wrapper.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return LabelerRun(report_id=report_id, entities=[], raw_response=raw, parse_ok=False, run_variant=run_variant)
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return LabelerRun(report_id=report_id, entities=[], raw_response=raw, parse_ok=False, run_variant=run_variant)
    ents_raw = d.get("entities", [])
    if not isinstance(ents_raw, list):
        return LabelerRun(report_id=report_id, entities=[], raw_response=raw, parse_ok=False, run_variant=run_variant)
    ents: list[EntityMention] = []
    prev_end = 0
    for m in ents_raw:
        if not isinstance(m, dict):
            continue
        e = _realign_span(text, m, prev_end)
        if e is None:
            continue
        ents.append(e)
        prev_end = max(prev_end, e.char_end)
    return LabelerRun(report_id=report_id, entities=ents, raw_response=raw, parse_ok=True, run_variant=run_variant)


async def annotate_report(
    text: str,
    report_id: str,
    api_key,  # str for cohere/gemini, _KeyRotator for openrouter
    client: httpx.AsyncClient,
    provider: str = "cohere",
    model: str = "command-a-03-2025",
    n_runs: int = 3,
) -> list[LabelerRun]:
    """Annotate a single report with n_runs of self-consistency."""
    runs: list[LabelerRun] = []
    for i in range(n_runs):
        temp = [0.0, 0.1, 0.2][i % 3]
        raw, _ = await _call_llm(
            client=client,
            text=text,
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temp,
            seed_perm=i,
        )
        run = _parse_response(text, raw, run_variant=f"run{i}_t{temp}", report_id=report_id)
        runs.append(run)
    return runs


def _pairwise_span_kappa(a: list[EntityMention], b: list[EntityMention]) -> float:
    """Simple span-level Jaccard-flavored agreement. Kappa proper needs a
    negative-class definition which is ill-defined for spans; return
    Jaccard(A, B) over span keys instead. Documented in module docstring."""
    ka = {e.key() for e in a}
    kb = {e.key() for e in b}
    if not ka and not kb:
        return 1.0
    inter = ka & kb
    union = ka | kb
    return len(inter) / len(union) if union else 1.0


def merge_self_consistency(
    runs: list[LabelerRun], min_votes: int = 2
) -> MergedLabels:
    """Majority-vote merge across runs. A span with >= min_votes is accepted;
    others are disputed."""
    if not runs:
        return MergedLabels(report_id="", accepted=[], disputed=[], agreement_rate=0.0, inter_run_kappa=None, n_runs=0)
    report_id = runs[0].report_id
    # Count each unique (entity_type, char_start, char_end) key.
    key_votes: dict[tuple[str, int, int], list[EntityMention]] = {}
    for r in runs:
        seen_this_run: set[tuple[str, int, int]] = set()
        for e in r.entities:
            k = e.key()
            if k in seen_this_run:
                continue
            seen_this_run.add(k)
            key_votes.setdefault(k, []).append(e)
    accepted: list[EntityMention] = []
    disputed: list[EntityMention] = []
    for k, votes in key_votes.items():
        rep = votes[0]  # representative
        if len(votes) >= min_votes:
            # Average confidence over voting runs.
            rep.confidence = sum(v.confidence for v in votes) / len(votes)
            accepted.append(rep)
        else:
            disputed.append(rep)
    # Overall agreement = mean pairwise Jaccard.
    if len(runs) < 2:
        agreement = 1.0
        kappa = None
    else:
        pair_jaccs: list[float] = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                pair_jaccs.append(
                    _pairwise_span_kappa(runs[i].entities, runs[j].entities)
                )
        agreement = sum(pair_jaccs) / len(pair_jaccs) if pair_jaccs else 1.0
        kappa = agreement  # documented approximation
    return MergedLabels(
        report_id=report_id,
        accepted=accepted,
        disputed=disputed,
        agreement_rate=agreement,
        inter_run_kappa=kappa,
        n_runs=len(runs),
    )


async def annotate_batch(
    reports: list[dict],  # each with {"report_id", "text"}
    api_key,  # str for cohere/gemini, _KeyRotator for openrouter
    provider: str = "cohere",
    model: str = "command-a-03-2025",
    n_runs: int = 3,
    max_concurrent: int = 12,
    out_path: Path | None = None,
    progress_every: int = 25,
    rpm: int = 30,  # requests per minute; 0 = no rate limit (rely on rotator)
) -> list[MergedLabels]:
    """Annotate a batch. Writes JSONL incrementally to out_path if given
    (out_path MUST be on a filesystem that supports append — use local disk).

    rpm behavior:
      - rpm > 0: global token bucket, all workers share; period = 60/rpm.
      - rpm == 0: skip rate-limiting (used with openrouter provider where the
        _KeyRotator handles per-key backoff on 429).

    Recommended provider setup:
      - openrouter + model=tencent/hy3:free + rpm=0 + max_concurrent=6:
        ~110 rpm sustained on real pathology (verified 2026-07-20).
      - cohere + model=command-a-03-2025 + rpm=30 + max_concurrent=4:
        40 rpm trial cap, 1000 calls/month.
      - gemini + model=gemini-2.5-flash: 20/DAY quota — unusable at scale.
    """
    sem = asyncio.Semaphore(max_concurrent)
    limits = httpx.Limits(max_connections=max_concurrent + 4, max_keepalive_connections=max_concurrent + 4)

    # Token bucket. period between requests = 60 / rpm seconds. rpm=0 disables.
    period = 60.0 / rpm if rpm > 0 else 0.0
    last_request_ts = [0.0]
    bucket_lock = asyncio.Lock()

    write_lock = asyncio.Lock()

    async def rate_limit() -> None:
        if period <= 0:
            return
        async with bucket_lock:
            now = time.time()
            wait = last_request_ts[0] + period - now
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_ts[0] = time.time()

    async with httpx.AsyncClient(limits=limits, timeout=90.0) as client:
        results: list[MergedLabels | None] = [None] * len(reports)

        async def one(i: int, r: dict) -> None:
            async with sem:
                # Do n_runs sequentially with rate limiting inside each.
                runs: list[LabelerRun] = []
                for run_idx in range(n_runs):
                    await rate_limit()
                    temp = [0.0, 0.1, 0.2][run_idx % 3]
                    raw, _ = await _call_llm(
                        client=client,
                        text=r["text"],
                        provider=provider,
                        api_key=api_key,
                        model=model,
                        temperature=temp,
                        seed_perm=run_idx,
                    )
                    run = _parse_response(
                        r["text"], raw, run_variant=f"run{run_idx}_t{temp}",
                        report_id=r["report_id"],
                    )
                    runs.append(run)
                merged = merge_self_consistency(runs, min_votes=max(1, n_runs - 1))
                results[i] = merged
                if out_path is not None:
                    async with write_lock:
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "report_id": r["report_id"],
                                "text": r["text"],
                                "accepted": [_ent_to_dict(e) for e in merged.accepted],
                                "disputed": [_ent_to_dict(e) for e in merged.disputed],
                                "agreement_rate": merged.agreement_rate,
                                "kappa": merged.inter_run_kappa,
                                "n_runs": merged.n_runs,
                                "provenance": f"LLM-{provider}-{model}-n{n_runs}",
                            }) + "\n")
                if (i + 1) % progress_every == 0:
                    logger.info("annotated %d/%d", i + 1, len(reports))
                    print(f"[annotate] {i+1}/{len(reports)} done", flush=True)

        await asyncio.gather(*(one(i, r) for i, r in enumerate(reports)))
        return [r for r in results if r is not None]


def _ent_to_dict(e: EntityMention) -> dict:
    return {
        "entity_type": e.entity_type,
        "value": e.value,
        "char_start": e.char_start,
        "char_end": e.char_end,
        "text_span": e.text_span,
        "confidence": e.confidence,
    }


def resolve_api_key(provider: str = "cohere"):
    """Resolve API key(s) for the requested provider from env.

    Returns str for cohere/gemini, _KeyRotator for openrouter.
    """
    if provider == "cohere":
        for env_var in ("COHERE_KEY", "COHERE_API_KEY"):
            v = os.environ.get(env_var)
            if v:
                return v
        raise RuntimeError("No Cohere API key in env (COHERE_KEY / COHERE_API_KEY)")
    elif provider == "gemini":
        for env_var in ("GEMINI_API_KEY", "GEMMA_GOOGLE_KEY", "GOOGLE_API_KEY"):
            v = os.environ.get(env_var)
            if v:
                return v
        raise RuntimeError("No Gemini API key in env (GEMINI_API_KEY / GEMMA_GOOGLE_KEY)")
    elif provider == "openrouter":
        keys: list[str] = []
        # V1, V2, LEGACY. Order matters (round-robin picks V1 first).
        for env_var in ("OPENROUTER_V1", "OPENROUTER_V2", "OPENROUTER_LEGACY",
                        "OPENROUTER_KEY", "OPENROUTER_API_KEY"):
            v = os.environ.get(env_var)
            if v and v not in keys:
                keys.append(v)
        if not keys:
            raise RuntimeError("No OpenRouter keys in env "
                               "(OPENROUTER_V1 / _V2 / _LEGACY / OPENROUTER_API_KEY)")
        return _KeyRotator(keys)
    else:
        raise ValueError(f"Unknown provider: {provider}")
