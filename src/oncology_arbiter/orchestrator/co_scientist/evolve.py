"""Phase 4 — generative (live) or string-swap (offline) hypothesis evolution."""
from __future__ import annotations

import json
from typing import Any, Literal

from .const import (
    BIOPSY_EVOLVE_SWAP_MAP,
    DEFAULT_N_VARIANTS,
    DEFAULT_TOP_N_EVOLVE,
    EVOLVE_ALLOWED_URLS_CAP,
    EVOLVE_CONTEXT_JSON_CHARS,
    EVOLVE_RATIONALE_CLIP,
    EVOLVE_SYSTEM,
    MEDGEMMA_EVOLVE_MAX_TOKENS,
    MEDGEMMA_EVOLVE_TEMPERATURE,
    OFFLINE_SCREENING_CONF_DELTAS,
    OFFLINE_THERAPY_LINE_DELTAS,
)
from .json_util import chat_json_with_retry
from .models import EloEntry, Hypothesis


def _evolve_offline(
    ranked: list[EloEntry],
    *,
    top_n: int,
    n_variants: int,
) -> list[Hypothesis]:
    """Legacy string-swap / conf-perturb evolve — offline tests only."""
    variants: list[Hypothesis] = []
    for entry in ranked[:top_n]:
        parent = entry.hypothesis
        if parent.stage == "screening":
            for delta in OFFLINE_SCREENING_CONF_DELTAS[:n_variants]:
                new_conf = max(0.0, min(1.0, parent.confidence + delta))
                variants.append(Hypothesis(
                    hyp_id=f"{parent.hyp_id}:conf{new_conf:.2f}",
                    stage=parent.stage,
                    statement=f"{parent.statement} (perturb conf {delta:+.2f})",
                    confidence=new_conf,
                    evidence=list(parent.evidence),
                    honesty_markers=dict(parent.honesty_markers),
                    derived_from=parent.hyp_id,
                ))
        elif parent.stage == "biopsy":
            new_subtype = None
            for needle, swap_to in BIOPSY_EVOLVE_SWAP_MAP.items():
                if needle in parent.hyp_id:
                    new_subtype = swap_to
                    break
            if new_subtype is None:
                continue
            variants.append(Hypothesis(
                hyp_id=f"{parent.hyp_id}:swap:{new_subtype}",
                stage=parent.stage,
                statement=f"{parent.statement} → swap subtype to {new_subtype}",
                confidence=parent.confidence,
                evidence=list(parent.evidence),
                honesty_markers=dict(parent.honesty_markers),
                derived_from=parent.hyp_id,
            ))
        elif parent.stage == "therapy":
            for delta in OFFLINE_THERAPY_LINE_DELTAS[:n_variants]:
                variants.append(Hypothesis(
                    hyp_id=f"{parent.hyp_id}:line{delta:+d}",
                    stage=parent.stage,
                    statement=f"{parent.statement} → line_of_therapy {delta:+d}",
                    confidence=parent.confidence,
                    evidence=list(parent.evidence),
                    honesty_markers=dict(parent.honesty_markers),
                    derived_from=parent.hyp_id,
                ))
    return variants


def _evolve_medgemma(
    ranked: list[EloEntry],
    *,
    top_n: int,
    n_variants: int,
    patient_context: dict[str, Any] | None,
    llm: Any,
    seen_urls: set[str],
) -> list[Hypothesis]:
    """Generative clinical variants; evidence clipped to ``seen_urls``."""
    variants: list[Hypothesis] = []
    allowed = sorted(seen_urls)[:EVOLVE_ALLOWED_URLS_CAP]
    system = EVOLVE_SYSTEM.replace("exactly N clinical", f"exactly {n_variants} clinical")
    for entry in ranked[:top_n]:
        parent = entry.hypothesis
        user = (
            f"N={n_variants}\n"
            f"Given winning regimen [{parent.statement}] and patient profile "
            f"[{json.dumps(patient_context or {}, ensure_ascii=False)[:EVOLVE_CONTEXT_JSON_CHARS]}], "
            "propose clinical adaptations: (1) A toxicity de-escalation variant, "
            "(2) A combinatorial resistance-interception partner.\n"
            f"STAGE={parent.stage} HYP_ID={parent.hyp_id}\n"
            f"ALLOWED_URLS:\n{json.dumps(allowed, ensure_ascii=False)}\n"
            "JSON:"
        )
        _resp, parsed = chat_json_with_retry(
            llm,
            system=system,
            user=user,
            max_tokens=MEDGEMMA_EVOLVE_MAX_TOKENS,
            temperature=MEDGEMMA_EVOLVE_TEMPERATURE,
        )
        rows = parsed.get("variants") if isinstance(parsed.get("variants"), list) else []
        for idx, row in enumerate(rows[:n_variants]):
            if not isinstance(row, dict):
                continue
            statement = str(row.get("statement") or "").strip()
            if not statement:
                continue
            try:
                conf = float(
                    row.get("confidence")
                    if row.get("confidence") is not None
                    else parent.confidence
                )
            except (TypeError, ValueError):
                conf = parent.confidence
            conf = max(0.0, min(1.0, conf))
            rationale = str(row.get("rationale") or "").strip()
            evidence = [
                e for e in parent.evidence
                if isinstance(e, dict) and e.get("url") in seen_urls
            ]
            variants.append(Hypothesis(
                hyp_id=f"{parent.hyp_id}:gen{idx}:{abs(hash(statement)) % 10_000_000:07d}",
                stage=parent.stage,
                statement=(
                    statement
                    if not rationale
                    else f"{statement} — {rationale[:EVOLVE_RATIONALE_CLIP]}"
                ),
                confidence=conf,
                evidence=evidence,
                honesty_markers={
                    **dict(parent.honesty_markers),
                    "loaded": True,
                    "generative_evolve": True,
                },
                derived_from=parent.hyp_id,
            ))
    return variants


def evolve_hypotheses(
    ranked: list[EloEntry],
    *,
    top_n: int = DEFAULT_TOP_N_EVOLVE,
    n_variants: int = DEFAULT_N_VARIANTS,
    mode: Literal["live", "offline"] = "offline",
    patient_context: dict[str, Any] | None = None,
    llm: Any | None = None,
    seen_urls: set[str] | None = None,
) -> list[Hypothesis]:
    """Spawn variants of the top hypotheses."""
    if mode == "live":
        if llm is None:
            raise RuntimeError("live evolve requires an LLM client with .chat()")
        return _evolve_medgemma(
            ranked,
            top_n=top_n,
            n_variants=n_variants,
            patient_context=patient_context,
            llm=llm,
            seen_urls=set(seen_urls or []),
        )
    return _evolve_offline(ranked, top_n=top_n, n_variants=n_variants)
