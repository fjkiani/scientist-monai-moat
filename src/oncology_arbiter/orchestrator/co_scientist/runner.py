"""Top-level Co-Scientist driver."""
from __future__ import annotations

from typing import Any, Iterable, Literal

from .const import (
    DEFAULT_N_VARIANTS,
    DEFAULT_RETURN_TOP,
    DEFAULT_TOP_N_EVOLVE,
    ENGINE_OFFLINE,
    PHASES,
)
from .evolve import evolve_hypotheses
from .generate import generate_hypotheses
from .llm_backend import build_live_llm, resolve_live_available
from .rank import rank_hypotheses
from .reflect_phase import reflect_hypotheses


def resolve_mode(mode: Literal["live", "offline", "auto"]) -> Literal["live", "offline"]:
    if mode == "live":
        return "live"
    if mode == "offline":
        return "offline"
    return "live" if resolve_live_available() else "offline"


def run_co_scientist(
    *,
    screening: dict[str, Any] | None,
    biopsy: dict[str, Any] | None,
    therapy: dict[str, Any] | None,
    seen_urls: Iterable[str] | None = None,
    top_n_evolve: int = DEFAULT_TOP_N_EVOLVE,
    n_variants: int = DEFAULT_N_VARIANTS,
    return_top: int = DEFAULT_RETURN_TOP,
    patient_context: dict[str, Any] | None = None,
    mode: Literal["live", "offline", "auto"] = "auto",
    llm: Any | None = None,
    llm_backend: str | None = None,
) -> dict[str, Any]:
    """Run generate → reflect → rank → evolve → rank.

    Live mode uses the selected LLM backend (MedGemma / Gemini Argon / …)
    for pairwise judging + generative evolve. Offline mode is the
    deterministic scalar path for ``/v1/offline_ranker/run``.
    """
    resolved = resolve_mode(mode)
    seen_urls_set: set[str] = set(seen_urls or [])
    warnings: list[str] = []
    transcript: list[dict[str, Any]] = []

    live_llm = llm
    engine = ENGINE_OFFLINE
    backend_name: str | None = None
    if resolved == "live":
        if live_llm is None:
            live_llm, engine, backend_name = build_live_llm(backend=llm_backend)  # type: ignore[arg-type]
        else:
            from .llm_backend import resolve_backend
            from .const import engine_id_for_backend

            backend_name = resolve_backend(llm_backend)
            engine = engine_id_for_backend(backend_name)

    ctx = patient_context or {
        "screening_model_state": (screening or {}).get("provenance", {}).get("model_state"),
        "biopsy_subtype": (biopsy or {}).get("subtype_prediction"),
        "biopsy_receptors": (biopsy or {}).get("receptor_panel"),
        "therapy_options": [
            o.get("regimen") for o in ((therapy or {}).get("recommended_options") or [])
        ],
    }

    initial = generate_hypotheses(
        screening=screening, biopsy=biopsy, therapy=therapy,
    )
    initial_count = len(initial)

    reflected, ref_warnings = reflect_hypotheses(initial, seen_urls_set)
    warnings.extend(ref_warnings)
    after_reflect = len(reflected)

    ranked_pre = rank_hypotheses(
        reflected,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        transcript=transcript,
    )

    variants = evolve_hypotheses(
        ranked_pre,
        top_n=top_n_evolve,
        n_variants=n_variants,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        seen_urls=seen_urls_set,
    )
    variants, var_warnings = reflect_hypotheses(variants, seen_urls_set)
    warnings.extend(var_warnings)

    all_hyps = [entry.hypothesis for entry in ranked_pre] + variants
    after_evolve = len(all_hyps)

    ranked_final = rank_hypotheses(
        all_hyps,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        transcript=transcript,
    )

    top = []
    for entry in ranked_final[:return_top]:
        h = entry.hypothesis
        top.append({
            "hyp_id": h.hyp_id,
            "stage": h.stage,
            "statement": h.statement,
            "confidence": h.confidence,
            "evidence": list(h.evidence),
            "honesty_markers": dict(h.honesty_markers),
            "derived_from": h.derived_from,
            "rating": round(entry.rating, 4),
            "wins": entry.wins,
            "losses": entry.losses,
            "draws": entry.draws,
        })

    return {
        "phases": list(PHASES),
        "engine": engine if resolved == "live" else ENGINE_OFFLINE,
        "llm_backend": backend_name if resolved == "live" else "offline",
        "mode": resolved,
        "warnings": warnings,
        "initial_count": initial_count,
        "after_reflect": after_reflect,
        "after_evolve": after_evolve,
        "hypotheses": top,
        "tournament_transcript": transcript,
        "patient_context": ctx,
        "honesty_gate": {
            "seen_urls_count": len(seen_urls_set),
            "warnings": [w for w in warnings if w.startswith("no_evidence_after_reflect:")],
        },
    }
