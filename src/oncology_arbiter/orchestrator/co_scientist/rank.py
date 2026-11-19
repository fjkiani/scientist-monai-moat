"""Phase 3 — Elo tournament (live MedGemma judge or offline scalar)."""
from __future__ import annotations

import random
from typing import Any, Literal

from .const import (
    ELO_K_FACTOR,
    ELO_SCALE,
    LIVE_MAX_PAIRS,
    OFFLINE_RATIONALE,
    TOURNAMENT_SEED,
)
from .judge import score_match
from .models import EloEntry, Hypothesis
from .offline_scalar import score_hypothesis


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


# Back-compat for unit tests.
_expected = expected_score


def rank_hypotheses(
    hyps: list[Hypothesis],
    *,
    k_factor: int = ELO_K_FACTOR,
    seed: int = TOURNAMENT_SEED,
    mode: Literal["live", "offline"] = "offline",
    patient_context: dict[str, Any] | None = None,
    llm: Any | None = None,
    transcript: list[dict[str, Any]] | None = None,
) -> list[EloEntry]:
    """Round-robin Elo tournament."""
    rng = random.Random(seed)
    entries = {h.hyp_id: EloEntry(hypothesis=h) for h in hyps}
    ids = sorted(entries.keys())
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    rng.shuffle(pairs)
    if mode == "live" and len(pairs) > LIVE_MAX_PAIRS:
        # Prefer therapy↔therapy then therapy↔other so clinical regimens are
        # not starved by the live pair budget (screening/biopsy noise).
        def _pair_priority(pair: tuple[str, str]) -> tuple[int, str, str]:
            a, b = pair
            sa = entries[a].hypothesis.stage
            sb = entries[b].hypothesis.stage
            if sa == "therapy" and sb == "therapy":
                tier = 0
            elif "therapy" in (sa, sb):
                tier = 1
            else:
                tier = 2
            return (tier, a, b)

        pairs = sorted(pairs, key=_pair_priority)[:LIVE_MAX_PAIRS]

    for pair_i, (a, b) in enumerate(pairs, start=1):
        if mode == "live":
            print(
                f"[co_scientist] live match {pair_i}/{len(pairs)}: {a} vs {b}",
                flush=True,
            )
        ea = entries[a]
        eb = entries[b]
        match_rec: dict[str, Any] = {
            "hypothesis_a_id": a,
            "hypothesis_b_id": b,
            "hyp_a": a,
            "hyp_b": b,
            "hypothesis_a_statement": ea.hypothesis.statement,
            "hypothesis_b_statement": eb.hypothesis.statement,
            "elo_a_before": round(ea.rating, 3),
            "elo_b_before": round(eb.rating, 3),
            "mode": mode,
        }
        if mode == "live":
            if llm is None:
                raise RuntimeError("live rank requires an LLM client with .chat()")
            judgment = score_match(
                ea.hypothesis,
                eb.hypothesis,
                patient_context=patient_context,
                llm=llm,
            )
            winner = judgment["winner"]
            match_rec.update(judgment)
            match_rec["rationale"] = judgment.get("clinical_rationale")
            if winner == "A":
                score_a, score_b = 1.0, 0.0
                ea.wins += 1
                eb.losses += 1
            elif winner == "B":
                score_a, score_b = 0.0, 1.0
                ea.losses += 1
                eb.wins += 1
            else:
                score_a, score_b = 0.5, 0.5
                ea.draws += 1
                eb.draws += 1
        else:
            sa = score_hypothesis(ea.hypothesis)
            sb = score_hypothesis(eb.hypothesis)
            if sa > sb:
                score_a, score_b = 1.0, 0.0
                ea.wins += 1
                eb.losses += 1
                match_rec["winner"] = "A"
            elif sb > sa:
                score_a, score_b = 0.0, 1.0
                ea.losses += 1
                eb.wins += 1
                match_rec["winner"] = "B"
            else:
                if a < b:
                    score_a, score_b = 1.0, 0.0
                    ea.wins += 1
                    eb.losses += 1
                    match_rec["winner"] = "A"
                else:
                    score_a, score_b = 0.0, 1.0
                    ea.losses += 1
                    eb.wins += 1
                    match_rec["winner"] = "B"
                ea.draws += 1
                eb.draws += 1
            match_rec["clinical_rationale"] = OFFLINE_RATIONALE
            match_rec["rationale"] = OFFLINE_RATIONALE
            match_rec["margin"] = 0.0
        exp_a = expected_score(ea.rating, eb.rating)
        exp_b = 1.0 - exp_a
        ea.rating += k_factor * (score_a - exp_a)
        eb.rating += k_factor * (score_b - exp_b)
        match_rec["elo_a_after"] = round(ea.rating, 3)
        match_rec["elo_b_after"] = round(eb.rating, 3)
        if transcript is not None:
            transcript.append(match_rec)

    return sorted(
        entries.values(),
        key=lambda e: (-e.rating, -e.wins, e.losses, e.hypothesis.hyp_id),
    )
