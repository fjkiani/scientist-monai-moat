"""Helpers for mapping MedGemma Co-Scientist Elo onto case envelopes."""
from __future__ import annotations

from typing import Any


class TumorBoardAssembler:
    """Thin helpers shared by case/full and Co-Scientist contract tests."""

    @staticmethod
    def elo_ranked_from_medgemma(
        output: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Map MedGemma pairwise Elo hypotheses onto the case_full ranked field.

        Empty when Co-Scientist did not execute. Never filled from
        ``/v1/elo/rank`` or ``orchestrator.co_scientist``.
        """
        if not output:
            return []
        ranked: list[dict[str, Any]] = []
        for item in output.get("hypotheses") or []:
            if not isinstance(item, dict):
                continue
            ranked.append({
                "hyp_id": item.get("hypothesis_id"),
                "statement": item.get("claim"),
                "confidence": item.get("initial_confidence"),
                "rating": item.get("elo"),
                "elo": item.get("elo"),
                "stage": item.get("source_scope") or output.get("stage") or "case_full",
                "evidence": [
                    {"url": url} for url in (item.get("evidence_urls") or []) if url
                ],
                "ranking_kind": "medgemma_pairwise_elo",
                "provider": output.get("llm_provider"),
                "tournament_wins": item.get("tournament_wins"),
                "tournament_losses": item.get("tournament_losses"),
                "evidence_alignment": item.get("evidence_alignment"),
            })
        return ranked
