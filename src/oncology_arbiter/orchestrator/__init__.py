"""Orchestrator layer — the Co-Scientist-inspired reasoning loop.

Phase 1 ships only the core primitives: the ReflectionLoopResult envelope and
the evidence-honesty gate that filters model-claimed URLs against tools that
were actually invoked. The full multi-agent tournament (Generation → Reflection
→ Ranking → Evolution → Supervisor) lands in Phase 5.
"""
from __future__ import annotations

from .reflection import (
    LoopResult,
    filter_evidence,
    reflect_and_filter,
)

__all__ = ["LoopResult", "filter_evidence", "reflect_and_filter"]
