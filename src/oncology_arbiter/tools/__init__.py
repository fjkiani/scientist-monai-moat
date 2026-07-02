"""Tool layer — Co-Scientist-compatible tool protocol + our L2 evidence tools.

Lean port of `Co-Scientist/co_scientist/tools/` — enough to (a) run our own
science-skills, (b) fetch PubMed / arXiv / Europe PMC / arbitrary URLs, and
(c) enforce the `seen_urls` honesty filter on any evidence surfaced to an
arbiter.

The SQLite artifact persistence in the upstream is deferred to Phase 5 when
the full multi-agent orchestrator lands; for now, tool artifacts land on disk
under `<artifacts_dir>/tool_runs/<skill>/<run_id>.json` and web-fetch
extractions under `<artifacts_dir>/papers/<sha1(url)>.json`.
"""
from __future__ import annotations

from .arxiv_search import ArxivSearchTool
from .base import Tool, ToolCtx, ToolResult, to_anthropic_tool
from .europe_pmc_search import EuropePMCSearchTool
from .honesty import filter_evidence_by_seen_urls
from .pubmed_search import PubmedSearchTool
from .science_skills import (
    ScienceSkillTool,
    SkillMeta,
    discover_skills,
    parse_skill_md,
)
from .web_fetch import WebFetchTool

__all__ = [
    # Protocol
    "Tool",
    "ToolCtx",
    "ToolResult",
    "to_anthropic_tool",
    # Honesty
    "filter_evidence_by_seen_urls",
    # L2 evidence tools
    "PubmedSearchTool",
    "ArxivSearchTool",
    "EuropePMCSearchTool",
    "WebFetchTool",
    # Skill bridge
    "ScienceSkillTool",
    "SkillMeta",
    "discover_skills",
    "parse_skill_md",
]
