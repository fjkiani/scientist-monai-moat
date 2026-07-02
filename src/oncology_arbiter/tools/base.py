"""Tool protocol shared by every tool the arbiters and orchestrator will call.

Ported from Co-Scientist `co_scientist/tools/base.py`. SQLite/aiosqlite ctx is
dropped for Phase 1; a bare dict is used for the persistence hook. When Phase
5 wires the full Co-Scientist orchestrator we swap the persistence hook back
to aiosqlite for durable session state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCtx:
    """Per-call context passed to every tool invocation."""

    artifacts_dir: Path                      # where tool_runs land
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None                # ULID for this invocation
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Standard result envelope."""

    is_error: bool = False
    content: Any = None
    artifact_path: str | None = None
    error_message: str | None = None
    duration_ms: int = 0
    result_bytes: int = 0


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]             # JSONSchema

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult: ...


def to_anthropic_tool(t: Tool) -> dict[str, Any]:
    """Render a Tool as the dict Anthropic / OpenAI-tool-format param expects."""
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }
