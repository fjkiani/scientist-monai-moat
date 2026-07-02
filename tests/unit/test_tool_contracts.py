"""Smoke tests for the L2 evidence tools.

These do NOT hit live endpoints — that would make CI flaky and hostage to
NCBI/arXiv/Europe PMC availability. We only assert:
    1. Tool classes construct without error
    2. name / description / input_schema conform to the Anthropic tool shape
    3. Empty-query short-circuits return is_error=True BEFORE any network call
    4. `to_anthropic_tool` renders each one to the expected dict

A dedicated live-integration suite lands in Phase 5 alongside the orchestrator.
"""
from __future__ import annotations

import pytest

from oncology_arbiter.tools import (
    ArxivSearchTool,
    EuropePMCSearchTool,
    PubmedSearchTool,
    WebFetchTool,
    to_anthropic_tool,
)
from oncology_arbiter.tools.base import ToolCtx


TOOLS_UNDER_TEST = [
    (PubmedSearchTool, "pubmed_search"),
    (ArxivSearchTool, "arxiv_search"),
    (EuropePMCSearchTool, "europe_pmc_search"),
    (WebFetchTool, "web_fetch"),
]


@pytest.mark.parametrize("cls,expected_name", TOOLS_UNDER_TEST)
def test_tool_has_anthropic_shape(cls, expected_name: str) -> None:
    """Every tool must expose (name, description, input_schema)."""
    t = cls()
    assert t.name == expected_name
    assert isinstance(t.description, str) and len(t.description) > 10
    assert isinstance(t.input_schema, dict)
    assert t.input_schema.get("type") == "object"
    assert "properties" in t.input_schema
    rendered = to_anthropic_tool(t)
    assert set(rendered.keys()) == {"name", "description", "input_schema"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,_name", TOOLS_UNDER_TEST[:3])  # skip web_fetch — no query arg
async def test_search_tools_empty_query_short_circuits(
    cls, _name: str, tmp_path
) -> None:
    """Empty query MUST return is_error BEFORE opening an httpx client."""
    t = cls()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await t.call({"query": ""}, ctx)
    assert result.is_error
    assert result.error_message is not None
    assert "empty" in result.error_message.lower()


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_url(tmp_path) -> None:
    """file:// and other non-http(s) schemes must be rejected upfront."""
    t = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await t.call({"url": "file:///etc/passwd"}, ctx)
    assert result.is_error
    assert "http" in result.error_message.lower()


@pytest.mark.asyncio
async def test_web_fetch_ssrf_blocks_loopback(tmp_path) -> None:
    """SSRF: localhost resolution must be blocked BEFORE any network I/O."""
    t = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await t.call(
        {"url": "http://127.0.0.1:8080/aws/latest/meta-data/"}, ctx
    )
    assert result.is_error
    assert (
        "private" in result.error_message.lower()
        or "loopback" in result.error_message.lower()
    )


@pytest.mark.asyncio
async def test_web_fetch_ssrf_blocks_metadata_service(tmp_path) -> None:
    """The AWS metadata service address must be blocked."""
    t = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await t.call({"url": "http://169.254.169.254/latest/meta-data/"}, ctx)
    assert result.is_error
    assert "private" in result.error_message.lower() or "loopback" in result.error_message.lower()
