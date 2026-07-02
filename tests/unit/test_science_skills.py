"""Tests for the science-skills subprocess bridge.

We drive the real subprocess boundary with a tiny inline fixture skill.
Path-traversal is exercised by pointing an entrypoint outside the skill dir.
"""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from oncology_arbiter.tools.base import ToolCtx
from oncology_arbiter.tools.science_skills import (
    ScienceSkillTool,
    discover_skills,
    parse_skill_md,
)


def _make_skill(tmp_path: Path, name: str, script_body: str) -> Path:
    """Create <tmp_path>/skills/<name>/ with SKILL.md + scripts/run.py."""
    skill_dir = tmp_path / "skills" / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: fixture skill for tests\n"
        f"entrypoint: scripts/run.py\ntimeout_seconds: 10\n---\n"
    )
    (skill_dir / "scripts" / "run.py").write_text(textwrap.dedent(script_body))
    return skill_dir


def test_parse_skill_md_reads_frontmatter(tmp_path: Path) -> None:
    _make_skill(tmp_path, "echo_skill", "import sys, json; print(sys.stdin.read())")
    meta = parse_skill_md(tmp_path / "skills" / "echo_skill")
    assert meta is not None
    assert meta.name == "echo_skill"
    assert meta.description == "fixture skill for tests"
    assert meta.timeout_seconds == 10
    assert meta.entrypoint is not None
    assert meta.entrypoint.name == "run.py"


def test_parse_skill_md_returns_none_for_missing_dir(tmp_path: Path) -> None:
    assert parse_skill_md(tmp_path / "no_such_skill") is None


def test_parse_skill_md_rejects_path_traversal(tmp_path: Path) -> None:
    """Entrypoint pointing outside skill_dir must be ignored (returned None)."""
    skill_dir = tmp_path / "skills" / "malicious"
    skill_dir.mkdir(parents=True)
    (tmp_path / "outside.py").write_text("print('escaped')")
    (skill_dir / "SKILL.md").write_text(
        "---\nname: malicious\ndescription: tries to escape\n"
        "entrypoint: ../../outside.py\n---\n"
    )
    meta = parse_skill_md(skill_dir)
    assert meta is not None
    # entrypoint was outside skill_dir → guard rejects → None
    assert meta.entrypoint is None


def test_parse_skill_md_falls_back_to_scripts_run_py(tmp_path: Path) -> None:
    """No entrypoint declared → look for scripts/run.py."""
    skill_dir = tmp_path / "skills" / "fallback"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fallback\ndescription: no entrypoint declared\n---\n"
    )
    (skill_dir / "scripts" / "run.py").write_text("print('{}')")
    meta = parse_skill_md(skill_dir)
    assert meta is not None
    assert meta.entrypoint is not None
    assert meta.entrypoint.name == "run.py"


def test_discover_skills_returns_all_valid(tmp_path: Path) -> None:
    _make_skill(tmp_path, "a", "print('{}')")
    _make_skill(tmp_path, "b", "print('{}')")
    (tmp_path / "skills" / "not_a_skill").mkdir()  # missing SKILL.md — skipped
    metas = discover_skills(tmp_path / "skills")
    names = {m.name for m in metas}
    assert names == {"a", "b"}


@pytest.mark.asyncio
async def test_science_skill_tool_roundtrip(tmp_path: Path) -> None:
    """End-to-end: skill reads stdin JSON, echoes doubled value, tool parses it."""
    _make_skill(
        tmp_path,
        "double",
        """
        import json, sys
        args = json.loads(sys.stdin.read() or "{}")
        result = {"input": args, "doubled": args.get("value", 0) * 2}
        print(json.dumps(result))
        """,
    )
    meta = parse_skill_md(tmp_path / "skills" / "double")
    assert meta is not None
    tool = ScienceSkillTool(meta)
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts", run_id="test_run")
    result = await tool.call({"args": {"value": 21}}, ctx)
    assert not result.is_error, result.error_message
    assert result.content is not None
    assert result.content["result"]["doubled"] == 42
    # artifact was persisted
    assert result.artifact_path is not None
    persisted = json.loads(
        (tmp_path / "artifacts" / result.artifact_path).read_text()
    )
    assert persisted["returncode"] == 0
    assert persisted["skill"] == "double"


@pytest.mark.asyncio
async def test_science_skill_tool_timeout(tmp_path: Path) -> None:
    """A skill that exceeds timeout_seconds must return a timeout error, not hang."""
    _make_skill(
        tmp_path,
        "slow",
        """
        import time, sys
        time.sleep(30)
        print("{}")
        """,
    )
    meta = parse_skill_md(tmp_path / "skills" / "slow")
    assert meta is not None
    meta.timeout_seconds = 1  # override to make the test fast
    tool = ScienceSkillTool(meta)
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts", run_id="slow_run")
    result = await tool.call({"args": {}}, ctx)
    assert result.is_error
    assert result.error_message is not None
    assert "timeout" in result.error_message.lower()


@pytest.mark.asyncio
async def test_science_skill_tool_nonzero_exit(tmp_path: Path) -> None:
    """Non-zero exit code surfaces as is_error with stderr in the message."""
    _make_skill(
        tmp_path,
        "failing",
        """
        import sys
        print("partial output", file=sys.stderr)
        sys.exit(2)
        """,
    )
    meta = parse_skill_md(tmp_path / "skills" / "failing")
    assert meta is not None
    tool = ScienceSkillTool(meta)
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts", run_id="fail_run")
    result = await tool.call({"args": {}}, ctx)
    assert result.is_error
    assert result.error_message is not None
    assert "exit 2" in result.error_message


def test_science_skill_name_sanitization(tmp_path: Path) -> None:
    """Anthropic tool names: only [a-z0-9_-]{1,64}."""
    skill_dir = tmp_path / "skills" / "weird"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: 'Weird Name/With Spaces!'\ndescription: x\n"
        "entrypoint: scripts/run.py\n---\n"
    )
    (skill_dir / "scripts" / "run.py").write_text("print('{}')")
    meta = parse_skill_md(skill_dir)
    assert meta is not None
    tool = ScienceSkillTool(meta)
    assert tool.name.replace("_", "").isalnum() or tool.name == "skill"
    # Anthropic ceiling
    assert len(tool.name) <= 64
