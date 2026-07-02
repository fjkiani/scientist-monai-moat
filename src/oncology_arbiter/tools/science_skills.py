"""Bridge to arbitrary skill directories.

For every subdirectory under `<science_skills.path>/skills/` that contains a
SKILL.md, we expose a `ScienceSkillTool` named `<dirname>`.

Ported from `Co-Scientist/co_scientist/tools/science_skills.py`. Kept:
- YAML front-matter parsing with plain-KV fallback
- Path-traversal guard on entrypoint
- Env sanitization
- Anthropic tool-name sanitization regex

Dropped for Phase 1:
- SQLite artifact write (Co-Scientist's `write_json` calls into a session DB).
  We write to disk under `<artifacts_dir>/tool_runs/<skill>/<run_id>.json`.
- The `cfg.secrets` config object — we just read from `os.environ`.

When Phase 5 wires the full Co-Scientist orchestrator, we restore both.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import ToolCtx, ToolResult

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$")


@dataclass
class SkillMeta:
    name: str
    description: str
    entrypoint: Path | None
    timeout_seconds: int = 120
    inputs_schema: dict[str, Any] | None = None
    requires_keys: list[str] = field(default_factory=list)


def parse_skill_md(skill_dir: Path) -> SkillMeta | None:
    """Return SkillMeta if this directory looks like a skill, else None."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    body = skill_md.read_text(errors="ignore")

    front: dict[str, Any] = {}
    m = _FRONT_MATTER_RE.match(body)
    if m:
        try:
            import yaml  # type: ignore[import-not-found]
            front = yaml.safe_load(m.group(1)) or {}
        except Exception:
            for line in m.group(1).splitlines():
                mm = _KV_RE.match(line.strip())
                if mm:
                    front[mm.group(1)] = mm.group(2).strip().strip("\"'")
        body_after = body[m.end():]
    else:
        body_after = body

    name = (front.get("name") or skill_dir.name).strip()
    desc = (front.get("description") or "").strip()
    if not desc:
        for line in body_after.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                desc = line[:400]
                break
    desc = desc or f"skill: {name}"

    # Path-traversal guard: entrypoint must stay inside skill_dir.
    entry: Path | None = None
    if front.get("entrypoint"):
        candidate = (skill_dir / str(front["entrypoint"])).resolve()
        try:
            candidate.relative_to(skill_dir.resolve())
            entry = candidate if candidate.exists() else None
        except ValueError:
            entry = None
    if entry is None:
        scripts = skill_dir / "scripts"
        if scripts.is_dir():
            for cand in ("run.py", "main.py", "cli.py", "run.sh"):
                p = scripts / cand
                if p.exists():
                    entry = p
                    break
            if entry is None:
                files = sorted([p for p in scripts.iterdir() if p.is_file()])
                if len(files) == 1:
                    entry = files[0]

    timeout = int(front.get("timeout_seconds") or 120)
    inputs_schema = front.get("inputs_schema") or front.get("inputs")
    if not isinstance(inputs_schema, dict):
        inputs_schema = None
    requires_keys: list[str] = []
    if isinstance(front.get("requires"), list):
        requires_keys = [str(x) for x in front["requires"]]

    return SkillMeta(
        name=name,
        description=desc,
        entrypoint=entry,
        timeout_seconds=timeout,
        inputs_schema=inputs_schema,
        requires_keys=requires_keys,
    )


def discover_skills(skills_root: Path) -> list[SkillMeta]:
    """Scan skills_root/*/SKILL.md and return all valid SkillMeta objects."""
    if not skills_root.exists():
        return []
    out: list[SkillMeta] = []
    for sub in sorted(skills_root.iterdir()):
        if not sub.is_dir():
            continue
        meta = parse_skill_md(sub)
        if meta is not None:
            out.append(meta)
    return out


class ScienceSkillTool:
    """One tool exposed to the orchestrator per discovered skill."""

    def __init__(self, meta: SkillMeta) -> None:
        self.meta = meta
        self.name = _sanitize_name(meta.name)
        self.description = meta.description[:1024]
        self.input_schema = meta.inputs_schema or {
            "type": "object",
            "properties": {
                "args": {
                    "type": "object",
                    "description": "Free-form arguments forwarded to the skill's script.",
                }
            },
            "required": [],
        }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        if self.meta.entrypoint is None:
            return ToolResult(
                is_error=True,
                error_message=f"skill {self.meta.name!r} has no entrypoint",
            )
        entry = self.meta.entrypoint
        run_id = ctx.run_id or _fallback_run_id()

        env = _sanitized_env(self.meta.requires_keys or [])
        cwd = ctx.artifacts_dir / "tool_runs" / self.meta.name / run_id
        cwd.mkdir(parents=True, exist_ok=True)

        cmd: list[str]
        if entry.suffix == ".py":
            cmd = ["python", str(entry)]
        elif entry.suffix in (".sh", ""):
            cmd = ["bash", str(entry)]
        else:
            cmd = [str(entry)]

        payload_stdin = json.dumps(args.get("args", args)).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=payload_stdin),
                    timeout=self.meta.timeout_seconds,
                )
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                return ToolResult(
                    is_error=True,
                    error_message=f"timeout after {self.meta.timeout_seconds}s",
                )
        except FileNotFoundError as e:
            return ToolResult(
                is_error=True,
                error_message=f"could not exec {cmd[0]}: {e}",
            )

        rc = proc.returncode or 0
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        # Persist raw artifact to disk (SQLite dropped for Phase 1)
        artifact_dir = ctx.artifacts_dir / "tool_runs" / self.meta.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{run_id}.json"
        artifact_path.write_text(json.dumps({
            "skill": self.meta.name,
            "args": args,
            "cmd": cmd,
            "returncode": rc,
            "stdout": stdout_text[:200_000],
            "stderr": stderr_text[:50_000],
        }, indent=2))

        parsed: Any
        parse_error: str | None = None
        try:
            parsed = json.loads(stdout_text) if stdout_text.strip() else {}
        except json.JSONDecodeError as e:
            parsed = {"raw": stdout_text[:8000]}
            parse_error = str(e)

        if rc != 0:
            return ToolResult(
                is_error=True,
                error_message=f"skill {self.meta.name} exit {rc}: {stderr_text[:600]}",
                content=parsed,
                duration_ms=int((time.monotonic() - t0) * 1000),
                artifact_path=str(artifact_path.relative_to(ctx.artifacts_dir)),
            )

        out_content: dict[str, Any] = {"result": parsed}
        if parse_error:
            out_content["parse_error"] = parse_error
        return ToolResult(
            content=out_content,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(stdout_text),
            artifact_path=str(artifact_path.relative_to(ctx.artifacts_dir)),
        )


# ---------------------------- helpers -------------------------------------- #

_ALLOWED_ENV_KEYS = {
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    # API keys downstream scripts may need
    "ANTHROPIC_API_KEY",
    "NCBI_API_KEY",
    "OPENALEX_API_KEY",
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    # Google Health AI Developer Foundations tokens
    "HF_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
}


def _sanitized_env(extra_required: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    allowed = _ALLOWED_ENV_KEYS | set(extra_required)
    for k, v in os.environ.items():
        if k in allowed:
            env[k] = v
    return env


_BAD_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_name(name: str) -> str:
    """Anthropic / OpenAI-style tool names: ^[a-zA-Z0-9_-]{1,64}$."""
    n = _BAD_NAME_RE.sub("_", name.lower()).strip("_")
    return (n or "skill")[:64]


def _fallback_run_id() -> str:
    """Fallback ULID-ish run id when caller didn't provide one."""
    import secrets
    return f"run_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
