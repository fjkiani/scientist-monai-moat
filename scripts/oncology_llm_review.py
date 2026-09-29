#!/usr/bin/env python3
"""Optional advisory LLM review of a deterministic oncology gauntlet report."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

MAX_REPORT_CHARS = 60_000


def extract_text(response: dict) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--report", default="artifacts/gauntlet/report.json"
    )
    parser.add_argument(
        "--output", default="artifacts/gauntlet/llm-review.md"
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        output.write_text(
            "# Advisory LLM review\n\n"
            "Skipped: `OPENAI_API_KEY` is not configured. "
            "Deterministic verdict remains authoritative.\n"
        )
        return 0

    report_path = (repo / args.report).resolve()
    if not report_path.is_file() or report_path.is_symlink():
        output.write_text(
            "# Advisory LLM review\n\n"
            "Skipped: trusted deterministic report is unavailable.\n"
        )
        return 0
    report = json.loads(report_path.read_text()[:MAX_REPORT_CHARS])
    skill_index = (repo / ".cursor/skills/INDEX.md").read_text()
    prompt = f"""
You are the adversarial Oncology Delivery reviewer for scientist-monai-moat.
The deterministic gauntlet is authoritative: you MUST NOT turn REJECT into
PASS. Diagnose exactly what the pushing agent failed to deliver and return a
concise markdown rejection package with:
1. Verdict (copy deterministic verdict exactly).
2. Capability-by-capability missing artifacts.
3. Concrete file-level fixes and commands.
4. Any suspicious receipt theater, fake training, synthetic data, missing
   model SHA, or unwired checkpoint.
5. A resubmission checklist.

No capability may be deleted, parked, deferred, or replaced by warnings.
Never print or request secrets.

SKILL INDEX:
{skill_index}

DETERMINISTIC REPORT:
{json.dumps(report, indent=2)}

No candidate files, source, diffs, or secrets are available to you. Diagnose
only the trusted deterministic report above.
""".strip()
    body = {
        "model": os.getenv("ONCOLOGY_REVIEW_MODEL") or "gpt-5.2",
        "input": prompt,
        "max_output_tokens": 4000,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        output.write_text(
            "# Advisory LLM review\n\n"
            f"LLM call failed: `{type(error).__name__}`. "
            "Deterministic verdict remains authoritative.\n"
        )
        return 0

    text = extract_text(payload)
    if not text:
        text = (
            "LLM returned no text. Deterministic verdict remains authoritative."
        )
    output.write_text("# Advisory LLM review\n\n" + text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
