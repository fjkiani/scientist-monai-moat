#!/usr/bin/env python3
"""Driver for LLM entity annotation on pathology reports.

Reads a JSONL of {report_id, text}, writes merged self-consistency labels
to a target JSONL. Skips reports already present in the output (resumable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oncology_arbiter.nlp.llm_labeler import annotate_batch, resolve_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _load_input(fp: Path) -> list[dict]:
    out = []
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _completed_ids(fp: Path) -> set[str]:
    if not fp.exists():
        return set()
    done = set()
    with fp.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add(d["report_id"])
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--provider", default="openrouter",
                    choices=["cohere", "gemini", "openrouter"],
                    help="LLM provider. Default openrouter (tencent/hy3:free with 3-key "
                         "rotation - 110 rpm sustained). cohere/gemini are fallbacks.")
    ap.add_argument("--model", default="tencent/hy3:free",
                    help="Model id. Default tencent/hy3:free via OpenRouter. "
                         "Cohere: command-a-03-2025. Gemini: gemini-2.5-flash.")
    ap.add_argument("--n-runs", type=int, default=3, help="Self-consistency runs.")
    ap.add_argument("--max-concurrent", type=int, default=6)
    ap.add_argument("--rpm", type=int, default=0,
                    help="Requests per minute (0 = disable rate limiting, rely on provider "
                         "rotator/backoff). For cohere use 30. For openrouter with hy3, 0 "
                         "is fine (rotator handles 429).")
    ap.add_argument("--limit", type=int, default=None, help="Cap total reports (dev).")
    args = ap.parse_args()

    api_key = resolve_api_key(provider=args.provider)
    reports = _load_input(args.input)
    if args.limit:
        reports = reports[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_ids(args.output)
    remaining = [r for r in reports if r["report_id"] not in done]
    print(f"[annotate] total={len(reports)} done={len(done)} remaining={len(remaining)}", flush=True)
    if not remaining:
        print("[annotate] nothing to do")
        return

    async def run():
        await annotate_batch(
            reports=remaining,
            api_key=api_key,
            provider=args.provider,
            model=args.model,
            n_runs=args.n_runs,
            max_concurrent=args.max_concurrent,
            out_path=args.output,
            progress_every=25,
            rpm=args.rpm,
        )

    asyncio.run(run())
    print(f"[annotate] complete: {args.output}")


if __name__ == "__main__":
    main()
