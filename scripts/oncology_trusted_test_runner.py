#!/usr/bin/env python3
"""Import the trusted test runner before exposing candidate source code."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    candidate = Path(args.candidate).resolve(strict=True)
    source = candidate / "src"
    if not source.is_dir():
        print("TRUSTED_TEST_RUNNER_FAIL candidate src/ is missing", file=sys.stderr)
        return 1
    os.chdir(candidate)
    sys.path.insert(0, str(source))
    return int(
        pytest.main(
            [
                "-q",
                "--tb=short",
                "-c",
                "/dev/null",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                *args.tests,
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
