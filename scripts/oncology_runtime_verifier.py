#!/usr/bin/env python3
"""Sandbox-only loadability and declared integration-test verifier.

This script loads candidate joblib artifacts and executes candidate-declared
tests. It must run only inside the guardian's networkless, read-only container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest


def contained(repo: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing path")
    raw = repo / value
    path = raw.resolve(strict=True)
    if raw.is_symlink() or path.is_symlink() or (repo != path and repo not in path.parents):
        raise ValueError(f"unsafe path: {value}")
    if not path.is_file():
        raise ValueError(f"not a file: {value}")
    return path


def estimator_like(value: object) -> bool:
    if callable(getattr(value, "predict", None)) or callable(getattr(value, "predict_proba", None)):
        return True
    if isinstance(value, dict):
        return any(estimator_like(item) for item in value.values())
    return False


def verify_artifact(name: str, artifact: Path) -> None:
    suffix = artifact.suffix.lower()
    if suffix == ".joblib":
        import joblib

        loaded = joblib.load(artifact)
        if not estimator_like(loaded):
            raise ValueError(f"{name}: joblib contains no predictor")
    elif suffix == ".json":
        payload = json.loads(artifact.read_text())
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"{name}: JSON model is empty")
    elif suffix in {".pt", ".pth", ".ckpt", ".bin"}:
        if not zipfile.is_zipfile(artifact):
            raise ValueError(f"{name}: checkpoint is not a zip weight container")
    elif suffix == ".safetensors":
        with artifact.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
        if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
            raise ValueError(f"{name}: safetensors has no tensors")
    elif suffix == ".onnx":
        if artifact.stat().st_size < 100_000:
            raise ValueError(f"{name}: ONNX artifact is implausibly small")
    else:
        raise ValueError(f"{name}: unsupported artifact suffix {suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", default="artifacts/tonight_delivery.json")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    manifest = contained(repo, args.manifest)
    payload = json.loads(manifest.read_text())
    entries = payload.get("capabilities")
    if not isinstance(entries, list) or not entries:
        raise ValueError("delivery manifest capabilities are missing")

    test_paths: list[str] = []
    for entry in entries:
        name = str(entry.get("name", "unnamed"))
        artifact = contained(repo, entry.get("artifact_path"))
        verify_artifact(name, artifact)
        for value in entry.get("test_paths", []):
            path = contained(repo, value)
            if path.suffix != ".py" or "test" not in path.name:
                raise ValueError(f"{name}: invalid test path {value}")
            test_paths.append(str(path))
        print(f"RUNTIME_LOAD_OK {name} {artifact.name}")

    unique_tests = sorted(set(test_paths))
    if args.run_tests:
        if not unique_tests or len(unique_tests) > 50:
            raise ValueError("declared integration-test count must be 1..50")
        os.chdir(repo)
        sys.path.insert(0, str(repo / "src"))
        result = pytest.main(
            [
                "-q",
                "--tb=short",
                "-c",
                "/dev/null",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                *unique_tests,
            ]
        )
        if result:
            return int(result)
        print(f"DECLARED_INTEGRATION_TESTS_OK count={len(unique_tests)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RUNTIME_VERIFIER_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
