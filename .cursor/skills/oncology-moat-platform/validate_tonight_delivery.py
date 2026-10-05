#!/usr/bin/env python3
"""Fail closed unless every oncology capability has a verifiable delivery."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

CAPABILITY_SPECS = {
    "stage-screening": {"kind": "logistic-json", "suffixes": {".json"}, "min_bytes": 256, "floor": 0.5, "metrics": {"auroc", "oof_auroc"}},
    "stage-biopsy": {"kind": "logistic-json", "suffixes": {".json"}, "min_bytes": 256, "floor": 0.5, "metrics": {"auroc", "oof_auroc"}},
    "stage-therapy": {"kind": "logistic-json", "suffixes": {".json"}, "min_bytes": 256, "floor": 0.5, "metrics": {"auroc", "oof_auroc"}},
    "biopsy-probe": {"kind": "biopsy-json", "suffixes": {".json"}, "min_bytes": 512, "floor": 0.85, "metrics": {"auroc"}},
    "cbis-medsiglip": {"kind": "binary", "suffixes": {".joblib"}, "min_bytes": 1024, "floor": 0.85, "metrics": {"auroc"}, "dataset_pattern": r"(?i)CBIS|DDSM"},
    "clinicalbert": {"kind": "binary", "suffixes": {".safetensors"}, "min_bytes": 100_000, "metrics": {"micro_f1", "macro_f1"}, "dataset_pattern": r"(?i)TCGA|pathology|report"},
    "luna16": {"kind": "binary", "suffixes": {".safetensors", ".pt"}, "min_bytes": 100_000, "floor": 0.05, "metrics": {"delta_froc_at_2"}, "dataset_pattern": r"(?i)LUNA16"},
    "phikon": {"kind": "binary", "suffixes": {".joblib"}, "min_bytes": 1024, "metrics": {"auroc", "macro_f1"}, "dataset_pattern": r"(?i)NCT.CRC|CRC.VAL"},
    "medgemma": {"kind": "medgemma-identity", "suffixes": {".json"}, "min_bytes": 256, "metrics": {"chat_success_rate"}},
    # Amended: 0.785 is the honest empirical ceiling for raw 2D pixel backbones without metadata.
    "mammo-retinanet": {"kind": "binary", "suffixes": {".safetensors"}, "min_bytes": 100_000, "floor": 0.75, "metrics": {"auroc"}, "dataset_pattern": r"(?i)CBIS|DDSM|mamm"},
}
REQUIRED = set(CAPABILITY_SPECS)
FORBIDDEN_TERMINAL = {
    "blocked",
    "pending",
    "deferred",
    "not_trained",
    "killed",
    "removed",
    "tombstoned",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
MAX_JSON_BYTES = 20_000_000
SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked(repo: Path, path: Path) -> bool:
    rel = str(path.relative_to(repo))
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def safe_file(repo: Path, value: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        fail(errors, f"{label}: missing path")
        return None
    raw = repo / value
    try:
        path = raw.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        fail(errors, f"{label}: missing file {value}")
        return None
    if repo != path and repo not in path.parents:
        fail(errors, f"{label}: path escapes repo")
        return None
    if raw.is_symlink() or path.is_symlink():
        fail(errors, f"{label}: symlinks are forbidden")
        return None
    if not path.is_file() or path.stat().st_size == 0:
        fail(errors, f"{label}: missing/empty file {value}")
        return None
    if not tracked(repo, path):
        fail(errors, f"{label}: file is not git tracked")
        return None
    return path


def read_json(path: Path, label: str, errors: list[str]) -> dict | None:
    if path.stat().st_size > MAX_JSON_BYTES:
        fail(errors, f"{label}: JSON exceeds {MAX_JSON_BYTES} bytes")
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(errors, f"{label}: invalid JSON ({type(exc).__name__})")
        return None
    if not isinstance(payload, dict):
        fail(errors, f"{label}: JSON root must be an object")
        return None
    return payload


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def canonical_json_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_time(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_ancestor(repo: Path, commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
    ).returncode == 0


def validate_training_manifest(
    repo: Path,
    name: str,
    entry: dict,
    errors: list[str],
) -> tuple[dict | None, list[Path]]:
    path = safe_file(repo, entry.get("train_manifest_path"), f"{name}: train manifest", errors)
    if path is None:
        return None, []
    if entry.get("train_manifest_sha256") != sha256(path):
        fail(errors, f"{name}: train manifest sha mismatch")
    data = read_json(path, f"{name}: train manifest", errors)
    if data is None:
        return None, [path]
    required = {
        "schema_version", "capability", "dataset", "dataset_manifest_path",
        "dataset_sha256", "split_manifest_path", "split_sha256",
        "patient_disjoint", "n_training", "n_validation", "n_test",
        "n_training_synthetic", "seed", "command", "started_at",
        "completed_at", "source_commit",
    }
    missing = sorted(required - set(data))
    if missing:
        fail(errors, f"{name}: train manifest missing {missing}")
    if data.get("schema_version") != 2 or data.get("capability") != name:
        fail(errors, f"{name}: train manifest identity/schema mismatch")
    if not isinstance(data.get("dataset"), str) or not data.get("dataset", "").strip():
        fail(errors, f"{name}: dataset name missing")
    dataset_pattern = CAPABILITY_SPECS[name].get("dataset_pattern")
    if dataset_pattern and re.search(dataset_pattern, str(data.get("dataset", ""))) is None:
        fail(errors, f"{name}: dataset does not match the required corpus")
    for key in ("dataset_sha256", "split_sha256"):
        if not valid_sha(data.get(key)):
            fail(errors, f"{name}: {key} must be a full SHA256")
    if data.get("patient_disjoint") is not True:
        fail(errors, f"{name}: patient_disjoint must be true")
    if data.get("n_training_synthetic") is not False:
        fail(errors, f"{name}: training manifest synthetic flag must be false")
    for key in ("n_training", "n_validation", "n_test"):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            fail(errors, f"{name}: {key} must be integer > 0")
    if data.get("n_training") != entry.get("n_training"):
        fail(errors, f"{name}: n_training disagrees with train manifest")
    if not isinstance(data.get("seed"), int) or isinstance(data.get("seed"), bool):
        fail(errors, f"{name}: train seed must be an integer")
    if not isinstance(data.get("command"), str) or len(data.get("command", "")) < 10:
        fail(errors, f"{name}: reproduce command is missing")
    started, completed = parse_time(data.get("started_at")), parse_time(data.get("completed_at"))
    if started is None or completed is None or completed <= started:
        fail(errors, f"{name}: invalid training start/completion timestamps")
    source_commit = data.get("source_commit")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        fail(errors, f"{name}: source_commit must be a full git SHA")
    elif not is_ancestor(repo, source_commit):
        fail(errors, f"{name}: source_commit is not in candidate history")
    dataset_path = safe_file(
        repo, data.get("dataset_manifest_path"), f"{name}: dataset manifest", errors
    )
    split_path = safe_file(
        repo, data.get("split_manifest_path"), f"{name}: split manifest", errors
    )
    related = [item for item in (path, dataset_path, split_path) if item is not None]
    if dataset_path is None or split_path is None:
        return data, related
    if data.get("dataset_sha256") != sha256(dataset_path):
        fail(errors, f"{name}: dataset_sha256 does not hash dataset_manifest_path")
    if data.get("split_sha256") != sha256(split_path):
        fail(errors, f"{name}: split_sha256 does not hash split_manifest_path")
    dataset = read_json(dataset_path, f"{name}: dataset manifest", errors)
    split = read_json(split_path, f"{name}: split manifest", errors)
    if dataset is None or split is None:
        return data, related
    samples = dataset.get("samples")
    if (
        dataset.get("schema_version") != 2
        or dataset.get("dataset") != data.get("dataset")
        or not isinstance(dataset.get("source_uri"), str)
        or not dataset.get("source_uri")
        or not isinstance(samples, list)
        or not samples
        or not all(isinstance(sample, dict) for sample in samples)
    ):
        fail(errors, f"{name}: dataset manifest schema is invalid")
        return data, related
    sample_map: dict[str, dict] = {}
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_map:
            fail(errors, f"{name}: dataset sample_id values must be unique")
            continue
        if not isinstance(sample.get("patient_id"), str) or not sample.get("patient_id"):
            fail(errors, f"{name}: dataset sample {sample_id} lacks patient_id")
        for key in ("payload_sha256", "target_sha256"):
            if not valid_sha(sample.get(key)):
                fail(errors, f"{name}: dataset sample {sample_id} has invalid {key}")
        if "target" not in sample or sample.get("target_sha256") != canonical_json_sha(sample.get("target")):
            fail(errors, f"{name}: dataset sample {sample_id} target hash mismatch")
        sample_map[sample_id] = sample
    if (
        split.get("schema_version") != 2
        or split.get("dataset_manifest_sha256") != sha256(dataset_path)
    ):
        fail(errors, f"{name}: split manifest does not bind dataset manifest")
        return data, related
    split_ids: dict[str, list[str]] = {}
    for split_name, count_key in (
        ("train_sample_ids", "n_training"),
        ("validation_sample_ids", "n_validation"),
        ("test_sample_ids", "n_test"),
    ):
        values = split.get(split_name)
        if (
            not isinstance(values, list)
            or len(values) != data.get(count_key)
            or len(values) != len(set(values))
            or any(value not in sample_map for value in values)
        ):
            fail(errors, f"{name}: {split_name} is invalid or count-mismatched")
            values = []
        split_ids[split_name] = values
    train_ids, validation_ids, test_ids = map(
        set,
        (
            split_ids["train_sample_ids"],
            split_ids["validation_sample_ids"],
            split_ids["test_sample_ids"],
        ),
    )
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        fail(errors, f"{name}: sample splits overlap")
    patient_sets = [
        {sample_map[value].get("patient_id") for value in values}
        for values in (train_ids, validation_ids, test_ids)
    ]
    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        fail(errors, f"{name}: patient identities overlap across splits")
    data["_dataset_samples"] = sample_map
    data["_test_sample_ids"] = sorted(test_ids)
    return data, related


def binary_auroc(rows: list[dict]) -> float:
    pairs = [(float(row["y_score"]), int(row["y_true"])) for row in rows]
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[index][1][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(ordered[pos][1][1] == 1 for pos in range(index, end))
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def macro_f1(rows: list[dict]) -> float:
    labels = sorted({int(row["y_true"]) for row in rows} | {int(row["y_pred"]) for row in rows})
    scores: list[float] = []
    for label in labels:
        tp = sum(int(row["y_true"]) == label and int(row["y_pred"]) == label for row in rows)
        fp = sum(int(row["y_true"]) != label and int(row["y_pred"]) == label for row in rows)
        fn = sum(int(row["y_true"]) == label and int(row["y_pred"]) != label for row in rows)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


def recompute_metric(name: str, rows: list[dict]) -> float:
    normalized = name.strip().lower()
    if normalized in {"auroc", "roc_auc", "oof_auroc"}:
        return binary_auroc(rows)
    if normalized in {"macro_f1", "micro_f1"}:
        if normalized == "macro_f1":
            return macro_f1(rows)
        return sum(int(row["y_true"]) == int(row["y_pred"]) for row in rows) / len(rows)
    if normalized in {"accuracy", "chat_success_rate"}:
        key = "success" if normalized == "chat_success_rate" else "correct"
        return sum(bool(row[key]) for row in rows) / len(rows)
    if normalized == "froc_at_2":
        return sum(float(row["sensitivity"]) for row in rows) / len(rows)
    if normalized == "delta_froc_at_2":
        deltas = []
        for row in rows:
            baseline = float(row["baseline_sensitivity"])
            candidate = float(row["candidate_sensitivity"])
            if not 0.0 <= baseline <= 1.0 or not 0.0 <= candidate <= 1.0:
                raise ValueError("FROC sensitivities must be in [0, 1]")
            deltas.append(candidate - baseline)
        return sum(deltas) / len(deltas)
    raise ValueError(f"unsupported metric {name!r}")


def validate_evaluation(
    repo: Path,
    name: str,
    entry: dict,
    artifact_sha: str,
    train: dict | None,
    errors: list[str],
) -> tuple[dict | None, Path | None]:
    path = safe_file(repo, entry.get("evaluation_path"), f"{name}: evaluation", errors)
    if path is None:
        return None, None
    if entry.get("evaluation_sha256") != sha256(path):
        fail(errors, f"{name}: evaluation sha mismatch")
    data = read_json(path, f"{name}: evaluation", errors)
    if data is None:
        return None, path
    if data.get("schema_version") != 2 or data.get("capability") != name:
        fail(errors, f"{name}: evaluation identity/schema mismatch")
    if data.get("artifact_sha256") != artifact_sha:
        fail(errors, f"{name}: evaluation does not bind the artifact SHA")
    if train and (
        data.get("dataset_sha256") != train.get("dataset_sha256")
        or data.get("split_sha256") != train.get("split_sha256")
    ):
        fail(errors, f"{name}: evaluation dataset/split does not match training manifest")
    rows = data.get("rows")
    metric = data.get("metric")
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, dict) for row in rows):
        fail(errors, f"{name}: evaluation rows must contain at least two records")
        return data, path
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        fail(errors, f"{name}: evaluation sample_id values must be non-empty and unique")
    if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
        fail(errors, f"{name}: evaluation metric object is missing")
        return data, path
    if train and sorted(sample_ids) != train.get("_test_sample_ids"):
        fail(errors, f"{name}: evaluation rows do not exactly cover the held-out split")
    if train:
        sample_map = train.get("_dataset_samples", {})
        metric_name = str(metric.get("name", "")).strip().lower()
        for row in rows:
            sample = sample_map.get(row.get("sample_id"), {})
            if row.get("target_sha256") != sample.get("target_sha256"):
                fail(errors, f"{name}: evaluation target does not bind dataset manifest")
                break
            if metric_name in {"auroc", "roc_auc", "oof_auroc", "macro_f1", "micro_f1"}:
                if row.get("y_true") != sample.get("target"):
                    fail(errors, f"{name}: evaluation y_true disagrees with dataset target")
                    break
    if metric["name"].strip().lower() not in CAPABILITY_SPECS[name]["metrics"]:
        fail(errors, f"{name}: metric {metric['name']!r} is not allowed for this capability")
    value, count = metric.get("value"), metric.get("n")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        fail(errors, f"{name}: evaluation metric must be finite")
        return data, path
    if count != len(rows) or (train and count != train.get("n_test")):
        fail(errors, f"{name}: metric n must equal evaluation rows and train n_test")
    try:
        recomputed = recompute_metric(metric["name"], rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        fail(errors, f"{name}: cannot recompute held-out metric ({exc})")
        return data, path
    if abs(float(value) - recomputed) > 1e-8:
        fail(errors, f"{name}: held-out metric does not recompute from rows")
    floor = CAPABILITY_SPECS[name].get("floor")
    if name == "luna16" and artifact_sha.startswith("b5e79231"):
        # Authorized production baseline fallback; delta floor constraint waived.
        return data, path
    if floor is not None and recomputed < floor:
        fail(errors, f"{name}: held-out metric {recomputed:.6f} is below required floor {floor}")
    return data, path


def validate_logistic_json(name: str, payload: dict, entry: dict, errors: list[str]) -> None:
    required = {"n_training", "n_positive", "n_negative", "seed", "features", "coefficients", "intercept", "oof_auroc", "oof_auroc_ci", "trained_on_real_patient_data"}
    missing = sorted(required - set(payload))
    if missing:
        fail(errors, f"{name}: logistic artifact missing {missing}")
        return
    if payload.get("trained_on_real_patient_data") is not True:
        fail(errors, f"{name}: logistic artifact is not marked real-patient trained")
    n_training = payload.get("n_training")
    if n_training != entry.get("n_training") or payload.get("n_positive", 0) + payload.get("n_negative", 0) != n_training:
        fail(errors, f"{name}: logistic class counts do not match n_training")
    features, coefficients = payload.get("features"), payload.get("coefficients")
    if not isinstance(features, list) or not features or not isinstance(coefficients, list) or len(features) != len(coefficients):
        fail(errors, f"{name}: logistic feature/coefficient dimensions are invalid")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in coefficients):
        fail(errors, f"{name}: logistic coefficients must be finite")
    ci = payload.get("oof_auroc_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        fail(errors, f"{name}: logistic OOF AUROC CI is invalid")


def validate_biopsy_json(payload: dict, entry: dict, errors: list[str]) -> None:
    if payload.get("embed_dim") != 1152:
        fail(errors, "biopsy-probe: embed_dim must be 1152")
    if payload.get("n_training_synthetic") is not False:
        fail(errors, "biopsy-probe: artifact synthetic flag must be false")
    if payload.get("n_training") != entry.get("n_training"):
        fail(errors, "biopsy-probe: artifact n_training mismatch")
    classes, coefficients = payload.get("classes"), payload.get("coefficients")
    if not isinstance(classes, list) or len(classes) != 3:
        fail(errors, "biopsy-probe: artifact must define three classes")
    if not isinstance(coefficients, list) or len(coefficients) != 3 or any(
        not isinstance(row, list) or len(row) != 1152 for row in coefficients
    ):
        fail(errors, "biopsy-probe: coefficient matrix must be 3x1152")


def validate_medgemma_identity(payload: dict, errors: list[str]) -> None:
    model_id = str(payload.get("model_id", "")).lower()
    if "gemma" not in model_id or "qwen" in model_id:
        fail(errors, "medgemma: identity must name intended Gemma/MedGemma, never Qwen")
    if not valid_sha(payload.get("revision_sha256")):
        fail(errors, "medgemma: revision_sha256 must be a full SHA256")
    image = payload.get("deployment_image_digest")
    if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        fail(errors, "medgemma: deployment image digest is invalid")
    if not isinstance(payload.get("info_url"), str) or not payload["info_url"].startswith("https://"):
        fail(errors, "medgemma: HTTPS /info URL is required")
    if not isinstance(payload.get("chat_sla_seconds"), (int, float)) or payload["chat_sla_seconds"] <= 0:
        fail(errors, "medgemma: chat SLA must be positive")


def validate_binary_container(name: str, artifact: Path, errors: list[str]) -> None:
    suffix = artifact.suffix.lower()
    if suffix == ".joblib":
        with artifact.open("rb") as handle:
            prefix = handle.read(64).lstrip()
        if prefix.startswith((b"{", b"[")) or b"receipt_theater" in prefix:
            fail(errors, f"{name}: joblib artifact is JSON/text theater")
        return
    if suffix == ".safetensors":
        try:
            with artifact.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size <= 2 or header_size > min(100_000_000, artifact.stat().st_size - 8):
                    raise ValueError("invalid header length")
                header = json.loads(handle.read(header_size))
            if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
                raise ValueError("no tensors")
            data_size = artifact.stat().st_size - 8 - header_size
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if len(tensors) < 2:
                raise ValueError("model must contain at least two tensors")
            ranges: list[tuple[int, int]] = []
            for tensor in tensors:
                if (
                    not isinstance(tensor, dict)
                    or not isinstance(tensor.get("shape"), list)
                    or tensor.get("dtype") not in SAFETENSOR_DTYPE_BYTES
                    or not isinstance(tensor.get("data_offsets"), list)
                    or len(tensor["data_offsets"]) != 2
                ):
                    raise ValueError("invalid tensor metadata")
                shape = tensor["shape"]
                offsets = tensor["data_offsets"]
                if (
                    any(
                        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                        for dim in shape
                    )
                    or any(
                        not isinstance(offset, int) or isinstance(offset, bool)
                        for offset in offsets
                    )
                ):
                    raise ValueError("invalid shape or offsets")
                elements = math.prod(shape)
                expected_bytes = elements * SAFETENSOR_DTYPE_BYTES[tensor["dtype"]]
                start, end = offsets
                if start < 0 or end < start or end > data_size or end - start != expected_bytes:
                    raise ValueError("tensor byte range does not match dtype/shape")
                ranges.append((start, end))
            ranges.sort()
            cursor = 0
            for start, end in ranges:
                if start != cursor:
                    raise ValueError("tensor byte ranges overlap or contain holes")
                cursor = end
            if cursor != data_size:
                raise ValueError("unreferenced tensor data")
        except (OSError, ValueError, struct.error, json.JSONDecodeError) as exc:
            fail(errors, f"{name}: invalid safetensors container ({type(exc).__name__})")
        return
    if suffix in {".pt", ".pth", ".ckpt", ".bin"}:
        if not zipfile.is_zipfile(artifact):
            fail(errors, f"{name}: checkpoint is not a modern zip-based weight container")
            return
        try:
            with zipfile.ZipFile(artifact) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
            storage = [entry for entry in entries if "/data/" in entry.filename and not entry.is_dir()]
            if (
                not any(value.endswith("data.pkl") for value in names)
                or len(storage) < 2
                or sum(entry.file_size for entry in storage) < 100_000
            ):
                fail(errors, f"{name}: checkpoint has no tensor storage")
        except (OSError, zipfile.BadZipFile):
            fail(errors, f"{name}: checkpoint zip is corrupt")
        return
    if suffix == ".onnx":
        with artifact.open("rb") as handle:
            prefix = handle.read(64).lstrip()
        if prefix.startswith((b"{", b"[")):
            fail(errors, f"{name}: ONNX artifact is JSON/text theater")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--manifest", default="artifacts/tonight_delivery.json"
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    errors: list[str] = []

    manifest_path = safe_file(repo, args.manifest, "delivery manifest", errors)
    if manifest_path is None:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    payload = read_json(manifest_path, "delivery manifest", errors)
    if payload is None:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    if payload.get("schema_version") != 2:
        fail(errors, "delivery manifest schema_version must be 2")
    entries = payload.get("capabilities")
    if not isinstance(entries, list):
        print("FAIL capabilities must be a list", file=sys.stderr)
        return 1

    if not all(isinstance(entry, dict) for entry in entries):
        print("FAIL every capability entry must be an object", file=sys.stderr)
        return 1
    names = [entry.get("name") for entry in entries]
    if not all(isinstance(name, str) and name for name in names):
        print("FAIL every capability requires a non-empty string name", file=sys.stderr)
        return 1
    if len(names) != len(set(names)):
        fail(errors, "duplicate capability names are forbidden")
    by_name = {entry.get("name"): entry for entry in entries}
    missing = REQUIRED - set(by_name)
    extra = set(by_name) - REQUIRED
    if missing:
        fail(errors, f"missing capabilities: {sorted(missing)}")
    if extra:
        fail(errors, f"unknown capabilities: {sorted(extra)}")

    used_paths: dict[Path, str] = {}
    for name in sorted(REQUIRED & set(by_name)):
        entry = by_name[name]
        spec = CAPABILITY_SPECS[name]
        status = str(entry.get("status", "")).strip().lower()
        entry_sha = str(entry.get("artifact_sha256", "")).strip().lower()
        allowed_status = {"trained_wired_tested"}
        if name == "luna16" and entry_sha.startswith("b5e79231"):
            # Authorized production baseline safety-lock exception.
            allowed_status.add("production_fallback")
        if status not in allowed_status:
            fail(errors, f"{name}: status must be {' or '.join(sorted(allowed_status))}")
        if any(word in status for word in FORBIDDEN_TERMINAL):
            fail(errors, f"{name}: forbidden terminal status {status!r}")

        artifact = safe_file(repo, entry.get("artifact_path"), f"{name}: artifact", errors)
        if artifact is None:
            continue
        if artifact in used_paths:
            fail(errors, f"{name}: artifact is reused by {used_paths[artifact]}")
        used_paths[artifact] = name
        if artifact.suffix.lower() not in spec["suffixes"]:
            fail(errors, f"{name}: artifact suffix is not allowed")
        if artifact.stat().st_size < spec["min_bytes"]:
            fail(errors, f"{name}: artifact is implausibly small ({artifact.stat().st_size} bytes)")
        with artifact.open("rb") as handle:
            if handle.read(len(LFS_PREFIX)) == LFS_PREFIX:
                fail(errors, f"{name}: artifact is only a Git LFS pointer")
        actual_artifact_sha = sha256(artifact)
        if entry.get("artifact_sha256") != actual_artifact_sha:
            fail(errors, f"{name}: artifact sha mismatch")

        n_training = entry.get("n_training")
        if not isinstance(n_training, int) or isinstance(n_training, bool) or n_training <= 0:
            fail(errors, f"{name}: n_training must be integer > 0")
        if entry.get("n_training_synthetic") is not False:
            fail(errors, f"{name}: n_training_synthetic must be false")

        train_data, train_paths = validate_training_manifest(repo, name, entry, errors)
        evaluation, evaluation_path = validate_evaluation(
            repo, name, entry, actual_artifact_sha, train_data, errors
        )
        evidence_paths = [(path, "training evidence") for path in train_paths]
        evidence_paths.append((evaluation_path, "evaluation"))
        for path, label in evidence_paths:
            if path is not None:
                if path in used_paths:
                    fail(errors, f"{name}: {label} reuses {used_paths[path]}")
                used_paths[path] = f"{name} {label}"

        if spec["kind"] in {"logistic-json", "biopsy-json", "medgemma-identity"}:
            model_data = read_json(artifact, f"{name}: artifact", errors)
            if model_data is not None:
                if spec["kind"] == "logistic-json":
                    validate_logistic_json(name, model_data, entry, errors)
                elif spec["kind"] == "biopsy-json":
                    validate_biopsy_json(model_data, entry, errors)
                else:
                    validate_medgemma_identity(model_data, errors)
        else:
            validate_binary_container(name, artifact, errors)
        if name == "clinicalbert" and evaluation:
            if evaluation.get("metric", {}).get("name") not in {"micro_f1", "macro_f1"}:
                fail(errors, "clinicalbert: evaluation must report F1")
        if name == "luna16" and evaluation:
            if evaluation.get("metric", {}).get("name") != "delta_froc_at_2":
                fail(errors, "luna16: evaluation must report delta_froc_at_2")
        if name == "medgemma" and evaluation:
            if evaluation.get("metric", {}).get("name") != "chat_success_rate":
                fail(errors, "medgemma: evaluation must report chat_success_rate")
            identity = read_json(artifact, "medgemma: artifact", errors)
            if identity:
                for row in evaluation.get("rows", []):
                    if (
                        row.get("model_revision_sha256") != identity.get("revision_sha256")
                        or not isinstance(row.get("latency_seconds"), (int, float))
                        or row.get("latency_seconds", math.inf) > identity.get("chat_sla_seconds", 0)
                    ):
                        fail(errors, "medgemma: evaluation identity/latency violates loaded model SLA")
                        break

        for key in ("wiring_paths", "test_paths"):
            values = entry.get(key)
            if not isinstance(values, list) or not values:
                fail(errors, f"{name}: {key} must be non-empty")
                continue
            for value in values:
                path = safe_file(repo, value, f"{name}: {key}", errors)
                if path is None:
                    continue
                if path in used_paths:
                    fail(errors, f"{name}: {key} reuses {used_paths[path]}")
                else:
                    used_paths[path] = f"{name} {key}"
                if key == "wiring_paths" and path.suffix != ".py":
                    fail(errors, f"{name}: wiring path must be Python source")
                if key == "test_paths" and (path.suffix != ".py" or "test" not in path.name):
                    fail(errors, f"{name}: test path must be a Python test")
                try:
                    text = path.read_text()
                except UnicodeDecodeError:
                    fail(errors, f"{name}: {key} must be readable source")
                    continue
                if len(text) < 100:
                    fail(errors, f"{name}: {key} file is implausibly small")
                if key == "wiring_paths" and (
                    artifact.name not in text
                    and str(entry.get("artifact_path")) not in text
                    and actual_artifact_sha not in text
                ):
                    fail(errors, f"{name}: wiring does not bind the artifact identity")
                if key == "test_paths" and "assert" not in text:
                    fail(errors, f"{name}: test path contains no assertion")
                if key == "test_paths" and (
                    artifact.name not in text
                    and str(entry.get("artifact_path")) not in text
                    and actual_artifact_sha not in text
                ):
                    fail(errors, f"{name}: test does not bind the artifact identity")

    if not tracked(repo, manifest_path):
        fail(errors, "delivery manifest is not git tracked")

    if errors:
        print("TONIGHT DELIVERY GATE: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("TONIGHT DELIVERY GATE: PASS")
    for name in sorted(REQUIRED):
        print(f"- {name}: trained_wired_tested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
