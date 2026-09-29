#!/usr/bin/env python3
"""Embed real BACH images with MedSigLIP-448 and train biopsy_probe_v1.

Scientific contract:
* immutable BACH source revision and official partial patient metadata;
* unresolved patient identities form one conservative train-only equivalence group;
* three honest classes: benign/normal, in-situ carcinoma, invasive carcinoma;
* hyperparameters are chosen on train/validation only;
* the patient-disjoint test split is opened once for final evaluation;
* the validator-bound AUROC is prespecified as invasive-carcinoma one-vs-rest;
* secondary evaluations interrogate class-specific and identity-stratum behavior.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

SEED = 20260928
CAPABILITY = "biopsy-probe"
CLASSES = ["benign_or_normal", "in_situ_carcinoma", "invasive_carcinoma"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
POSITIVE_CLASS = "invasive_carcinoma"
MODEL_REPO = "google/medsiglip-448"
EXPECTED_MODEL_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
TRANSFORMS = ("raw", "standard", "l2")
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")
TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
RUO = "RESEARCH USE ONLY — not validated for clinical decision-making."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def source_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def verify_source_table(source_table_path: Path, images_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = json.loads(source_table_path.read_text())
    samples = table.get("samples")
    if table.get("schema_version") != 2 or not isinstance(samples, list) or len(samples) != 400:
        raise ValueError("source table must be schema-v2 with exactly 400 BACH images")
    if table.get("patient_disjoint") is not True:
        raise ValueError("source table is not marked patient-disjoint")
    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs")
    for row in samples:
        path = images_dir / row["payload_name"]
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["payload_bytes"]) or file_sha(path) != row["payload_sha256"]:
            raise ValueError(f"payload mismatch for {row['sample_id']}")
        if row["multiclass_target"] not in CLASS_TO_INDEX:
            raise ValueError(f"unsupported class {row['multiclass_target']}")
        expected_binary = int(row["multiclass_target"] == POSITIVE_CLASS)
        if int(row["target"]) != expected_binary:
            raise ValueError(f"binary target mismatch for {row['sample_id']}")
    by_split = {name: [r for r in samples if r["split"] == name] for name in ("train", "validation", "test")}
    if [len(by_split[name]) for name in ("train", "validation", "test")] != [292, 54, 54]:
        raise ValueError("unexpected train/validation/test counts")
    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in by_split.items()}
    if any(
        patient_sets[a] & patient_sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("patient overlap across splits")
    return table, samples


def resolve_model_revision() -> str:
    response = requests.get(f"https://huggingface.co/api/models/{MODEL_REPO}", timeout=60)
    response.raise_for_status()
    revision = str(response.json().get("sha", ""))
    if revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            f"MedSigLIP revision drift: expected {EXPECTED_MODEL_REVISION}, observed {revision}"
        )
    return revision


def embed_images(
    *,
    samples: list[dict[str, Any]],
    images_dir: Path,
    endpoint_base: str,
    cache_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "bach_medsiglip_embeddings.npy"
    manifest_path = cache_dir / "bach_medsiglip_embedding_cache.json"
    info_url = endpoint_base.rstrip("/") + "-info.modal.run"
    embed_url = endpoint_base.rstrip("/") + "-embed-batch.modal.run"
    info_response = requests.get(info_url, timeout=180)
    info_response.raise_for_status()
    info = info_response.json()
    revision = resolve_model_revision()
    if info.get("model_repo") != MODEL_REPO or int(info.get("embedding_dim", 0)) != 1152:
        raise RuntimeError(f"unexpected MedSigLIP endpoint identity: {info}")
    cache_identity = {
        "sample_ids": [row["sample_id"] for row in samples],
        "payload_sha256": [row["payload_sha256"] for row in samples],
        "model_repo": MODEL_REPO,
        "model_revision": revision,
        "endpoint": embed_url,
        "endpoint_app_version": info.get("app_version"),
    }
    identity_sha = canonical_sha(cache_identity)
    if matrix_path.is_file() and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        matrix = np.load(matrix_path)
        if (
            previous.get("cache_identity_sha256") == identity_sha
            and previous.get("matrix_file_sha256") == file_sha(matrix_path)
            and matrix.shape == (len(samples), 1152)
            and np.isfinite(matrix).all()
        ):
            return matrix.astype(np.float64), previous
    matrix = np.empty((len(samples), 1152), dtype=np.float32)
    batch_receipts: list[dict[str, Any]] = []
    start = time.perf_counter()
    for start_index in range(0, len(samples), batch_size):
        batch = samples[start_index : start_index + batch_size]
        payload = {
            "pixels_b64": [
                base64.b64encode((images_dir / row["payload_name"]).read_bytes()).decode("ascii")
                for row in batch
            ]
        }
        t0 = time.perf_counter()
        response = requests.post(embed_url, json=payload, timeout=600)
        elapsed = time.perf_counter() - t0
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"MedSigLIP endpoint error: {body['error']}")
        vectors = np.asarray(body.get("embeddings"), dtype=np.float32)
        if vectors.shape != (len(batch), 1152) or not np.isfinite(vectors).all():
            raise RuntimeError(f"invalid embedding batch shape/content {vectors.shape}")
        if np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise RuntimeError("zero-norm MedSigLIP embedding")
        matrix[start_index : start_index + len(batch)] = vectors
        batch_receipts.append(
            {
                "start_index": start_index,
                "n": len(batch),
                "elapsed_seconds": elapsed,
                "response": {key: value for key, value in body.items() if key != "embeddings"},
            }
        )
    total_elapsed = time.perf_counter() - start
    np.save(matrix_path, matrix)
    manifest = {
        "schema_version": 1,
        "cache_identity_sha256": identity_sha,
        **cache_identity,
        "endpoint_info": info,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": True,
        "matrix_array_sha256": array_sha(matrix),
        "matrix_file_sha256": file_sha(matrix_path),
        "elapsed_seconds": total_elapsed,
        "images_per_second": len(samples) / total_elapsed,
        "batch_receipts": batch_receipts,
    }
    write_json(manifest_path, manifest)
    return matrix.astype(np.float64), manifest


def transform_fit(X: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if name == "raw":
        return X.copy(), {}
    if name == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("zero norm in l2 transform")
        return X / norms, {}
    if name == "standard":
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return (X - mean) / scale, {"mean": mean, "scale": scale}
    raise ValueError(name)


def transform_apply(X: np.ndarray, name: str, params: dict[str, np.ndarray]) -> np.ndarray:
    if name == "raw":
        return X
    if name == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)
    if name == "standard":
        return (X - params["mean"]) / params["scale"]
    raise ValueError(name)


def fit_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    transform: str,
    c_value: float,
    class_weight: str | None,
) -> tuple[LogisticRegression, dict[str, np.ndarray]]:
    transformed, params = transform_fit(X_train, transform)
    model = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        solver="lbfgs",
        max_iter=5000,
        class_weight=class_weight,
        random_state=SEED,
    )
    model.fit(transformed, y_train)
    if model.coef_.shape != (3, 1152):
        raise RuntimeError(f"unexpected coefficient shape {model.coef_.shape}")
    return model, params


def candidate_probabilities(
    model: LogisticRegression,
    params: dict[str, np.ndarray],
    transform: str,
    X: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    logits = model.decision_function(transform_apply(X, transform, params)) / float(temperature)
    return softmax(np.asarray(logits, dtype=np.float64))


def multiclass_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    per_class = {
        CLASSES[index]: float(roc_auc_score((y == index).astype(int), probabilities[:, index]))
        for index in range(3)
    }
    return {
        "macro_ovr_auroc": float(np.mean(list(per_class.values()))),
        "per_class_ovr_auroc": per_class,
        "invasive_ovr_auroc": per_class[POSITIVE_CLASS],
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "accuracy": float(np.mean(probabilities.argmax(axis=1) == y)),
    }


def select_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[dict[str, Any], LogisticRegression, dict[str, np.ndarray]]:
    candidates: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float, str | None], tuple[LogisticRegression, dict[str, np.ndarray]]] = {}
    for transform in TRANSFORMS:
        for c_value in C_GRID:
            for class_weight in CLASS_WEIGHTS:
                model, params = fit_candidate(X_train, y_train, transform, c_value, class_weight)
                probabilities = candidate_probabilities(model, params, transform, X_val)
                metrics = multiclass_metrics(y_val, probabilities)
                row = {
                    "transform": transform,
                    "c": c_value,
                    "class_weight": class_weight,
                    **metrics,
                }
                candidates.append(row)
                fitted[(transform, c_value, class_weight)] = (model, params)
    # Prespecified hierarchy: macro AUROC, invasive AUROC, lower log loss,
    # then stronger regularization and deterministic transform order.
    preference = {name: -index for index, name in enumerate(TRANSFORMS)}
    best = max(
        candidates,
        key=lambda row: (
            row["macro_ovr_auroc"],
            row["invasive_ovr_auroc"],
            -row["log_loss"],
            -row["c"],
            preference[row["transform"]],
            row["class_weight"] is None,
        ),
    )
    key = (str(best["transform"]), float(best["c"]), best["class_weight"])
    model, params = fitted[key]
    base_probs = {
        temperature: candidate_probabilities(model, params, key[0], X_val, temperature)
        for temperature in TEMPERATURE_GRID
    }
    temperature_losses = {
        str(temperature): float(log_loss(y_val, probabilities, labels=[0, 1, 2]))
        for temperature, probabilities in base_probs.items()
    }
    best_temperature = min(TEMPERATURE_GRID, key=lambda value: (temperature_losses[str(value)], value))
    selection = {
        "selection_metric": "validation macro one-vs-rest AUROC; invasive AUROC and log loss tie-breaks",
        "selected": {**best, "temperature": best_temperature},
        "temperature_validation_log_loss": temperature_losses,
        "all_candidates": candidates,
    }
    return selection, model, params


def export_linear_parameters(
    model: LogisticRegression,
    transform: str,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    biases = np.asarray(model.intercept_, dtype=np.float64)
    if transform == "standard":
        coefficients = coefficients / params["scale"][None, :]
        biases = biases - coefficients @ params["mean"]
        runtime_transform = "none"
    elif transform == "l2":
        runtime_transform = "l2_normalize"
    else:
        runtime_transform = "none"
    return coefficients, biases, runtime_transform


def artifact_probabilities(
    X: np.ndarray,
    coefficients: np.ndarray,
    biases: np.ndarray,
    runtime_transform: str,
    temperature: float,
) -> np.ndarray:
    if runtime_transform == "l2_normalize":
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return softmax((X @ coefficients.T + biases[None, :]) / temperature)


def grouped_bootstrap_auc(y: np.ndarray, score: np.ndarray, groups: np.ndarray, draws: int = 4000) -> list[float]:
    unique = np.array(sorted(set(groups.tolist())), dtype=object)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED + 31)
    values: list[float] = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([lookup[group] for group in selected])
        if len(np.unique(y[indices])) == 2:
            values.append(float(roc_auc_score(y[indices], score[indices])))
    if len(values) < 1000:
        raise RuntimeError("too few valid grouped bootstrap replicates")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def secondary_evaluation(
    *,
    X: np.ndarray,
    y_class: np.ndarray,
    probabilities: np.ndarray,
    samples: list[dict[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    selection: dict[str, Any],
) -> dict[str, Any]:
    y_binary = (y_class == CLASS_TO_INDEX[POSITIVE_CLASS]).astype(int)
    invasive_score = probabilities[:, CLASS_TO_INDEX[POSITIVE_CLASS]]
    test_y = y_binary[test_indices]
    test_score = invasive_score[test_indices]
    test_groups = np.asarray([samples[index]["patient_id"] for index in test_indices], dtype=object)
    pairwise: dict[str, Any] = {}
    for negative_class in ("benign_or_normal", "in_situ_carcinoma"):
        keep = np.array(
            [y_class[index] in {CLASS_TO_INDEX[negative_class], CLASS_TO_INDEX[POSITIVE_CLASS]} for index in test_indices]
        )
        yy = test_y[keep]
        ss = test_score[keep]
        pairwise[f"invasive_vs_{negative_class}"] = {
            "n": int(len(yy)),
            "n_positive": int(yy.sum()),
            "auroc": float(roc_auc_score(yy, ss)),
        }
    leave_one_patient_out: list[float] = []
    for patient in sorted(set(test_groups.tolist())):
        keep = test_groups != patient
        if len(np.unique(test_y[keep])) == 2:
            leave_one_patient_out.append(float(roc_auc_score(test_y[keep], test_score[keep])))
    unresolved = np.array(
        [samples[index]["patient_id"] == "BACH-UNRESOLVED-IDENTITY-EQUIVALENCE" for index in train_indices]
    )
    unresolved_indices = train_indices[unresolved]
    selected = selection["selected"]
    identity_model, identity_params = fit_candidate(
        X[unresolved_indices],
        y_class[unresolved_indices],
        str(selected["transform"]),
        float(selected["c"]),
        selected["class_weight"],
    )
    identity_probs = candidate_probabilities(
        identity_model,
        identity_params,
        str(selected["transform"]),
        X[validation_indices],
        float(selected["temperature"]),
    )
    mann_whitney = mannwhitneyu(test_score[test_y == 1], test_score[test_y == 0], alternative="greater")
    return {
        "patient_grouped_bootstrap_auroc_ci": grouped_bootstrap_auc(test_y, test_score, test_groups),
        "mann_whitney_one_sided_p": float(mann_whitney.pvalue),
        "test_per_class": multiclass_metrics(y_class[test_indices], probabilities[test_indices]),
        "test_pairwise_invasive_auroc": pairwise,
        "leave_one_test_patient_out_auroc": {
            "n_estimable": len(leave_one_patient_out),
            "min": float(min(leave_one_patient_out)),
            "max": float(max(leave_one_patient_out)),
            "mean": float(np.mean(leave_one_patient_out)),
        },
        "cross_identity_status_validation": {
            "description": "fit on all unresolved-identity train-only images; evaluate on official-ID validation patients",
            "n_train_unresolved": int(len(unresolved_indices)),
            "n_validation_identified": int(len(validation_indices)),
            **multiclass_metrics(y_class[validation_indices], identity_probs),
        },
        "source_shortcut_assessment": {
            "status": "not_applicable_single_source",
            "evidence": "all 400 samples come from one immutable BACH revision, acquisition protocol, and preprocessing path",
        },
        "propensity_or_synthetic_control": {
            "status": "not_applicable_supervised_prediction_cohort",
            "reason": "no single-arm treatment-effect estimand is present",
        },
    }


def update_delivery_manifest(repo_root: Path, entry: dict[str, Any]) -> None:
    path = repo_root / "artifacts" / "tonight_delivery.json"
    existing = json.loads(path.read_text()) if path.is_file() else {"schema_version": 2, "capabilities": []}
    capabilities = [item for item in existing.get("capabilities", []) if item.get("name") != CAPABILITY]
    capabilities.append(entry)
    write_json(path, {"schema_version": 2, "capabilities": capabilities})


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    started_at = utc_now()
    start_clock = time.perf_counter()
    source_table, samples = verify_source_table(args.source_table, args.images_dir)
    X, embedding_cache = embed_images(
        samples=samples,
        images_dir=args.images_dir,
        endpoint_base=args.endpoint_base,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    if X.shape != (400, 1152) or not np.isfinite(X).all():
        raise RuntimeError(f"invalid full embedding matrix {X.shape}")
    y_class = np.asarray([CLASS_TO_INDEX[row["multiclass_target"]] for row in samples], dtype=int)
    split_indices = {
        name: np.asarray([index for index, row in enumerate(samples) if row["split"] == name], dtype=int)
        for name in ("train", "validation", "test")
    }
    train_idx, val_idx, test_idx = (split_indices[name] for name in ("train", "validation", "test"))
    selection, model, transform_params = select_model(X[train_idx], y_class[train_idx], X[val_idx], y_class[val_idx])
    selected = selection["selected"]
    coefficients, biases, runtime_transform = export_linear_parameters(
        model, str(selected["transform"]), transform_params
    )
    temperature = float(selected["temperature"])
    probabilities = artifact_probabilities(X, coefficients, biases, runtime_transform, temperature)
    reference_probabilities = candidate_probabilities(
        model, transform_params, str(selected["transform"]), X, temperature
    )
    if not np.allclose(probabilities, reference_probabilities, atol=1e-10, rtol=1e-10):
        raise RuntimeError("exported artifact probabilities do not match fitted model")
    validation_metrics = multiclass_metrics(y_class[val_idx], probabilities[val_idx])
    # Test is opened only after the complete train/validation selection and export checks.
    test_metrics = multiclass_metrics(y_class[test_idx], probabilities[test_idx])
    secondary = secondary_evaluation(
        X=X,
        y_class=y_class,
        probabilities=probabilities,
        samples=samples,
        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx,
        selection=selection,
    )
    completed_at = utc_now()
    while completed_at == started_at:
        time.sleep(0.001)
        completed_at = utc_now()

    evidence_dir = repo_root / "artifacts" / "biopsy_probe"
    dataset_rel = "artifacts/biopsy_probe/biopsy_probe_dataset_v2.json"
    split_rel = "artifacts/biopsy_probe/biopsy_probe_split_v2.json"
    train_rel = "artifacts/biopsy_probe/biopsy_probe_train_v2.json"
    evaluation_rel = "artifacts/biopsy_probe/biopsy_probe_evaluation_v2.json"
    embedding_rel = "artifacts/biopsy_probe/biopsy_probe_embedding_manifest_v1.json"
    artifact_rel = "src/oncology_arbiter/arbiter/models/biopsy_probe_v1.json"

    embedding_matrix_path = args.cache_dir / "bach_medsiglip_embeddings.npy"
    embedding_manifest = {
        "schema_version": 1,
        "capability": CAPABILITY,
        "model_repo": MODEL_REPO,
        "model_revision": EXPECTED_MODEL_REVISION,
        "endpoint": embedding_cache["endpoint"],
        "endpoint_info": embedding_cache["endpoint_info"],
        "source_table_sha256": file_sha(args.source_table),
        "embedding_matrix_file_sha256": file_sha(embedding_matrix_path),
        "embedding_matrix_array_sha256": array_sha(X.astype(np.float32)),
        "shape": [400, 1152],
        "finite": True,
        "rows": [
            {
                "sample_id": row["sample_id"],
                "payload_sha256": row["payload_sha256"],
                "embedding_sha256": array_sha(X[index].astype(np.float32)),
            }
            for index, row in enumerate(samples)
        ],
    }
    embedding_path = repo_root / embedding_rel
    write_json(embedding_path, embedding_manifest)
    embedding_manifest_sha = file_sha(embedding_path)

    dataset_manifest = {
        "schema_version": 2,
        "capability": CAPABILITY,
        "dataset": source_table["dataset"],
        "source_uri": source_table["source_uri"],
        "source_revision": source_table["source_revision"],
        "license": source_table["license"],
        "patient_metadata_uri": source_table["patient_metadata_uri"],
        "patient_metadata_sha256": source_table["patient_metadata_sha256"],
        "target_definition": "binary validator target: invasive carcinoma one-vs-rest; model is trained on three source-grounded classes",
        "multiclass_classes": CLASSES,
        "embedding_manifest_path": embedding_rel,
        "embedding_manifest_sha256": embedding_manifest_sha,
        "samples": [
            {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "payload_sha256": row["payload_sha256"],
                "target": int(row["target"]),
                "target_sha256": canonical_sha(int(row["target"])),
                "source_name": row["source_name"],
                "source_payload_sha256": row["source_payload_sha256"],
                "source_label": row["source_label"],
                "multiclass_target": row["multiclass_target"],
                "patient_identity_basis": row["patient_identity_basis"],
            }
            for row in samples
        ],
    }
    dataset_path = repo_root / dataset_rel
    write_json(dataset_path, dataset_manifest)
    dataset_sha = file_sha(dataset_path)
    split_manifest = {
        "schema_version": 2,
        "capability": CAPABILITY,
        "dataset_manifest_path": dataset_rel,
        "dataset_manifest_sha256": dataset_sha,
        "seed": SEED,
        "strategy": source_table["split_method"],
        "train_sample_ids": [samples[index]["sample_id"] for index in train_idx],
        "validation_sample_ids": [samples[index]["sample_id"] for index in val_idx],
        "test_sample_ids": [samples[index]["sample_id"] for index in test_idx],
        "train_patient_ids": sorted({samples[index]["patient_id"] for index in train_idx}),
        "validation_patient_ids": sorted({samples[index]["patient_id"] for index in val_idx}),
        "test_patient_ids": sorted({samples[index]["patient_id"] for index in test_idx}),
    }
    split_path = repo_root / split_rel
    write_json(split_path, split_manifest)
    split_sha = file_sha(split_path)

    artifact = {
        "$schema_version": "biopsy_probe_v1",
        "schema_version": 1,
        "model_name": "biopsy_probe_v1",
        "model_type": "three_class_L2_multinomial_logistic_regression",
        "classes": CLASSES,
        "coefficients": coefficients.tolist(),
        "biases": biases.tolist(),
        "temperature": temperature,
        "embedding_transform": runtime_transform,
        "fit_transform": selected["transform"],
        "embed_dim": 1152,
        "embedding_model_repo": MODEL_REPO,
        "embedding_model_revision": EXPECTED_MODEL_REVISION,
        "embedding_endpoint_app_version": embedding_cache["endpoint_info"].get("app_version"),
        "embedding_manifest_path": embedding_rel,
        "embedding_manifest_sha256": embedding_manifest_sha,
        "n_training": int(len(train_idx)),
        "n_training_synthetic": False,
        "trained_on_real_patient_data": True,
        "training_dataset": source_table["dataset"],
        "seed": SEED,
        "selected_c": float(selected["c"]),
        "selected_class_weight": selected["class_weight"],
        "positive_class_for_reported_auroc": POSITIVE_CLASS,
        "validation_metrics": validation_metrics,
        "held_out_test_metrics": test_metrics,
        "disclaimer": RUO,
    }
    artifact_path = repo_root / artifact_rel
    write_json(artifact_path, artifact)
    artifact_sha = file_sha(artifact_path)

    invasive_index = CLASS_TO_INDEX[POSITIVE_CLASS]
    evaluation_rows = []
    for index in test_idx.tolist():
        row = samples[index]
        evaluation_rows.append(
            {
                "sample_id": row["sample_id"],
                "target_sha256": canonical_sha(int(row["target"])),
                "y_true": int(row["target"]),
                "y_score": float(probabilities[index, invasive_index]),
                "multiclass_y_true": row["multiclass_target"],
                "multiclass_y_pred": CLASSES[int(probabilities[index].argmax())],
                "class_probabilities": {
                    class_name: float(probabilities[index, class_index])
                    for class_index, class_name in enumerate(CLASSES)
                },
                "patient_id": row["patient_id"],
                "source_label": row["source_label"],
            }
        )
    invasive_auc = float(test_metrics["invasive_ovr_auroc"])
    evaluation = {
        "schema_version": 2,
        "capability": CAPABILITY,
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "metric": {"name": "auroc", "value": invasive_auc, "n": int(len(test_idx))},
        "metric_definition": "held-out invasive_carcinoma one-vs-rest AUROC; positive class prespecified before test opening",
        "rows": evaluation_rows,
        "validation_metrics": validation_metrics,
        "test_multiclass_metrics": test_metrics,
        "secondary_evaluation": secondary,
    }
    evaluation_path = repo_root / evaluation_rel
    write_json(evaluation_path, evaluation)

    command = " ".join([sys.executable, *sys.argv])
    train_manifest = {
        "schema_version": 2,
        "capability": CAPABILITY,
        "dataset": source_table["dataset"],
        "dataset_manifest_path": dataset_rel,
        "dataset_sha256": dataset_sha,
        "split_manifest_path": split_rel,
        "split_sha256": split_sha,
        "patient_disjoint": True,
        "n_training": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_training_synthetic": False,
        "seed": SEED,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "source_commit": source_commit(repo_root),
        "model_selection": selection,
        "secondary_evaluation_summary": secondary,
    }
    train_path = repo_root / train_rel
    write_json(train_path, train_manifest)

    summary = {
        "capability": CAPABILITY,
        "n_training": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "artifact_path": artifact_rel,
        "artifact_sha256": artifact_sha,
        "dataset_manifest_path": dataset_rel,
        "dataset_sha256": dataset_sha,
        "split_manifest_path": split_rel,
        "split_sha256": split_sha,
        "train_manifest_path": train_rel,
        "train_manifest_sha256": file_sha(train_path),
        "evaluation_path": evaluation_rel,
        "evaluation_sha256": file_sha(evaluation_path),
        "embedding_manifest_path": embedding_rel,
        "embedding_manifest_sha256": embedding_manifest_sha,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "secondary_evaluation": secondary,
        "selection": selection["selected"],
        "embedding_cache": {
            "model_repo": embedding_cache["model_repo"],
            "model_revision": embedding_cache["model_revision"],
            "endpoint": embedding_cache["endpoint"],
            "shape": embedding_cache["shape"],
            "images_per_second": embedding_cache["images_per_second"],
        },
        "runtime_seconds": time.perf_counter() - start_clock,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "biopsy_probe_v1_run_summary.json", summary)
    update_delivery_manifest(
        repo_root,
        {
            "name": CAPABILITY,
            "status": "trained_wired_tested",
            "artifact_path": artifact_rel,
            "artifact_sha256": artifact_sha,
            "n_training": int(len(train_idx)),
            "n_training_synthetic": False,
            "train_manifest_path": train_rel,
            "train_manifest_sha256": file_sha(train_path),
            "evaluation_path": evaluation_rel,
            "evaluation_sha256": file_sha(evaluation_path),
            "wiring_paths": ["src/oncology_arbiter/models/biopsy_probe_v1_wiring.py"],
            "test_paths": ["tests/unit/test_biopsy_probe_v1_identity.py"],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--endpoint-base", default="https://crispro-test--medsiglip")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_size <= 32:
        raise SystemExit("--batch-size must be in [1, 32]")
    summary = train(args)
    print(
        f"biopsy-probe: n_train={summary['n_training']} "
        f"validation_macro_auc={summary['validation_metrics']['macro_ovr_auroc']:.4f} "
        f"test_invasive_auc={summary['test_metrics']['invasive_ovr_auroc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
