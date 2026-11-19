#!/usr/bin/env python3
"""Recompute ClinicalBERT v2 evidence and interrogate null/anomalous signals.

This script never selects a threshold on test or external rows. It validates the
frozen prediction rows, recomputes span metrics, constructs validator-compatible
schema-v2 evidence, and runs patient-clustered uncertainty, permutation-null,
stratified-regression, and cross-cohort analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import beta, norm


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def entity_key(entity: dict[str, Any]) -> tuple[str, int, int]:
    return entity["entity_type"], int(entity["char_start"]), int(entity["char_end"])


def row_counts(row: dict[str, Any], predicted_override: set[tuple[str, int, int]] | None = None) -> Counter[str]:
    gold = {entity_key(item) for item in row["gold_entities"]}
    predicted = predicted_override if predicted_override is not None else {entity_key(item) for item in row["predicted_entities"]}
    return Counter(tp=len(gold & predicted), fp=len(predicted - gold), fn=len(gold - predicted))


def aggregate_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row_counts(row))
    return counts


def metrics_from_counts(counts: Counter[str]) -> dict[str, Any]:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> list[float] | None:
    if trials <= 0:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [lower, upper]


def bootstrap_metrics(
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    values = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(patients), size=len(patients))
        counts: Counter[str] = Counter()
        for index in sampled:
            for row in by_patient[patients[int(index)]]:
                counts.update(row_counts(row))
        metric = metrics_from_counts(counts)
        for name in values:
            values[name].append(float(metric[name]))
    result: dict[str, Any] = {
        "method": "patient-cluster percentile bootstrap",
        "replicates": n_bootstrap,
        "seed": seed,
        "n_patient_clusters": len(patients),
    }
    point = metrics_from_counts(aggregate_counts(rows))
    result.update({
        name: {
            "estimate": point[name],
            "ci_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
        for name, samples in values.items()
    })
    return result


def length_strata(rows: list[dict[str, Any]], bins: int = 4) -> dict[int, tuple[str, int]]:
    strata: dict[int, tuple[str, int]] = {}
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["tumor_source"]].append(index)
    for source, indices in sorted(by_source.items()):
        ordered = sorted(indices, key=lambda i: (int(rows[i]["report_length_tokens"]), rows[i]["sample_id"]))
        n = len(ordered)
        for rank, index in enumerate(ordered):
            strata[index] = (source, min(bins - 1, (rank * bins) // max(n, 1)))
    return strata


def permutation_null(
    rows: list[dict[str, Any]],
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = metrics_from_counts(aggregate_counts(rows))
    observed_exact = sum(bool(row["strict_exact_match"]) for row in rows) / len(rows)
    predictions = [{entity_key(item) for item in row["predicted_entities"]} for row in rows]
    strata = length_strata(rows)
    members: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, stratum in strata.items():
        members[stratum].append(index)
    rng = np.random.default_rng(seed)
    null_f1 = np.zeros(n_permutations, dtype=np.float64)
    null_exact = np.zeros(n_permutations, dtype=np.float64)
    for permutation in range(n_permutations):
        assignment = np.arange(len(rows))
        for indices in members.values():
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            assignment[np.asarray(indices, dtype=np.int64)] = shuffled
        counts: Counter[str] = Counter()
        exact = 0
        for row_index, prediction_index in enumerate(assignment):
            counts.update(row_counts(rows[row_index], predictions[int(prediction_index)]))
            gold = {entity_key(item) for item in rows[row_index]["gold_entities"]}
            exact += int(gold == predictions[int(prediction_index)])
        null_f1[permutation] = metrics_from_counts(counts)["f1"]
        null_exact[permutation] = exact / len(rows)
    f1_sd = float(null_f1.std(ddof=1))
    return {
        "method": "prediction-set permutation within tumor-source and report-length quartile",
        "permutations": n_permutations,
        "seed": seed,
        "span_micro_f1": {
            "observed": observed["f1"],
            "null_mean": float(null_f1.mean()),
            "null_ci_95": [float(np.quantile(null_f1, 0.025)), float(np.quantile(null_f1, 0.975))],
            "z_vs_null": None if f1_sd == 0.0 else float((observed["f1"] - null_f1.mean()) / f1_sd),
            "p_one_sided_greater": float((1 + np.count_nonzero(null_f1 >= observed["f1"])) / (n_permutations + 1)),
        },
        "strict_report_exact_match": {
            "observed": observed_exact,
            "null_mean": float(null_exact.mean()),
            "null_ci_95": [float(np.quantile(null_exact, 0.025)), float(np.quantile(null_exact, 0.975))],
            "p_one_sided_greater": float((1 + np.count_nonzero(null_exact >= observed_exact)) / (n_permutations + 1)),
        },
    }


def per_entity_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = {entity_key(item) for item in row["gold_entities"]}
        predicted = {entity_key(item) for item in row["predicted_entities"]}
        for entity_type, _, _ in gold & predicted:
            counts[entity_type]["tp"] += 1
        for entity_type, _, _ in predicted - gold:
            counts[entity_type]["fp"] += 1
        for entity_type, _, _ in gold - predicted:
            counts[entity_type]["fn"] += 1
    result: dict[str, dict[str, Any]] = {}
    for entity_type in sorted(counts):
        metric = metrics_from_counts(counts[entity_type])
        metric["precision_ci_95_exact"] = clopper_pearson(metric["tp"], metric["tp"] + metric["fp"])
        metric["recall_ci_95_exact"] = clopper_pearson(metric["tp"], metric["tp"] + metric["fn"])
        result[entity_type] = metric
    return result


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if not math.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def entity_detection_frame(
    rows: list[dict[str, Any]],
    training_entity_frequency: Counter[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        predicted = {entity_key(item) for item in row["predicted_entities"]}
        for entity in row["gold_entities"]:
            key = entity_key(entity)
            entity_type = entity["entity_type"]
            records.append({
                "detected": int(key in predicted),
                "patient_id": row["patient_id"],
                "sample_id": row["sample_id"],
                "tumor_source": row["tumor_source"],
                "entity_type": entity_type,
                "log_length": math.log1p(int(row["report_length_tokens"])),
                "log_train_frequency": math.log1p(int(training_entity_frequency[entity_type])),
                "log_negation": math.log1p(int(row["negation_term_count"])),
                "log_uncertainty": math.log1p(int(row["uncertainty_term_count"])),
            })
    frame = pd.DataFrame.from_records(records)
    for source, target in (
        ("log_length", "z_log_length"),
        ("log_train_frequency", "z_log_train_frequency"),
        ("log_negation", "z_log_negation"),
        ("log_uncertainty", "z_log_uncertainty"),
    ):
        frame[target] = zscore(frame[source])
    return frame


def fit_clustered_regression(frame: pd.DataFrame, formula: str) -> dict[str, Any]:
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": frame["patient_id"]}
            )
            caught = [str(item.message) for item in captured]
        terms: dict[str, Any] = {}
        for term in fit.params.index:
            coefficient = float(fit.params[term])
            standard_error = float(fit.bse[term])
            terms[term] = {
                "coefficient_log_odds": coefficient,
                "cluster_robust_se": standard_error,
                "odds_ratio": float(math.exp(coefficient)) if coefficient < 700 else None,
                "ci_95_log_odds": [float(coefficient - 1.959963984540054 * standard_error), float(coefficient + 1.959963984540054 * standard_error)],
                "p_value_wald": finite_or_none(fit.pvalues[term]),
            }
        return {
            "status": "fit",
            "formula": formula,
            "family": "binomial logit",
            "covariance": "patient-cluster robust",
            "n_entity_mentions": int(len(frame)),
            "n_patient_clusters": int(frame["patient_id"].nunique()),
            "converged": bool(fit.converged),
            "terms": terms,
            "warnings": caught,
        }
    except Exception as exc:  # evidence-preserving failure, never silently ignored
        return {
            "status": "failed",
            "formula": formula,
            "n_entity_mentions": int(len(frame)),
            "n_patient_clusters": int(frame["patient_id"].nunique()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "warnings": caught,
        }


def stratified_detection(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["report_length_quartile"] = pd.qcut(work["log_length"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    work["entity_frequency_tercile"] = pd.qcut(work["log_train_frequency"].rank(method="first"), 3, labels=["low", "middle", "high"])
    work["negation_stratum"] = np.where(work["log_negation"] > work["log_negation"].median(), "above_median", "at_or_below_median")
    work["uncertainty_stratum"] = np.where(work["log_uncertainty"] > 0, "present", "absent")
    result: dict[str, Any] = {}
    for variable in (
        "report_length_quartile", "tumor_source", "entity_frequency_tercile",
        "negation_stratum", "uncertainty_stratum",
    ):
        levels: dict[str, Any] = {}
        for level, group in work.groupby(variable, observed=True):
            successes = int(group["detected"].sum())
            trials = int(len(group))
            levels[str(level)] = {
                "detected": successes,
                "gold_mentions": trials,
                "recall": successes / trials if trials else 0.0,
                "recall_ci_95_exact": clopper_pearson(successes, trials),
            }
        result[variable] = levels
    return result


def validate_predictions(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    expected_ids: list[str],
    artifact_sha: str,
) -> None:
    expected_map = {row["sample_id"]: row for row in expected_rows}
    if len(rows) != len(expected_ids) or {row["sample_id"] for row in rows} != set(expected_ids):
        raise ValueError("prediction rows do not exactly cover the expected split")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("prediction sample IDs are not unique")
    for row in rows:
        expected = expected_map[row["sample_id"]]
        if row["patient_id"] != expected["patient_id"]:
            raise ValueError(f"patient mismatch for {row['sample_id']}")
        if row["target_sha256"] != expected["target_sha256"] or row["y_true"] != expected["target"]:
            raise ValueError(f"target mismatch for {row['sample_id']}")
        if row["gold_entities"] != expected["structured_target"]:
            raise ValueError(f"gold entity mismatch for {row['sample_id']}")
        if row["artifact_sha256"] != artifact_sha:
            raise ValueError(f"artifact mismatch for {row['sample_id']}")
        if row["payload_sha256"] != expected["payload_sha256"]:
            raise ValueError(f"payload mismatch for {row['sample_id']}")
        for entity in row["predicted_entities"]:
            start, end = int(entity["char_start"]), int(entity["char_end"])
            if not (0 <= start < end <= len(expected["text"])):
                raise ValueError(f"predicted span outside report for {row['sample_id']}")
            if expected["text"][start:end] != entity["surface"]:
                raise ValueError(f"predicted surface mismatch for {row['sample_id']}")
        predicted_keys = [entity_key(item) for item in row["predicted_entities"]]
        if len(predicted_keys) != len(set(predicted_keys)):
            raise ValueError(f"duplicate predicted spans for {row['sample_id']}")
        expected_predicted_target = int(canonical_sha(row["predicted_entities"])[:15], 16)
        if row["y_pred"] != expected_predicted_target:
            raise ValueError(f"predicted target-code mismatch for {row['sample_id']}")
        if bool(row["strict_exact_match"]) != (row["y_true"] == row["y_pred"]):
            raise ValueError(f"strict-match flag mismatch for {row['sample_id']}")


def split_patient_audit(dataset: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    sample_map = {sample["sample_id"]: sample for sample in dataset["samples"]}
    split_keys = ["train_sample_ids", "validation_sample_ids", "test_sample_ids", "external_validation_sample_ids"]
    sample_sets = {key: set(split[key]) for key in split_keys}
    patient_sets = {
        key: {sample_map[sample_id]["patient_id"] for sample_id in values}
        for key, values in sample_sets.items()
    }
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(split_keys):
        for right in split_keys[left_index + 1:]:
            name = f"{left}__{right}"
            pairwise[name] = {
                "sample_overlap": len(sample_sets[left] & sample_sets[right]),
                "patient_overlap": len(patient_sets[left] & patient_sets[right]),
            }
    if any(item["sample_overlap"] or item["patient_overlap"] for item in pairwise.values()):
        raise ValueError("sample or patient leakage detected")
    return pairwise


def cohort_analysis(
    rows: list[dict[str, Any]],
    training_entity_frequency: Counter[str],
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    overall = metrics_from_counts(aggregate_counts(rows))
    strict_successes = sum(bool(row["strict_exact_match"]) for row in rows)
    any_tp_successes = sum(row_counts(row)["tp"] > 0 for row in rows)
    gold_counts = [len(row["gold_entities"]) for row in rows]
    predicted_counts = [len(row["predicted_entities"]) for row in rows]
    frame = entity_detection_frame(rows, training_entity_frequency)
    covariate_formula = (
        "detected ~ z_log_length + C(tumor_source) + z_log_train_frequency "
        "+ z_log_negation + z_log_uncertainty"
    )
    entity_fixed_effect_formula = (
        "detected ~ z_log_length + C(tumor_source) + C(entity_type) "
        "+ z_log_negation + z_log_uncertainty"
    )
    return {
        "n_reports": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "span_micro": overall,
        "span_micro_cluster_bootstrap": bootstrap_metrics(rows, n_bootstrap, seed),
        "span_precision_ci_95_exact": clopper_pearson(overall["tp"], overall["tp"] + overall["fp"]),
        "span_recall_ci_95_exact": clopper_pearson(overall["tp"], overall["tp"] + overall["fn"]),
        "reports_with_any_exact_span": {
            "successes": any_tp_successes,
            "trials": len(rows),
            "rate": any_tp_successes / len(rows),
            "ci_95_exact": clopper_pearson(any_tp_successes, len(rows)),
        },
        "prediction_burden": {
            "gold_entities": sum(gold_counts),
            "predicted_entities": sum(predicted_counts),
            "predicted_to_gold_ratio": sum(predicted_counts) / sum(gold_counts),
            "median_gold_entities_per_report": float(np.median(gold_counts)),
            "median_predicted_entities_per_report": float(np.median(predicted_counts)),
        },
        "strict_report_exact_match": {
            "successes": strict_successes,
            "trials": len(rows),
            "rate": strict_successes / len(rows),
            "ci_95_exact": clopper_pearson(strict_successes, len(rows)),
        },
        "per_entity": per_entity_metrics(rows),
        "permutation_null": permutation_null(rows, n_permutations, seed + 1),
        "stratified_regression": fit_clustered_regression(frame, covariate_formula),
        "entity_type_fixed_effect_sensitivity": fit_clustered_regression(frame, entity_fixed_effect_formula),
        "stratified_detection": stratified_detection(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest-path", default="artifacts/clinicalbert/clinicalbert_dataset_v2.json")
    parser.add_argument("--split-manifest-path", default="artifacts/clinicalbert/clinicalbert_split_v2.json")
    parser.add_argument("--artifact-path", default="artifacts/clinicalbert/clinicalbert_head_v2.safetensors")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.bootstrap <= 0 or args.permutations <= 0:
        raise ValueError("bootstrap and permutations must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.data_dir / "clinicalbert_dataset_v2.json"
    split_path = args.data_dir / "clinicalbert_split_v2.json"
    dataset = load_json(dataset_path)
    split = load_json(split_path)
    summary_path = args.training_output_dir / "clinicalbert_training_summary_v2.json"
    summary = load_json(summary_path)
    metrics_path = args.training_output_dir / "clinicalbert_metrics_v2.json"
    training_metrics = load_json(metrics_path)
    artifact_path = args.training_output_dir / "clinicalbert_head_v2.safetensors"
    artifact_sha = sha_file(artifact_path)
    if artifact_sha != summary["artifact_sha256"] or artifact_sha != training_metrics["artifact_sha256"]:
        raise ValueError("artifact hash disagrees with training outputs")
    dataset_sha, split_sha = sha_file(dataset_path), sha_file(split_path)
    if dataset_sha != summary["dataset_sha256"] or split_sha != summary["split_sha256"]:
        raise ValueError("dataset/split hash disagrees with training summary")
    if split["dataset_manifest_sha256"] != dataset_sha:
        raise ValueError("split manifest does not bind dataset manifest")
    leakage_audit = split_patient_audit(dataset, split)

    train_rows = load_jsonl(args.data_dir / "clinicalbert_train_reports.jsonl")
    test_source_rows = load_jsonl(args.data_dir / "clinicalbert_test_reports.jsonl")
    external_source_rows = load_jsonl(args.data_dir / "clinicalbert_external_reports.jsonl")
    test_predictions_path = args.training_output_dir / "clinicalbert_test_predictions_v2.jsonl"
    external_predictions_path = args.training_output_dir / "clinicalbert_external_predictions_v2.jsonl"
    test_rows = load_jsonl(test_predictions_path)
    external_rows = load_jsonl(external_predictions_path)
    validate_predictions(test_rows, test_source_rows, split["test_sample_ids"], artifact_sha)
    validate_predictions(external_rows, external_source_rows, split["external_validation_sample_ids"], artifact_sha)

    training_entity_frequency: Counter[str] = Counter(
        entity["entity_type"] for row in train_rows for entity in row["structured_target"]
    )
    primary = cohort_analysis(test_rows, training_entity_frequency, args.bootstrap, args.permutations, args.seed)
    external = cohort_analysis(external_rows, training_entity_frequency, args.bootstrap, args.permutations, args.seed + 100)

    cross_cohort: dict[str, Any] = {}
    entity_types = sorted(set(primary["per_entity"]) | set(external["per_entity"]))
    for entity_type in entity_types:
        p = primary["per_entity"].get(entity_type, {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0})
        e = external["per_entity"].get(entity_type, {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0})
        p_gold, e_gold = int(p["tp"] + p["fn"]), int(e["tp"] + e["fn"])
        if p_gold and e_gold:
            if p["tp"] == 0 and e["tp"] == 0:
                status = "null_replicated_across_cohorts"
            elif p["tp"] == 0 and e["tp"] > 0:
                status = "primary_null_not_replicated_external_signal"
            elif p["tp"] > 0 and e["tp"] == 0:
                status = "primary_signal_not_replicated_external_null"
            else:
                status = "nonzero_signal_in_both_cohorts"
        elif p_gold:
            status = "primary_only_no_external_gold"
        elif e_gold:
            status = "external_only_no_primary_gold"
        else:
            status = "no_gold_mentions"
        cross_cohort[entity_type] = {
            "training_mentions": int(training_entity_frequency[entity_type]),
            "primary_gold_mentions": p_gold,
            "primary_tp": int(p["tp"]),
            "primary_f1": float(p["f1"]),
            "external_gold_mentions": e_gold,
            "external_tp": int(e["tp"]),
            "external_f1": float(e["f1"]),
            "status": status,
        }

    exact_rows = [{
        "sample_id": row["sample_id"],
        "target_sha256": row["target_sha256"],
        "y_true": row["y_true"],
        "y_pred": row["y_pred"],
        "strict_exact_match": bool(row["strict_exact_match"]),
    } for row in test_rows]
    exact_rate = sum(int(row["y_true"] == row["y_pred"]) for row in exact_rows) / len(exact_rows)
    training_manifest = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "dataset": dataset["dataset"],
        "dataset_manifest_path": args.dataset_manifest_path,
        "dataset_sha256": dataset_sha,
        "split_manifest_path": args.split_manifest_path,
        "split_sha256": split_sha,
        "patient_disjoint": True,
        "n_training": int(summary["n_training"]),
        "n_validation": int(summary["n_validation"]),
        "n_test": int(summary["n_test"]),
        "n_external_validation": int(summary["n_external_validation"]),
        "n_training_synthetic": False,
        "seed": int(summary["seed"]),
        "command": summary["command"],
        "started_at": summary["started_at"],
        "completed_at": summary["completed_at"],
        "source_commit": summary["source_commit"],
        "artifact_path": args.artifact_path,
        "artifact_sha256": artifact_sha,
        "base_model": summary["base_model"],
        "base_revision": summary["base_revision"],
        "epochs": int(summary["epochs"]),
        "batch_size": int(summary["batch_size"]),
        "learning_rate": float(summary["learning_rate"]),
        "weight_decay": float(summary["weight_decay"]),
        "max_length": int(summary["max_length"]),
        "stride": int(summary["stride"]),
        "selected_non_o_margin_threshold": float(summary["selected_non_o_margin_threshold"]),
        "threshold_selection_split": "validation",
        "n_training_windows": int(summary["n_training_windows"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "peak_cuda_memory_bytes": int(summary["peak_cuda_memory_bytes"]),
    }
    evaluation = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "metric": {"name": "micro_f1", "value": exact_rate, "n": len(exact_rows)},
        "metric_semantics": "validator-compatible exact report-level structured-entity-set equality",
        "span_micro_f1": primary["span_micro"]["f1"],
        "span_micro_precision": primary["span_micro"]["precision"],
        "span_micro_recall": primary["span_micro"]["recall"],
        "rows": exact_rows,
    }
    external_evaluation = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "cohort": "independent_external_validation",
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "span_micro": external["span_micro"],
        "strict_report_exact_match": external["strict_report_exact_match"],
        "n": len(external_rows),
        "prediction_rows_sha256": sha_file(external_predictions_path),
    }
    statistical_analysis = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "analysis_seed": args.seed,
        "primary_pathologist_gold": primary,
        "independent_external_validation": external,
        "cross_cohort_by_entity": cross_cohort,
        "leakage_audit": leakage_audit,
        "prediction_file_hashes": {
            "primary_test": sha_file(test_predictions_path),
            "external": sha_file(external_predictions_path),
        },
        "training_output_hashes": {
            "summary": sha_file(summary_path),
            "metrics": sha_file(metrics_path),
        },
        "null_interpretation": {
            "strict_report_exact_match": "unresolved null when the exact binomial interval includes nontrivial success; span-level signal is assessed separately",
            "entity_rule": "zero-TP entities are cross-checked only where the entity has gold mentions in both cohorts",
        },
        "causal_methods": "N/A — supervised extraction evaluation has no treatment assignment or single-arm causal estimand; propensity matching and synthetic controls are not scientifically applicable",
    }

    write_json(args.out_dir / "clinicalbert_train_manifest_v2.json", training_manifest)
    write_json(args.out_dir / "clinicalbert_evaluation_v2.json", evaluation)
    write_json(args.out_dir / "clinicalbert_external_evaluation_v2.json", external_evaluation)
    write_json(args.out_dir / "clinicalbert_statistical_analysis_v2.json", statistical_analysis)
    receipt = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "n_test": len(test_rows),
        "n_external": len(external_rows),
        "primary_span_micro_f1": primary["span_micro"]["f1"],
        "external_span_micro_f1": external["span_micro"]["f1"],
        "strict_report_exact_match_rate": exact_rate,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha_file(path)}
            for path in sorted(args.out_dir.iterdir()) if path.is_file()
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
