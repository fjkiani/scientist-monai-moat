#!/usr/bin/env python3
"""Train the three stage arbiters on real patient-indexed public data.

Screening and biopsy use the official CBIS-DDSM training case-description
cohort. Therapy models observed chemotherapy receipt in public METABRIC data;
it does not claim treatment benefit or causal efficacy.

The implementation is deliberately leakage-resistant:

* patients, not rows, are assigned to train/validation/test;
* hyperparameters are selected with grouped CV inside the training split only;
* grouped nested-CV predictions estimate training OOF AUROC;
* the held-out test split is scored once by the final training-only fit;
* secondary within-stratum and leave-stratum-out regressions interrogate weak or
  apparently null signals rather than dismissing them as confounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED_DEFAULT = 20260928
C_GRID = (0.01, 0.1, 1.0, 10.0)
CAPABILITIES = ("stage-screening", "stage-biopsy", "stage-therapy")

CBIS_SOURCE_URI = "https://www.cancerimagingarchive.net/collection/cbis-ddsm/"
METABRIC_SOURCE_URI = (
    "https://www.cbioportal.org/api/studies/brca_metabric/clinical-data"
    "?clinicalDataType=PATIENT&projection=DETAILED&pageSize=10000000"
)
RUO = "RESEARCH USE ONLY — not validated for clinical decision-making. Not FDA-cleared. Not CE-marked."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _source_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _inverse_group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.array([1.0 / counts[g] for g in groups], dtype=float)
    return weights / weights.mean()


def _fit_lr(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        solver="lbfgs",
        max_iter=4000,
        random_state=0,
    )
    model.fit(X, y, sample_weight=_inverse_group_weights(groups))
    return model


def _usable_group_folds(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    by_group: dict[str, set[int]] = defaultdict(set)
    for label, group in zip(y.tolist(), groups.tolist(), strict=True):
        by_group[str(group)].add(int(label))
    positive_groups = sum(1 in labels for labels in by_group.values())
    negative_groups = sum(0 in labels for labels in by_group.values())
    return max(2, min(requested, positive_groups, negative_groups, len(by_group)))


def _group_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    requested: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = _usable_group_folds(y, groups, requested)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, valid_idx in splitter.split(X, y, groups):
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[valid_idx])) == 2:
            splits.append((train_idx, valid_idx))
    if len(splits) < 2:
        raise RuntimeError("fewer than two valid grouped stratified folds")
    return splits


def _select_c(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    requested_folds: int = 3,
) -> tuple[float, dict[str, float]]:
    splits = _group_cv_splits(X, y, groups, requested=requested_folds, seed=seed)
    means: dict[str, float] = {}
    for c_value in C_GRID:
        scores: list[float] = []
        for train_idx, valid_idx in splits:
            model = _fit_lr(X[train_idx], y[train_idx], groups[train_idx], c_value)
            pred = model.predict_proba(X[valid_idx])[:, 1]
            scores.append(float(roc_auc_score(y[valid_idx], pred)))
        means[str(c_value)] = float(np.mean(scores))
    # Conservative tie-break: choose stronger regularisation (smallest C).
    best = max(C_GRID, key=lambda c: (means[str(c)], -c))
    return float(best), means


def _nested_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    pilot: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    splits = _group_cv_splits(X, y, groups, requested=4 if pilot else 5, seed=seed)
    oof = np.full(len(y), np.nan, dtype=float)
    details: list[dict[str, Any]] = []
    for fold, (outer_train, outer_valid) in enumerate(splits):
        selected_c, inner = _select_c(
            X[outer_train],
            y[outer_train],
            groups[outer_train],
            seed=seed + 100 + fold,
            requested_folds=2 if pilot else 3,
        )
        model = _fit_lr(X[outer_train], y[outer_train], groups[outer_train], selected_c)
        scores = model.predict_proba(X[outer_valid])[:, 1]
        oof[outer_valid] = scores
        details.append(
            {
                "fold": fold,
                "selected_c": selected_c,
                "inner_mean_auroc_by_c": inner,
                "n_train": int(len(outer_train)),
                "n_valid": int(len(outer_valid)),
                "outer_auroc": float(roc_auc_score(y[outer_valid], scores)),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("nested OOF predictions did not cover every training row")
    return oof, details


def _bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> list[float]:
    unique = np.array(sorted(set(groups.tolist())), dtype=object)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(y[idx])) != 2:
            continue
        values.append(float(roc_auc_score(y[idx], score[idx])))
    if len(values) < max(50, draws // 5):
        raise RuntimeError("insufficient valid patient bootstrap AUROC replicates")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _classify_null(ci: list[float]) -> str:
    if ci[0] > 0.5:
        return "signal_above_null"
    if ci[1] < 0.5:
        return "inverted_signal_below_null"
    return "inconclusive_vs_null"


def _metric_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    if len(y) < 4 or len(np.unique(y)) != 2:
        return {
            "status": "not_estimable",
            "reason": "requires at least four rows and both classes",
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }
    auc = float(roc_auc_score(y, score))
    ci = _bootstrap_auc(y, score, groups, seed=seed, draws=draws)
    return {
        "status": "estimated",
        "auroc": auc,
        "auroc_ci": ci,
        "null_signal_result": _classify_null(ci),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "n_patients": int(len(set(groups.tolist()))),
    }


def _secondary_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    final_model: LogisticRegression,
    *,
    selected_c: float,
    seed: int,
    pilot: bool,
) -> dict[str, Any]:
    """Run prespecified stratified and cross-stratum tests without retuning C."""
    draws = 120 if pilot else 600
    result: dict[str, Any] = {
        "purpose": (
            "Mathematically interrogate weak/null signals using test-stratified AUROC, "
            "within-stratum regressions, and leave-stratum-out cross-cohort validation."
        ),
        "selected_c_fixed_before_test": selected_c,
        "strata": {},
    }
    for offset, stratum in enumerate(sorted(set(strata[test_idx].tolist()))):
        test_sub = test_idx[strata[test_idx] == stratum]
        row: dict[str, Any] = {
            "final_model_test": _metric_with_ci(
                y[test_sub],
                final_model.predict_proba(X[test_sub])[:, 1],
                groups[test_sub],
                seed=seed + 1000 + offset,
                draws=draws,
            )
        }
        within_train = train_idx[strata[train_idx] == stratum]
        if len(within_train) >= 12 and len(np.unique(y[within_train])) == 2:
            within_model = _fit_lr(
                X[within_train], y[within_train], groups[within_train], selected_c
            )
            row["within_stratum_regression"] = _metric_with_ci(
                y[test_sub],
                within_model.predict_proba(X[test_sub])[:, 1],
                groups[test_sub],
                seed=seed + 2000 + offset,
                draws=draws,
            )
            row["within_stratum_regression"]["n_training"] = int(len(within_train))
        else:
            row["within_stratum_regression"] = {
                "status": "not_estimable",
                "reason": "training stratum lacks >=12 rows with both classes",
                "n_training": int(len(within_train)),
            }
        outside_train = train_idx[strata[train_idx] != stratum]
        if len(outside_train) >= 12 and len(np.unique(y[outside_train])) == 2:
            outside_model = _fit_lr(
                X[outside_train], y[outside_train], groups[outside_train], selected_c
            )
            row["leave_stratum_out_validation"] = _metric_with_ci(
                y[test_sub],
                outside_model.predict_proba(X[test_sub])[:, 1],
                groups[test_sub],
                seed=seed + 3000 + offset,
                draws=draws,
            )
            row["leave_stratum_out_validation"]["n_training"] = int(len(outside_train))
        else:
            row["leave_stratum_out_validation"] = {
                "status": "not_estimable",
                "reason": "outside-stratum training data lack >=12 rows with both classes",
                "n_training": int(len(outside_train)),
            }
        result["strata"][str(stratum)] = row
    return result


def _patient_split(
    patient_ids: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_targets: dict[str, int] = {}
    for patient in sorted(set(patient_ids.tolist())):
        labels = y[patient_ids == patient]
        patient_targets[patient] = int(labels.max())
    patients = np.array(sorted(patient_targets), dtype=object)
    targets = np.array([patient_targets[p] for p in patients], dtype=int)
    dev_patients, test_patients = train_test_split(
        patients, test_size=0.15, random_state=seed, stratify=targets
    )
    dev_targets = np.array([patient_targets[p] for p in dev_patients], dtype=int)
    train_patients, validation_patients = train_test_split(
        dev_patients,
        test_size=0.17647058823529413,
        random_state=seed + 1,
        stratify=dev_targets,
    )
    train_set, validation_set, test_set = map(
        set, (train_patients.tolist(), validation_patients.tolist(), test_patients.tolist())
    )
    assert not (train_set & validation_set or train_set & test_set or validation_set & test_set)
    train_idx = np.flatnonzero(np.isin(patient_ids, list(train_set)))
    validation_idx = np.flatnonzero(np.isin(patient_ids, list(validation_set)))
    test_idx = np.flatnonzero(np.isin(patient_ids, list(test_set)))
    return train_idx, validation_idx, test_idx


def _pilot_patient_subset(
    patient_ids: np.ndarray,
    y: np.ndarray,
    *,
    max_patients: int,
    seed: int,
) -> np.ndarray:
    unique = np.array(sorted(set(patient_ids.tolist())), dtype=object)
    if len(unique) <= max_patients:
        return np.arange(len(patient_ids))
    target_by_patient = {
        patient: int(y[patient_ids == patient].max()) for patient in unique.tolist()
    }
    targets = np.array([target_by_patient[p] for p in unique], dtype=int)
    chosen, _ = train_test_split(
        unique,
        train_size=max_patients,
        random_state=seed,
        stratify=targets,
    )
    return np.flatnonzero(np.isin(patient_ids, chosen))


def _load_cbis(cbis_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = (
        ("mass", cbis_dir / "mass_case_description_train_set.csv"),
        ("calc", cbis_dir / "calc_case_description_train_set.csv"),
    )
    for source, path in specs:
        frame = pd.read_csv(path).copy()
        frame.columns = [column.strip() for column in frame.columns]
        frame["source_type"] = source
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(len(frame), dtype=int)
        density_col = "breast_density" if "breast_density" in frame else "breast density"
        frame["density_value"] = pd.to_numeric(frame[density_col], errors="coerce")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["patient_id"] = data["patient_id"].astype(str)
    data["target"] = (data["pathology"].astype(str).str.upper() == "MALIGNANT").astype(int)
    data["assessment_value"] = pd.to_numeric(data["assessment"], errors="coerce")
    data["sample_id"] = [
        f"cbis-train-{source}-{row:05d}"
        for source, row in zip(data["source_type"], data["source_row"], strict=True)
    ]
    if data["sample_id"].duplicated().any():
        raise RuntimeError("CBIS sample IDs are not unique")
    return data


def _screening_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    features = [f"birads_BI_RADS_{value}" for value in range(6)] + [
        "density_A_almost_entirely_fatty",
        "density_B_scattered_fibroglandular",
        "density_C_heterogeneously_dense",
        "density_D_extremely_dense",
    ]
    rows: list[list[float]] = []
    payloads: list[dict[str, Any]] = []
    density_names = {
        1: "A_almost_entirely_fatty",
        2: "B_scattered_fibroglandular",
        3: "C_heterogeneously_dense",
        4: "D_extremely_dense",
    }
    for row in data.to_dict(orient="records"):
        birads = int(row["assessment_value"]) if pd.notna(row["assessment_value"]) else None
        density = int(row["density_value"]) if pd.notna(row["density_value"]) else None
        vector = [float(birads == value) for value in range(6)]
        vector += [float(density_names.get(density) == level) for level in density_names.values()]
        rows.append(vector)
        payloads.append(
            {
                "source_file": row["source_file"],
                "source_row": int(row["source_row"]),
                "abnormality_type": str(row.get("abnormality type", "")),
                "assessment": birads,
                "density": density,
                "view": str(row.get("image view", "")),
                "laterality": str(row.get("left or right breast", "")),
                "pathology": str(row.get("pathology", "")),
            }
        )
    return np.asarray(rows, dtype=float), features, payloads


def _biopsy_lesion_level(row: dict[str, Any]) -> str:
    if row["source_type"] == "calc":
        calc_type = str(row.get("calc type", "")).upper()
        if "PLEOMORPHIC" in calc_type or "FINE_LINEAR" in calc_type:
            return "calcification_pleomorphic"
        if "AMORPHOUS" in calc_type:
            return "calcification_amorphous"
        return "OTHER_OR_UNCLEAR"
    margins = str(row.get("mass margins", "")).upper()
    shape = str(row.get("mass shape", "")).upper()
    if "SPICULATED" in margins:
        return "mass_spiculated"
    if "CIRCUMSCRIBED" in margins:
        return "mass_circumscribed"
    if "ARCHITECTURAL_DISTORTION" in shape:
        return "architectural_distortion"
    return "OTHER_OR_UNCLEAR"


def _biopsy_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    levels = [
        "calcification_pleomorphic",
        "calcification_amorphous",
        "mass_spiculated",
        "mass_circumscribed",
        "architectural_distortion",
    ]
    features = [f"lesion_type_{level}" for level in levels]
    rows: list[list[float]] = []
    payloads: list[dict[str, Any]] = []
    for row in data.to_dict(orient="records"):
        level = _biopsy_lesion_level(row)
        rows.append([float(level == expected) for expected in levels])
        payloads.append(
            {
                "source_file": row["source_file"],
                "source_row": int(row["source_row"]),
                "abnormality_type": str(row.get("abnormality type", "")),
                "lesion_type": level,
                "pathology": str(row.get("pathology", "")),
            }
        )
    return np.asarray(rows, dtype=float), features, payloads


def _load_metabric(
    patient_clinical_path: Path,
    training_table_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    clinical_rows = json.loads(patient_clinical_path.read_text())
    by_patient: dict[str, dict[str, str]] = defaultdict(dict)
    for row in clinical_rows:
        patient = str(row.get("patientId", ""))
        attribute = str(row.get("clinicalAttributeId", ""))
        if patient and attribute:
            by_patient[patient][attribute] = str(row.get("value", ""))
    sample_table = json.loads(training_table_path.read_text())
    records: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for patient in sorted(set(by_patient) & set(sample_table)):
        patient_values = by_patient[patient]
        target_text = patient_values.get("CHEMOTHERAPY")
        if target_text not in {"YES", "NO"}:
            continue
        sample_values = sample_table[patient]
        merged = {**sample_values, **patient_values}
        record = {"patient_id": patient, "sample_id": f"metabric-{patient}"}
        record["target"] = int(target_text == "YES")
        record["cohort"] = patient_values.get("COHORT", "UNKNOWN")
        record["histology"] = patient_values.get("HISTOLOGICAL_SUBTYPE", "Other")
        record["grade"] = sample_values.get("GRADE")
        record["er"] = sample_values.get("ER_STATUS")
        record["pr"] = sample_values.get("PR_STATUS")
        record["her2"] = sample_values.get("HER2_STATUS")
        record["tumor_size"] = sample_values.get("TUMOR_SIZE")
        record["nodes_positive"] = sample_values.get("LYMPH_NODES_EXAMINED_POSITIVE")
        record["age"] = sample_values.get("AGE_AT_DIAGNOSIS")
        records.append(record)
        payloads.append(
            {
                "patient_id": patient,
                "target_field": "CHEMOTHERAPY",
                "chemotherapy_receipt": target_text,
                "cohort": record["cohort"],
                "histological_subtype": record["histology"],
                "grade": record["grade"],
                "er_status": record["er"],
                "pr_status": record["pr"],
                "her2_status": record["her2"],
                "tumor_size": record["tumor_size"],
                "lymph_nodes_examined_positive": record["nodes_positive"],
                "age_at_diagnosis": record["age"],
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty or frame["sample_id"].duplicated().any():
        raise RuntimeError("METABRIC treatment table is empty or has duplicate sample IDs")
    return frame, payloads


def _float_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _therapy_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    features = [
        "histology_invasive_ductal",
        "histology_invasive_lobular",
        "histology_ductal_carcinoma_in_situ",
        "histology_lobular_carcinoma_in_situ",
        "grade_1",
        "grade_2",
        "grade_3",
        "er_status_positive",
        "pr_status_positive",
        "her2_status_positive",
        "tumor_size_norm",
        "node_status_positive",
        "age_at_diagnosis_norm",
    ]
    rows: list[list[float]] = []
    for row in data.to_dict(orient="records"):
        histology = str(row["histology"]).lower()
        grade = str(row["grade"])
        vector = [
            float("ductal" in histology or "nst" in histology),
            float("lobular" in histology),
            0.0,
            0.0,
            float(grade == "1"),
            float(grade == "2"),
            float(grade == "3"),
            float(str(row["er"]).lower() == "positive"),
            float(str(row["pr"]).lower() == "positive"),
            float(str(row["her2"]).lower() == "positive"),
            _float_or_zero(row["tumor_size"]) / 50.0,
            float(_float_or_zero(row["nodes_positive"]) > 0),
            _float_or_zero(row["age"]) / 100.0,
        ]
        rows.append(vector)
    return np.asarray(rows, dtype=float), features


def _feature_encodings(capability: str) -> dict[str, Any]:
    if capability == "stage-screening":
        return {
            "birads": {"ONE_HOT": [f"BI_RADS_{value}" for value in range(6)], "REFERENCE": "BI_RADS_6"},
            "density": {
                "ONE_HOT": [
                    "A_almost_entirely_fatty",
                    "B_scattered_fibroglandular",
                    "C_heterogeneously_dense",
                    "D_extremely_dense",
                ],
                "REFERENCE": "UNKNOWN",
            },
            "prior_biopsy_history": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "family_history_first_degree": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "brca_status_known_pathogenic": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "age_norm": "raw_age_years / 100.0",
            "years_since_last_mammo_norm": "raw_years / 5.0",
        }
    if capability == "stage-biopsy":
        return {
            "lesion_type": {
                "ONE_HOT": [
                    "calcification_pleomorphic",
                    "calcification_amorphous",
                    "mass_spiculated",
                    "mass_circumscribed",
                    "architectural_distortion",
                ],
                "REFERENCE": "OTHER_OR_UNCLEAR",
            },
            "size_norm": "size_mm / 30.0",
            "growth_delta_norm": "delta_mm_per_6mo / 5.0",
            "prior_biopsy_benign_at_site": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "family_history_first_degree": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "brca_status_known_pathogenic": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "us_correlate_hypoechoic_mass": {"true": 1.0, "false": 0.0, "unknown": 0.5},
            "us_correlate_simple_cyst": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        }
    return {
        "histology": {
            "ONE_HOT": [
                "invasive_ductal",
                "invasive_lobular",
                "ductal_carcinoma_in_situ",
                "lobular_carcinoma_in_situ",
            ],
            "REFERENCE": "OTHER_OR_UNCLEAR",
        },
        "grade": {"ONE_HOT": ["1", "2", "3"], "REFERENCE": "UNKNOWN"},
        "er_status_positive": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        "pr_status_positive": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        "her2_status_positive": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        "node_status_positive": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        "brca_status_known_pathogenic": {"true": 1.0, "false": 0.0, "unknown": 0.5},
        "ki67_norm": "ki67_pct / 100.0",
        "tumor_size_norm": "size_mm / 50.0",
        "age_at_diagnosis_norm": "age_years / 100.0",
    }


def _recommendations(capability: str) -> dict[str, str]:
    if capability == "stage-screening":
        return {
            "LOW": "ROUTINE_1YR_FOLLOWUP",
            "MID": "SHORT_INTERVAL_6MO_FOLLOWUP",
            "HIGH": "RECALL_FOR_DIAGNOSTIC_WORKUP",
        }
    if capability == "stage-biopsy":
        return {
            "LOW": "SHORT_INTERVAL_6MO_FOLLOWUP",
            "MID": "ADDITIONAL_IMAGING_WORKUP",
            "HIGH": "PROCEED_TO_CORE_NEEDLE_BIOPSY",
        }
    return {
        "LOW": "LOW_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "MID": "INTERMEDIATE_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
        "HIGH": "HIGH_OBSERVED_CHEMOTHERAPY_RECEIPT_PROFILE",
    }


def _model_name(capability: str) -> str:
    return capability.removeprefix("stage-") + "_arbiter_v1"


def _artifact_relative_path(capability: str) -> str:
    return f"src/oncology_arbiter/arbiter/models/{_model_name(capability)}.json"


def _evidence_stem(capability: str) -> str:
    return capability.replace("-", "_")


def _make_dataset_manifest(
    capability: str,
    dataset_name: str,
    source_uri: str,
    sample_ids: np.ndarray,
    patients: np.ndarray,
    y: np.ndarray,
    payloads: list[dict[str, Any]],
    raw_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = []
    for sample_id, patient, target, payload in zip(
        sample_ids.tolist(), patients.tolist(), y.tolist(), payloads, strict=True
    ):
        samples.append(
            {
                "sample_id": str(sample_id),
                "patient_id": str(patient),
                "payload_sha256": _canonical_sha(payload),
                "target": int(target),
                "target_sha256": _canonical_sha(int(target)),
                "source_record": {
                    key: payload[key]
                    for key in ("source_file", "source_row", "target_field")
                    if key in payload
                },
            }
        )
    return {
        "schema_version": 2,
        "capability": capability,
        "dataset": dataset_name,
        "source_uri": source_uri,
        "raw_sources": raw_sources,
        "target_definition": (
            "pathology equals MALIGNANT"
            if capability != "stage-therapy"
            else "METABRIC patient clinical attribute CHEMOTHERAPY equals YES; observed receipt, not benefit"
        ),
        "samples": samples,
    }


def _train_capability(
    *,
    capability: str,
    dataset_name: str,
    source_uri: str,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    strata: np.ndarray,
    payloads: list[dict[str, Any]],
    raw_sources: list[dict[str, Any]],
    feature_names: list[str],
    seed: int,
    pilot: bool,
    out_dir: Path,
    repo_root: Path,
    write_repo: bool,
    command: str,
) -> dict[str, Any]:
    started_at = _utc_now()
    start_clock = time.perf_counter()
    if pilot:
        keep = _pilot_patient_subset(
            patient_ids, y, max_patients=240 if capability != "stage-therapy" else 420, seed=seed
        )
        X, y = X[keep], y[keep]
        patient_ids, sample_ids, strata = (
            patient_ids[keep], sample_ids[keep], strata[keep]
        )
        payloads = [payloads[index] for index in keep.tolist()]
    train_idx, validation_idx, test_idx = _patient_split(patient_ids, y, seed=seed)
    oof, nested_details = _nested_oof(
        X[train_idx], y[train_idx], patient_ids[train_idx], seed=seed, pilot=pilot
    )
    oof_auc = float(roc_auc_score(y[train_idx], oof))
    oof_ci = _bootstrap_auc(
        y[train_idx],
        oof,
        patient_ids[train_idx],
        seed=seed + 10,
        draws=180 if pilot else 1200,
    )
    selected_c, final_cv = _select_c(
        X[train_idx],
        y[train_idx],
        patient_ids[train_idx],
        seed=seed + 20,
        requested_folds=3 if pilot else 5,
    )
    model = _fit_lr(X[train_idx], y[train_idx], patient_ids[train_idx], selected_c)
    validation_scores = model.predict_proba(X[validation_idx])[:, 1]
    test_scores = model.predict_proba(X[test_idx])[:, 1]
    validation_auc = float(roc_auc_score(y[validation_idx], validation_scores))
    test_auc = float(roc_auc_score(y[test_idx], test_scores))
    test_ci = _bootstrap_auc(
        y[test_idx],
        test_scores,
        patient_ids[test_idx],
        seed=seed + 30,
        draws=180 if pilot else 1200,
    )
    secondary = _secondary_evaluation(
        X,
        y,
        patient_ids,
        strata,
        train_idx,
        test_idx,
        model,
        selected_c=selected_c,
        seed=seed + 40,
        pilot=pilot,
    )
    completed_at = _utc_now()
    while completed_at == started_at:
        time.sleep(0.001)
        completed_at = _utc_now()

    summary: dict[str, Any] = {
        "capability": capability,
        "dataset": dataset_name,
        "pilot": pilot,
        "n_total": int(len(y)),
        "n_patients": int(len(set(patient_ids.tolist()))),
        "n_training": int(len(train_idx)),
        "n_validation": int(len(validation_idx)),
        "n_test": int(len(test_idx)),
        "selected_c": selected_c,
        "oof_auroc": oof_auc,
        "oof_auroc_ci": oof_ci,
        "validation_auroc": validation_auc,
        "test_auroc": test_auc,
        "test_auroc_ci": test_ci,
        "nested_cv": nested_details,
        "final_cv_mean_auroc_by_c": final_cv,
        "secondary_evaluation": secondary,
        "runtime_seconds": time.perf_counter() - start_clock,
        "started_at": started_at,
        "completed_at": completed_at,
        "feature_names": [str(value) for value in feature_names],
    }
    _write_json(out_dir / f"{_evidence_stem(capability)}_run_summary.json", summary)
    if not write_repo:
        return summary

    stem = _evidence_stem(capability)
    evidence_dir = repo_root / "artifacts" / "stage_arbiters"
    artifact_rel = _artifact_relative_path(capability)
    artifact_path = repo_root / artifact_rel
    dataset_rel = f"artifacts/stage_arbiters/{stem}_dataset_v2.json"
    split_rel = f"artifacts/stage_arbiters/{stem}_split_v2.json"
    train_rel = f"artifacts/stage_arbiters/{stem}_train_v2.json"
    evaluation_rel = f"artifacts/stage_arbiters/{stem}_evaluation_v2.json"

    dataset_manifest = _make_dataset_manifest(
        capability,
        dataset_name,
        source_uri,
        sample_ids,
        patient_ids,
        y,
        payloads,
        raw_sources,
    )
    dataset_path = repo_root / dataset_rel
    _write_json(dataset_path, dataset_manifest)
    dataset_sha = _file_sha(dataset_path)
    split_manifest = {
        "schema_version": 2,
        "capability": capability,
        "dataset_manifest_path": dataset_rel,
        "dataset_manifest_sha256": dataset_sha,
        "seed": seed,
        "strategy": "patient-level 70/15/15 stratified split; target=max per patient",
        "train_sample_ids": sample_ids[train_idx].tolist(),
        "validation_sample_ids": sample_ids[validation_idx].tolist(),
        "test_sample_ids": sample_ids[test_idx].tolist(),
        "train_patient_ids": sorted(set(patient_ids[train_idx].tolist())),
        "validation_patient_ids": sorted(set(patient_ids[validation_idx].tolist())),
        "test_patient_ids": sorted(set(patient_ids[test_idx].tolist())),
    }
    split_path = repo_root / split_rel
    _write_json(split_path, split_manifest)
    split_sha = _file_sha(split_path)

    positive_class = (
        "malignant_pathology"
        if capability != "stage-therapy"
        else "observed_chemotherapy_receipt"
    )
    caveat = (
        "Real-patient retrospective grouped training and held-out evaluation; research use only. "
        "The score is not prospectively validated."
    )
    if capability == "stage-therapy":
        caveat += (
            " The target is observed chemotherapy receipt in METABRIC, not treatment response, "
            "benefit, efficacy, or a causal recommendation."
        )
    artifact = {
        "schema_version": 1,
        "model_name": _model_name(capability),
        "model_type": "L2_regularized_logistic_regression",
        "lambda": 1.0 / selected_c,
        "selected_c": selected_c,
        "n_training": int(len(train_idx)),
        "n_positive": int(y[train_idx].sum()),
        "n_negative": int(len(train_idx) - y[train_idx].sum()),
        "n_training_synthetic": False,
        "trained_on_real_patient_data": True,
        "seed": seed,
        "positive_class": positive_class,
        "features": [str(value) for value in feature_names],
        "coefficients": [float(value) for value in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "feature_encodings": _feature_encodings(capability),
        "recommendations": _recommendations(capability),
        "oof_auroc": oof_auc,
        "oof_auroc_ci": oof_ci,
        "performance": {
            "oof_auroc": oof_auc,
            "oof_auroc_ci": oof_ci,
            "validation_auroc": validation_auc,
            "held_out_test_auroc": test_auc,
            "held_out_test_auroc_ci": test_ci,
            "AUROC_CAVEAT": caveat,
        },
        "target_definition": dataset_manifest["target_definition"],
        "training_dataset": dataset_name,
        "disclaimer": RUO,
    }
    # Features are passed separately to keep model fitting code generic.
    artifact["features"] = [str(value) for value in summary["feature_names"]]
    _write_json(artifact_path, artifact)
    artifact_sha = _file_sha(artifact_path)

    evaluation_rows = [
        {
            "sample_id": str(sample_ids[index]),
            "target_sha256": _canonical_sha(int(y[index])),
            "y_true": int(y[index]),
            "y_score": float(score),
            "patient_id": str(patient_ids[index]),
            "stratum": str(strata[index]),
        }
        for index, score in zip(test_idx.tolist(), test_scores.tolist(), strict=True)
    ]
    evaluation = {
        "schema_version": 2,
        "capability": capability,
        "artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "metric": {"name": "auroc", "value": test_auc, "n": int(len(test_idx))},
        "bootstrap_patient_auroc_ci": test_ci,
        "rows": evaluation_rows,
        "secondary_evaluation": secondary,
    }
    evaluation_path = repo_root / evaluation_rel
    _write_json(evaluation_path, evaluation)

    train_manifest = {
        "schema_version": 2,
        "capability": capability,
        "dataset": dataset_name,
        "dataset_manifest_path": dataset_rel,
        "dataset_sha256": dataset_sha,
        "split_manifest_path": split_rel,
        "split_sha256": split_sha,
        "patient_disjoint": True,
        "n_training": int(len(train_idx)),
        "n_validation": int(len(validation_idx)),
        "n_test": int(len(test_idx)),
        "n_training_synthetic": False,
        "seed": seed,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "source_commit": _source_commit(repo_root),
        "selected_c": selected_c,
        "nested_grouped_cv": nested_details,
        "secondary_evaluation_summary": secondary,
    }
    train_path = repo_root / train_rel
    _write_json(train_path, train_manifest)

    summary.update(
        {
            "feature_names": artifact["features"],
            "artifact_path": artifact_rel,
            "artifact_sha256": artifact_sha,
            "dataset_manifest_path": dataset_rel,
            "dataset_sha256": dataset_sha,
            "split_manifest_path": split_rel,
            "split_sha256": split_sha,
            "train_manifest_path": train_rel,
            "train_manifest_sha256": _file_sha(train_path),
            "evaluation_path": evaluation_rel,
            "evaluation_sha256": _file_sha(evaluation_path),
        }
    )
    _write_json(out_dir / f"{stem}_run_summary.json", summary)
    return summary


def _raw_source(path: Path, source_uri: str) -> dict[str, Any]:
    return {
        "filename": path.name,
        "source_uri": source_uri,
        "bytes": path.stat().st_size,
        "sha256": _file_sha(path),
    }


def _update_delivery_manifest(repo_root: Path, summaries: list[dict[str, Any]]) -> None:
    wiring = {
        "stage-screening": "src/oncology_arbiter/arbiter/stage_screening_wiring.py",
        "stage-biopsy": "src/oncology_arbiter/arbiter/stage_biopsy_wiring.py",
        "stage-therapy": "src/oncology_arbiter/arbiter/stage_therapy_wiring.py",
    }
    tests = {
        "stage-screening": "tests/unit/test_stage_screening_v1_identity.py",
        "stage-biopsy": "tests/unit/test_stage_biopsy_v1_identity.py",
        "stage-therapy": "tests/unit/test_stage_therapy_v1_identity.py",
    }
    entries = []
    for summary in summaries:
        capability = summary["capability"]
        entries.append(
            {
                "name": capability,
                "status": "trained_wired_tested",
                "artifact_path": summary["artifact_path"],
                "artifact_sha256": summary["artifact_sha256"],
                "n_training": summary["n_training"],
                "n_training_synthetic": False,
                "train_manifest_path": summary["train_manifest_path"],
                "train_manifest_sha256": summary["train_manifest_sha256"],
                "evaluation_path": summary["evaluation_path"],
                "evaluation_sha256": summary["evaluation_sha256"],
                "wiring_paths": [wiring[capability]],
                "test_paths": [tests[capability]],
            }
        )
    _write_json(
        repo_root / "artifacts" / "tonight_delivery.json",
        {"schema_version": 2, "capabilities": entries},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="fit a patient-subset pilot")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--write-repo", action="store_true", help="write v1 artifacts and schema-v2 evidence")
    parser.add_argument(
        "--cbis-dir", type=Path, default=Path("/mnt/shared-workspace/cbis_ddsm/raw")
    )
    parser.add_argument(
        "--metabric-patient-clinical",
        type=Path,
        default=Path(
            "/mnt/shared-workspace/shared/ten-capability-delivery/stage-arbiters/w2/"
            "metabric_patient_clinical_full.json"
        ),
    )
    parser.add_argument(
        "--metabric-training-table",
        type=Path,
        default=Path(
            "/mnt/shared-workspace/shared/wave6/data/metabric/metabric_training_table.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pilot and args.write_repo:
        raise SystemExit("--pilot cannot be combined with --write-repo")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve()
    command = " ".join([sys.executable, *sys.argv])

    cbis = _load_cbis(args.cbis_dir)
    cbis_patients = cbis["patient_id"].to_numpy(dtype=object)
    cbis_samples = cbis["sample_id"].to_numpy(dtype=object)
    cbis_y = cbis["target"].to_numpy(dtype=int)
    cbis_strata = cbis["source_type"].to_numpy(dtype=object)
    screening_X, screening_features, screening_payloads = _screening_matrix(cbis)
    biopsy_X, biopsy_features, biopsy_payloads = _biopsy_matrix(cbis)
    cbis_sources = [
        _raw_source(args.cbis_dir / "mass_case_description_train_set.csv", CBIS_SOURCE_URI),
        _raw_source(args.cbis_dir / "calc_case_description_train_set.csv", CBIS_SOURCE_URI),
    ]

    therapy, therapy_payloads = _load_metabric(
        args.metabric_patient_clinical, args.metabric_training_table
    )
    therapy_X, therapy_features = _therapy_matrix(therapy)
    therapy_y = therapy["target"].to_numpy(dtype=int)
    therapy_patients = therapy["patient_id"].to_numpy(dtype=object)
    therapy_samples = therapy["sample_id"].to_numpy(dtype=object)
    therapy_strata = therapy["cohort"].to_numpy(dtype=object)
    therapy_sources = [
        _raw_source(args.metabric_patient_clinical, METABRIC_SOURCE_URI),
        _raw_source(args.metabric_training_table, "https://www.cbioportal.org/study/summary?id=brca_metabric"),
    ]

    jobs = [
        {
            "capability": "stage-screening",
            "dataset_name": "CBIS-DDSM official training case descriptions (screening malignancy-risk proxy)",
            "source_uri": CBIS_SOURCE_URI,
            "X": screening_X,
            "y": cbis_y,
            "patient_ids": cbis_patients,
            "sample_ids": cbis_samples,
            "strata": cbis_strata,
            "payloads": screening_payloads,
            "raw_sources": cbis_sources,
            "feature_names": screening_features,
        },
        {
            "capability": "stage-biopsy",
            "dataset_name": "CBIS-DDSM official training case descriptions (biopsy pathology)",
            "source_uri": CBIS_SOURCE_URI,
            "X": biopsy_X,
            "y": cbis_y,
            "patient_ids": cbis_patients,
            "sample_ids": cbis_samples,
            "strata": cbis_strata,
            "payloads": biopsy_payloads,
            "raw_sources": cbis_sources,
            "feature_names": biopsy_features,
        },
        {
            "capability": "stage-therapy",
            "dataset_name": "METABRIC patient clinical cohort (observed chemotherapy receipt)",
            "source_uri": METABRIC_SOURCE_URI,
            "X": therapy_X,
            "y": therapy_y,
            "patient_ids": therapy_patients,
            "sample_ids": therapy_samples,
            "strata": therapy_strata,
            "payloads": therapy_payloads,
            "raw_sources": therapy_sources,
            "feature_names": therapy_features,
        },
    ]
    summaries = []
    for offset, job in enumerate(jobs):
        summary = _train_capability(
            **job,
            seed=args.seed + offset,
            pilot=args.pilot,
            out_dir=args.out_dir,
            repo_root=repo_root,
            write_repo=args.write_repo,
            command=command,
        )
        summaries.append(summary)
        print(
            f"{summary['capability']}: n_train={summary['n_training']} "
            f"oof_auroc={summary['oof_auroc']:.4f} test_auroc={summary['test_auroc']:.4f}",
            flush=True,
        )
    if args.write_repo:
        _update_delivery_manifest(repo_root, summaries)
    _write_json(
        args.out_dir / ("stage_pilot_summary.json" if args.pilot else "stage_full_summary.json"),
        {
            "schema_version": 1,
            "pilot": args.pilot,
            "seed": args.seed,
            "capabilities": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
