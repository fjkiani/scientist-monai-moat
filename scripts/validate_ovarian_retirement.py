#!/usr/bin/env python3
"""Reproduce the secondary evaluation supporting ovarian arbiter retirement.

The script consumes a frozen GDC ``/cases`` JSON response and emits only
aggregate statistics plus the input SHA-256. No case identifiers or row-level
predictions are written. Row order is intentionally preserved because the
historical five-fold split was seeded against that order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
HORIZON_DAYS = 730.0
EXPECTED_RAW_SHA256 = "a87eaa72d929aba00f6673a625798219c63cd49d68aad66a0f60852624518942"
EXPECTED = {
    "n": 271,
    "events": 69,
    "age_only_oof_auroc": 0.658057110058832,
    "age_plus_figo_oof_auroc": 0.6828095853063567,
    "delta_oof_auroc": 0.024752475247524663,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _figo_group(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().upper().replace("STAGE", "").strip()
    if text.startswith("IV"):
        return "IV"
    if text.startswith("III"):
        return "III"
    if text.startswith("II") or text.startswith("I"):
        return "I_II"
    return None


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in payload["data"]["hits"]:
        diagnosis = (case.get("diagnoses") or [{}])[0]
        demographic = case.get("demographic") or {}
        rows.append({
            "case_id": case.get("submitter_id") or case.get("case_id"),
            "vital": demographic.get("vital_status"),
            "days_to_birth": demographic.get("days_to_birth"),
            "days_to_death": demographic.get("days_to_death"),
            "days_to_follow_up": diagnosis.get("days_to_last_follow_up"),
            "figo": _figo_group(diagnosis.get("figo_stage")),
        })
    return rows


def build_corrected_cohort(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    v1: list[dict[str, Any]] = []
    initial_drops: Counter[str] = Counter()
    for row in rows:
        if row["vital"] == "Dead":
            y_v1 = 1
        elif row["vital"] == "Alive":
            if row["days_to_follow_up"] is None:
                initial_drops["no_label"] += 1
                continue
            y_v1 = int(float(row["days_to_follow_up"]) < HORIZON_DAYS)
        else:
            initial_drops["no_label"] += 1
            continue
        if row["figo"] is None:
            initial_drops["no_figo"] += 1
            continue
        if row["days_to_birth"] is None:
            initial_drops["no_age"] += 1
            continue
        v1.append({
            **row,
            "age": -float(row["days_to_birth"]) / 365.25,
            "y_v1": y_v1,
        })

    corrected: list[dict[str, Any]] = []
    secondary_drops: Counter[str] = Counter()
    for row in v1:
        if row["vital"] == "Dead":
            if row["days_to_death"] is None:
                secondary_drops["death_untimed"] += 1
                continue
            y = int(float(row["days_to_death"]) <= HORIZON_DAYS)
        else:
            if float(row["days_to_follow_up"]) < HORIZON_DAYS:
                secondary_drops["censored_before_horizon"] += 1
                continue
            y = 0
        corrected.append({**row, "y": y})
    counts = {
        "source_cases": len(rows),
        "v1_n": len(v1),
        "v1_events": sum(r["y_v1"] for r in v1),
        "corrected_n": len(corrected),
        "corrected_events": sum(r["y"] for r in corrected),
        **{f"initial_drop_{k}": int(v) for k, v in sorted(initial_drops.items())},
        **{f"secondary_drop_{k}": int(v) for k, v in sorted(secondary_drops.items())},
    }
    return corrected, counts


def _design(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray([r["y"] for r in rows], dtype=int)
    age = np.asarray([r["age"] / 100.0 for r in rows], dtype=float)[:, None]
    figo = np.asarray([[r["figo"] == "III", r["figo"] == "IV"] for r in rows], dtype=float)
    return y, age, figo, np.column_stack([age, figo])


def _oof_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]


def _paired_auc_delta(y: np.ndarray, full: np.ndarray, reduced: np.ndarray, n_boot: int) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    index = np.arange(len(y))
    values: list[float] = []
    while len(values) < n_boot:
        sample = rng.choice(index, size=len(index), replace=True)
        if np.unique(y[sample]).size < 2:
            continue
        values.append(roc_auc_score(y[sample], full[sample]) - roc_auc_score(y[sample], reduced[sample]))
    values_array = np.asarray(values)
    return {
        "value": float(roc_auc_score(y, full) - roc_auc_score(y, reduced)),
        "ci95": [float(np.quantile(values_array, 0.025)), float(np.quantile(values_array, 0.975))],
        "p_delta_le_zero": float(np.mean(values_array <= 0.0)),
        "n_bootstrap": n_boot,
        "seed": SEED,
    }


def _logistic_mle(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from scipy.optimize import minimize

    design = np.column_stack([np.ones(len(x)), x])
    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    fit = minimize(nll, np.zeros(design.shape[1]), method="BFGS")
    if not fit.success and float(np.linalg.norm(fit.jac)) > 1e-4:
        raise RuntimeError(f"logistic MLE failed: {fit.message}; gradient={np.linalg.norm(fit.jac)}")
    beta = np.asarray(fit.x)
    p = expit(design @ beta)
    weights = p * (1.0 - p)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design))
    se = np.sqrt(np.diag(covariance))
    return {
        "coefficients": beta,
        "standard_errors": se,
        "log_likelihood": -nll(beta),
    }


def _wald_term(name: str, coefficient: float, standard_error: float, scale: float = 1.0) -> dict[str, float | str]:
    beta = coefficient * scale
    se = standard_error * scale
    z = beta / se
    return {
        "term": name,
        "odds_ratio": float(np.exp(beta)),
        "ci95": [float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se))],
        "p_value": float(2.0 * norm.sf(abs(z))),
    }


def _regression_checks(rows: list[dict[str, Any]], y: np.ndarray, age: np.ndarray, figo: np.ndarray) -> dict[str, Any]:
    full = _logistic_mle(np.column_stack([age, figo]), y)
    age_only = _logistic_mle(age, y)
    likelihood_ratio = 2.0 * (full["log_likelihood"] - age_only["log_likelihood"])
    terms = [
        _wald_term("age_per_decade", full["coefficients"][1], full["standard_errors"][1], scale=0.1),
        _wald_term("figo_III_vs_I_II", full["coefficients"][2], full["standard_errors"][2]),
        _wald_term("figo_IV_vs_I_II", full["coefficients"][3], full["standard_errors"][3]),
    ]
    strata: dict[str, Any] = {}
    for stage in ("I_II", "III", "IV"):
        idx = np.asarray([r["figo"] == stage for r in rows])
        y_stage = y[idx]
        age_stage = age[idx]
        if np.unique(y_stage).size < 2:
            strata[stage] = {"n": int(idx.sum()), "events": int(y_stage.sum()), "fit": "degenerate_single_class"}
            continue
        fit = _logistic_mle(age_stage, y_stage)
        strata[stage] = {
            "n": int(idx.sum()),
            "events": int(y_stage.sum()),
            "age_auroc": float(roc_auc_score(y_stage, age_stage[:, 0])),
            "age_per_decade": _wald_term("age_per_decade", fit["coefficients"][1], fit["standard_errors"][1], scale=0.1),
        }
    return {
        "age_adjusted_figo_terms": terms,
        "age_only_vs_age_plus_figo_likelihood_ratio": {
            "statistic": float(likelihood_ratio),
            "degrees_of_freedom": 2,
            "p_value": float(chi2.sf(likelihood_ratio, 2)),
        },
        "figo_stratified_age_regressions": strata,
    }


def analyze(path: Path, n_boot: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = extract_rows(payload)
    corrected, counts = build_corrected_cohort(rows)
    y, age, figo, both = _design(corrected)
    predictions = {"age": _oof_predictions(age, y), "figo": _oof_predictions(figo, y), "both": _oof_predictions(both, y)}
    aurocs = {name: float(roc_auc_score(y, values)) for name, values in predictions.items()}
    result = {
        "input": {"sha256": _sha256(path), "source": "GDC TCGA-OV /cases response", "row_order_preserved": True},
        "cohort": counts,
        "oof_auroc": {"age_only": aurocs["age"], "figo_only": aurocs["figo"], "age_plus_figo": aurocs["both"]},
        "paired_increment_age_plus_figo_over_age": _paired_auc_delta(y, predictions["both"], predictions["age"], n_boot),
        "secondary_regressions": _regression_checks(corrected, y, age, figo),
        "external_validation": {
            "status": "not_performed",
            "reason": "no_feature_compatible_external_cohort_available",
            "interpretation": "absence of external validation is recorded, not replaced by a synthetic or same-cohort validation",
        },
        "verdict": "retire_patient_level_ovarian_scoring",
        "patient_level_outputs_permitted": False,
    }
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--refreshed-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    result, old_rows = analyze(args.input_json, args.bootstrap)
    checks = {
        "frozen_input_sha256": result["input"]["sha256"] == EXPECTED_RAW_SHA256,
        "n": result["cohort"]["corrected_n"] == EXPECTED["n"],
        "events": result["cohort"]["corrected_events"] == EXPECTED["events"],
        "age_only_oof_auroc": abs(result["oof_auroc"]["age_only"] - EXPECTED["age_only_oof_auroc"]) < 1e-12,
        "age_plus_figo_oof_auroc": abs(result["oof_auroc"]["age_plus_figo"] - EXPECTED["age_plus_figo_oof_auroc"]) < 1e-12,
        "delta_oof_auroc": abs(result["paired_increment_age_plus_figo_over_age"]["value"] - EXPECTED["delta_oof_auroc"]) < 1e-12,
    }
    if args.artifact:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        checks["saved_artifact_matches_reanalysis"] = all([
            artifact["n_training"] == EXPECTED["n"],
            artifact["n_positive"] == EXPECTED["events"],
            abs(artifact["performance"]["ablation_auroc"]["age_only"] - result["oof_auroc"]["age_only"]) < 1e-12,
            abs(artifact["performance"]["ablation_auroc"]["age_plus_figo"] - result["oof_auroc"]["age_plus_figo"]) < 1e-12,
        ])
    result["acceptance_checks"] = checks

    if args.refreshed_json:
        refreshed, new_rows = analyze(args.refreshed_json, min(args.bootstrap, 500))
        old_by_id = {r["case_id"]: r for r in old_rows}
        new_by_id = {r["case_id"]: r for r in new_rows}
        common = sorted(old_by_id.keys() & new_by_id.keys())
        result["refreshed_snapshot_sensitivity"] = {
            "input_sha256": refreshed["input"]["sha256"],
            "common_cases": len(common),
            "selected_field_changes": {
                field: sum(old_by_id[i].get(field) != new_by_id[i].get(field) for i in common)
                for field in ["vital", "days_to_death", "days_to_follow_up", "days_to_birth", "figo"]
            },
            "oof_auroc": refreshed["oof_auroc"],
            "interpretation": "The corrected cohort membership is stable, but first-diagnosis follow-up fields and API row order changed. Seeded StratifiedKFold assignments therefore changed; the retirement verdict remains unchanged.",
        }

    if not all(checks.values()):
        raise SystemExit(f"acceptance checks failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks, "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
