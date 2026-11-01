"""Frozen METABRIC breast disease-specific-survival prognostic model.

This is a prognosis model, not a treatment-benefit model.  It predicts the
METABRIC disease-specific mortality label from seven explicit clinicopathologic
features.  It must never substitute missing values from report-parser output:
the endpoint scores only when all seven validated inputs are supplied.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

ARTIFACT_NAME = "breast_dss_arbiter_v3_metabric"
_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "breast_dss_arbiter_v3_metabric.json"
)


class BreastDssArbiterError(ValueError):
    """Raised when the artifact or an inference payload violates contract."""


@dataclass(frozen=True)
class BreastDssResult:
    model_name: str
    score: float
    logit: float
    term_contributions: dict[str, float]
    artifact_sha256: str
    n_training: int
    events: int
    oof_auroc: float
    caveats: tuple[str, ...]


@lru_cache(maxsize=1)
def load_breast_dss_artifact() -> tuple[dict, str]:
    raw = _ARTIFACT_PATH.read_bytes()
    artifact = json.loads(raw)
    if artifact.get("artifact_name") != ARTIFACT_NAME:
        raise BreastDssArbiterError(
            f"unexpected artifact_name={artifact.get('artifact_name')!r}"
        )
    expected = [
        "age", "tumor_size_mm", "nodes_positive", "grade",
        "er_pos", "pr_pos", "her2_pos",
    ]
    if artifact.get("features") != expected:
        raise BreastDssArbiterError("artifact feature order drift")
    model = artifact.get("model") or {}
    if not model.get("standardised"):
        raise BreastDssArbiterError("artifact must carry standardisation parameters")
    if len(model.get("scaler_mean", [])) != len(expected):
        raise BreastDssArbiterError("artifact scaler_mean length drift")
    if len(model.get("scaler_scale", [])) != len(expected):
        raise BreastDssArbiterError("artifact scaler_scale length drift")
    return artifact, hashlib.sha256(raw).hexdigest()


def score_breast_dss(features: Mapping[str, float | int | bool]) -> BreastDssResult:
    """Apply the frozen standardised logistic regression exactly once.

    No imputation is permitted.  The API's Pydantic request model handles
    clinical range validation; this function independently enforces complete,
    finite numerics so non-API callers cannot bypass the contract.
    """
    artifact, digest = load_breast_dss_artifact()
    names = artifact["features"]
    missing = [name for name in names if name not in features]
    extra = sorted(set(features) - set(names))
    if missing or extra:
        raise BreastDssArbiterError(
            f"feature contract mismatch: missing={missing}, extra={extra}"
        )

    values = []
    for name in names:
        value = float(features[name])
        if not math.isfinite(value):
            raise BreastDssArbiterError(f"{name} must be finite")
        values.append(value)

    model = artifact["model"]
    means = model["scaler_mean"]
    scales = model["scaler_scale"]
    coefs = model["coef_standardised"]
    contributions = {
        name: float(coefs[name]) * ((value - float(mean)) / float(scale))
        for name, value, mean, scale in zip(names, values, means, scales)
    }
    logit = float(model["intercept"]) + sum(contributions.values())
    # Stable logistic transform.
    if logit >= 0:
        score = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        score = e / (1.0 + e)

    return BreastDssResult(
        model_name=ARTIFACT_NAME,
        score=float(score),
        logit=float(logit),
        term_contributions=contributions,
        artifact_sha256=digest,
        n_training=int(artifact["n"]),
        events=int(artifact["events"]),
        oof_auroc=float(artifact["out_of_fold_discrimination"]["arbiter"]),
        caveats=tuple(str(x) for x in artifact["deployment_caveats"]),
    )
