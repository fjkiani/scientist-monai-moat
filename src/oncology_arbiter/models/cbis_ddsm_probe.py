"""Trained LogReg probe on top of Modal-served MedSigLIP-448 embeddings.

Loads ``models/cbis_ddsm_logreg_v2.joblib`` (sha256 ``e5f5b67caf8ba18006e3d582e91becf95fbb20b8b34417366c7ef07b906a211b``; a scikit-learn Pipeline of
``StandardScaler + LogisticRegression`` fit on 2445 CBIS-DDSM_1024 training
images). Features are the 1152-d MedSigLIP embedding only. No clinical
metadata and no radiologist ``assessment``. Held-out rescore on the cached
test embeddings is AUROC 0.747552 (n=641). The C=0.01 sweep row 0.752120 in
``docs/proofs/cbis_ddsm_logreg_v2_metrics.json`` is not this joblib's
``test_metrics.auroc`` (0.747551988693721).

**Off-label for mammography.** MedSigLIP-448 was not pretrained on
mammography; this probe is a supervised head that recovers useful signal
from an off-label backbone. Test AUC ≈ 0.75 is below what fine-tuned
mammography CNNs achieve (typical CBIS-DDSM benchmarks: 0.85–0.90) —
callers must surface this ceiling to the UI.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


EXPECTED_JOBLIB_SHA256 = "e5f5b67caf8ba18006e3d582e91becf95fbb20b8b34417366c7ef07b906a211b"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cbis_ddsm_logreg_v2.joblib"
DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "proofs" / "cbis_ddsm_logreg_v2_metrics.json"
)

# Threshold selected from the metrics dossier for the "high-sensitivity" op
# point (recall = 0.85 on held-out test). Callers can override per-request.
DEFAULT_THRESHOLD: float = 0.2836
DEFAULT_THRESHOLD_LABEL: str = "recall_0.85"

# Honesty text surfaced to the API caller.
CBIS_DDSM_PROBE_WARNING: str = (
    "This probe is a supervised classifier trained on 2445 CBIS-DDSM images "
    "using 1152-d MedSigLIP embeddings only (held-out test AUROC = 0.747552 "
    "on n=641). No clinical metadata and no radiologist assessment. "
    "MedSigLIP-448 was NOT pretrained on mammography — the backbone is used "
    "off-label. Fine-tuned mammography CNNs on CBIS-DDSM typically reach "
    "AUC 0.85-0.90, so this probe is a research prototype, not a clinical tool."
)


@dataclass
class CbisDdsmProbeResult:
    """Output of a single probe forward pass."""

    proba_cancer: float                 # sigmoid probability from LR
    threshold: float                    # decision threshold used
    threshold_label: str                # "recall_0.85", "youden", "0.5", or "custom"
    predicted_class: int                # 1 if proba_cancer >= threshold else 0
    predicted_label: str                # "cancer" / "not_cancer"
    embedding_dim: int                  # 1152 (SigLIP-So400m/14)
    model_repo: str = "google/medsiglip-448"
    probe_version: str = "cbis_ddsm_logreg_v2"
    warnings: list[str] = field(default_factory=lambda: [CBIS_DDSM_PROBE_WARNING])


class CbisDdsmProbe:
    """Lazy-loading wrapper around the trained LogReg probe."""

    _CACHE: dict[str, "CbisDdsmProbe"] = {}

    def __init__(self, model_path: os.PathLike[str] | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipe: Any = None

    @classmethod
    def get(cls) -> "CbisDdsmProbe":
        """Process-level singleton."""
        key = str(DEFAULT_MODEL_PATH)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls(DEFAULT_MODEL_PATH)
        return cls._CACHE[key]

    def _load(self) -> None:
        if self._pipe is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"trained probe not found at {self.model_path}. "
                f"Run scripts/train_cbis_ddsm_logreg.py or check the deploy artifact."
            )
        import hashlib

        from joblib import load as joblib_load

        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest != EXPECTED_JOBLIB_SHA256:
            raise RuntimeError(
                f"cbis joblib identity mismatch: expected {EXPECTED_JOBLIB_SHA256}, got {digest}"
            )
        if self.model_path.name != "cbis_ddsm_logreg_v2.joblib":
            raise RuntimeError(
                f"cbis_v2_path_refuse:{self.model_path.name}:1152-d MedSigLIP "
                "must not enter a fusion head"
            )
        self._pipe = joblib_load(self.model_path)
        scaler = getattr(self._pipe, "named_steps", {}).get("scaler")
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in != 1152:
            raise RuntimeError(
                f"cbis_v2_shape_refuse:scaler_n_features={n_in}:expected_1152"
            )
        self.artifact_sha256 = digest
        logger.info("Loaded CBIS-DDSM probe: %s n_features=%s", self.model_path, n_in)

    def predict_proba(self, embedding: Sequence[float]) -> float:
        """Return P(cancer) for a single 1152-d MedSigLIP embedding."""
        import numpy as np

        self._load()
        arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != 1152:
            raise ValueError(f"embedding must be 1152-d; got {arr.shape[1]}")
        proba = self._pipe.predict_proba(arr)[:, 1]
        return float(proba[0])

    def predict(
        self,
        embedding: Sequence[float],
        *,
        threshold: float | None = None,
        threshold_label: str | None = None,
    ) -> CbisDdsmProbeResult:
        thr = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        thr_lbl = threshold_label or ("custom" if threshold is not None else DEFAULT_THRESHOLD_LABEL)
        p = self.predict_proba(embedding)
        pred = 1 if p >= thr else 0
        return CbisDdsmProbeResult(
            proba_cancer=p,
            threshold=thr,
            threshold_label=thr_lbl,
            predicted_class=pred,
            predicted_label="cancer" if pred == 1 else "not_cancer",
            embedding_dim=len(embedding),
        )


__all__ = [
    "CBIS_DDSM_PROBE_WARNING",
    "CbisDdsmProbe",
    "CbisDdsmProbeResult",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLD_LABEL",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_MODEL_PATH",
]
