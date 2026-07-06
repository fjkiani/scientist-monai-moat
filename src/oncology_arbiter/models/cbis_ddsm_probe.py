"""Trained LogReg probe on top of Modal-served MedSigLIP-448 embeddings.

Loads ``models/cbis_ddsm_logreg_v1.joblib`` (a scikit-learn Pipeline of
``StandardScaler + LogisticRegression`` fit on 2445 CBIS-DDSM_1024 training
images, held-out test AUC = 0.7526 on n=641). See
``docs/proofs/cbis_ddsm_logreg_v1_metrics.json`` for the full metrics
dossier including the CV per-fold breakdown, threshold sweep, and honesty
caveats.

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


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cbis_ddsm_logreg_v1.joblib"
DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "proofs" / "cbis_ddsm_logreg_v1_metrics.json"
)

# Threshold selected from the metrics dossier for the "high-sensitivity" op
# point (recall = 0.85 on held-out test). Callers can override per-request.
DEFAULT_THRESHOLD: float = 0.2836
DEFAULT_THRESHOLD_LABEL: str = "recall_0.85"

# Honesty text surfaced to the API caller.
CBIS_DDSM_PROBE_WARNING: str = (
    "This probe is a supervised classifier trained on 2445 CBIS-DDSM images "
    "using MedSigLIP-448 embeddings (held-out test AUC = 0.7526 on n=641). "
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
    probe_version: str = "cbis_ddsm_logreg_v1"
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
        from joblib import load as joblib_load

        self._pipe = joblib_load(self.model_path)
        logger.info("Loaded CBIS-DDSM probe: %s", self.model_path)

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
