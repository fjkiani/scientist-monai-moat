"""RSNA MedSigLIP fusion (LogReg) — deterministic API scorer.

This is the *exact* linear head fit in Modal by
`deploy/modal/rsna_medsiglip_embed_full.py::fusion_clinical_proof`:

- inputs: 1152-d L2-normalized MedSigLIP embedding + one-hot metadata vector
- model: StandardScaler(with_mean=True, with_std=True) + LogisticRegression
- output: sigmoid probability

Research use only (RUO). Not validated prospectively. Not a diagnostic device.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


_HERE = Path(__file__).resolve()
# In packaged wheels, `models/` is shipped alongside the python package under site-packages/.
# In local dev (src-layout), `models/` lives at repo root (one level above `src/`).
_DEFAULT_IN_SITEPKG = _HERE.parents[3] / "models" / "rsna_medsiglip_fusion_logreg_v1.json"
_DEFAULT_IN_REPO = _HERE.parents[4] / "models" / "rsna_medsiglip_fusion_logreg_v1.json"
DEFAULT_WEIGHTS_PATH = _DEFAULT_IN_SITEPKG if _DEFAULT_IN_SITEPKG.exists() else _DEFAULT_IN_REPO

RSNA_FUSION_WARNING: str = (
    "RUO: This score is a supervised logistic-regression head fit on the RSNA Breast Cancer Detection "
    "Kaggle training cohort using MedSigLIP-448 embeddings + limited clinical metadata. "
    "It is retrospective, image-level, and not validated for clinical decision-making."
)


@dataclass(frozen=True)
class RsnaFusionInputs:
    """Minimal clinical metadata required by the fused head."""

    age: float
    density: str
    laterality: str | None = None
    view: str | None = None
    implant: str | None = None
    site_id: str | None = None
    machine_id: str | None = None


class RsnaMedsiglipFusionLogReg:
    """Loads the frozen fused head weights and provides a predict_proba API."""

    _CACHE: dict[str, "RsnaMedsiglipFusionLogReg"] = {}

    def __init__(self, weights_path: os.PathLike[str] | str = DEFAULT_WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        self._w: dict[str, Any] | None = None

    @classmethod
    def get(cls) -> "RsnaMedsiglipFusionLogReg":
        key = str(DEFAULT_WEIGHTS_PATH)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls(DEFAULT_WEIGHTS_PATH)
        return cls._CACHE[key]

    def _load(self) -> None:
        if self._w is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"RSNA fusion weights not found at {self.weights_path}. "
                "Pull cohort_full/fusion_weights.json from Modal and commit it under models/."
            )
        self._w = json.loads(self.weights_path.read_text(encoding="utf-8"))
        if self._w.get("schema_version") != "rsna_medsiglip_fusion_logreg_v1":
            raise ValueError(f"unexpected schema_version: {self._w.get('schema_version')!r}")
        logger.info("Loaded RSNA fusion weights: %s", self.weights_path)

    @property
    def embedding_dim(self) -> int:
        self._load()
        assert self._w is not None
        return int(self._w["embedding_dim"])

    @property
    def xmeta_columns(self) -> list[str]:
        self._load()
        assert self._w is not None
        return list(self._w["xmeta_columns"])

    def _xmeta_vector(self, meta: RsnaFusionInputs) -> list[float]:
        """Build the one-hot metadata vector in the training column order.

        Unknown / missing categorical fields are encoded as all-zeros for that field.
        """
        cols = self.xmeta_columns
        v = [0.0] * len(cols)

        def set_one(prefix: str, value: str | None) -> None:
            if value is None:
                return
            key = f"{prefix}_{value}"
            try:
                j = cols.index(key)
            except ValueError:
                return
            v[j] = 1.0

        # Categorical (training used str-cast values)
        set_one("density", str(meta.density))
        set_one("view", str(meta.view) if meta.view is not None else None)
        set_one("laterality", str(meta.laterality) if meta.laterality is not None else None)
        set_one("implant", str(meta.implant) if meta.implant is not None else None)
        set_one("site_id", str(meta.site_id) if meta.site_id is not None else None)
        set_one("machine_id", str(meta.machine_id) if meta.machine_id is not None else None)

        # Continuous fields (if present in column set)
        if "age" in cols:
            v[cols.index("age")] = float(meta.age)
        if "age2" in cols:
            v[cols.index("age2")] = float(meta.age) ** 2
        return v

    def predict_proba(
        self,
        *,
        embedding: Sequence[float],
        meta: RsnaFusionInputs,
    ) -> float:
        """Return fused probability for a single mammogram."""
        self._load()
        assert self._w is not None

        emb = [float(x) for x in embedding]
        if len(emb) != int(self._w["embedding_dim"]):
            raise ValueError(f"embedding must be {self._w['embedding_dim']}-d; got {len(emb)}")

        # L2 normalize (training did this)
        nrm = math.sqrt(sum(x * x for x in emb))
        if nrm == 0.0:
            raise ValueError("embedding is all zeros")
        emb = [x / nrm for x in emb]

        xmeta = self._xmeta_vector(meta)
        x = emb + xmeta

        mean = self._w["standard_scaler"]["mean"]
        scale = self._w["standard_scaler"]["scale"]
        coef = self._w["logistic_regression"]["coef"]
        intercept = float(self._w["logistic_regression"]["intercept"])

        if not (len(x) == len(mean) == len(scale) == len(coef)):
            raise ValueError(
                "weights shape mismatch: "
                f"len(x)={len(x)} len(mean)={len(mean)} len(scale)={len(scale)} len(coef)={len(coef)}"
            )

        # Standardize then linear + sigmoid
        z = intercept
        for xi, mu, sc, wi in zip(x, mean, scale, coef):
            if sc == 0.0:
                continue
            z += (float(xi) - float(mu)) / float(sc) * float(wi)
        # stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return float(1.0 / (1.0 + ez))
        ez = math.exp(z)
        return float(ez / (1.0 + ez))


__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "RSNA_FUSION_WARNING",
    "RsnaFusionInputs",
    "RsnaMedsiglipFusionLogReg",
]

