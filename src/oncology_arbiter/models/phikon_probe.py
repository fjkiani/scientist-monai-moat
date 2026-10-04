"""Phikon NCT-CRC linear probe loader.

Loads ``models/phikon_probe_v1.joblib`` (sha256 ``7c77d0759dbcc00440dac6c9b6d99255523ff9706041ce275e8c56724ff8a713``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "phikon_probe_v1.joblib"
EXPECTED_SHA256 = "7c77d0759dbcc00440dac6c9b6d99255523ff9706041ce275e8c56724ff8a713"
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


class PhikonProbe:
    _CACHE: dict[str, "PhikonProbe"] = {}

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipe: Any = None

    @classmethod
    def get(cls) -> "PhikonProbe":
        key = str(DEFAULT_MODEL_PATH)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls()
        return cls._CACHE[key]

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from joblib import load as joblib_load

        self._pipe = joblib_load(self.model_path)
        logger.info("Loaded Phikon probe %s", self.model_path)

    def predict(self, embedding: Sequence[float]) -> dict[str, Any]:
        import numpy as np

        self._load()
        arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != 768:
            raise ValueError(f"Phikon embedding must be 768-d; got {arr.shape[1]}")
        pred = int(self._pipe.predict(arr)[0])
        proba = self._pipe.predict_proba(arr)[0]
        return {
            "pred_class": pred,
            "pred_label": CLASSES[pred],
            "proba": {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))},
            "probe_version": "phikon_probe_v1",
            "artifact_sha256": EXPECTED_SHA256,
        }
