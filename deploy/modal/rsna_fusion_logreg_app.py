"""Modal app: RSNA MedSigLIP fusion head (logistic regression) scoring service.

This service exists so oncology-arbiter can run the fusion head as a real remote
ML capability on Modal, rather than embedding coefficients in the API server.

Inputs:
- 1,152-D MedSigLIP embedding (float list)
- clinical metadata (age, density, optional view/laterality/implant/site_id/machine_id)

Weights source:
- Modal volume `rsna-mammo-embeddings` path `cohort_full/fusion_weights.json`

Deploy:
  modal deploy deploy/modal/rsna_fusion_logreg_app.py
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional

import modal

APP_VERSION = "rsna-fusion-logreg-modal-v0.1.0"
APP_NAME = "rsna-fusion-logreg"
VOLUME_EMB = "rsna-mammo-embeddings"
WEIGHTS_PATH = "/emb/cohort_full/fusion_weights.json"

_MODAL_MODE = (os.environ.get("RSNA_FUSION_MODAL_MODE") or "staging").lower()
_MIN_CONTAINERS = 1 if _MODAL_MODE == "prod" else 0
_SCALEDOWN_S = 900 if _MODAL_MODE == "prod" else 300

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.0",
    )
)

app = modal.App(APP_NAME)
vol_emb = modal.Volume.from_name(VOLUME_EMB, create_if_missing=False)


@app.function(image=IMAGE)
@modal.fastapi_endpoint(method="GET", label="rsna-fusion-healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


@app.cls(
    image=IMAGE,
    secrets=[],
    volumes={"/emb": vol_emb},
    timeout=600,
    min_containers=_MIN_CONTAINERS,
    scaledown_window=_SCALEDOWN_S,
)
class RsnaFusionLogReg:
    @modal.enter()
    def load(self) -> None:
        t0 = time.time()
        raw = open(WEIGHTS_PATH, "rb").read()
        self.weights = json.loads(raw.decode("utf-8"))
        if self.weights.get("schema_version") != "rsna_medsiglip_fusion_logreg_v1":
            raise ValueError(f"unexpected schema_version: {self.weights.get('schema_version')!r}")
        self.xmeta_columns = list(self.weights["xmeta_columns"])
        self.mean = list(self.weights["standard_scaler"]["mean"])
        self.scale = list(self.weights["standard_scaler"]["scale"])
        self.coef = list(self.weights["logistic_regression"]["coef"])
        self.intercept = float(self.weights["logistic_regression"]["intercept"])
        self.embedding_dim = int(self.weights["embedding_dim"])
        self.load_seconds = round(time.time() - t0, 3)
        self.warmed_at = time.time()

    def _xmeta(self, meta: Dict[str, Any]) -> list[float]:
        cols = self.xmeta_columns
        v = [0.0] * len(cols)

        def set_one(prefix: str, value: Any | None) -> None:
            if value is None:
                return
            key = f"{prefix}_{value}"
            try:
                j = cols.index(key)
            except ValueError:
                return
            v[j] = 1.0

        # categorical
        set_one("density", meta.get("density"))
        set_one("view", meta.get("view"))
        set_one("laterality", meta.get("laterality"))
        set_one("implant", meta.get("implant"))
        set_one("site_id", meta.get("site_id"))
        set_one("machine_id", meta.get("machine_id"))

        # continuous
        age = meta.get("age")
        if age is not None:
            if "age" in cols:
                v[cols.index("age")] = float(age)
            if "age2" in cols:
                v[cols.index("age2")] = float(age) ** 2
        return v

    def _sigmoid(self, z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    @modal.fastapi_endpoint(method="GET", label="rsna-fusion-info")
    def info(self) -> Dict[str, Any]:
        return {
            "app": APP_NAME,
            "version": APP_VERSION,
            "schema_version": self.weights.get("schema_version"),
            "weights_created_at": self.weights.get("created_at"),
            "embedding_dim": self.embedding_dim,
            "xmeta_columns": len(self.xmeta_columns),
            "load_seconds": self.load_seconds,
            "warmed_at": self.warmed_at,
        }

    @modal.fastapi_endpoint(method="POST", label="rsna-fusion-score")
    def score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        emb = payload.get("embedding")
        meta = payload.get("clinical_metadata") or {}
        if not isinstance(emb, list) or not emb:
            return {"error": "embedding (list[float]) required"}
        if len(emb) != self.embedding_dim:
            return {"error": f"embedding_dim_mismatch:{len(emb)}"}

        # L2 normalize (training did this)
        emb = [float(x) for x in emb]
        nrm = math.sqrt(sum(x * x for x in emb))
        if nrm == 0.0:
            return {"error": "embedding_all_zeros"}
        emb = [x / nrm for x in emb]

        x = emb + self._xmeta(meta)
        if not (len(x) == len(self.mean) == len(self.scale) == len(self.coef)):
            return {"error": "weights_shape_mismatch"}

        z = self.intercept
        for xi, mu, sc, wi in zip(x, self.mean, self.scale, self.coef):
            if float(sc) == 0.0:
                continue
            z += (float(xi) - float(mu)) / float(sc) * float(wi)
        p = float(self._sigmoid(z))
        return {
            "fusion_score": p,
            "model_name": "rsna_medsiglip_fusion_logreg_v1",
            "weights_created_at": self.weights.get("created_at"),
            "seconds": round(time.time() - t0, 3),
        }

