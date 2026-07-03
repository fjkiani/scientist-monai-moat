"""L4b biopsy stage — MedSigLIP-448 embedding + calibrated linear probe.

Design contract
---------------
Reuse the same MedSigLIP-448 encoder that L4a screening already ships (so we
don't ship two copies of an 800M-parameter model). Pool the vision-tower
output to a 768-dim embedding, then feed through a Platt-calibrated 3-class
logistic regression head trained on TCGA-BRCA slides.

Classes:
  * IDC     — invasive ductal carcinoma
  * DCIS    — ductal carcinoma in situ
  * benign  — no evidence of malignancy

Honesty policy (SAME as MedSigLip):
  * Every ``run()`` call preflight-probes the HAI-DEF gate on
    ``google/medsiglip-448``. Denied → GatedAccessError propagates.
  * Weights are **synthetic** and clearly labeled so in
    ``arbiter/models/biopsy_probe_v0.json`` (``n_training_synthetic=True``).
    We MUST NOT quote a real AUROC — the model card carries the caveat.
  * ``BiopsyProbeResult.warnings`` always includes the AUROC_CAVEAT string
    when the probe runs on a real image.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
    check_hai_def_access,
)
from oncology_arbiter.models.medsiglip import MEDSIGLIP_REPO, MedSigLip


# --------------------------------------------------------------------------- #
# Weights file
# --------------------------------------------------------------------------- #

_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "arbiter"
    / "models"
    / "biopsy_probe_v0.json"
)


@dataclass(frozen=True)
class BiopsyProbeWeights:
    """L2-regularised logistic regression head weights."""

    classes: List[str]
    weights: np.ndarray          # (n_classes, embed_dim), float32
    biases: np.ndarray           # (n_classes,), float32
    temperature: float           # Platt scaling
    embed_dim: int
    n_training: int
    n_training_synthetic: bool
    caveat: str

    @classmethod
    def load(cls, path: Path = _WEIGHTS_PATH) -> "BiopsyProbeWeights":
        if not path.exists():
            raise FileNotFoundError(
                f"biopsy probe weights not found at {path} — did the "
                "arbiter/models/biopsy_probe_v0.json ship?"
            )
        blob = json.loads(path.read_text())
        classes = list(blob["classes"])
        weights = np.array(blob["weights"], dtype=np.float32)
        biases = np.array(blob["biases"], dtype=np.float32)
        n_classes, embed_dim = weights.shape
        if biases.shape != (n_classes,):
            raise ValueError(
                f"weights/biases shape mismatch: weights={weights.shape} "
                f"biases={biases.shape}"
            )
        if len(classes) != n_classes:
            raise ValueError(
                f"classes/weights shape mismatch: classes={len(classes)} "
                f"weights.rows={n_classes}"
            )
        return cls(
            classes=classes,
            weights=weights,
            biases=biases,
            temperature=float(blob.get("temperature", 1.0)),
            embed_dim=embed_dim,
            n_training=int(blob.get("n_training", 0)),
            n_training_synthetic=bool(blob.get("n_training_synthetic", True)),
            caveat=str(blob.get("AUROC_CAVEAT", AUROC_CAVEAT)),
        )


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #


@dataclass
class BiopsyProbeResult:
    subtype: str                         # top-1 class label
    subtype_probs: dict[str, float]      # calibrated probs, sum == 1
    embedding_dim: int                   # 768 for medsiglip-448
    model_state: str                     # "loaded_biopsy_probe" (only, since gated raises)
    model_name: str                      # "medsiglip-448 + biopsy-probe-v0"
    weights_n_training: int
    weights_n_training_synthetic: bool
    gate_report: GateReport | None
    warnings: List[str]
    caveat: str = AUROC_CAVEAT
    disclaimer: str = RUO_DISCLAIMER


# --------------------------------------------------------------------------- #
# Probe class
# --------------------------------------------------------------------------- #


class BiopsyMedSigLipProbe:
    """MedSigLIP-448 image encoder + calibrated logistic head.

    Reuses the encoder singleton if one is already loaded (via
    ``_shared_client``); otherwise loads its own copy. The head is small
    (~5 KB JSON) and always loaded from disk.
    """

    def __init__(
        self,
        repo_id: str = MEDSIGLIP_REPO,
        weights_path: Path = _WEIGHTS_PATH,
        preflight_fn: Any = None,
        _shared_client: Optional[MedSigLip] = None,
    ) -> None:
        self.repo_id = repo_id
        self._preflight_fn = preflight_fn or check_hai_def_access
        self._weights = BiopsyProbeWeights.load(weights_path)
        self._client = _shared_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        preprocessed_image: np.ndarray | None = None,
    ) -> BiopsyProbeResult:
        """Run the biopsy probe.

        Exactly one of ``image_bytes`` / ``image_url`` / ``preprocessed_image``
        must be provided. Raises :class:`GatedAccessError` if the HAI-DEF
        gate denies access to the encoder repo.
        """
        # 1) preflight the gate
        gate_report = self._preflight_fn(self.repo_id)
        if not gate_report.allowed:
            raise GatedAccessError(
                repo_id=self.repo_id,
                access_level=gate_report.access_level,
                status_code=gate_report.status_code,
                reason=gate_report.reason,
            )

        # 2) lazy-load the encoder (shared or dedicated)
        if self._client is None:
            self._client = MedSigLip(repo_id=self.repo_id, preflight_fn=self._preflight_fn)
        embedding = self._client.embed_image(
            image_bytes=image_bytes,
            image_url=image_url,
            preprocessed_image=preprocessed_image,
        )
        if embedding.shape != (self._weights.embed_dim,):
            raise ValueError(
                f"embedding shape {embedding.shape} does not match probe "
                f"expected embed_dim={self._weights.embed_dim}"
            )

        # 3) linear head + softmax + Platt temperature
        logits = self._weights.weights @ embedding + self._weights.biases
        scaled = logits / max(1e-3, self._weights.temperature)
        # softmax with numerical stability
        scaled = scaled - scaled.max()
        exps = np.exp(scaled)
        probs = exps / exps.sum()

        subtype_idx = int(probs.argmax())
        subtype = self._weights.classes[subtype_idx]
        subtype_probs = {
            self._weights.classes[i]: float(probs[i])
            for i in range(len(self._weights.classes))
        }

        warnings = [
            f"biopsy_probe_synthetic_weights: n_training={self._weights.n_training} "
            f"synthetic={self._weights.n_training_synthetic}",
            f"biopsy_probe_caveat: {self._weights.caveat}",
        ]

        return BiopsyProbeResult(
            subtype=subtype,
            subtype_probs=subtype_probs,
            embedding_dim=self._weights.embed_dim,
            model_state="loaded_biopsy_probe",
            model_name=f"{self.repo_id} + biopsy-probe-v0",
            weights_n_training=self._weights.n_training,
            weights_n_training_synthetic=self._weights.n_training_synthetic,
            gate_report=gate_report,
            warnings=warnings,
        )


__all__ = [
    "BiopsyMedSigLipProbe",
    "BiopsyProbeResult",
    "BiopsyProbeWeights",
]
