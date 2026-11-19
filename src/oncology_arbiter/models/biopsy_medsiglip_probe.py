"""BACH biopsy microscopy probe on the pinned MedSigLIP-448 vision tower.

The production v1 head is a real three-class multinomial logistic regression
trained on patient-disjoint BACH microscopy images.  Its labels deliberately
preserve the source ontology: ``invasive_carcinoma`` is not relabelled IDC and
``in_situ_carcinoma`` is not relabelled DCIS.

The head was fit on the unprojected 1,152-dimensional
``vision_model(...).pooler_output`` representation.  Both local and Modal
clients therefore must expose exactly that representation; projected
``get_image_features()`` vectors are not interchangeable with this artifact.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from oncology_arbiter import RUO_DISCLAIMER
from oncology_arbiter.models.hai_def import (
    GateReport,
    GatedAccessError,
    check_hai_def_access,
)
from oncology_arbiter.models.medsiglip import (
    MEDSIGLIP_REPO,
    MEDSIGLIP_REVISION,
    MedSigLip,
)


_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "arbiter"
    / "models"
    / "biopsy_probe_v1.json"
)

BUNDLED_BIOPSY_PROBE_V1_SHA256 = (
    "58f1699d88cced9f76b93ee7bc0d1df820ef48e0c30442585914d2412d5a9d8c"
)
BACH_BIOPSY_CLASSES = (
    "benign_or_normal",
    "in_situ_carcinoma",
    "invasive_carcinoma",
)
BIOPSY_PROBE_EMBED_DIM = 1152
BIOPSY_PROBE_MODEL_NAME = (
    f"{MEDSIGLIP_REPO}@{MEDSIGLIP_REVISION}"
    f"+biopsy_probe_v1@sha256:{BUNDLED_BIOPSY_PROBE_V1_SHA256}"
)
BIOPSY_PROBE_CAVEAT = (
    "BACH microscopy three-class research probe. Broad BACH labels do not "
    "establish histologic subtype: invasive_carcinoma must not be interpreted "
    "as IDC and in_situ_carcinoma must not be interpreted as DCIS."
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BiopsyProbeWeights:
    """Identity-bearing multinomial logistic-regression head."""

    classes: List[str]
    coefficients: np.ndarray       # (3, 1152), float64
    biases: np.ndarray             # (3,), float64
    temperature: float
    embedding_transform: str
    embed_dim: int
    n_training: int
    n_training_synthetic: bool
    trained_on_real_patient_data: bool
    embedding_model_repo: str
    embedding_model_revision: str
    model_name: str
    held_out_test_metrics: dict[str, Any]
    artifact_sha256: str
    disclaimer: str

    @property
    def weights(self) -> np.ndarray:
        """Backward-compatible alias for older callers."""
        return self.coefficients

    @classmethod
    def load(
        cls,
        path: Path = _WEIGHTS_PATH,
        *,
        expected_sha256: str | None = BUNDLED_BIOPSY_PROBE_V1_SHA256,
    ) -> "BiopsyProbeWeights":
        if not path.is_file():
            raise FileNotFoundError(f"biopsy probe v1 artifact not found at {path}")

        raw = path.read_bytes()
        actual_sha256 = _sha256_bytes(raw)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise RuntimeError(
                "biopsy-probe artifact identity mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        blob = json.loads(raw)
        if blob.get("$schema_version") != "biopsy_probe_v1":
            raise ValueError(
                "production biopsy loader requires $schema_version=biopsy_probe_v1"
            )

        classes = [str(value) for value in blob["classes"]]
        if tuple(classes) != BACH_BIOPSY_CLASSES:
            raise ValueError(
                f"unexpected biopsy class order {classes!r}; expected "
                f"{list(BACH_BIOPSY_CLASSES)!r}"
            )

        coefficients = np.asarray(blob["coefficients"], dtype=np.float64)
        biases = np.asarray(blob["biases"], dtype=np.float64)
        embed_dim = int(blob["embed_dim"])
        expected_shape = (len(BACH_BIOPSY_CLASSES), BIOPSY_PROBE_EMBED_DIM)
        if embed_dim != BIOPSY_PROBE_EMBED_DIM or coefficients.shape != expected_shape:
            raise ValueError(
                "biopsy-probe coefficient contract mismatch: "
                f"embed_dim={embed_dim}, coefficients={coefficients.shape}, "
                f"expected embed_dim={BIOPSY_PROBE_EMBED_DIM}, shape={expected_shape}"
            )
        if biases.shape != (len(BACH_BIOPSY_CLASSES),):
            raise ValueError(
                f"coefficient/bias shape mismatch: {coefficients.shape} vs {biases.shape}"
            )
        if not np.isfinite(coefficients).all() or not np.isfinite(biases).all():
            raise ValueError("biopsy-probe coefficients and biases must be finite")

        temperature = float(blob["temperature"])
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"invalid biopsy-probe temperature={temperature!r}")

        embedding_transform = str(blob.get("embedding_transform", "none"))
        if embedding_transform not in {"none", "l2_normalize"}:
            raise ValueError(
                f"unsupported biopsy-probe embedding_transform={embedding_transform!r}"
            )

        n_training = int(blob.get("n_training", 0))
        n_training_synthetic = bool(blob.get("n_training_synthetic", True))
        trained_on_real_patient_data = bool(
            blob.get("trained_on_real_patient_data", False)
        )
        if n_training <= 0 or n_training_synthetic or not trained_on_real_patient_data:
            raise ValueError(
                "production biopsy probe refuses template/synthetic lineage: "
                f"n_training={n_training}, synthetic={n_training_synthetic}, "
                f"real_patient_data={trained_on_real_patient_data}"
            )

        embedding_model_repo = str(blob.get("embedding_model_repo", ""))
        embedding_model_revision = str(blob.get("embedding_model_revision", ""))
        if embedding_model_repo != MEDSIGLIP_REPO:
            raise ValueError(
                f"unexpected embedding repo {embedding_model_repo!r}; "
                f"expected {MEDSIGLIP_REPO!r}"
            )
        if embedding_model_revision != MEDSIGLIP_REVISION:
            raise ValueError(
                "embedding revision mismatch: "
                f"artifact={embedding_model_revision!r}, expected={MEDSIGLIP_REVISION!r}"
            )

        return cls(
            classes=classes,
            coefficients=coefficients,
            biases=biases,
            temperature=temperature,
            embedding_transform=embedding_transform,
            embed_dim=embed_dim,
            n_training=n_training,
            n_training_synthetic=n_training_synthetic,
            trained_on_real_patient_data=trained_on_real_patient_data,
            embedding_model_repo=embedding_model_repo,
            embedding_model_revision=embedding_model_revision,
            model_name=str(blob.get("model_name", "biopsy_probe_v1")),
            held_out_test_metrics=dict(blob.get("held_out_test_metrics", {})),
            artifact_sha256=actual_sha256,
            disclaimer=str(blob.get("disclaimer", RUO_DISCLAIMER)),
        )


@dataclass
class BiopsyProbeResult:
    subtype: str
    subtype_probs: dict[str, float]
    embedding_dim: int
    model_state: str
    model_name: str
    weights_n_training: int
    weights_n_training_synthetic: bool
    artifact_sha256: str
    embedding_model_revision: str
    gate_report: GateReport | None
    warnings: List[str]
    caveat: str = BIOPSY_PROBE_CAVEAT
    disclaimer: str = RUO_DISCLAIMER


class BiopsyMedSigLipProbe:
    """Pinned MedSigLIP pooled embedding plus the real BACH v1 head."""

    def __init__(
        self,
        repo_id: str = MEDSIGLIP_REPO,
        weights_path: Path = _WEIGHTS_PATH,
        preflight_fn: Any = None,
        _shared_client: Optional[Any] = None,
        *,
        expected_artifact_sha256: str | None = BUNDLED_BIOPSY_PROBE_V1_SHA256,
        expected_model_revision: str = MEDSIGLIP_REVISION,
    ) -> None:
        self.repo_id = repo_id
        self._weights = BiopsyProbeWeights.load(
            weights_path,
            expected_sha256=expected_artifact_sha256,
        )
        if repo_id != self._weights.embedding_model_repo:
            raise ValueError(
                f"probe repo_id={repo_id!r} does not match trained repo "
                f"{self._weights.embedding_model_repo!r}"
            )
        if expected_model_revision != self._weights.embedding_model_revision:
            raise ValueError(
                "requested model revision does not match trained artifact: "
                f"{expected_model_revision!r} != "
                f"{self._weights.embedding_model_revision!r}"
            )

        self._client = _shared_client
        if _shared_client is not None:
            client_repo = getattr(_shared_client, "repo_id", None)
            if client_repo is not None and client_repo != repo_id:
                raise ValueError(
                    f"embedding client repo mismatch: {client_repo!r} != {repo_id!r}"
                )
            client_revision = getattr(
                _shared_client,
                "model_revision",
                getattr(_shared_client, "revision", None),
            )
            if client_revision is not None and client_revision != expected_model_revision:
                raise ValueError(
                    "embedding client revision mismatch: "
                    f"{client_revision!r} != {expected_model_revision!r}"
                )

        if preflight_fn is not None:
            self._preflight_fn = preflight_fn
        elif _shared_client is not None and callable(
            getattr(_shared_client, "preflight", None)
        ):
            self._preflight_fn = lambda _repo: _shared_client.preflight()
        else:
            self._preflight_fn = check_hai_def_access

    def run(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        preprocessed_image: np.ndarray | None = None,
    ) -> BiopsyProbeResult:
        """Run one microscopy image through the identity-locked v1 probe."""
        gate_report = self._preflight_fn(self.repo_id)
        if not gate_report.allowed:
            raise GatedAccessError(
                repo_id=self.repo_id,
                access_level=gate_report.access_level,
                status_code=gate_report.status_code,
                reason=gate_report.reason,
            )

        if self._client is None:
            # Reuse the successful report so the local wrapper does not repeat
            # the remote HAI-DEF probe on the same request.
            self._client = MedSigLip(
                repo_id=self.repo_id,
                revision=self._weights.embedding_model_revision,
                preflight_fn=lambda _repo: gate_report,
            )

        embedding = np.asarray(
            self._client.embed_image(
                image_bytes=image_bytes,
                image_url=image_url,
                preprocessed_image=preprocessed_image,
            ),
            dtype=np.float64,
        )
        if embedding.shape != (self._weights.embed_dim,):
            raise ValueError(
                f"embedding shape {embedding.shape} does not match trained "
                f"embed_dim={self._weights.embed_dim}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("embedding contains non-finite values")

        if self._weights.embedding_transform == "l2_normalize":
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError("cannot L2-normalize a zero/non-finite embedding")
            embedding = embedding / norm

        logits = self._weights.coefficients @ embedding + self._weights.biases
        scaled = logits / self._weights.temperature
        scaled = scaled - float(np.max(scaled))
        exps = np.exp(scaled)
        probs = exps / float(np.sum(exps))

        subtype_idx = int(np.argmax(probs))
        subtype = self._weights.classes[subtype_idx]
        subtype_probs = {
            label: float(probs[index])
            for index, label in enumerate(self._weights.classes)
        }

        test_metrics = self._weights.held_out_test_metrics
        warnings = [
            "biopsy_probe_label_scope:BACH broad labels only; "
            "invasive_carcinoma is not IDC and in_situ_carcinoma is not DCIS.",
            "biopsy_probe_validation_scope:"
            f"held_out_invasive_ovr_auroc="
            f"{float(test_metrics.get('invasive_ovr_auroc', float('nan'))):.6f};"
            f"multiclass_top1_accuracy="
            f"{float(test_metrics.get('accuracy', float('nan'))):.6f};"
            "in_situ patient-level generalization is not established.",
        ]

        return BiopsyProbeResult(
            subtype=subtype,
            subtype_probs=subtype_probs,
            embedding_dim=self._weights.embed_dim,
            model_state="loaded_biopsy_probe",
            model_name=(
                f"{self.repo_id}@{self._weights.embedding_model_revision}"
                f"+{self._weights.model_name}@sha256:{self._weights.artifact_sha256}"
            ),
            weights_n_training=self._weights.n_training,
            weights_n_training_synthetic=self._weights.n_training_synthetic,
            artifact_sha256=self._weights.artifact_sha256,
            embedding_model_revision=self._weights.embedding_model_revision,
            gate_report=gate_report,
            warnings=warnings,
            caveat=BIOPSY_PROBE_CAVEAT,
            disclaimer=self._weights.disclaimer,
        )


__all__ = [
    "BACH_BIOPSY_CLASSES",
    "BIOPSY_PROBE_CAVEAT",
    "BIOPSY_PROBE_EMBED_DIM",
    "BIOPSY_PROBE_MODEL_NAME",
    "BUNDLED_BIOPSY_PROBE_V1_SHA256",
    "BiopsyMedSigLipProbe",
    "BiopsyProbeResult",
    "BiopsyProbeWeights",
]
