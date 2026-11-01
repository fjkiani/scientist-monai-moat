"""Unit tests for the real BACH biopsy MedSigLIP probe v1.

The encoder and HAI-DEF preflight are stubbed; no model download or network
request occurs.
"""
from __future__ import annotations

import numpy as np
import pytest

from oncology_arbiter.models.biopsy_medsiglip_probe import (
    BACH_BIOPSY_CLASSES,
    BIOPSY_PROBE_EMBED_DIM,
    BIOPSY_PROBE_MODEL_NAME,
    BiopsyMedSigLipProbe,
    BiopsyProbeResult,
    BiopsyProbeWeights,
)
from oncology_arbiter.models.hai_def import AccessLevel, GateReport, GatedAccessError
from oncology_arbiter.models.medsiglip import MEDSIGLIP_REPO, MEDSIGLIP_REVISION


class _FakeMedSigLip:
    """Deterministic pinned pooled-embedding client."""

    repo_id = MEDSIGLIP_REPO
    model_revision = MEDSIGLIP_REVISION

    def __init__(self, embedding_seed: int = 12345, dim: int = BIOPSY_PROBE_EMBED_DIM):
        self._rng = np.random.default_rng(seed=embedding_seed)
        self._dim = dim
        self._called_with = None

    def embed_image(
        self,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        preprocessed_image: np.ndarray | None = None,
    ) -> np.ndarray:
        self._called_with = {
            "image_bytes": image_bytes,
            "image_url": image_url,
            "preprocessed_image": (
                None if preprocessed_image is None else preprocessed_image.shape
            ),
        }
        return self._rng.standard_normal(self._dim).astype(np.float32)


def _preflight_allowed(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="OK",
        has_token=True,
    )


def _preflight_forbidden(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.FORBIDDEN,
        status_code=403,
        reason="terms not accepted",
        has_token=True,
    )


def _preflight_unauthenticated(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.UNAUTHENTICATED,
        status_code=401,
        reason="no HF_TOKEN provided",
        has_token=False,
    )


def test_weights_file_exists_and_loads_real_v1() -> None:
    weights = BiopsyProbeWeights.load()
    assert weights.embed_dim == BIOPSY_PROBE_EMBED_DIM == 1152
    assert tuple(weights.classes) == BACH_BIOPSY_CLASSES
    assert weights.n_training == 292
    assert weights.n_training_synthetic is False
    assert weights.trained_on_real_patient_data is True
    assert weights.embedding_model_revision == MEDSIGLIP_REVISION
    assert weights.coefficients.shape == (3, 1152)
    assert weights.biases.shape == (3,)
    assert np.isfinite(weights.coefficients).all()


def test_probe_run_returns_valid_broad_bach_label() -> None:
    encoder = _FakeMedSigLip(embedding_seed=42)
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=encoder,
    )
    result = probe.run(image_bytes=b"fake-png-bytes")
    assert isinstance(result, BiopsyProbeResult)
    assert result.subtype in BACH_BIOPSY_CLASSES
    assert result.embedding_dim == 1152
    assert result.model_state == "loaded_biopsy_probe"
    assert result.model_name == BIOPSY_PROBE_MODEL_NAME
    assert result.weights_n_training_synthetic is False


def test_subtype_probs_sum_to_one() -> None:
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=999),
    )
    result = probe.run(image_bytes=b"fake")
    assert set(result.subtype_probs) == set(BACH_BIOPSY_CLASSES)
    assert abs(sum(result.subtype_probs.values()) - 1.0) < 1e-12


def test_probe_deterministic_with_same_encoder_seed() -> None:
    r1 = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=77),
    ).run(image_bytes=b"same")
    r2 = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=77),
    ).run(image_bytes=b"same")
    assert r1.subtype == r2.subtype
    for label in r1.subtype_probs:
        assert abs(r1.subtype_probs[label] - r2.subtype_probs[label]) < 1e-12


def test_gated_access_forbidden_raises() -> None:
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_forbidden,
        _shared_client=_FakeMedSigLip(),
    )
    with pytest.raises(GatedAccessError) as exc:
        probe.run(image_bytes=b"whatever")
    assert exc.value.access_level == AccessLevel.FORBIDDEN
    assert exc.value.status_code == 403


def test_gated_access_unauthenticated_raises() -> None:
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_unauthenticated,
        _shared_client=_FakeMedSigLip(),
    )
    with pytest.raises(GatedAccessError) as exc:
        probe.run(image_bytes=b"whatever")
    assert exc.value.access_level == AccessLevel.UNAUTHENTICATED


def test_dimension_mismatch_is_rejected() -> None:
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(dim=768),
    )
    with pytest.raises(ValueError, match="embedding shape"):
        probe.run(image_bytes=b"wrong-representation")


def test_warnings_preserve_label_and_validation_scope() -> None:
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=11),
    )
    result = probe.run(image_bytes=b"x")
    assert not any("synthetic" in warning.lower() for warning in result.warnings)
    assert any("invasive_carcinoma is not IDC" in warning for warning in result.warnings)
    assert any("patient-level generalization" in warning for warning in result.warnings)
    assert result.gate_report is not None
    assert result.gate_report.repo_id == MEDSIGLIP_REPO
    assert result.gate_report.allowed is True
