"""Unit tests for L4b BiopsyMedSigLipProbe.

Tests stub the MedSigLip encoder and HAI-DEF preflight — no HF downloads,
no real gate probing. Deterministic across runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oncology_arbiter.models.biopsy_medsiglip_probe import (
    BiopsyMedSigLipProbe,
    BiopsyProbeResult,
    BiopsyProbeWeights,
)
from oncology_arbiter.models.hai_def import AccessLevel, GateReport, GatedAccessError


# --------------------------------------------------------------------------- #
# Fixtures — deterministic fake encoder
# --------------------------------------------------------------------------- #


class _FakeMedSigLip:
    """Fake MedSigLip that returns deterministic embeddings without any HF."""

    def __init__(self, embedding_seed: int = 12345):
        self._rng = np.random.default_rng(seed=embedding_seed)
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
        return self._rng.standard_normal(768).astype(np.float32)


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


# --------------------------------------------------------------------------- #
# 1. Import + weights load
# --------------------------------------------------------------------------- #


def test_weights_file_exists_and_loads() -> None:
    weights = BiopsyProbeWeights.load()
    assert weights.embed_dim == 768
    assert len(weights.classes) == 3
    assert set(weights.classes) == {"IDC", "DCIS", "benign"}
    assert weights.n_training_synthetic is True
    assert weights.weights.shape == (3, 768)
    assert weights.biases.shape == (3,)


# --------------------------------------------------------------------------- #
# 2. Fake image → probe returns valid subtype ∈ {IDC, DCIS, benign}
# --------------------------------------------------------------------------- #


def test_probe_run_returns_valid_subtype() -> None:
    encoder = _FakeMedSigLip(embedding_seed=42)
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=encoder,
    )
    result = probe.run(image_bytes=b"fake-png-bytes")
    assert isinstance(result, BiopsyProbeResult)
    assert result.subtype in {"IDC", "DCIS", "benign"}
    assert result.embedding_dim == 768
    assert result.model_state == "loaded_biopsy_probe"


# --------------------------------------------------------------------------- #
# 3. Probabilities sum to 1
# --------------------------------------------------------------------------- #


def test_subtype_probs_sum_to_one() -> None:
    encoder = _FakeMedSigLip(embedding_seed=999)
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=encoder,
    )
    result = probe.run(image_bytes=b"fake")
    total = sum(result.subtype_probs.values())
    assert abs(total - 1.0) < 1e-6, f"probs sum={total}, expected 1.0"


# --------------------------------------------------------------------------- #
# 4. Deterministic across runs with the same encoder seed
# --------------------------------------------------------------------------- #


def test_probe_deterministic_with_same_encoder() -> None:
    r1 = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=77),
    ).run(image_bytes=b"same")
    r2 = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=_FakeMedSigLip(embedding_seed=77),
    ).run(image_bytes=b"same")
    assert r1.subtype == r2.subtype
    for k in r1.subtype_probs:
        assert abs(r1.subtype_probs[k] - r2.subtype_probs[k]) < 1e-9


# --------------------------------------------------------------------------- #
# 5. GatedAccessError propagates when preflight denies
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# 6. Warnings + gate_report populated
# --------------------------------------------------------------------------- #


def test_warnings_and_gate_report_populated() -> None:
    encoder = _FakeMedSigLip(embedding_seed=11)
    probe = BiopsyMedSigLipProbe(
        preflight_fn=_preflight_allowed,
        _shared_client=encoder,
    )
    result = probe.run(image_bytes=b"x")
    assert len(result.warnings) >= 2, "expected at least 2 honesty warnings"
    assert any("synthetic" in w.lower() for w in result.warnings), (
        f"expected synthetic-weights warning; got {result.warnings}"
    )
    assert result.gate_report is not None
    assert result.gate_report.repo_id == "google/medsiglip-448"
    assert result.gate_report.allowed is True
