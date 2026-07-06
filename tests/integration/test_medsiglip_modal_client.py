"""Live integration test for the Modal-backed MedSigLIP client.

Skipped unless the environment carries ``MODAL_MEDSIGLIP_URL`` (i.e. the
Modal app is deployed and reachable from this host). This is intentionally
NOT a unit test — it hits the real endpoints so we can catch drift between
the deploy artifact and the repo-side client.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest


LIVE = pytest.mark.skipif(
    not os.environ.get("MODAL_MEDSIGLIP_URL"),
    reason="MODAL_MEDSIGLIP_URL not set (skipping live Modal test)",
)

# The three files below are the mammography fixtures staged under
# /mnt/shared-workspace/shared/ — the test skips gracefully if they are
# not present on the current host.
FIXTURE_ROOT = Path("/mnt/shared-workspace/shared")
DEFAULT_DICOM = FIXTURE_ROOT / "smoke_test_dicom.dcm"
DEFAULT_PNG_CANCER = Path(
    "/workspace/cbis_ddsm_1024/train/cancer/"
    "1.3.6.1.4.1.9590.100.1.2.100079897611795347840188527952364733954_1-1.png"
)
DEFAULT_PNG_NOTCANCER = Path(
    "/workspace/cbis_ddsm_1024/train/not_cancer/"
    "1.3.6.1.4.1.9590.100.1.2.100398235711369262725667293542266145456_1-1.png"
)


def _skip_if_missing(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"fixture missing: {p}")


@LIVE
def test_preflight_reports_allowed_and_dim_1152() -> None:
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    c = MedSigLipModalClient()
    gr = c.preflight()
    assert gr.access_level.value == "allowed", gr.reason
    assert gr.repo_id == "google/medsiglip-448"
    # The reason string embeds "dim=1152" — regression fence against silent
    # model swaps that would change the embedding width.
    assert "dim=1152" in gr.reason, gr.reason


@LIVE
def test_embed_png_returns_1152_vector() -> None:
    _skip_if_missing(DEFAULT_PNG_CANCER)
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    c = MedSigLipModalClient()
    emb = c.embed_dicom(DEFAULT_PNG_CANCER)
    assert isinstance(emb, list)
    assert len(emb) == 1152
    # Non-degenerate (not all zeros / not saturated NaNs).
    l2 = math.sqrt(sum(x * x for x in emb))
    assert 1.0 < l2 < 100.0, f"L2 norm {l2} outside expected range"


@LIVE
def test_embed_dicoms_batch_matches_singleton() -> None:
    _skip_if_missing(DEFAULT_PNG_CANCER, DEFAULT_PNG_NOTCANCER)
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    c = MedSigLipModalClient()
    single = c.embed_dicom(DEFAULT_PNG_CANCER)
    batch = c.embed_dicoms([DEFAULT_PNG_CANCER, DEFAULT_PNG_NOTCANCER])
    assert len(batch) == 2
    # Determinism: /embed and /embed_batch must return the same vector for
    # the same input (within tight float32 tolerance).
    for a, b in zip(single, batch[0]):
        assert abs(a - b) < 1e-3, f"drift: single={a} batch={b}"


@LIVE
def test_run_returns_medsiglip_result_contract() -> None:
    _skip_if_missing(DEFAULT_PNG_CANCER)
    from oncology_arbiter.api.schemas import ModelState
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    c = MedSigLipModalClient()
    res = c.run(DEFAULT_PNG_CANCER)
    assert res.model_repo == "google/medsiglip-448"
    assert res.input_resolution == 448
    assert res.model_state is ModelState.LOADED_MEDSIGLIP
    assert res.source_path == str(DEFAULT_PNG_CANCER)
    assert len(res.labels) >= 2
    assert len(res.probs) == len(res.labels)
    assert res.top_label in res.labels
    assert 0.0 <= res.top_prob <= 1.0
    # Every SigLIP prob is a valid sigmoid output.
    assert all(0.0 <= p <= 1.0 for p in res.probs)
    # The mammography off-label warning is a policy invariant.
    assert any("MedSigLIP" in w for w in res.warnings)
    # Gate report must be ALLOWED (test would not reach here otherwise).
    assert res.gate_report is not None
    assert res.gate_report.access_level.value == "allowed"


@LIVE
def test_factory_selects_modal_via_env() -> None:
    monkey_backend = os.environ.get("MEDSIGLIP_BACKEND")
    os.environ["MEDSIGLIP_BACKEND"] = "modal"
    try:
        from oncology_arbiter.models.medsiglip_modal_client import (
            MedSigLipModalClient,
            get_medsiglip_client,
        )

        c = get_medsiglip_client()
        assert isinstance(c, MedSigLipModalClient)
    finally:
        if monkey_backend is None:
            del os.environ["MEDSIGLIP_BACKEND"]
        else:
            os.environ["MEDSIGLIP_BACKEND"] = monkey_backend
