"""v0.2.2: /health.models_loaded is computed from env vars at request time.

Every slot's precedence mirrors the corresponding endpoint's precedence.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from oncology_arbiter.api.app import _compute_models_loaded
from oncology_arbiter.api.schemas import ModelState


ALL_ENV_KEYS = [
    "ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP",
    "ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY",
    "ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR",
    "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
    "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
    "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY",
    "ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST",
]


def _envset(**overrides: str) -> dict[str, str]:
    """Return an env dict with all model-related keys cleared, then
    overrides applied. Used with patch.dict(clear=False)."""
    env: dict[str, str] = {k: "" for k in ALL_ENV_KEYS}
    env.update(overrides)
    return env


# ---------- Screening slot precedence ----------

def test_screening_default_placeholder() -> None:
    with patch.dict(os.environ, _envset(), clear=False):
        assert _compute_models_loaded()["monai_screening"] == ModelState.PLACEHOLDER


def test_screening_medsiglip_wins() -> None:
    """MedSigLIP takes precedence over SigLIP proxy + MONAI heuristic."""
    with patch.dict(
        os.environ,
        _envset(
            ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP="1",
            ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY="1",
            ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR="1",
        ),
        clear=False,
    ):
        assert _compute_models_loaded()["monai_screening"] == ModelState.LOADED_MEDSIGLIP


def test_screening_siglip_proxy_beats_monai() -> None:
    """SigLIP proxy runs if MedSigLIP is off but SigLIP proxy is on."""
    with patch.dict(
        os.environ,
        _envset(
            ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY="1",
            ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR="1",
        ),
        clear=False,
    ):
        assert _compute_models_loaded()["monai_screening"] == ModelState.PROXY_SIGLIP


def test_screening_monai_heuristic() -> None:
    with patch.dict(
        os.environ,
        _envset(ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR="1"),
        clear=False,
    ):
        assert (
            _compute_models_loaded()["monai_screening"]
            == ModelState.PROXY_MONAI_HEURISTIC
        )


# ---------- Biopsy classifier + report parser ----------

def test_biopsy_classifier_default_placeholder() -> None:
    with patch.dict(os.environ, _envset(), clear=False):
        assert _compute_models_loaded()["medsiglip_biopsy"] == ModelState.PLACEHOLDER


def test_biopsy_medsiglip_probe_on() -> None:
    with patch.dict(
        os.environ,
        _envset(ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP="1"),
        clear=False,
    ):
        assert (
            _compute_models_loaded()["medsiglip_biopsy"]
            == ModelState.LOADED_BIOPSY_PROBE
        )


def test_biopsy_report_parser_always_proxy_regex() -> None:
    """The regex parser is stateless code — reports proxy_regex_v0 always."""
    # With everything off
    with patch.dict(os.environ, _envset(), clear=False):
        assert (
            _compute_models_loaded()["biopsy_report_parser"]
            == ModelState.PROXY_REGEX_V0
        )
    # With biopsy medsiglip probe on
    with patch.dict(
        os.environ,
        _envset(ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP="1"),
        clear=False,
    ):
        assert (
            _compute_models_loaded()["biopsy_report_parser"]
            == ModelState.PROXY_REGEX_V0
        )


# ---------- Therapy slot precedence ----------

def test_therapy_default_placeholder() -> None:
    with patch.dict(os.environ, _envset(), clear=False):
        assert _compute_models_loaded()["txgemma_therapy"] == ModelState.PLACEHOLDER


def test_therapy_txgemma_wins() -> None:
    with patch.dict(
        os.environ,
        _envset(
            ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA="1",
            ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY="1",
        ),
        clear=False,
    ):
        assert _compute_models_loaded()["txgemma_therapy"] == ModelState.LOADED_TXGEMMA


def test_therapy_rules_lite_fallback() -> None:
    with patch.dict(
        os.environ,
        _envset(ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY="1"),
        clear=False,
    ):
        assert _compute_models_loaded()["txgemma_therapy"] == ModelState.PROXY_RULES_LITE


# ---------- Co-Scientist + L3 arbiter ----------

def test_co_scientist_default_placeholder() -> None:
    with patch.dict(os.environ, _envset(), clear=False):
        assert _compute_models_loaded()["co_scientist"] == ModelState.PLACEHOLDER


def test_co_scientist_enabled_reports_proxy_co_scientist() -> None:
    with patch.dict(
        os.environ,
        _envset(ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST="1"),
        clear=False,
    ):
        assert (
            _compute_models_loaded()["co_scientist"]
            == ModelState.PROXY_CO_SCIENTIST
        )


def test_l3_arbiter_always_template() -> None:
    """L3 arbiter templates load from disk unconditionally; reports 'template'."""
    with patch.dict(os.environ, _envset(), clear=False):
        assert _compute_models_loaded()["l3_arbiter"] == ModelState.TEMPLATE


def test_nsclc_always_proxy_lung_heuristic() -> None:
    with patch.dict(os.environ, _envset(), clear=False):
        assert (
            _compute_models_loaded()["nsclc_pipeline"]
            == ModelState.PROXY_LUNG_HEURISTIC
        )


# ---------- End-to-end shape ----------

def test_health_includes_all_expected_slots() -> None:
    """/v0.2.2 adds biopsy_report_parser; keeps l3_arbiter and nsclc_pipeline."""
    with patch.dict(os.environ, _envset(), clear=False):
        result = _compute_models_loaded()
    expected = {
        "monai_screening",
        "medsiglip_biopsy",
        "biopsy_report_parser",
        "txgemma_therapy",
        "co_scientist",
        "l3_arbiter",
        "nsclc_pipeline",
    }
    assert set(result.keys()) == expected


def test_render_env_reports_realistic_state() -> None:
    """The exact Render env config → the values we expect in /health."""
    with patch.dict(
        os.environ,
        _envset(
            ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY="1",
            ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST="1",
        ),
        clear=False,
    ):
        result = _compute_models_loaded()
    assert result["monai_screening"] == ModelState.PLACEHOLDER
    assert result["medsiglip_biopsy"] == ModelState.PLACEHOLDER
    assert result["biopsy_report_parser"] == ModelState.PROXY_REGEX_V0
    assert result["txgemma_therapy"] == ModelState.PROXY_RULES_LITE
    assert result["co_scientist"] == ModelState.PROXY_CO_SCIENTIST
    assert result["l3_arbiter"] == ModelState.TEMPLATE
    assert result["nsclc_pipeline"] == ModelState.PROXY_LUNG_HEURISTIC
