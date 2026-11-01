"""Regression lock: luna16-infer detect path must contain Spacingd-equivalent resample.

Fails if deploy/modal/luna16_infer_app.py drops `_resample_volume_to_luna16`
or stops calling it from detect(). Source-level (no Modal network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "deploy" / "modal" / "luna16_infer_app.py"


def test_luna16_infer_app_defines_resample_helper() -> None:
    src = APP_PATH.read_text(encoding="utf-8")
    assert "_resample_volume_to_luna16" in src, (
        "detect path missing _resample_volume_to_luna16 helper "
        "(Spacingd-equivalent required by oncology-moat-luna16)"
    )
    assert "sitk.ResampleImageFilter" in src or "ResampleImageFilter" in src
    assert "LUNA16_TARGET_SPACING_MM" in src


def test_detect_calls_resample_before_clip() -> None:
    src = APP_PATH.read_text(encoding="utf-8")
    # Function must be invoked inside detect body, not only defined.
    call_idx = src.find("_resample_volume_to_luna16(volume")
    clip_idx = src.find("np.clip(volume")
    assert call_idx > 0, "detect() never calls _resample_volume_to_luna16(volume...)"
    assert clip_idx > call_idx, "HU clip must run after resample"


def test_preprocessing_summary_emits_actual_and_source_spacing() -> None:
    src = APP_PATH.read_text(encoding="utf-8")
    assert '"source_spacing_mm"' in src or "'source_spacing_mm'" in src
    assert '"was_resampled"' in src or "'was_resampled'" in src
    assert "actual_spacing_mm" in src
