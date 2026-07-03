"""Unit tests for L4a MonaiDetector — mask-gradient heuristic mode.

Runs on synthetic images (no DICOM required). Deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from oncology_arbiter.models.monai_detector import (
    MONAI_DETECTOR_WARNING,
    DetectionBox,
    MonaiDetector,
    MonaiDetectorResult,
)


# --------------------------------------------------------------------------- #
# 1. Empty breast mask → 0 boxes, warning still present
# --------------------------------------------------------------------------- #


def test_empty_breast_mask_yields_no_boxes() -> None:
    img = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=bool)
    result = MonaiDetector().detect(img, mask)
    assert result.n_boxes == 0
    assert result.weights_loaded is False
    assert MONAI_DETECTOR_WARNING in result.warnings


# --------------------------------------------------------------------------- #
# 2. Injected bright square → detector localizes at least one box near it
# --------------------------------------------------------------------------- #


def test_bright_square_is_localized() -> None:
    rng = np.random.default_rng(seed=42)
    H, W = 256, 256
    img = rng.random((H, W)).astype(np.float32) * 0.3
    img[100:130, 100:130] += 0.5  # injected lesion

    mask = np.zeros((H, W), dtype=bool)
    mask[50:250, 30:200] = True

    result = MonaiDetector(max_boxes=3).detect(img, mask)
    assert result.n_boxes >= 1
    # At least one box overlaps the injected square in normalized coords.
    target_x = 115 / W  # center of injected square
    target_y = 115 / H
    overlaps = [
        b for b in result.boxes
        if b.x0 <= target_x <= b.x1 and b.y0 <= target_y <= b.y1
    ]
    assert overlaps, (
        f"expected a detection covering injected lesion at ({target_x:.3f}, "
        f"{target_y:.3f}); got boxes: "
        f"{[(b.x0, b.y0, b.x1, b.y1) for b in result.boxes]}"
    )


# --------------------------------------------------------------------------- #
# 3. weights_loaded=False on every box + at result level
# --------------------------------------------------------------------------- #


def test_weights_loaded_is_false_on_every_finding() -> None:
    rng = np.random.default_rng(seed=1)
    img = rng.random((128, 128)).astype(np.float32)
    mask = np.ones((128, 128), dtype=bool)
    result = MonaiDetector().detect(img, mask)
    assert result.weights_loaded is False
    for b in result.boxes:
        assert b.weights_loaded is False
        assert b.heuristic is True


# --------------------------------------------------------------------------- #
# 4. All box coordinates normalized to [0, 1]
# --------------------------------------------------------------------------- #


def test_boxes_normalized() -> None:
    rng = np.random.default_rng(seed=2)
    img = rng.random((200, 200)).astype(np.float32)
    mask = np.ones((200, 200), dtype=bool)
    result = MonaiDetector(max_boxes=10).detect(img, mask)
    for b in result.boxes:
        assert 0.0 <= b.x0 <= 1.0
        assert 0.0 <= b.y0 <= 1.0
        assert 0.0 <= b.x1 <= 1.0
        assert 0.0 <= b.y1 <= 1.0
        assert b.x0 < b.x1 and b.y0 < b.y1


# --------------------------------------------------------------------------- #
# 5. Deterministic — same input → same boxes
# --------------------------------------------------------------------------- #


def test_deterministic() -> None:
    rng = np.random.default_rng(seed=3)
    img = rng.random((128, 128)).astype(np.float32)
    mask = np.ones((128, 128), dtype=bool)
    r1 = MonaiDetector().detect(img, mask)
    r2 = MonaiDetector().detect(img, mask)
    assert r1.n_boxes == r2.n_boxes
    for b1, b2 in zip(r1.boxes, r2.boxes):
        assert b1.x0 == b2.x0
        assert b1.x1 == b2.x1
        assert b1.score == b2.score


# --------------------------------------------------------------------------- #
# 6. model_state honest: proxy_monai_heuristic
# --------------------------------------------------------------------------- #


def test_model_state_is_proxy_heuristic() -> None:
    rng = np.random.default_rng(seed=4)
    img = rng.random((128, 128)).astype(np.float32)
    mask = np.ones((128, 128), dtype=bool)
    result = MonaiDetector().detect(img, mask)
    assert result.model_state == "proxy_monai_heuristic"
    assert result.model_name.startswith("monai-detector-heuristic")


# --------------------------------------------------------------------------- #
# 7. Warning always contains "heuristic" and "not"
# --------------------------------------------------------------------------- #


def test_warning_labels_heuristic_and_not_trained() -> None:
    rng = np.random.default_rng(seed=5)
    img = rng.random((128, 128)).astype(np.float32)
    mask = np.ones((128, 128), dtype=bool)
    result = MonaiDetector().detect(img, mask)
    w = MONAI_DETECTOR_WARNING.lower()
    assert "heuristic" in w
    assert "not a diagnosis" in w
    assert "no trained weights" in w
    assert MONAI_DETECTOR_WARNING in result.warnings
