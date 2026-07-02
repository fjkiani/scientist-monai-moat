"""Tests for oncology_arbiter.models.siglip_baseline

Two families:

  1. **Structural / stub tests** (always run) — assert constants, wire-up,
     result-envelope honesty, and injection of processor/model stubs.
  2. **Real-data smoke tests** (skipped without CBIS fixtures OR network) —
     actually download `google/siglip-base-patch16-224` and run one forward
     pass on a real CBIS-DDSM DICOM. Marked `slow`.

Real-data honesty invariants that are gated as tests:
  * result.model_state MUST be ModelState.PROXY_SIGLIP, NOT LOADED
  * result.model_repo MUST equal "google/siglip-base-patch16-224"
  * result.warnings MUST contain the PROXY_MAMMOGRAPHY_WARNING string
  * The warning MUST reference (a) proxy nature, (b) MedSigLIP,
    (c) mammography, so no downstream code can silently strip it
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest

from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.data.cbis_ddsm import default_fixture_dir
from oncology_arbiter.models.siglip_baseline import (
    DEFAULT_ZERO_SHOT_LABELS,
    PROXY_MAMMOGRAPHY_WARNING,
    SIGLIP_PROXY_ARCH,
    SIGLIP_PROXY_INPUT_RES,
    SIGLIP_PROXY_LICENSE,
    SIGLIP_PROXY_REPO,
    SiglipBaseline,
    SiglipBaselineResult,
    _to_pil_from_float01,
)


# --------------------------------------------------------------------------- #
# Fixture wiring

FIXTURE_DIR = default_fixture_dir()
CANONICAL_STEM = "Calc-Test_P_00038_LEFT_CC"  # RIGHT-non-mirror CC view, well behaved
CANONICAL_DICOM = FIXTURE_DIR / f"{CANONICAL_STEM}.dcm"


def _fixtures_present() -> bool:
    return CANONICAL_DICOM.is_file()


needs_fixtures = pytest.mark.skipif(
    not _fixtures_present(),
    reason="CBIS-DDSM DICOM fixture not on disk; run download_cbis_ddsm_fixtures.py",
)


def _network_ok() -> bool:
    if os.environ.get("OA_SKIP_NETWORK_TESTS"):
        return False
    try:
        import huggingface_hub  # noqa: F401
        import transformers     # noqa: F401
        import torch            # noqa: F401
        import pydicom          # noqa: F401
    except ImportError:
        return False
    return True


needs_network_stack = pytest.mark.skipif(
    not _network_ok(),
    reason="OA_SKIP_NETWORK_TESTS is set OR transformers/torch/pydicom missing",
)


# --------------------------------------------------------------------------- #
# Constants (verbatim from google/siglip-base-patch16-224 model card)


def test_siglip_proxy_repo_verbatim() -> None:
    assert SIGLIP_PROXY_REPO == "google/siglip-base-patch16-224"


def test_siglip_proxy_input_resolution_is_224() -> None:
    assert SIGLIP_PROXY_INPUT_RES == 224


def test_siglip_proxy_architecture_is_vit_b_16() -> None:
    assert SIGLIP_PROXY_ARCH == "ViT-B/16"


def test_siglip_proxy_license_is_apache() -> None:
    assert SIGLIP_PROXY_LICENSE == "Apache-2.0"


def test_default_labels_are_two_neutral_prompts() -> None:
    assert len(DEFAULT_ZERO_SHOT_LABELS) == 2
    joined = " || ".join(DEFAULT_ZERO_SHOT_LABELS).lower()
    assert "mammogram" in joined
    # One prompt must describe malignancy, the other its absence, so top-1
    # is a real binary claim, not a coin flip mask.
    assert "malignant" in joined


# --------------------------------------------------------------------------- #
# Honesty warning content


def test_proxy_warning_references_proxy_and_medsiglip() -> None:
    w = PROXY_MAMMOGRAPHY_WARNING.lower()
    assert "proxy" in w
    assert "medsiglip" in w


def test_proxy_warning_references_mammography_absence() -> None:
    w = PROXY_MAMMOGRAPHY_WARNING.lower()
    assert "mammography" in w
    # Must state "does not include mammography" or equivalent negation
    assert (
        "does not include mammography" in w
        or "not include mammography" in w
    )


def test_proxy_warning_cites_pathology_anchor() -> None:
    """Anchor: 0.933 zero-shot Invasive Breast Cancer PATHOLOGY row."""
    assert "0.933" in PROXY_MAMMOGRAPHY_WARNING
    assert "histopathology" in PROXY_MAMMOGRAPHY_WARNING.lower()


def test_proxy_warning_names_apache_or_general_domain() -> None:
    w = PROXY_MAMMOGRAPHY_WARNING.lower()
    assert (
        "apache" in w
        or "general-domain" in w
        or "not medically fine-tuned" in w
    )


def test_proxy_warning_forbids_reporting_as_medsiglip() -> None:
    """The 'do not report' language must ban substitution claims."""
    w = PROXY_MAMMOGRAPHY_WARNING.lower()
    assert "do not report" in w
    assert "medsiglip" in w


# --------------------------------------------------------------------------- #
# _to_pil_from_float01


def test_to_pil_grayscale_to_rgb() -> None:
    arr = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    img = _to_pil_from_float01(arr)
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_to_pil_clips_out_of_range_values() -> None:
    arr = np.full((8, 8), 2.0, dtype=np.float32)  # above [0,1]
    img = _to_pil_from_float01(arr)
    # PIL RGB image; sample corner pixel should be (255,255,255) after clip.
    r, g, b = img.getpixel((0, 0))
    assert (r, g, b) == (255, 255, 255)


def test_to_pil_rejects_3d_input() -> None:
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        _to_pil_from_float01(arr)


# --------------------------------------------------------------------------- #
# SiglipBaseline structural / stub tests


class _StubProcessor:
    """Records call args and returns dict-of-tensors of controlled shapes."""

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, repo_id):
        inst = cls()
        inst.repo_id = repo_id
        return inst

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        import torch  # type: ignore
        _StubProcessor.calls.append({
            "text": list(text) if not isinstance(text, str) else [text],
            "num_images": 1 if images is not None else 0,
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
        })
        return {
            "input_ids": torch.zeros((len(text), 4), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        }


class _StubModel:
    """Deterministic 2-label output: label 0 is favoured."""

    device_called: list[str] = []
    eval_called: list[bool] = []

    @classmethod
    def from_pretrained(cls, repo_id):
        inst = cls()
        inst.repo_id = repo_id
        return inst

    def to(self, device):
        _StubModel.device_called.append(device)
        return self

    def eval(self):
        _StubModel.eval_called.append(True)
        return self

    def __call__(self, **inputs):
        import torch  # type: ignore
        n_labels = int(inputs["input_ids"].shape[0])
        # Rig logits so label 0 is dominant.
        logits = torch.tensor([[3.0] + [-3.0] * (n_labels - 1)], dtype=torch.float32)
        return types.SimpleNamespace(logits_per_image=logits)


def _stub_preprocess(_path):
    """Return a preprocessed 32x32 mammogram-ish tile without touching DICOM."""
    return types.SimpleNamespace(
        image=np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32),
        breast_mask=np.ones((32, 32), dtype=bool),
        pectoral_removed=False,
        metadata=None,
    )


def test_baseline_run_returns_proxy_siglip_state(tmp_path) -> None:
    _StubProcessor.calls.clear()
    _StubModel.device_called.clear()
    _StubModel.eval_called.clear()
    dcm = tmp_path / "does_not_matter.dcm"
    dcm.write_bytes(b"stub")
    baseline = SiglipBaseline(
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess,
    )
    result = baseline.run(dcm, labels=["a mammogram of a breast with a malignant mass",
                                       "a mammogram of a breast without a malignant mass"])
    assert isinstance(result, SiglipBaselineResult)
    assert result.model_state is ModelState.PROXY_SIGLIP
    assert result.model_repo == SIGLIP_PROXY_REPO
    assert result.input_resolution == 224
    assert result.logits_shape == (1, 2)
    assert result.top_label == "a mammogram of a breast with a malignant mass"
    assert 0.0 < result.top_prob <= 1.0
    # SigLIP uses sigmoid per label, so probs are independent and can sum to != 1.
    assert 0.0 <= result.probs[1] <= 1.0
    # Honesty warning MUST ride along in the result envelope.
    assert PROXY_MAMMOGRAPHY_WARNING in result.warnings
    # Stub was actually used (no accidental real download).
    assert _StubProcessor.calls
    assert _StubModel.eval_called


def test_baseline_rejects_single_label() -> None:
    baseline = SiglipBaseline(
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess,
    )
    with pytest.raises(ValueError):
        baseline.run("ignored.dcm", labels=["only one label"])


def test_baseline_calls_processor_with_sigLIP_padding(tmp_path) -> None:
    _StubProcessor.calls.clear()
    dcm = tmp_path / "x.dcm"
    dcm.write_bytes(b"stub")
    baseline = SiglipBaseline(
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess,
    )
    baseline.run(dcm)
    assert _StubProcessor.calls, "processor never invoked"
    call = _StubProcessor.calls[-1]
    # SigLIP requires padding='max_length' + truncation=True at the tokenizer.
    assert call["padding"] == "max_length"
    assert call["truncation"] is True
    assert call["return_tensors"] == "pt"


def test_baseline_close_drops_model() -> None:
    baseline = SiglipBaseline(
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess,
    )
    # Force load once so we know close() actually drops something.
    baseline._load()
    assert baseline._model is not None
    baseline.close()
    assert baseline._model is None
    assert baseline._processor is None


def test_baseline_load_is_idempotent() -> None:
    baseline = SiglipBaseline(
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess,
    )
    baseline._load()
    ref1 = baseline._model
    baseline._load()
    ref2 = baseline._model
    assert ref1 is ref2, "second _load() replaced an already-loaded model"


def test_baseline_defaults_to_ungated_proxy_repo() -> None:
    """DEFENSIVE: the class MUST default to the ungated proxy, never to
    google/medsiglip-448. Substituting HAI-DEF-gated repo here would
    silently attempt to pull weights that require a token."""
    baseline = SiglipBaseline()
    assert baseline.repo_id == "google/siglip-base-patch16-224"
    assert baseline.repo_id != "google/medsiglip-448"


# --------------------------------------------------------------------------- #
# Real-data smoke tests (slow; skip without fixtures OR network)


@pytest.mark.slow
@needs_fixtures
@needs_network_stack
def test_real_siglip_smoke_on_cbis_dicom():
    """End-to-end: real DICOM → preprocess → SigLIP proxy → real logits.

    This is the smoke test that proves the entire pipeline works on one
    real CBIS-DDSM screening mammogram. It downloads the SigLIP proxy
    weights (~800MB) from HuggingFace on first run; subsequent runs use
    the local HF cache.
    """
    baseline = SiglipBaseline()
    result = baseline.run(CANONICAL_DICOM)
    assert isinstance(result, SiglipBaselineResult)
    assert result.model_repo == SIGLIP_PROXY_REPO
    assert result.model_state is ModelState.PROXY_SIGLIP
    assert result.logits_shape == (1, 2)
    assert len(result.probs) == 2
    # SigLIP sigmoid probs are in (0, 1) and independent per label; they do
    # NOT have to sum to 1.
    for p in result.probs:
        assert 0.0 < p < 1.0
    # Honesty warning present verbatim
    assert PROXY_MAMMOGRAPHY_WARNING in result.warnings
    assert result.source_path.endswith(f"{CANONICAL_STEM}.dcm")


@pytest.mark.slow
@needs_fixtures
@needs_network_stack
def test_real_siglip_two_fixtures_produce_distinct_logits():
    """Different DICOMs MUST produce different logits — a smoke test that
    catches a silent 'always same output' bug (e.g., pixel_values were
    dropped on the input dict)."""
    a = SiglipBaseline().run(CANONICAL_DICOM).probs
    b = SiglipBaseline().run(FIXTURE_DIR / "Mass-Test_P_00016_LEFT_CC.dcm").probs
    assert a != b, "SigLIP produced identical probs for two distinct mammograms"
