"""Tests for oncology_arbiter.models.medsiglip

Covers:
  * Repo-id, input resolution and honesty warning are the values Google's
    2026-07-02 model card documents
  * HAI-DEF preflight is required before any weight download, and:
      - FORBIDDEN → GatedAccessError, no weight download
      - UNAUTHENTICATED → GatedAccessError, no weight download
      - ALLOWED → weight download attempted, MedSigLipResult returned
  * Preflight result cached across calls (only one HEAD per process)
  * Result carries the honest MEDSIGLIP_MAMMOGRAPHY_WARNING
  * Module MUST NOT import SiglipBaseline (endpoint-level fallback policy)

Injects stubs for ``processor_cls`` / ``model_cls`` / ``preprocess_fn`` /
``preflight_fn`` — no real network I/O or DICOM I/O required.
"""
from __future__ import annotations

import ast
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
)
from oncology_arbiter.models.medsiglip import (
    MEDSIGLIP_IMAGE_MEAN,
    MEDSIGLIP_IMAGE_STD,
    MEDSIGLIP_INPUT_RES,
    MEDSIGLIP_MAMMOGRAPHY_WARNING,
    MEDSIGLIP_REPO,
    MedSigLip,
    MedSigLipResult,
)


# --------------------------------------------------------------------------- #
# Card-anchored constants


def test_repo_id_is_medsiglip_448_exact() -> None:
    """Repo id must be the gated 448-res variant. There is no medsiglip-256
    variant on HuggingFace (confirmed 2026-07-02 — HfApi returned 404)."""
    assert MEDSIGLIP_REPO == "google/medsiglip-448"


def test_input_resolution_is_448() -> None:
    assert MEDSIGLIP_INPUT_RES == 448


def test_image_mean_and_std_are_half() -> None:
    """MedSigLIP's preprocessor_config.json (fetched 2026-07-02) reports
    image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5] → after rescale
    factor 1/255 pixel range is (-1, 1)."""
    assert MEDSIGLIP_IMAGE_MEAN == (0.5, 0.5, 0.5)
    assert MEDSIGLIP_IMAGE_STD == (0.5, 0.5, 0.5)


def test_warning_mentions_out_of_distribution_mammography() -> None:
    """The honesty warning MUST call out the mammography off-label caveat
    verbatim so downstream UIs cannot mis-render the score as a validated
    mammographic prediction."""
    w = MEDSIGLIP_MAMMOGRAPHY_WARNING.lower()
    assert "medsiglip" in w
    assert "mammograph" in w
    # The caveat must mention that training data does NOT include mammography.
    assert "does not include mammography" in w or "off-label" in w
    # Must reference the one Google-published breast AUROC for MedSigLIP
    # (histopathology invasive breast cancer, n=5000, 3-class).
    assert "histopathology" in w
    assert "0.933" in w or "0.930" in w or "0.943" in w


# --------------------------------------------------------------------------- #
# Stub factories


def _allowed_gate_report(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="preflight succeeded",
        has_token=True,
    )


def _forbidden_gate_report(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.FORBIDDEN,
        status_code=403,
        reason="HAI-DEF terms have not been accepted for this repo",
        has_token=True,
    )


def _unauthenticated_gate_report(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.UNAUTHENTICATED,
        status_code=401,
        reason="no valid HuggingFace token; set HF_TOKEN and re-run",
        has_token=False,
    )


class _StubProcessor:
    """Fake AutoProcessor — returns a dict of ``fake_tensor``s.

    The MedSigLip client tries to call ``.to()`` on those tensors; our
    stub silently returns self so the code path exercises but no torch
    ops actually execute.
    """
    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "_StubProcessor":
        obj = cls()
        obj.repo_id = repo_id
        obj.kwargs = kwargs
        return obj

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        # Return a small dict of "tensors" that support .to()
        class _T:
            def to(self, *_a, **_kw): return self
        self.last_labels = list(text)
        return {"pixel_values": _T(), "input_ids": _T()}


class _StubModel:
    """Fake AutoModel that produces deterministic logits matching a label pair."""
    def __init__(self, logits_row: list[float]) -> None:
        self._logits = logits_row

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "_StubModel":
        # Bit-approximate MedSigLIP live output on Calc-Test_P_00038_LEFT_CC.dcm
        # (probs[0]=7.977e-06 malignant, probs[1]=1.391e-05 without).
        # We store *logits* not probs; sigmoid inverse of 7.977e-06 ≈ -11.74,
        # sigmoid inverse of 1.391e-05 ≈ -11.19.
        return cls([-11.74, -11.19])

    def to(self, *_a, **_kw): return self
    def eval(self): return self

    def __call__(self, **_inputs):
        import torch  # type: ignore
        # Shape (1, len(labels)) — SigLIP's logits_per_image contract.
        logits = torch.tensor([self._logits], dtype=torch.float32)
        return types.SimpleNamespace(logits_per_image=logits)


def _stub_preprocess_fn(_dicom_path: str):
    """Fake preprocessor returning a preprocessed float32 [0,1] 2D array.
    Signature matches ``preprocess_mammogram(path)`` -> object with .image."""
    class _Pre:
        image = np.linspace(0.0, 1.0, num=256 * 256, dtype=np.float32).reshape(256, 256)
    return _Pre()


# --------------------------------------------------------------------------- #
# Preflight caching + gate enforcement


def test_preflight_is_cached_across_calls() -> None:
    """After the first .run(), a second .run() must NOT call preflight
    again — that's a HEAD to HF and we want at most one per process."""
    calls: list[str] = []

    def _preflight(repo_id: str) -> GateReport:
        calls.append(repo_id)
        return _allowed_gate_report(repo_id)

    ms = MedSigLip(
        preflight_fn=_preflight,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    r1 = ms.run("/fake.dcm")
    r2 = ms.run("/fake.dcm")
    assert isinstance(r1, MedSigLipResult)
    assert isinstance(r2, MedSigLipResult)
    assert calls == [MEDSIGLIP_REPO], (
        f"preflight should be called exactly once, got {calls}"
    )


def test_load_raises_gated_access_error_on_forbidden_report() -> None:
    """When preflight reports FORBIDDEN, .run() (and hence ._load()) must
    raise GatedAccessError BEFORE any weight download attempt."""
    ms = MedSigLip(
        preflight_fn=_forbidden_gate_report,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    with pytest.raises(GatedAccessError) as ei:
        ms.run("/fake.dcm")
    assert ei.value.access_level is AccessLevel.FORBIDDEN
    assert ei.value.status_code == 403


def test_load_raises_gated_access_error_on_unauthenticated_report() -> None:
    ms = MedSigLip(
        preflight_fn=_unauthenticated_gate_report,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    with pytest.raises(GatedAccessError) as ei:
        ms.run("/fake.dcm")
    assert ei.value.access_level is AccessLevel.UNAUTHENTICATED
    assert ei.value.status_code == 401


def test_run_raises_gated_access_error_on_forbidden_report() -> None:
    """Duplicate coverage: .run() is the public entrypoint, must be gate-
    enforced independently of ._load()."""
    ms = MedSigLip(preflight_fn=_forbidden_gate_report)
    with pytest.raises(GatedAccessError):
        ms.run("/fake.dcm")


def test_run_returns_medsiglip_result_with_honest_warning() -> None:
    """Happy path: ALLOWED preflight → real MedSigLipResult with the
    LOADED_MEDSIGLIP state + mammography warning."""
    ms = MedSigLip(
        preflight_fn=_allowed_gate_report,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    r = ms.run("/fake.dcm")
    assert isinstance(r, MedSigLipResult)
    assert r.model_state is ModelState.LOADED_MEDSIGLIP
    assert r.model_repo == MEDSIGLIP_REPO
    assert r.input_resolution == 448
    # Two labels → two probs
    assert len(r.probs) == 2
    # Sigmoid of stubbed logits [-11.74, -11.19] ≈ [7.97e-06, 1.39e-05]
    assert r.probs[0] < 1e-4
    assert r.probs[1] < 1e-4
    # Honesty warning present
    assert any("MedSigLIP" in w for w in r.warnings)
    assert any("mammograph" in w.lower() for w in r.warnings)
    # gate_report attached for audit
    assert r.gate_report is not None
    assert r.gate_report.access_level is AccessLevel.ALLOWED


def test_close_preserves_gate_report() -> None:
    """After .close(), model/processor are dropped but gate_report stays
    so audit can still tell 'ran once, cleared memory' from 'never ran'."""
    ms = MedSigLip(
        preflight_fn=_allowed_gate_report,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    ms.run("/fake.dcm")
    assert ms.gate_report is not None
    ms.close()
    assert ms._processor is None
    assert ms._model is None
    assert ms.gate_report is not None, "gate_report must survive close()"


def test_run_rejects_single_label() -> None:
    """Zero-shot SigLIP requires ≥2 label prompts (contrastive)."""
    ms = MedSigLip(
        preflight_fn=_allowed_gate_report,
        processor_cls=_StubProcessor,
        model_cls=_StubModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    with pytest.raises(ValueError):
        ms.run("/fake.dcm", labels=["only one label"])


def test_load_is_idempotent() -> None:
    """Multiple .run() calls must share one loaded processor + model
    instance (no re-download)."""
    load_calls: list[str] = []

    class _CountingProcessor(_StubProcessor):
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "_CountingProcessor":
            load_calls.append(("processor", repo_id))
            return super().from_pretrained(repo_id, **kwargs)  # type: ignore[return-value]

    class _CountingModel(_StubModel):
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: Any) -> "_CountingModel":
            load_calls.append(("model", repo_id))
            return cls([-11.74, -11.19])

    ms = MedSigLip(
        preflight_fn=_allowed_gate_report,
        processor_cls=_CountingProcessor,
        model_cls=_CountingModel,
        preprocess_fn=_stub_preprocess_fn,
    )
    ms.run("/fake.dcm")
    ms.run("/fake.dcm")
    proc_loads = [c for c in load_calls if c[0] == "processor"]
    model_loads = [c for c in load_calls if c[0] == "model"]
    assert len(proc_loads) == 1, f"processor should load once, loaded {len(proc_loads)}x"
    assert len(model_loads) == 1, f"model should load once, loaded {len(model_loads)}x"


# --------------------------------------------------------------------------- #
# Endpoint-level fallback policy: MedSigLIP module MUST NOT know about proxy


def test_no_proxy_fallback_reference_in_module() -> None:
    """The MedSigLIP module must NOT import the ``SiglipBaseline`` class.

    Rationale: the proxy fallback is an endpoint-level policy decision
    (deployment may prefer 'gated → GATED response' over 'gated → proxy').
    Baking the proxy fallback into the MedSigLIP client would violate
    that layering and make the wire-level ``ModelState`` ambiguous.

    We ALLOW importing ``_to_pil_from_float01`` and ``DEFAULT_ZERO_SHOT_LABELS``
    from ``siglip_baseline`` (generic image I/O and label constants), but
    NOT the ``SiglipBaseline`` model class itself. Enforced by:
      (a) attribute check — the module must not expose the SiglipBaseline name
      (b) AST walk — no ``from ... import SiglipBaseline`` and no bare
          ``SiglipBaseline`` reference in the AST

    Also fails if we accidentally use ``ModelState.PROXY_SIGLIP`` in the
    module (an easy typo when copy-pasting from the proxy file).
    """
    import oncology_arbiter.models.medsiglip as m

    # (a) Attribute check on the loaded module
    assert not hasattr(m, "SiglipBaseline"), (
        "medsiglip.py must not expose SiglipBaseline — endpoint-level policy only"
    )

    # (b) AST walk of the source file
    src = Path(m.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        # any Name node referencing SiglipBaseline is a violation
        if isinstance(node, ast.Name) and node.id == "SiglipBaseline":
            raise AssertionError(
                f"medsiglip.py references SiglipBaseline at line {node.lineno} — "
                "fallback is an endpoint-level policy, not a client-level one"
            )
        # `from x import SiglipBaseline` shows up as ImportFrom with a name.
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "SiglipBaseline", (
                    f"medsiglip.py imports SiglipBaseline (line {node.lineno}) — "
                    "forbidden"
                )
        # `ModelState.PROXY_SIGLIP` shows up as Attribute(value=Name('ModelState'), attr='PROXY_SIGLIP').
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "PROXY_SIGLIP"
            and isinstance(node.value, ast.Name)
            and node.value.id == "ModelState"
        ):
            raise AssertionError(
                f"medsiglip.py references ModelState.PROXY_SIGLIP at line "
                f"{node.lineno} — this module must emit LOADED_MEDSIGLIP only"
            )
