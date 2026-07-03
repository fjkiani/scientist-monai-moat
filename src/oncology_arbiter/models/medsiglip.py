"""MedSigLIP (google/medsiglip-448) zero-shot backbone, gated by HAI-DEF.

This module wires the real Google Health MedSigLIP-448 weights into the
oncology-arbiter, replacing the general-domain SigLIP proxy in
:mod:`oncology_arbiter.models.siglip_baseline`. It differs from the proxy
in three important ways:

1. **Gated access**. MedSigLIP is distributed under the Health AI Developer
   Foundations (HAI-DEF) Terms of Use and is gated on HuggingFace. We call
   :func:`oncology_arbiter.models.hai_def.check_hai_def_access` at
   ``_load()`` time and raise :class:`GatedAccessError` (from
   ``hai_def``) BEFORE any weight download if the gate is not ALLOWED.
   This is deliberate: silent proxy fallback is an endpoint-level policy,
   not a client-level one, and this module MUST NOT reference the proxy
   class by name (the test
   ``tests/unit/test_medsiglip_wiring.py::test_no_proxy_fallback_reference_in_module``
   is a static-analysis guard that would break if we did).

2. **Different preprocessing**. MedSigLIP takes 448×448 RGB at
   ``mean=[0.5,0.5,0.5]``, ``std=[0.5,0.5,0.5]``, ``rescale_factor=1/255``
   → pixel range (-1, 1). The proxy takes 224×224 at ImageNet stats. We
   defer the actual normalization to ``AutoProcessor.from_pretrained
   ("google/medsiglip-448")`` because HF's SiglipImageProcessor reads
   ``preprocessor_config.json`` and applies it correctly; we only supply a
   3-channel RGB PIL image at whatever resolution — the processor resizes.

3. **Distinct honesty warning**. MedSigLIP IS a medical model, but its
   training corpus (per Google's model card fetched 2026-07-02) covers
   chest X-ray (MIMIC-CXR), dermatology (SCIN, PAD-UFES-20), ophthalmology
   (EyePACS), histopathology (TCGA, CAMELYON), and CT/MRI slices — NOT
   mammography. The only Google-published breast-related AUROC is
   histopathology invasive breast cancer (n=5000, 3-class: zero-shot
   0.933 / linear-probe 0.930 / HAI-DEF LP 0.943). Using MedSigLIP
   zero-shot on mammograms is therefore off-label and must surface that
   caveat in every result.

Injection points (mirroring ``SiglipBaseline`` so screening-endpoint tests
can stub identically):

* ``preflight_fn`` overrides ``check_hai_def_access`` — tests supply
  stubs that return a synthetic ``GateReport`` and never hit the network.
* ``processor_cls`` / ``model_cls`` override transformers class objects.
* ``preprocess_fn`` overrides ``preprocess_mammogram`` for tests that skip
  real DICOM I/O.

Result carries the ``GateReport`` used at preflight so audit callers can
inspect why access was granted (200 direct vs. 302 CDN redirect etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
    _discover_hf_token,
    check_hai_def_access,
)
# NOTE: we import ONE helper from siglip_baseline (the PIL float-array
# converter is generic image I/O, not a model reference) and the shared
# label pair, so the honesty test can still assert that we do NOT import
# the ``SiglipBaseline`` *class* itself.
from oncology_arbiter.models.siglip_baseline import (
    DEFAULT_ZERO_SHOT_LABELS,
    _to_pil_from_float01,
)


# --------------------------------------------------------------------------- #
# Constants (verbatim from google/medsiglip-448 model card + preprocessor
# config, fetched 2026-07-02 with a valid HF token)


MEDSIGLIP_REPO: str = "google/medsiglip-448"
MEDSIGLIP_INPUT_RES: int = 448
MEDSIGLIP_ARCH: str = "SigLIP-400M vision + 400M text"
MEDSIGLIP_LICENSE: str = "health-ai-developer-foundations"
MEDSIGLIP_TEXT_CONTEXT_TOKENS: int = 64
# From preprocessor_config.json: image_mean/image_std both 0.5, rescale_factor
# 1/255 → normalized range (-1, 1) at 448×448.
MEDSIGLIP_IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
MEDSIGLIP_IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


# Honesty warning appended to every result. Must mention that MedSigLIP is
# medical but its training corpus does not include mammography — using it
# on mammograms is off-label and any downstream decision must treat the
# score as an out-of-distribution zero-shot probe.
MEDSIGLIP_MAMMOGRAPHY_WARNING: str = (
    "This score is from google/medsiglip-448 (HAI-DEF-gated Health AI "
    "Developer Foundations model). MedSigLIP is a genuine medical "
    "vision-language model, but its published training corpus covers "
    "chest X-ray (MIMIC-CXR), dermatology (SCIN, PAD-UFES-20), "
    "ophthalmology (EyePACS), histopathology (TCGA, CAMELYON), and "
    "CT/MRI slices — it does NOT include mammography. Running MedSigLIP "
    "zero-shot on a mammogram is therefore OFF-LABEL and this score is "
    "an out-of-distribution probe, not a validated mammographic score. "
    "The only Google-published breast-related AUROC for MedSigLIP is "
    "histopathology (Invasive Breast Cancer, n=5000, 3-class, zero-shot "
    "0.933 / linear-probe 0.930 / HAI-DEF LP 0.943)."
)


# --------------------------------------------------------------------------- #
# Result type


@dataclass
class MedSigLipResult:
    """Output of one MedSigLIP zero-shot forward pass on a preprocessed mammogram.

    Distinct from ``SiglipBaselineResult`` so mypy/pydantic and downstream
    provenance code cannot silently reuse a proxy result as a MedSigLIP
    result (that has bitten us before — see the 2026-07-02 silent-fallback
    incident logged in ``docs/model_cards/errata.md``).
    """

    source_path: str
    labels: list[str]
    probs: list[float]                # per-label sigmoid probabilities
    top_label: str
    top_prob: float
    model_repo: str = MEDSIGLIP_REPO
    model_state: ModelState = ModelState.LOADED_MEDSIGLIP
    input_resolution: int = MEDSIGLIP_INPUT_RES
    logits_shape: tuple[int, ...] = ()
    warnings: list[str] = field(default_factory=list)
    # HAI-DEF preflight record — the GateReport that justified this run.
    # Endpoint code copies this onto the response envelope for audit.
    gate_report: GateReport | None = None


# --------------------------------------------------------------------------- #
# Model runner


PreflightFn = Callable[[str], GateReport]


class MedSigLip:
    """Lazy-loading wrapper around google/medsiglip-448.

    Use ``MedSigLip()`` (defers weight download and HAI-DEF preflight)
    and call ``.run(dicom_path)`` on each real DICOM. The first
    ``.run(...)`` triggers preflight → weight download → inference. Preflight
    is cached across calls so we only pay the ~1s HEAD once per process
    lifetime unless callers explicitly ``.close()``.

    Any preflight outcome other than ``AccessLevel.ALLOWED`` raises
    ``GatedAccessError`` — this client NEVER falls back to a proxy. The
    endpoint (``/v1/screening/analyze``) decides whether to translate
    ``GatedAccessError`` into ``ModelState.GATED`` (honest failure) or into
    a policy-driven proxy call; that policy lives in ``api/app.py``, not
    here.
    """

    def __init__(
        self,
        *,
        repo_id: str = MEDSIGLIP_REPO,
        device: str = "cpu",
        preflight_fn: PreflightFn | None = None,
        processor_cls: Any = None,
        model_cls: Any = None,
        preprocess_fn: Any = None,
    ) -> None:
        self.repo_id = repo_id
        self.device = device
        self._preflight_fn: PreflightFn = preflight_fn or check_hai_def_access
        self._processor_cls = processor_cls
        self._model_cls = model_cls
        self._preprocess_fn = preprocess_fn
        self._processor: Any = None
        self._model: Any = None
        self._gate_report: GateReport | None = None

    # Lazy loaders ---------------------------------------------------------
    def _load(self) -> None:
        # Preflight caching: if we've already checked and passed, don't
        # re-hit the network. If we've never checked, do it now; if we've
        # checked and been denied, we've already raised — we never reach
        # here in that path.
        if self._gate_report is None:
            self._gate_report = self._preflight_fn(self.repo_id)
        if self._gate_report.access_level is not AccessLevel.ALLOWED:
            raise GatedAccessError(
                repo_id=self.repo_id,
                access_level=self._gate_report.access_level,
                status_code=self._gate_report.status_code,
                reason=self._gate_report.reason,
            )
        # Weights already loaded? Nothing to do.
        if self._processor is not None and self._model is not None:
            return
        if self._processor_cls is None or self._model_cls is None:
            from transformers import AutoProcessor, AutoModel  # type: ignore
            processor_cls = self._processor_cls or AutoProcessor
            model_cls = self._model_cls or AutoModel
        else:
            processor_cls = self._processor_cls
            model_cls = self._model_cls
        # Use the discovered HF token for the weight download so HAI-DEF
        # gating is respected end-to-end (config.json probe succeeded → the
        # same token must be attached when transformers pulls model.safetensors).
        token = _discover_hf_token()
        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        self._processor = processor_cls.from_pretrained(self.repo_id, **kwargs)
        model = model_cls.from_pretrained(self.repo_id, **kwargs)
        try:
            model.to(self.device)
        except Exception:
            # Some test stubs won't implement .to()
            pass
        try:
            model.eval()
        except Exception:
            pass
        self._model = model

    def close(self) -> None:
        """Drop model + processor from memory. Preserves ``gate_report``
        so audit callers can still inspect the last known access grant."""
        self._processor = None
        self._model = None

    @property
    def gate_report(self) -> GateReport | None:
        return self._gate_report

    # Public API -----------------------------------------------------------
    def run(
        self,
        dicom_path: str | Path,
        *,
        labels: Iterable[str] = DEFAULT_ZERO_SHOT_LABELS,
    ) -> MedSigLipResult:
        """Preprocess a real mammography DICOM and run zero-shot MedSigLIP.

        Raises
        ------
        GatedAccessError
            If HAI-DEF preflight fails at ``_load()`` time. Callers MUST
            NOT catch this and silently fall back to a proxy; endpoint
            policy for fallback is opt-in and explicit.
        ValueError
            If fewer than 2 zero-shot label prompts are supplied.
        RuntimeError
            If the model output shape does not match SigLIP's expected
            ``logits_per_image`` layout (would indicate a wrong repo_id).
        """
        import torch  # type: ignore

        p = Path(dicom_path)
        labels_list = [str(x) for x in labels]
        if len(labels_list) < 2:
            raise ValueError("zero-shot MedSigLIP requires at least 2 label prompts")

        # Fail-fast on HAI-DEF gate BEFORE touching DICOM I/O or downloading
        # weights. This is the whole point of preflight — do it first, and
        # any GatedAccessError raised here surfaces cleanly to the endpoint
        # (no half-loaded DICOM state to unwind).
        self._load()

        # Preprocess via the shared pipeline so we get consistent [0,1]
        # float32; SiglipImageProcessor then resizes to 448 and normalises.
        if self._preprocess_fn is None:
            from oncology_arbiter.mammography.pipeline import preprocess_mammogram
            preprocess_fn = preprocess_mammogram
        else:
            preprocess_fn = self._preprocess_fn
        pre = preprocess_fn(str(p))
        img = _to_pil_from_float01(np.asarray(pre.image))
        proc = self._processor
        model = self._model

        inputs = proc(
            text=labels_list,
            images=img,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            outputs = model(**inputs)

        logits = getattr(outputs, "logits_per_image", None)
        if logits is None:
            raise RuntimeError(
                "MedSigLIP model output has no `logits_per_image` attribute; "
                f"check that repo_id is a SigLIP model (got {self.repo_id!r})"
            )
        logits_shape = tuple(int(d) for d in logits.shape)
        # SigLIP-family: sigmoid per-label (not softmax across labels).
        probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy().tolist()
        if not isinstance(probs, list):
            probs = [float(probs)]
        if len(probs) != len(labels_list):
            raise RuntimeError(
                f"probs length {len(probs)} does not match labels count "
                f"{len(labels_list)} — model output layout unexpected"
            )
        top_idx = int(np.argmax(probs))
        return MedSigLipResult(
            source_path=str(p),
            labels=labels_list,
            probs=[float(x) for x in probs],
            top_label=labels_list[top_idx],
            top_prob=float(probs[top_idx]),
            model_repo=self.repo_id,
            model_state=ModelState.LOADED_MEDSIGLIP,
            input_resolution=MEDSIGLIP_INPUT_RES,
            logits_shape=logits_shape,
            warnings=[MEDSIGLIP_MAMMOGRAPHY_WARNING],
            gate_report=self._gate_report,
        )


__all__ = [
    "MEDSIGLIP_REPO",
    "MEDSIGLIP_INPUT_RES",
    "MEDSIGLIP_ARCH",
    "MEDSIGLIP_LICENSE",
    "MEDSIGLIP_TEXT_CONTEXT_TOKENS",
    "MEDSIGLIP_IMAGE_MEAN",
    "MEDSIGLIP_IMAGE_STD",
    "MEDSIGLIP_MAMMOGRAPHY_WARNING",
    "MedSigLipResult",
    "MedSigLip",
]
