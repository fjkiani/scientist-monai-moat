"""SigLIP zero-shot baseline over real CBIS-DDSM DICOMs.

This module wires `google/siglip-base-patch16-224` (Apache-2.0, ungated,
general-domain, trained on WebLI) as a **PROXY** vision-language backbone
so the arbiter can run end-to-end on real mammography *without* HAI-DEF
gating during development. It is deliberately named `baseline` and NOT
`medsiglip` — this proxy is NOT a medical model and its zero-shot outputs
MUST NOT be reported as MedSigLIP mammography scores.

Real invariants (verbatim from `docs/model_cards/siglip_base_patch16_224.md`
and `docs/model_cards/medsiglip_448.md`):

  * SigLIP proxy repo:  google/siglip-base-patch16-224 (ViT-B/16, 224x224)
  * SigLIP proxy license: Apache-2.0 (ungated)
  * SigLIP proxy training data: WebLI (general web image-text, no medical curation)
  * MedSigLIP repo (gated): google/medsiglip-448 (400M+400M, 448x448)
  * MedSigLIP training data: does NOT include mammography

Any code that constructs a `SiglipBaselineResult` MUST set
`model_state=ModelState.PROXY_SIGLIP` and populate `warnings` with the
mammography honesty warning. Consumers that misuse this by relabeling the
result as `ModelState.LOADED` violate the honesty contract enforced in
`test_siglip_baseline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from oncology_arbiter.api.schemas import ModelState


# --------------------------------------------------------------------------- #
# Constants (verbatim from model cards)

SIGLIP_PROXY_REPO: str = "google/siglip-base-patch16-224"
SIGLIP_PROXY_INPUT_RES: int = 224
SIGLIP_PROXY_ARCH: str = "ViT-B/16"
SIGLIP_PROXY_LICENSE: str = "Apache-2.0"

# Honesty warning appended to every result. Must reference both the proxy
# nature AND the fact that MedSigLIP itself has no mammography training data.
PROXY_MAMMOGRAPHY_WARNING: str = (
    "This score is from google/siglip-base-patch16-224, an Apache-2.0 "
    "general-domain SigLIP baseline (not medically fine-tuned). It is a "
    "PROXY for MedSigLIP during development. Do NOT report proxy zero-shot "
    "AUCs on mammography as evidence of MedSigLIP's mammography performance. "
    "MedSigLIP's training data does not include mammography; the only "
    "Google-published breast-related AUROC for MedSigLIP is histopathology "
    "(Invasive Breast Cancer, n=5000, 3 classes, zero-shot 0.933 / "
    "linear-probe 0.930 / HAI-DEF LP 0.943)."
)

# Standard candidate label pair for a zero-shot breast screening probe.
# These are the neutral English prompts used across SigLIP-family papers
# (Chen et al., 2023, arXiv:2303.15343) and are NOT clinical decision text.
DEFAULT_ZERO_SHOT_LABELS: tuple[str, str] = (
    "a mammogram of a breast with a malignant mass",
    "a mammogram of a breast without a malignant mass",
)


# --------------------------------------------------------------------------- #
# Result type


@dataclass
class SiglipBaselineResult:
    """Output of one zero-shot forward pass on a preprocessed mammogram."""

    source_path: str
    labels: list[str]
    probs: list[float]                # sigmoid probabilities per label (SigLIP style)
    top_label: str
    top_prob: float
    model_repo: str = SIGLIP_PROXY_REPO
    model_state: ModelState = ModelState.PROXY_SIGLIP
    input_resolution: int = SIGLIP_PROXY_INPUT_RES
    logits_shape: tuple[int, ...] = ()
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Preprocessing bridge


def _to_pil_from_float01(arr: np.ndarray) -> Any:
    """Convert a preprocessed mammogram (float32 in [0,1], HxW) to a PIL image
    that SiglipProcessor can consume.

    * Grayscale [0,1] → uint8 → 3-channel RGB (SigLIP expects 3ch).
    * No resizing here — SiglipProcessor handles resize+center-crop to 224.
    """
    from PIL import Image
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale mammogram, got shape {arr.shape}")
    a = np.clip(arr, 0.0, 1.0)
    a8 = (a * 255.0 + 0.5).astype(np.uint8)
    rgb = np.stack([a8, a8, a8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


# --------------------------------------------------------------------------- #
# Model runner


class SiglipBaseline:
    """Lazy-loading wrapper around google/siglip-base-patch16-224.

    Use `SiglipBaseline()` (defers weight download) and call `.run(...)` on
    each real DICOM. The first `.run` triggers the model download; subsequent
    calls reuse the cached model. Use `.close()` to drop the model out of
    memory if the caller cares about VRAM/RAM footprint.

    Injection points for tests:
        * `processor_cls` / `model_cls` override transformers class objects
        * `_preprocess_fn` overrides `preprocess_mammogram` for tests that
          want to skip real DICOM I/O
    """

    def __init__(
        self,
        *,
        repo_id: str = SIGLIP_PROXY_REPO,
        device: str = "cpu",
        processor_cls: Any = None,
        model_cls: Any = None,
        preprocess_fn: Any = None,
    ) -> None:
        self.repo_id = repo_id
        self.device = device
        self._processor_cls = processor_cls
        self._model_cls = model_cls
        self._preprocess_fn = preprocess_fn
        self._processor: Any = None
        self._model: Any = None

    # Lazy loaders ---------------------------------------------------------
    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if self._processor_cls is None or self._model_cls is None:
            from transformers import AutoProcessor, AutoModel  # type: ignore
            processor_cls = self._processor_cls or AutoProcessor
            model_cls = self._model_cls or AutoModel
        else:
            processor_cls = self._processor_cls
            model_cls = self._model_cls
        self._processor = processor_cls.from_pretrained(self.repo_id)
        model = model_cls.from_pretrained(self.repo_id)
        try:
            model.to(self.device)
        except Exception:
            # Some test stubs won't have .to()
            pass
        model.eval()
        self._model = model

    def close(self) -> None:
        self._processor = None
        self._model = None

    # Public API -----------------------------------------------------------
    def run(
        self,
        dicom_path: str | Path,
        *,
        labels: Iterable[str] = DEFAULT_ZERO_SHOT_LABELS,
    ) -> SiglipBaselineResult:
        """Preprocess a real mammography DICOM and run zero-shot SigLIP."""
        import torch  # type: ignore

        p = Path(dicom_path)
        labels_list = [str(x) for x in labels]
        if len(labels_list) < 2:
            raise ValueError("zero-shot SigLIP requires at least 2 label prompts")

        # Preprocess via the shared pipeline so we get consistent [0,1] float32.
        if self._preprocess_fn is None:
            from oncology_arbiter.mammography.pipeline import preprocess_mammogram
            preprocess_fn = preprocess_mammogram
        else:
            preprocess_fn = self._preprocess_fn
        pre = preprocess_fn(str(p))
        img = _to_pil_from_float01(np.asarray(pre.image))

        # Model load (idempotent)
        self._load()
        proc = self._processor
        model = self._model

        # SiglipProcessor handles image resize/normalize + tokenizer with
        # padding='max_length' truncation=True for SigLIP-style inputs.
        inputs = proc(
            text=labels_list,
            images=img,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Move tensors to device (best-effort; stubs may return dicts).
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            outputs = model(**inputs)

        logits = getattr(outputs, "logits_per_image", None)
        if logits is None:
            raise RuntimeError(
                "SigLIP model output has no `logits_per_image` attribute; "
                "check that repo_id is a SigLIP model (got "
                f"{self.repo_id!r})"
            )
        logits_shape = tuple(int(d) for d in logits.shape)
        # SigLIP uses sigmoid per-label, not softmax across labels.
        probs = torch.sigmoid(logits).squeeze(0).cpu().float().numpy().tolist()
        if not isinstance(probs, list):
            probs = [float(probs)]
        # Guard shape (1, K) or (K,)
        if len(probs) != len(labels_list):
            raise RuntimeError(
                f"probs length {len(probs)} does not match labels count "
                f"{len(labels_list)} — model output layout unexpected"
            )
        top_idx = int(np.argmax(probs))
        return SiglipBaselineResult(
            source_path=str(p),
            labels=labels_list,
            probs=[float(x) for x in probs],
            top_label=labels_list[top_idx],
            top_prob=float(probs[top_idx]),
            model_repo=self.repo_id,
            model_state=ModelState.PROXY_SIGLIP,
            input_resolution=SIGLIP_PROXY_INPUT_RES,
            logits_shape=logits_shape,
            warnings=[PROXY_MAMMOGRAPHY_WARNING],
        )


__all__ = [
    "SIGLIP_PROXY_REPO",
    "SIGLIP_PROXY_INPUT_RES",
    "SIGLIP_PROXY_ARCH",
    "SIGLIP_PROXY_LICENSE",
    "PROXY_MAMMOGRAPHY_WARNING",
    "DEFAULT_ZERO_SHOT_LABELS",
    "SiglipBaselineResult",
    "SiglipBaseline",
]
