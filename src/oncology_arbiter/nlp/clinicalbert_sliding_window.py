"""Sliding-window token classification at the ClinicalBERT training horizon.

The v0.5.1 checkpoint was fine-tuned with ``max_len=192``.  Inference must
therefore not feed the token-classification head positions beyond 192, and it
must not truncate a document to its first window.  This module applies the
same head to overlapping 192-token windows with a 32-token overlap and merges
repeated word predictions by averaging logits before a document-level BIO
decode.

``window_tokens`` includes the tokenizer's special tokens.  Hugging Face's
``stride`` is the number of *content wordpieces* repeated between adjacent
windows.  Padding keeps every model input exactly 192 tokens long.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

WINDOW_TOKENS = 192
OVERLAP_TOKENS = 32
AGGREGATION = "mean_logits_per_word_across_overlapping_windows"


@dataclass(frozen=True)
class SlidingWindowPrediction:
    """Document-level word predictions reconstructed from all windows."""

    labels: list[str]
    confidences: list[float]
    n_windows: int
    window_tokens: int = WINDOW_TOKENS
    overlap_tokens: int = OVERLAP_TOKENS
    aggregation: str = AGGREGATION


def predict_words_sliding_window(
    *,
    tokenizer: Any,
    model: Any,
    words: Sequence[str],
    id2label: dict[int, str],
    device: Any | None = None,
    window_tokens: int = WINDOW_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> SlidingWindowPrediction:
    """Infer one label per source word over a complete document.

    The tokenizer must be a fast Hugging Face tokenizer because stable
    document-word reconstruction requires ``BatchEncoding.word_ids()``.
    Each source word contributes its first wordpiece logit in every window in
    which it appears.  Logits are averaged across those repeated observations,
    then softmax/argmax is applied once at document level.  Averaging logits
    (rather than independently decoded labels) avoids a window-edge vote from
    overriding a confident interior observation.
    """
    if window_tokens != WINDOW_TOKENS:
        raise ValueError(
            f"ClinicalBERT window_tokens is locked to the trained horizon "
            f"{WINDOW_TOKENS}; got {window_tokens}"
        )
    if overlap_tokens != OVERLAP_TOKENS:
        raise ValueError(
            f"ClinicalBERT overlap_tokens is locked to {OVERLAP_TOKENS}; "
            f"got {overlap_tokens}"
        )
    if overlap_tokens <= 0 or overlap_tokens >= window_tokens - 2:
        raise ValueError("overlap_tokens must be in [1, window_tokens-3]")
    if not words:
        return SlidingWindowPrediction(labels=[], confidences=[], n_windows=0)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ML extra is required here
        raise RuntimeError("sliding-window ClinicalBERT inference requires torch") from exc

    try:
        encoded = tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=WINDOW_TOKENS,
            stride=OVERLAP_TOKENS,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
    except Exception as exc:
        raise RuntimeError(f"sliding-window tokenization failed: {exc}") from exc

    if not hasattr(encoded, "word_ids"):
        raise RuntimeError(
            "ClinicalBERT sliding-window inference requires a fast tokenizer "
            "with BatchEncoding.word_ids()"
        )

    input_ids = encoded.get("input_ids")
    if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
        raise RuntimeError("tokenizer did not return batched 2-D input_ids")
    n_windows = int(input_ids.shape[0])
    if int(input_ids.shape[1]) != WINDOW_TOKENS:
        raise RuntimeError(
            f"tokenizer emitted width {int(input_ids.shape[1])}; expected exactly "
            f"{WINDOW_TOKENS} tokens per padded window"
        )

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            device = torch.device("cpu")

    model_inputs = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = encoded.get(key)
        if value is not None:
            model_inputs[key] = value.to(device)

    with torch.no_grad():
        logits = model(**model_inputs).logits.detach().cpu()

    if logits.ndim != 3 or int(logits.shape[0]) != n_windows:
        raise RuntimeError(
            f"model returned logits shape {tuple(logits.shape)} for {n_windows} windows"
        )
    n_labels = int(logits.shape[-1])
    if n_labels <= 0:
        raise RuntimeError("model returned zero labels")

    # CPU float64 makes the overlap merge deterministic across model devices.
    logit_sums = torch.zeros((len(words), n_labels), dtype=torch.float64)
    observations = torch.zeros(len(words), dtype=torch.int64)

    for window_idx in range(n_windows):
        word_ids = encoded.word_ids(batch_index=window_idx)
        if len(word_ids) != WINDOW_TOKENS:
            raise RuntimeError(
                f"window {window_idx} has {len(word_ids)} word ids; expected "
                f"{WINDOW_TOKENS}"
            )
        previous_word_id = None
        for token_idx, word_id in enumerate(word_ids):
            # Use one logit vector per source word per window.  At an overflow
            # boundary a continued word may be observed twice; the 32-token
            # overlap ensures it also has an interior observation, and averaging
            # prevents either edge observation from dominating.
            if word_id is None or word_id == previous_word_id:
                continue
            previous_word_id = word_id
            if not 0 <= int(word_id) < len(words):
                raise RuntimeError(
                    f"tokenizer returned out-of-range word id {word_id} for "
                    f"{len(words)} words"
                )
            logit_sums[int(word_id)] += logits[window_idx, token_idx].to(torch.float64)
            observations[int(word_id)] += 1

    missing = torch.nonzero(observations == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(
            f"sliding-window reconstruction missed {len(missing)} source words; "
            f"first missing indices: {missing[:10]}"
        )

    mean_logits = logit_sums / observations[:, None]
    probabilities = torch.softmax(mean_logits, dim=-1)
    confidences, predicted_ids = probabilities.max(dim=-1)
    labels = [id2label.get(int(i), "O") for i in predicted_ids.tolist()]

    return SlidingWindowPrediction(
        labels=labels,
        confidences=[float(x) for x in confidences.tolist()],
        n_windows=n_windows,
    )
