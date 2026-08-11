"""Regression tests for full-document ClinicalBERT inference."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from oncology_arbiter.nlp.clinicalbert_sliding_window import (
    AGGREGATION,
    OVERLAP_TOKENS,
    WINDOW_TOKENS,
    predict_words_sliding_window,
)
from oncology_arbiter.nlp.clinicalbert_local_client import ClinicalBertLocalClient


torch = pytest.importorskip("torch")


class _Encoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids_by_window):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids_by_window = word_ids_by_window

    def word_ids(self, batch_index: int):
        return self._word_ids_by_window[batch_index]


class _Tokenizer:
    """Two windows; source words 4 and 5 are deliberately overlapped."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, words, **kwargs):
        self.kwargs = kwargs
        w0 = [None, 0, 1, 2, 3, 4, 5, None] + [None] * (WINDOW_TOKENS - 8)
        w1 = [None, 4, 5, 6, 7, None] + [None] * (WINDOW_TOKENS - 6)
        return _Encoding(
            torch.zeros((2, WINDOW_TOKENS), dtype=torch.long),
            torch.ones((2, WINDOW_TOKENS), dtype=torch.long),
            [w0, w1],
        )


class _Model:
    def __call__(self, **kwargs):
        assert tuple(kwargs["input_ids"].shape) == (2, WINDOW_TOKENS)
        logits = torch.zeros((2, WINDOW_TOKENS, 3), dtype=torch.float32)
        logits[..., 0] = 5.0  # O by default
        # Tail entity appears only because the second window is inferred.
        logits[1, 3] = torch.tensor([0.0, 9.0, 0.0])  # word 6 = B-X
        logits[1, 4] = torch.tensor([0.0, 0.0, 9.0])  # word 7 = I-X
        # Overlap word 4 gets contradictory edge/interior evidence. Mean
        # logits must select B-X: mean([6,0,0], [0,10,0]) = [3,5,0].
        logits[0, 5] = torch.tensor([6.0, 0.0, 0.0])
        logits[1, 1] = torch.tensor([0.0, 10.0, 0.0])
        return SimpleNamespace(logits=logits)


def test_exact_192_32_windows_cover_tail_and_average_overlap() -> None:
    tok = _Tokenizer()
    out = predict_words_sliding_window(
        tokenizer=tok,
        model=_Model(),
        words=[f"w{i}" for i in range(8)],
        id2label={0: "O", 1: "B-X", 2: "I-X"},
        device=torch.device("cpu"),
    )

    assert tok.kwargs["max_length"] == 192
    assert tok.kwargs["stride"] == 32
    assert tok.kwargs["return_overflowing_tokens"] is True
    assert tok.kwargs["padding"] == "max_length"
    assert out.n_windows == 2
    assert out.window_tokens == 192
    assert out.overlap_tokens == 32
    assert out.aggregation == AGGREGATION
    assert out.labels[4] == "B-X"  # averaged overlap, not first-window winner
    assert out.labels[6:] == ["B-X", "I-X"]  # beyond first window
    assert all(0.0 <= x <= 1.0 for x in out.confidences)


def test_horizon_and_overlap_are_release_locked() -> None:
    common = dict(
        tokenizer=_Tokenizer(), model=_Model(), words=["x"],
        id2label={0: "O"}, device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="trained horizon 192"):
        predict_words_sliding_window(**common, window_tokens=256)
    with pytest.raises(ValueError, match="locked to 32"):
        predict_words_sliding_window(**common, overlap_tokens=16)
    with pytest.raises(Exception, match="trained horizon 192"):
        ClinicalBertLocalClient(max_length=512)


def test_local_client_surfaces_window_provenance(monkeypatch) -> None:
    client = ClinicalBertLocalClient(weight_dir="unused")
    tok = _Tokenizer()
    meta = {
        "provenance": "REAL-test",
        "base_model": "unit-test",
        "training_seed": 123,
        "test_micro_f1": None,
    }
    monkeypatch.setattr(
        client,
        "_get",
        lambda: (tok, _Model(), {0: "O", 1: "B-X", 2: "I-X"}, meta),
    )
    result = client.parse("a b c d e f g h")
    assert result["n_windows"] == 2
    assert result["window_tokens"] == WINDOW_TOKENS
    assert result["overlap_tokens"] == OVERLAP_TOKENS
    assert result["window_aggregation"] == AGGREGATION
    assert any(s["start_tok"] == 6 and s["end_tok"] == 8 for s in result["spans"])
