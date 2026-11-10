"""Weak supervision label model for pathology entity extraction.

Snorkel-style label model that combines multiple labeling functions (LFs) into
soft per-token labels. Each LF votes on a token's entity class (or ABSTAIN=-1),
and the label model learns per-LF accuracies from unlabeled agreement patterns.

Why not use the full snorkel-labeling package?
------------------------------------------------
The snorkel-labeling PyPI package depends on old numpy/scipy versions and
pulls TensorFlow-lite. We only need Snorkel's core insight (majority vote +
learned per-LF accuracies from covariance) which is small enough to
reimplement here without extra deps. Reference: Ratner et al., "Snorkel:
Rapid Training Data Creation with Weak Supervision", VLDB 2018,
sections 4.2-4.3.

Model
-----
For each token t and LF j, the LF emits a label y_{t,j} in {0..K, -1}
where -1 = ABSTAIN and 0..K are class labels.

We learn per-LF accuracies alpha_j by matrix decomposition of the LF
covariance matrix. Then soft labels for each token are computed as a
weighted majority vote using alpha_j as weights, with an EM-style
refinement step to sharpen the class posteriors.

Interface
---------
- LabelModel.fit(L_train: array (n_tokens, n_lfs), n_classes)
- LabelModel.predict_proba(L: array) -> array (n_tokens, n_classes)
- LabelModel.predict(L: array) -> array (n_tokens,)  # hard argmax
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


ABSTAIN = -1


@dataclass
class LabelModelStats:
    """Metadata about a fitted label model."""

    lf_accuracies: dict[str, float]
    lf_coverage: dict[str, float]  # fraction of tokens each LF votes on
    lf_conflict: dict[str, float]  # fraction where LFs disagree
    n_classes: int
    n_lfs: int
    n_tokens: int


class LabelModel:
    """Snorkel-style label model.

    We use a simplified version: given LF votes L (n_tokens, n_lfs), we
    compute the observable class-conditional probabilities and estimate
    per-LF accuracies via a maximum-likelihood iterative reweighting.

    This is roughly the "generative model" of Ratner et al. without the
    dependency structure learning (we assume LFs are conditionally
    independent given the true label — reasonable when LFs come from
    genuinely different signals like regex vs LLM vs ontology).
    """

    def __init__(self, n_classes: int, lf_names: list[str] | None = None):
        self.n_classes = n_classes
        self.lf_names = lf_names or []
        self.lf_accuracies_: np.ndarray | None = None  # shape (n_lfs,)
        self.class_prior_: np.ndarray | None = None  # shape (n_classes,)

    def fit(
        self,
        L: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.05,
        class_balance: np.ndarray | None = None,
    ) -> LabelModelStats:
        """Fit the label model.

        L: (n_tokens, n_lfs) integer array in {-1, 0..n_classes-1}.
        """
        n_tokens, n_lfs = L.shape
        assert L.min() >= -1 and L.max() < self.n_classes, "invalid LF votes"

        # Class prior — from majority vote initialization (excluding abstains)
        if class_balance is None:
            init_labels = self._majority_vote_init(L)
            counts = np.bincount(init_labels, minlength=self.n_classes)
            self.class_prior_ = (counts + 1) / (counts.sum() + self.n_classes)
        else:
            assert class_balance.shape == (self.n_classes,)
            self.class_prior_ = class_balance / class_balance.sum()

        # Initial per-LF accuracy = agreement with majority vote (assuming
        # majority is roughly right). Range [0.5, 0.99].
        init_labels = self._majority_vote_init(L)
        accuracies = np.zeros(n_lfs)
        for j in range(n_lfs):
            voted = L[:, j] != ABSTAIN
            n_v = voted.sum()
            if n_v == 0:
                accuracies[j] = 1.0 / self.n_classes  # random
                continue
            agree = (L[voted, j] == init_labels[voted]).sum()
            accuracies[j] = np.clip(agree / n_v, 0.51, 0.99)

        # EM-like iterative refinement: E-step compute soft labels using
        # current accuracies; M-step update per-LF accuracy from soft-label
        # agreement.
        for epoch in range(n_epochs):
            self.lf_accuracies_ = accuracies
            proba = self._compute_proba(L, accuracies)
            hard = proba.argmax(axis=1)
            new_acc = np.zeros(n_lfs)
            for j in range(n_lfs):
                voted = L[:, j] != ABSTAIN
                n_v = voted.sum()
                if n_v == 0:
                    new_acc[j] = 1.0 / self.n_classes
                    continue
                # Weighted agreement using proba mass on the LF's vote
                weighted_agree = proba[voted, L[voted, j]].sum()
                new_acc[j] = np.clip(weighted_agree / n_v, 0.51, 0.99)
            # Damped update
            delta = new_acc - accuracies
            if np.abs(delta).max() < 1e-4:
                break
            accuracies = accuracies + lr * delta

        self.lf_accuracies_ = accuracies

        # Stats
        lf_coverage = {}
        lf_conflict = {}
        for j in range(n_lfs):
            name = self.lf_names[j] if j < len(self.lf_names) else f"LF_{j}"
            voted = L[:, j] != ABSTAIN
            lf_coverage[name] = float(voted.sum() / n_tokens)
            # Conflict: fraction of voted tokens where LF disagrees with another voting LF
            n_conflict = 0
            for k in range(n_lfs):
                if j == k:
                    continue
                both = voted & (L[:, k] != ABSTAIN)
                n_conflict += (L[both, j] != L[both, k]).sum()
            lf_conflict[name] = float(n_conflict / max(1, voted.sum() * (n_lfs - 1)))
        lf_accuracies_out = {
            (self.lf_names[j] if j < len(self.lf_names) else f"LF_{j}"): float(accuracies[j])
            for j in range(n_lfs)
        }
        return LabelModelStats(
            lf_accuracies=lf_accuracies_out,
            lf_coverage=lf_coverage,
            lf_conflict=lf_conflict,
            n_classes=self.n_classes,
            n_lfs=n_lfs,
            n_tokens=n_tokens,
        )

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        assert self.lf_accuracies_ is not None, "call fit() first"
        return self._compute_proba(L, self.lf_accuracies_)

    def predict(self, L: np.ndarray) -> np.ndarray:
        return self.predict_proba(L).argmax(axis=1)

    def _majority_vote_init(self, L: np.ndarray) -> np.ndarray:
        """Simple majority vote, breaking ties in favor of class 0 (usually 'O')."""
        n_tokens = L.shape[0]
        out = np.zeros(n_tokens, dtype=np.int64)
        for i in range(n_tokens):
            votes = L[i, L[i] != ABSTAIN]
            if len(votes) == 0:
                out[i] = 0  # default to 'O'
            else:
                counts = np.bincount(votes, minlength=self.n_classes)
                out[i] = counts.argmax()
        return out

    def _compute_proba(self, L: np.ndarray, accuracies: np.ndarray) -> np.ndarray:
        """Compute per-token class posterior.

        For each token i and class c:
            P(y=c | L_i) prop_to prior[c] * prod_j P(L_{i,j} | y=c)
        where P(l | y=c) = alpha_j if l == c else (1 - alpha_j) / (K - 1) for l != -1,
        and P(l=-1 | y=c) = 1 (abstains carry no info).
        """
        n_tokens, n_lfs = L.shape
        K = self.n_classes
        log_prior = np.log(self.class_prior_ + 1e-12)  # (K,)
        # log_lik[i, c] = sum_j log P(L[i,j] | y=c)
        log_lik = np.zeros((n_tokens, K))
        for j in range(n_lfs):
            a = accuracies[j]
            log_correct = np.log(a + 1e-12)
            log_wrong = np.log((1 - a) / max(1, K - 1) + 1e-12)
            voted_mask = L[:, j] != ABSTAIN
            for c in range(K):
                # if LF votes c: log_correct; else log_wrong; abstain: 0
                match = (L[:, j] == c) & voted_mask
                mismatch = (L[:, j] != c) & voted_mask
                log_lik[match, c] += log_correct
                log_lik[mismatch, c] += log_wrong
        log_post = log_prior[None, :] + log_lik  # (n_tokens, K)
        # Normalize per token via softmax
        m = log_post.max(axis=1, keepdims=True)
        exp = np.exp(log_post - m)
        return exp / exp.sum(axis=1, keepdims=True)


def evaluate_on_gold(
    L_gold: np.ndarray,
    gold_labels: np.ndarray,
    label_model: LabelModel,
) -> dict[str, float]:
    """Evaluate a fitted label model on a gold-labeled sample.

    L_gold: (n, n_lfs) LF votes on gold tokens.
    gold_labels: (n,) true class labels.
    Returns micro-F1 and per-class F1.
    """
    y_pred = label_model.predict(L_gold)
    # Micro F1 (ignore class 0 = 'O' for entity-focused metric)
    K = label_model.n_classes
    tp = fp = fn = 0
    for c in range(1, K):
        tp += ((y_pred == c) & (gold_labels == c)).sum()
        fp += ((y_pred == c) & (gold_labels != c)).sum()
        fn += ((y_pred != c) & (gold_labels == c)).sum()
    micro_p = tp / max(1, tp + fp)
    micro_r = tp / max(1, tp + fn)
    micro_f1 = 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r)
    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "accuracy": float((y_pred == gold_labels).mean()),
    }


if __name__ == "__main__":
    # Self-check on synthetic 3-LF, 3-class problem
    np.random.seed(20260619)
    n_tokens = 1000
    n_classes = 3
    n_lfs = 3
    true_labels = np.random.randint(0, n_classes, n_tokens)
    L = np.full((n_tokens, n_lfs), ABSTAIN, dtype=np.int64)
    # LF 0: 80% accurate, 90% coverage
    for i in range(n_tokens):
        if np.random.rand() < 0.9:
            L[i, 0] = true_labels[i] if np.random.rand() < 0.8 else (true_labels[i] + 1) % n_classes
    # LF 1: 70% accurate, 60% coverage
    for i in range(n_tokens):
        if np.random.rand() < 0.6:
            L[i, 1] = true_labels[i] if np.random.rand() < 0.7 else (true_labels[i] + 2) % n_classes
    # LF 2: 90% accurate, 40% coverage
    for i in range(n_tokens):
        if np.random.rand() < 0.4:
            L[i, 2] = true_labels[i] if np.random.rand() < 0.9 else (true_labels[i] + 1) % n_classes

    lm = LabelModel(n_classes=n_classes, lf_names=["regex", "llm", "ontology"])
    stats = lm.fit(L)
    print("fitted stats:", stats)
    pred = lm.predict(L)
    accuracy = (pred == true_labels).mean()
    print(f"accuracy on synthetic gold: {accuracy:.3f}")
