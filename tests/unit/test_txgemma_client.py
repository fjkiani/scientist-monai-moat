"""Unit tests for L4c TxGemmaClient — preflight-first, no silent fallback.

Under the current session HF token, TxGemma repos return HTTP 403. These
tests use stubbed preflight fns so they run offline and stay deterministic.
"""
from __future__ import annotations

import pytest

from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GatedAccessError,
    GateReport,
)
from oncology_arbiter.models.txgemma_client import (
    TXGEMMA_CHAT_REPO,
    TXGEMMA_HONESTY_WARNING,
    TxGemmaClient,
)


def _preflight_forbidden(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.FORBIDDEN,
        status_code=403,
        reason="terms not accepted",
        has_token=True,
    )


def _preflight_unauth(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.UNAUTHENTICATED,
        status_code=401,
        reason="no HF_TOKEN provided",
        has_token=False,
    )


def _preflight_allowed(repo_id: str) -> GateReport:
    return GateReport(
        repo_id=repo_id,
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="OK",
        has_token=True,
    )


# --------------------------------------------------------------------------- #
# 1. Default repo = txgemma-9b-chat
# --------------------------------------------------------------------------- #


def test_default_repo_is_txgemma_9b_chat() -> None:
    client = TxGemmaClient(preflight_fn=_preflight_allowed)
    assert client.repo_id == TXGEMMA_CHAT_REPO


# --------------------------------------------------------------------------- #
# 2. Forbidden preflight → GatedAccessError (no silent proxy)
# --------------------------------------------------------------------------- #


def test_forbidden_preflight_raises_gated_access_error() -> None:
    client = TxGemmaClient(preflight_fn=_preflight_forbidden)
    with pytest.raises(GatedAccessError) as exc:
        client.preflight()
    assert exc.value.access_level == AccessLevel.FORBIDDEN
    assert exc.value.status_code == 403
    assert "txgemma_gated:forbidden" in exc.value.reason.lower()


# --------------------------------------------------------------------------- #
# 3. Unauthenticated preflight → GatedAccessError
# --------------------------------------------------------------------------- #


def test_unauthenticated_preflight_raises_gated_access_error() -> None:
    client = TxGemmaClient(preflight_fn=_preflight_unauth)
    with pytest.raises(GatedAccessError) as exc:
        client.preflight()
    assert exc.value.access_level == AccessLevel.UNAUTHENTICATED
    assert exc.value.status_code == 401


# --------------------------------------------------------------------------- #
# 4. recommend_therapy() also raises under forbidden preflight
# --------------------------------------------------------------------------- #


def test_recommend_therapy_raises_when_gated() -> None:
    client = TxGemmaClient(preflight_fn=_preflight_forbidden)
    with pytest.raises(GatedAccessError):
        client.recommend_therapy(
            receptor_status={"ER": True, "PR": True, "HER2": False},
            grade=2,
            stage="T1N0M0",
        )


# --------------------------------------------------------------------------- #
# 5. Allowed preflight returns a GateReport
# --------------------------------------------------------------------------- #


def test_allowed_preflight_returns_report() -> None:
    client = TxGemmaClient(preflight_fn=_preflight_allowed)
    report = client.preflight()
    assert report.repo_id == TXGEMMA_CHAT_REPO
    assert report.access_level == AccessLevel.ALLOWED
    assert report.allowed is True


# --------------------------------------------------------------------------- #
# 6. Honesty warning constant is non-empty and mentions research use
# --------------------------------------------------------------------------- #


def test_honesty_warning_is_present_and_labels_ruo() -> None:
    assert TXGEMMA_HONESTY_WARNING
    assert "research use only" in TXGEMMA_HONESTY_WARNING.lower()
    assert "not" in TXGEMMA_HONESTY_WARNING.lower()  # not clinical advice
