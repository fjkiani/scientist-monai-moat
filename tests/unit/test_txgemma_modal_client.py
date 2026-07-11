"""Unit tests for the TxGemma Modal client wrapper.

No live-network calls; every ``urlopen`` is monkey-patched. The tests
exercise:

- URL derivation from the base URL.
- Successful ``reason()`` response parsing.
- HTTP 403 mapped to :class:`GatedAccessError` with the FORBIDDEN level.
- Other HTTP errors propagate as :class:`RuntimeError`.
- Factory reads ``TXGEMMA_BACKEND`` and returns the right client type.
- Honesty warning + RUO disclaimer are always attached to the result.
"""
from __future__ import annotations

import io
import json
import os
from unittest.mock import patch
from urllib import error as urllib_error

import pytest

from oncology_arbiter.models.hai_def import AccessLevel, GatedAccessError
from oncology_arbiter.models.txgemma_client import TxGemmaClient
from oncology_arbiter.models.txgemma_modal_client import (
    TxGemmaModalClient,
    TxGemmaModalTherapyResult,
    TxGemmaModalUrls,
    TXGEMMA_HONESTY_WARNING,
    get_txgemma_client,
)


class TestUrlDerivation:
    def test_healthz(self):
        u = TxGemmaModalUrls(base="https://fjkiani--txgemma")
        assert u.healthz == "https://fjkiani--txgemma-healthz.modal.run"

    def test_info(self):
        u = TxGemmaModalUrls(base="https://fjkiani--txgemma")
        assert u.info == "https://fjkiani--txgemma-info.modal.run"

    def test_reason(self):
        u = TxGemmaModalUrls(base="https://fjkiani--txgemma")
        assert u.reason == "https://fjkiani--txgemma-reason.modal.run"


class TestClientInit:
    def test_env_var_supplies_base(self):
        with patch.dict(os.environ, {"TXGEMMA_MODAL_URL": "https://foo--txgemma"}, clear=False):
            c = TxGemmaModalClient()
            assert c.urls.base == "https://foo--txgemma"
            assert c.device == "modal:A10G"

    def test_missing_env_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="TXGEMMA_MODAL_URL not set"):
                TxGemmaModalClient()


class _FakeHttpOk:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


class TestReasonEndpoint:
    def test_successful_response_parses(self):
        payload = {
            "text": "Recommend paclitaxel + trastuzumab + pertuzumab for HER2+ pT2N1.",
            "elapsed_ms": 1420,
            "model_repo": "google/txgemma-9b-chat",
            "app_version": "txgemma-modal-v0.4.0-alpha",
            "honesty_warning": TXGEMMA_HONESTY_WARNING,
        }
        c = TxGemmaModalClient(base_url="https://foo--txgemma")
        with patch(
            "oncology_arbiter.models.txgemma_modal_client.urllib_request.urlopen",
            return_value=_FakeHttpOk(json.dumps(payload).encode("utf-8")),
        ):
            r = c.reason("62F ER+/HER2- pT2N1")
        assert isinstance(r, TxGemmaModalTherapyResult)
        assert "trastuzumab" in r.text
        assert r.model_repo == "google/txgemma-9b-chat"
        assert r.elapsed_ms == 1420
        assert r.honesty_warning == TXGEMMA_HONESTY_WARNING
        # RUO disclaimer is defaulted (mirrors medsiglip pattern)
        assert "RESEARCH" in r.ruo_disclaimer.upper()

    def test_403_maps_to_gated_access_error(self):
        c = TxGemmaModalClient(base_url="https://foo--txgemma")
        err = urllib_error.HTTPError(
            "https://foo--txgemma-reason.modal.run",
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b"gated:txgemma"),
        )
        with patch(
            "oncology_arbiter.models.txgemma_modal_client.urllib_request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(GatedAccessError) as excinfo:
                c.reason("test")
        assert excinfo.value.access_level == AccessLevel.FORBIDDEN
        assert "modal" in str(excinfo.value).lower()

    def test_500_stays_runtime_error(self):
        c = TxGemmaModalClient(base_url="https://foo--txgemma")
        err = urllib_error.HTTPError(
            "https://foo--txgemma-reason.modal.run",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"boom"),
        )
        with patch(
            "oncology_arbiter.models.txgemma_modal_client.urllib_request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                c.reason("test")

    def test_network_error_stays_runtime_error(self):
        c = TxGemmaModalClient(base_url="https://foo--txgemma")
        with patch(
            "oncology_arbiter.models.txgemma_modal_client.urllib_request.urlopen",
            side_effect=urllib_error.URLError("timeout"),
        ):
            with pytest.raises(RuntimeError, match="Network error"):
                c.reason("test")

    def test_prompt_forwarded_verbatim(self):
        c = TxGemmaModalClient(base_url="https://foo--txgemma")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = req.data
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = dict(req.header_items())
            return _FakeHttpOk(json.dumps({
                "text": "x", "elapsed_ms": 0,
                "model_repo": "google/txgemma-9b-chat",
                "app_version": "txgemma-modal-v0.4.0-alpha",
            }).encode("utf-8"))

        with patch(
            "oncology_arbiter.models.txgemma_modal_client.urllib_request.urlopen",
            side_effect=fake_urlopen,
        ):
            c.reason("EXACT PROMPT XYZ", max_new_tokens=256, temperature=0.2)

        assert captured["method"] == "POST"
        assert captured["url"].endswith("-reason.modal.run")
        body = json.loads(captured["data"].decode("utf-8"))
        assert body["prompt"] == "EXACT PROMPT XYZ"
        assert body["max_new_tokens"] == 256
        assert body["temperature"] == 0.2
        assert captured["headers"]["Content-type"] == "application/json"


class TestFactory:
    def test_default_returns_local_client(self):
        with patch.dict(os.environ, {}, clear=True):
            c = get_txgemma_client(force_backend="local")
        assert isinstance(c, TxGemmaClient)

    def test_env_var_modal_returns_modal_client(self):
        with patch.dict(
            os.environ,
            {"TXGEMMA_BACKEND": "modal", "TXGEMMA_MODAL_URL": "https://foo--txgemma"},
            clear=False,
        ):
            c = get_txgemma_client()
        assert isinstance(c, TxGemmaModalClient)

    def test_force_backend_wins_over_env(self):
        with patch.dict(
            os.environ,
            {"TXGEMMA_BACKEND": "modal", "TXGEMMA_MODAL_URL": "https://foo--txgemma"},
            clear=False,
        ):
            c = get_txgemma_client(force_backend="local")
        assert isinstance(c, TxGemmaClient)
