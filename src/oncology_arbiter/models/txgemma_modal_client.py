"""TxGemma Modal client — stdlib-only urllib wrapper.

Mirrors :mod:`oncology_arbiter.models.medsiglip_modal_client` in shape:
the Modal deployment (see ``deploy/modal/txgemma_app.py``) hosts
``google/txgemma-9b-chat`` behind three FastAPI endpoints:

    GET  <base>-healthz.modal.run
    GET  <base>-info.modal.run
    POST <base>-reason.modal.run

The client uses only stdlib (``urllib`` + ``json``). No transformers / no
torch imports on the caller side, so this ships fine on Render CPU boxes
even though the actual TxGemma weights are gated + heavy.

Environment
-----------
``TXGEMMA_MODAL_URL`` — base URL like ``https://<workspace>--txgemma``.
                       We derive endpoint URLs by appending the label suffix.
``TXGEMMA_BACKEND``   — ``modal`` or ``local``. The public factory
                       :func:`get_txgemma_client` reads this to decide.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from oncology_arbiter import RUO_DISCLAIMER
from oncology_arbiter.models.hai_def import AccessLevel, GatedAccessError


TXGEMMA_HONESTY_WARNING = (
    "TxGemma is a Google research LLM (HAI-DEF gated). Its outputs are "
    "recommendations from a generative language model, NOT verified "
    "clinical decisions. It MUST NOT be used to make treatment choices. "
    "Real clinical use requires a certified breast oncologist and a full "
    "guideline consultation. RESEARCH USE ONLY."
)


# --------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------- #

@dataclass
class TxGemmaModalUrls:
    """Modal deployment URL derivation, one string per endpoint."""

    base: str

    @property
    def healthz(self) -> str:
        return f"{self.base}-healthz.modal.run"

    @property
    def info(self) -> str:
        return f"{self.base}-info.modal.run"

    @property
    def reason(self) -> str:
        return f"{self.base}-reason.modal.run"


def _post_json(url: str, payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib_error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} calling {url}: {exc.read()[:512]!r}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:512]!r}") from exc


def _get_json(url: str, *, timeout: int) -> Dict[str, Any]:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:512]!r}") from exc


# --------------------------------------------------------------------- #
# Result dataclass (mirrors local TxGemmaTherapyResult minimally)
# --------------------------------------------------------------------- #

@dataclass
class TxGemmaModalTherapyResult:
    """Result returned by :meth:`TxGemmaModalClient.reason`."""

    text: str
    model_repo: str
    elapsed_ms: int
    app_version: str
    honesty_warning: str = TXGEMMA_HONESTY_WARNING
    ruo_disclaimer: str = RUO_DISCLAIMER


# --------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------- #

class TxGemmaModalClient:
    """Thin stdlib client to the TxGemma Modal endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        preflight_timeout: int = 10,
        reason_timeout: int = 180,
    ) -> None:
        base = base_url or os.environ.get("TXGEMMA_MODAL_URL")
        if not base:
            raise RuntimeError(
                "TXGEMMA_MODAL_URL not set. Deploy deploy/modal/txgemma_app.py "
                "and export TXGEMMA_MODAL_URL=<https://.../txgemma>."
            )
        self.urls = TxGemmaModalUrls(base=base)
        self.preflight_timeout = preflight_timeout
        self.reason_timeout = reason_timeout
        self.device = "modal:A10G"

    def healthz(self) -> Dict[str, Any]:
        return _get_json(self.urls.healthz, timeout=self.preflight_timeout)

    def info(self) -> Dict[str, Any]:
        return _get_json(self.urls.info, timeout=self.preflight_timeout)

    def reason(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> TxGemmaModalTherapyResult:
        """Call the TxGemma reason endpoint.

        Raises
        ------
        GatedAccessError
            When Modal returns HTTP 403 (the underlying HF gate refused).
        RuntimeError
            On any other HTTP or transport failure.
        """
        try:
            resp = _post_json(
                self.urls.reason,
                {
                    "prompt": prompt,
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                },
                timeout=self.reason_timeout,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "HTTP 403" in msg:
                raise GatedAccessError(
                    repo_id="google/txgemma-9b-chat",
                    access_level=AccessLevel.FORBIDDEN,
                    status_code=403,
                    reason=f"txgemma_modal_gated_forbidden: {msg[:256]}",
                ) from exc
            raise
        return TxGemmaModalTherapyResult(
            text=str(resp.get("text", "")),
            model_repo=str(resp.get("model_repo", "google/txgemma-9b-chat")),
            elapsed_ms=int(resp.get("elapsed_ms", 0)),
            app_version=str(resp.get("app_version", "")),
        )


def get_txgemma_client(force_backend: Optional[str] = None) -> Any:
    """Factory: reads ``TXGEMMA_BACKEND`` and returns the right client.

    Values
    ------
    ``modal`` → :class:`TxGemmaModalClient`
    ``local`` → :func:`oncology_arbiter.models.txgemma_client.load_txgemma`
                (may raise ``GatedAccessError`` immediately under the
                current HAI-DEF gate).
    """
    backend = (force_backend or os.environ.get("TXGEMMA_BACKEND", "local")).lower()
    if backend == "modal":
        return TxGemmaModalClient()
    from oncology_arbiter.models.txgemma_client import TxGemmaClient
    return TxGemmaClient()
