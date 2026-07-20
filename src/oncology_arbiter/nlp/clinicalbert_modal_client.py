"""Modal-backed drop-in for the ClinicalBERT report parser.

The Modal deployment (see ``deploy/modal/clinicalbert_app.py``) hosts the
fine-tuned Bio_ClinicalBERT + token-classification head behind three HTTPS
endpoints:

* ``GET  /clinicalbert-healthz`` — liveness (no model touch)
* ``GET  /clinicalbert-info``    — warmed model metadata (provenance,
                                    training seed, test micro-F1)
* ``POST /clinicalbert-parse``   — JSON ``{"report_text": "..."}`` →
                                    parsed pathology fields + BIO spans

This client mirrors the public surface of the local
:class:`oncology_arbiter.nlp.clinicalbert_parser.ClinicalBertReportParser`
so the API layer can swap backends via ``CLINICALBERT_BACKEND=modal``.
Stdlib-only (``urllib`` + ``json``) — no new deps for the Render dyno.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: int = int(os.environ.get("CLINICALBERT_MODAL_TIMEOUT", "30"))


@dataclass(frozen=True)
class ClinicalBertModalEndpointConfig:
    """URL layout for the deployed Modal app.

    Modal auto-generates URLs of the form
    ``https://<workspace>--<label>.modal.run``. The label is the ``label=...``
    argument in each ``@modal.fastapi_endpoint`` decorator. Given a base
    prefix (``https://crispro-test--clinicalbert``) the three endpoints are
    derived by suffixing ``-healthz``, ``-info``, ``-parse``.
    """

    base: str  # e.g. "https://crispro-test--clinicalbert"

    @property
    def healthz(self) -> str:
        return f"{self.base}-healthz.modal.run"

    @property
    def info(self) -> str:
        return f"{self.base}-info.modal.run"

    @property
    def parse(self) -> str:
        return f"{self.base}-parse.modal.run"

    @classmethod
    def from_env(cls) -> "ClinicalBertModalEndpointConfig":
        base = os.environ.get("CLINICALBERT_MODAL_URL")
        if not base:
            raise RuntimeError(
                "CLINICALBERT_MODAL_URL is not set; expected the Modal base URL, "
                "e.g. https://crispro-test--clinicalbert"
            )
        return cls(base=base.rstrip("/"))


class ClinicalBertModalError(RuntimeError):
    """Raised when a Modal request fails (HTTP error, network, or Modal-returned {'error': ...})."""


def _post_json(url: str, payload: dict, *, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib_error.HTTPError as exc:
        raise ClinicalBertModalError(
            f"HTTP {exc.code} calling {url}: {exc.read()[:512]!r}"
        ) from exc
    except urllib_error.URLError as exc:
        raise ClinicalBertModalError(f"Network error calling {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClinicalBertModalError(
            f"Modal response was not JSON ({url}): {raw[:512]!r}"
        ) from exc


def _get_json(url: str, *, timeout: int) -> dict:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib_error.URLError as exc:
        raise ClinicalBertModalError(f"Network error calling {url}: {exc}") from exc


class ClinicalBertModalClient:
    """Drop-in for the local ClinicalBERT parser, backed by the Modal deployment.

    Public surface:

    * :meth:`healthz` - returns ``{"status": "ok", ...}`` (no model touch)
    * :meth:`info`    - returns training metadata dict
    * :meth:`parse`   - ``parse(report_text: str) -> dict`` returns the
                         parsed fields (+ BIO spans + provenance).

    All methods raise :class:`ClinicalBertModalError` on failure. The Modal
    app's own error responses (``{"error": "..."}``) are surfaced as the
    same exception so callers get one uniform failure path.
    """

    def __init__(
        self,
        *,
        endpoints: Optional[ClinicalBertModalEndpointConfig] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoints = endpoints or ClinicalBertModalEndpointConfig.from_env()
        self.timeout = int(timeout)

    # -- Health / info ------------------------------------------------
    def healthz(self) -> Dict[str, Any]:
        return _get_json(self.endpoints.healthz, timeout=self.timeout)

    def info(self) -> Dict[str, Any]:
        d = _get_json(self.endpoints.info, timeout=self.timeout)
        if isinstance(d, dict) and "error" in d:
            raise ClinicalBertModalError(f"info: {d['error']}")
        return d

    # -- Parse --------------------------------------------------------
    def parse(self, report_text: str) -> Dict[str, Any]:
        """Parse a pathology report and return the structured fields.

        Returns a dict shaped like::

            {
              "provenance": "SYNTHETIC-v0.3.1",
              "base_model": "emilyalsentzer/Bio_ClinicalBERT",
              "training_seed": 42,
              "test_micro_f1": 0.94,
              "parsed": { "KRAS": {"surface": "...", "value": "mutated"}, ... },
              "spans": [...],  # raw BIO spans for auditability
              "n_tokens": 128,
              "seconds": 0.15,
              "app_version": "clinicalbert-modal-v0.4.1-alpha",
              "disclaimer": "Research Use Only. ..."
            }

        Raises :class:`ClinicalBertModalError` if:
        - the Modal endpoint is unreachable,
        - the response is not JSON,
        - the Modal app returned an ``{"error": ...}`` body.
        """
        if not isinstance(report_text, str) or not report_text.strip():
            raise ClinicalBertModalError("report_text must be a non-empty string")
        payload = {"report_text": report_text}
        d = _post_json(self.endpoints.parse, payload, timeout=self.timeout)
        if isinstance(d, dict) and "error" in d:
            raise ClinicalBertModalError(f"parse: {d['error']}")
        return d


def parse_report(
    report_text: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Module-level convenience: parse a report via the env-configured Modal URL.

    Equivalent to::

        ClinicalBertModalClient(timeout=timeout).parse(report_text)
    """
    return ClinicalBertModalClient(timeout=timeout).parse(report_text)
