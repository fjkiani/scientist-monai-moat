"""Stdlib product client for the identity-bound ClinicalBERT v2 Modal app."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

EXPECTED_ARTIFACT_SHA256 = "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50"
EXPECTED_BASE_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("CLINICALBERT_MODAL_TIMEOUT", "120"))


@dataclass(frozen=True)
class ClinicalBertModalEndpointConfig:
    base: str

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
            raise RuntimeError("CLINICALBERT_MODAL_URL is not set")
        return cls(base=base.rstrip("/"))


class ClinicalBertModalError(RuntimeError):
    """Raised for transport, schema, or model-identity failures."""


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib_error.HTTPError as exc:
        raise ClinicalBertModalError(f"HTTP {exc.code} calling {url}: {exc.read()[:512]!r}") from exc
    except urllib_error.URLError as exc:
        raise ClinicalBertModalError(f"Network error calling {url}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClinicalBertModalError(f"Modal response was not JSON: {raw[:512]!r}") from exc
    if not isinstance(value, dict):
        raise ClinicalBertModalError("Modal response must be a JSON object")
    return value


def _get_json(url: str, *, timeout: int) -> dict[str, Any]:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read())
    except (urllib_error.URLError, json.JSONDecodeError) as exc:
        raise ClinicalBertModalError(f"GET failed for {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClinicalBertModalError("Modal response must be a JSON object")
    return value


def _assert_identity(payload: dict[str, Any]) -> None:
    if payload.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        raise ClinicalBertModalError("ClinicalBERT artifact identity mismatch")
    if payload.get("base_revision") != EXPECTED_BASE_REVISION:
        raise ClinicalBertModalError("ClinicalBERT base revision mismatch")
    if payload.get("n_training_synthetic") is not False:
        raise ClinicalBertModalError("ClinicalBERT deployment does not attest real training")


class ClinicalBertModalClient:
    def __init__(
        self,
        *,
        endpoints: Optional[ClinicalBertModalEndpointConfig] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoints = endpoints or ClinicalBertModalEndpointConfig.from_env()
        self.timeout = int(timeout)

    def healthz(self) -> Dict[str, Any]:
        return _get_json(self.endpoints.healthz, timeout=self.timeout)

    def info(self) -> Dict[str, Any]:
        response = _get_json(self.endpoints.info, timeout=self.timeout)
        if "error" in response:
            raise ClinicalBertModalError(f"info: {response['error']}")
        _assert_identity(response)
        return response

    def parse(self, report_text: str) -> Dict[str, Any]:
        if not isinstance(report_text, str) or not report_text.strip():
            raise ClinicalBertModalError("report_text must be a non-empty string")
        response = _post_json(
            self.endpoints.parse,
            {"report_text": report_text},
            timeout=self.timeout,
        )
        if "error" in response:
            raise ClinicalBertModalError(f"parse: {response['error']}")
        _assert_identity(response)
        expected_report_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        if response.get("report_sha256") != expected_report_sha:
            raise ClinicalBertModalError("ClinicalBERT response report hash mismatch")
        if not isinstance(response.get("window_count"), int) or response["window_count"] <= 0:
            raise ClinicalBertModalError("ClinicalBERT response lacks sliding-window evidence")
        if not isinstance(response.get("spans"), list) or not isinstance(response.get("parsed"), dict):
            raise ClinicalBertModalError("ClinicalBERT response schema is incomplete")
        return response


def parse_report(report_text: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    return ClinicalBertModalClient(timeout=timeout).parse(report_text)
