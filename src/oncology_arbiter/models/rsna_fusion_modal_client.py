"""Modal-backed client for the RSNA MedSigLIP fusion head (LogReg).

This keeps the fusion head as a remote ML capability on Modal rather than a
local deterministic coefficient blob in the API server.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_TIMEOUT_SECONDS: int = int(os.environ.get("RSNA_FUSION_MODAL_TIMEOUT", "60"))


@dataclass(frozen=True)
class RsnaFusionEndpointConfig:
    base: str  # e.g. "https://crispro-test--rsna-fusion"

    @property
    def healthz(self) -> str:
        return f"{self.base}-healthz.modal.run"

    @property
    def info(self) -> str:
        return f"{self.base}-info.modal.run"

    @property
    def score(self) -> str:
        return f"{self.base}-score.modal.run"

    @classmethod
    def from_env(cls) -> "RsnaFusionEndpointConfig":
        base = (os.environ.get("RSNA_FUSION_MODAL_URL") or "").rstrip("/")
        if not base:
            raise RuntimeError(
                "RSNA_FUSION_MODAL_URL not set. Deploy deploy/modal/rsna_fusion_logreg_app.py "
                "and export RSNA_FUSION_MODAL_URL=<https://<workspace>--rsna-fusion>."
            )
        return cls(base=base)


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
    except urllib_error.HTTPError as exc:  # pragma: no cover
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {exc.read()[:512]!r}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(f"Modal response was not JSON ({url}): {raw[:512]!r}") from exc


class RsnaFusionModalClient:
    """Tiny client: embedding + clinical metadata → fusion_score."""

    def __init__(self, *, endpoints: RsnaFusionEndpointConfig | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.endpoints = endpoints or RsnaFusionEndpointConfig.from_env()
        self.timeout = int(timeout)

    def score(self, *, embedding: Sequence[float], clinical_metadata: dict[str, Any]) -> float:
        payload = {"embedding": [float(x) for x in embedding], "clinical_metadata": clinical_metadata}
        out = _post_json(self.endpoints.score, payload, timeout=self.timeout)
        if out.get("error"):
            raise RuntimeError(str(out["error"]))
        if "fusion_score" not in out:
            raise RuntimeError("fusion_score missing from response")
        return float(out["fusion_score"])


__all__ = [
    "RsnaFusionEndpointConfig",
    "RsnaFusionModalClient",
]

