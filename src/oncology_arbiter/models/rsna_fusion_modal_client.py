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
    """Explicit endpoint URLs for the deployed Modal functions.

    Modal deployments may be regional (e.g. `*.us-east.modal.run`), so we avoid
    hardcoding URL concatenation rules in the client. Prefer explicit URLs.
    """

    score: str
    info: str | None = None
    healthz: str | None = None

    @classmethod
    def from_env(cls) -> "RsnaFusionEndpointConfig":
        score = (os.environ.get("RSNA_FUSION_SCORE_URL") or "").strip()
        info = (os.environ.get("RSNA_FUSION_INFO_URL") or "").strip() or None
        healthz = (os.environ.get("RSNA_FUSION_HEALTHZ_URL") or "").strip() or None
        if not score:
            raise RuntimeError(
                "RSNA_FUSION_SCORE_URL not set. Deploy deploy/modal/rsna_fusion_logreg_app.py "
                "and export RSNA_FUSION_SCORE_URL=<https://<workspace>--rsna-fusion-score...modal.run>."
            )
        return cls(score=score, info=info, healthz=healthz)


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

