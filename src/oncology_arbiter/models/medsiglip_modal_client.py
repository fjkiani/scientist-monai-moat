"""Modal-backed drop-in for :class:`oncology_arbiter.models.medsiglip.MedSigLip`.

The Modal deployment (see ``deploy/modal/medsiglip_app.py``) hosts
``google/medsiglip-448`` on an A10G behind five HTTPS endpoints:

* ``GET  /healthz``      — liveness
* ``GET  /info``         — model card metadata + embedding_dim (1152) + load_seconds
* ``POST /embed``        — 1×1152 embedding for one DICOM **or** PNG/JPEG
* ``POST /embed_batch``  — N×1152 embeddings (server-side cap N ≤ 32)
* ``POST /zero_shot``    — SigLIP sigmoid probs for prompts + one image

This client mirrors the local :class:`MedSigLip` public surface so the API
layer can swap backends via ``MEDSIGLIP_BACKEND=modal`` without downstream
changes. It uses only stdlib (``urllib`` + ``base64`` + ``json``); model
loading, GPU work, and DICOM preprocessing all live on the Modal side.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .hai_def import AccessLevel, GateReport
from .medsiglip import (
    DEFAULT_ZERO_SHOT_LABELS,
    MEDSIGLIP_INPUT_RES,
    MEDSIGLIP_MAMMOGRAPHY_WARNING,
    MEDSIGLIP_REPO,
    MedSigLipResult,
)
from ..api.schemas import ModelState

logger = logging.getLogger(__name__)

# Server-side embed_batch caps at 32; keep a client-side margin for HTTP body size.
DEFAULT_BATCH_CHUNK: int = 16
DEFAULT_TIMEOUT_SECONDS: int = int(os.environ.get("MEDSIGLIP_MODAL_TIMEOUT", "300"))


@dataclass(frozen=True)
class ModalEndpointConfig:
    """URL layout for the deployed Modal app.

    Modal auto-generates URLs of the form
    ``https://<workspace>--<app>-<function>.modal.run``. We take a base
    prefix (``https://crispro-test--medsiglip``) and derive the five
    endpoints from it, so callers only need one env var.
    """

    base: str  # e.g. "https://crispro-test--medsiglip"

    @property
    def healthz(self) -> str:
        return f"{self.base}-healthz.modal.run"

    @property
    def info(self) -> str:
        return f"{self.base}-info.modal.run"

    @property
    def embed(self) -> str:
        return f"{self.base}-embed.modal.run"

    @property
    def embed_batch(self) -> str:
        return f"{self.base}-embed-batch.modal.run"

    @property
    def zero_shot(self) -> str:
        return f"{self.base}-zero-shot.modal.run"

    @classmethod
    def from_env(cls) -> "ModalEndpointConfig":
        base = os.environ.get("MODAL_MEDSIGLIP_URL")
        if not base:
            raise RuntimeError(
                "MODAL_MEDSIGLIP_URL is not set; expected the Modal base URL, "
                "e.g. https://crispro-test--medsiglip"
            )
        return cls(base=base.rstrip("/"))


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
    except urllib_error.HTTPError as exc:  # pragma: no cover - live-network
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {exc.read()[:512]!r}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(f"Modal response was not JSON ({url}): {raw[:512]!r}") from exc


def _get_json(url: str, *, timeout: int) -> dict:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib_error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _b64_of(path: os.PathLike[str] | str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _looks_like_dicom(path: os.PathLike[str] | str) -> bool:
    """DICOM detection: check ``DICM`` magic at offset 128 (Part 10 header).

    Falls back to extension check if the file cannot be read.
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(132)
    except OSError:
        return p.suffix.lower() in {".dcm", ".dicom"}
    return len(head) >= 132 and head[128:132] == b"DICM"


class MedSigLipModalClient:
    """Drop-in for :class:`MedSigLip`, backed by the Modal deployment.

    Public surface mirrors the local wrapper:

    * :meth:`preflight` returns a :class:`GateReport`
    * :meth:`embed_dicom` / :meth:`embed_dicoms` return raw 1152-d vectors
      (used by Track V training — no local counterpart)
    * :meth:`run` returns a :class:`MedSigLipResult` with the same fields
      the API layer already reads
    """

    def __init__(
        self,
        *,
        endpoints: ModalEndpointConfig | None = None,
        batch_chunk: int = DEFAULT_BATCH_CHUNK,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoints = endpoints or ModalEndpointConfig.from_env()
        self.batch_chunk = max(1, min(int(batch_chunk), 32))
        self.timeout = int(timeout)
        self._info_cache: dict | None = None
        self._gate_report: GateReport | None = None
        # Compat property so callers written against the local class don't crash.
        self.repo_id = MEDSIGLIP_REPO
        self.device = "modal:A10G"

    # ---------------------------------------------------------------- helpers
    def _info(self) -> dict:
        if self._info_cache is None:
            self._info_cache = _get_json(self.endpoints.info, timeout=self.timeout)
        return self._info_cache

    def _payload_for_path(self, path: os.PathLike[str] | str) -> dict:
        b64 = _b64_of(path)
        key = "dicom_b64" if _looks_like_dicom(path) else "pixels_b64"
        return {key: b64}

    def _payloads_for_paths(self, paths: Sequence[os.PathLike[str] | str]) -> dict:
        if not paths:
            return {"dicoms_b64": []}
        dicom_hits = [_looks_like_dicom(p) for p in paths]
        if all(dicom_hits):
            return {"dicoms_b64": [_b64_of(p) for p in paths]}
        if not any(dicom_hits):
            return {"pixels_b64": [_b64_of(p) for p in paths]}
        raise ValueError(
            "embed_dicoms received a mixed batch of DICOM and non-DICOM files; "
            "issue two separate calls."
        )

    # ---------------------------------------------------------------- preflight
    def preflight(self) -> GateReport:
        """Verify Modal reachability + model card match, return :class:`GateReport`.

        Modal-side loading already used the HF token baked into the
        deployment secret, so a 200 from ``/info`` means the deployment
        cleared HAI-DEF gating for us. We synthesize a GateReport with
        ``has_token=True`` because a token was necessarily present at
        deploy time.
        """
        if self._gate_report is not None:
            return self._gate_report
        try:
            info = self._info()
        except RuntimeError as exc:
            gr = GateReport(
                repo_id=MEDSIGLIP_REPO,
                access_level=AccessLevel.UNKNOWN,
                status_code=None,
                reason=f"modal-preflight-failed: {exc}",
                has_token=False,
            )
            self._gate_report = gr
            return gr

        model_repo = str(info.get("model_repo", ""))
        if model_repo != MEDSIGLIP_REPO:
            gr = GateReport(
                repo_id=MEDSIGLIP_REPO,
                access_level=AccessLevel.UNKNOWN,
                status_code=200,
                reason=f"unexpected model_repo={model_repo!r}",
                has_token=True,
            )
            self._gate_report = gr
            return gr

        dim = int(info.get("embedding_dim", 0) or 0)
        gr = GateReport(
            repo_id=MEDSIGLIP_REPO,
            access_level=AccessLevel.ALLOWED,
            status_code=200,
            reason=f"modal-remote model_repo={model_repo} dim={dim}",
            has_token=True,
        )
        self._gate_report = gr
        return gr

    @property
    def gate_report(self) -> GateReport | None:
        return self._gate_report

    def close(self) -> None:
        """No-op for compat with the local wrapper — Modal handles teardown."""
        self._info_cache = None

    # ---------------------------------------------------------------- embed
    def embed_dicom(self, path: os.PathLike[str] | str) -> list[float]:
        """Return a single 1152-d embedding for a DICOM or PNG/JPEG file."""
        payload = self._payload_for_path(path)
        resp = _post_json(self.endpoints.embed, payload, timeout=self.timeout)
        emb = resp.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"malformed embed response: {resp!r}")
        return [float(x) for x in emb]

    def embed_dicoms(
        self,
        paths: Sequence[os.PathLike[str] | str],
        *,
        chunk: int | None = None,
    ) -> list[list[float]]:
        """Batch embed. Chunks paths to :attr:`batch_chunk` (default 16)."""
        step = int(chunk) if chunk else self.batch_chunk
        out: list[list[float]] = []
        for i in range(0, len(paths), step):
            batch = list(paths[i : i + step])
            payload = self._payloads_for_paths(batch)
            resp = _post_json(self.endpoints.embed_batch, payload, timeout=self.timeout)
            embs = resp.get("embeddings")
            if not isinstance(embs, list) or len(embs) != len(batch):
                raise RuntimeError(
                    f"embed_batch returned {len(embs) if isinstance(embs, list) else '?'} "
                    f"embeddings for {len(batch)} inputs: {resp!r}"
                )
            out.extend([[float(x) for x in vec] for vec in embs])
        return out

    # ---------------------------------------------------------------- zero-shot
    def zero_shot_raw(
        self,
        path: os.PathLike[str] | str,
        *,
        labels: Iterable[str] = DEFAULT_ZERO_SHOT_LABELS,
    ) -> dict:
        labels_list = list(labels)
        payload = self._payload_for_path(path)
        payload["prompts"] = labels_list
        return _post_json(self.endpoints.zero_shot, payload, timeout=self.timeout)

    def run(
        self,
        dicom_path: os.PathLike[str] | str,
        *,
        labels: Iterable[str] = DEFAULT_ZERO_SHOT_LABELS,
    ) -> MedSigLipResult:
        """Zero-shot classify one DICOM/PNG on Modal, return a MedSigLipResult.

        Fields set:
        ``source_path`` = str(dicom_path)
        ``labels`` / ``probs`` = as returned by /zero_shot (sigmoid)
        ``top_label`` / ``top_prob`` = argmax over probs
        ``model_repo`` = "google/medsiglip-448"
        ``input_resolution`` = 448
        ``logits_shape`` = (1, len(labels))
        ``model_state`` = LOADED_MEDSIGLIP
        ``warnings`` = [MEDSIGLIP_MAMMOGRAPHY_WARNING]
        ``gate_report`` = synthesized ALLOWED report from /info
        """
        labels_list = [str(x) for x in labels]
        if len(labels_list) < 2:
            raise ValueError("zero-shot MedSigLIP requires at least 2 label prompts")

        gate = self.preflight()
        if gate.access_level is not AccessLevel.ALLOWED:
            # Match the local wrapper: raise so endpoint policy can decide.
            from .hai_def import GatedAccessError

            raise GatedAccessError(
                repo_id=MEDSIGLIP_REPO,
                access_level=gate.access_level,
                status_code=gate.status_code,
                reason=gate.reason,
            )

        resp = self.zero_shot_raw(dicom_path, labels=labels_list)
        probs = resp.get("probs")
        if not isinstance(probs, list) or len(probs) != len(labels_list):
            raise RuntimeError(f"malformed zero_shot response: {resp!r}")
        probs_f = [float(x) for x in probs]
        top_idx = max(range(len(probs_f)), key=lambda i: probs_f[i])
        return MedSigLipResult(
            source_path=str(dicom_path),
            labels=labels_list,
            probs=probs_f,
            top_label=labels_list[top_idx],
            top_prob=probs_f[top_idx],
            model_repo=MEDSIGLIP_REPO,
            model_state=ModelState.LOADED_MEDSIGLIP,
            input_resolution=MEDSIGLIP_INPUT_RES,
            logits_shape=(1, len(labels_list)),
            warnings=[MEDSIGLIP_MAMMOGRAPHY_WARNING],
            gate_report=gate,
        )


def get_medsiglip_client(*, force_backend: str | None = None):
    """Factory: reads ``MEDSIGLIP_BACKEND`` and returns the right client.

    * ``modal``    → :class:`MedSigLipModalClient`
    * anything else (default ``local``) → local :class:`MedSigLip`

    ``force_backend`` overrides the env var (used by tests).
    """
    backend = (force_backend or os.environ.get("MEDSIGLIP_BACKEND", "local")).lower()
    if backend == "modal":
        return MedSigLipModalClient()
    from .medsiglip import MedSigLip

    return MedSigLip()


__all__ = [
    "MedSigLipModalClient",
    "ModalEndpointConfig",
    "get_medsiglip_client",
]
