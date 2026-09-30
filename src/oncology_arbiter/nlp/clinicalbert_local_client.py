"""Local product client for the identity-bound ClinicalBERT v2 runtime."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from oncology_arbiter.nlp.clinicalbert_runtime_v2 import (
    DISCLAIMER,
    EXPECTED_ARTIFACT_SHA256,
    ClinicalBertV2Runtime,
)

APP_VERSION = "clinicalbert-local-v2.0.0-real-tcga"
EXPECTED_LOCAL_ARTIFACT_SHA256 = "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50"
if EXPECTED_LOCAL_ARTIFACT_SHA256 != EXPECTED_ARTIFACT_SHA256:
    raise RuntimeError("ClinicalBERT local/runtime artifact contract drift")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BUNDLE_DIR = os.environ.get(
    "CLINICALBERT_LOCAL_WEIGHT_DIR",
    str(_REPO_ROOT / "artifacts" / "clinicalbert"),
)
_DEFAULT_BASE_DIR = os.environ.get("CLINICALBERT_BASE_MODEL_DIR")
_RUNTIME_CACHE: dict[tuple[str, str | None], ClinicalBertV2Runtime] = {}
_RUNTIME_LOCK = threading.Lock()


class ClinicalBertLocalError(RuntimeError):
    """Raised when identity verification or local inference fails."""


def _runtime(bundle_dir: str, base_model_dir: str | None) -> ClinicalBertV2Runtime:
    key = (bundle_dir, base_model_dir)
    with _RUNTIME_LOCK:
        if key not in _RUNTIME_CACHE:
            _RUNTIME_CACHE[key] = ClinicalBertV2Runtime(
                bundle_dir,
                base_model_dir=base_model_dir,
                device="cpu",
            )
        return _RUNTIME_CACHE[key]


class ClinicalBertLocalClient:
    """Drop-in local equivalent of :class:`ClinicalBertModalClient`."""

    def __init__(
        self,
        *,
        weight_dir: str | None = None,
        base_model_dir: str | None = None,
    ) -> None:
        self.weight_dir = weight_dir or _DEFAULT_BUNDLE_DIR
        self.base_model_dir = base_model_dir if base_model_dir is not None else _DEFAULT_BASE_DIR

    def _get(self) -> ClinicalBertV2Runtime:
        try:
            return _runtime(self.weight_dir, self.base_model_dir)
        except Exception as exc:
            raise ClinicalBertLocalError(f"ClinicalBERT v2 load failed: {type(exc).__name__}: {exc}") from exc

    def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "app": "clinicalbert-local",
            "app_version": APP_VERSION,
            "expected_artifact_sha256": EXPECTED_LOCAL_ARTIFACT_SHA256,
            "disclaimer": DISCLAIMER,
        }

    def info(self) -> dict[str, Any]:
        identity = self._get().identity()
        return {"app": "clinicalbert-local", "app_version": APP_VERSION, **identity, "disclaimer": DISCLAIMER}

    def parse(self, report_text: str) -> dict[str, Any]:
        try:
            result = self._get().parse(report_text)
        except Exception as exc:
            raise ClinicalBertLocalError(f"ClinicalBERT v2 parse failed: {type(exc).__name__}: {exc}") from exc
        if result.get("artifact_sha256") != EXPECTED_LOCAL_ARTIFACT_SHA256:
            raise ClinicalBertLocalError("ClinicalBERT v2 response artifact identity mismatch")
        return {"app_version": APP_VERSION, **result}
