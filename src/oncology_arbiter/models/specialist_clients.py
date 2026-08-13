"""Strict stdlib clients for deployed specialist services.

Each client validates the stage-specific wire contract and returns a result plus an
additive receipt. No client performs a proxy fallback.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


class SpecialistServiceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SpecialistCall:
    output: dict[str, Any]
    receipt: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    req = urllib_request.Request(
        url,
        data=body,
        headers=request_headers,
        method="POST" if body is not None else "GET",
    )
    started = time.perf_counter()
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except urllib_error.HTTPError as exc:
        detail = exc.read(512).decode("utf-8", errors="replace")
        raise SpecialistServiceError("http_error", f"HTTP {exc.code}: {detail}") from exc
    except (urllib_error.URLError, TimeoutError) as exc:
        raise SpecialistServiceError("transport_error", str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if status != 200:
        raise SpecialistServiceError("http_error", f"unexpected HTTP {status}")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecialistServiceError("non_json_response", raw[:256].decode("utf-8", errors="replace")) from exc
    if not isinstance(decoded, dict):
        raise SpecialistServiceError("contract_mismatch", "response must be a JSON object")
    if decoded.get("error"):
        raise SpecialistServiceError("upstream_error", str(decoded["error"]))
    return decoded, elapsed_ms


def _success_receipt(
    *,
    stage: str,
    required: bool,
    request_id: str,
    service_name: str,
    endpoint_label: str,
    app_version: str | None,
    model_name: str | None,
    model_version: str | None,
    artifact_sha256: str | None,
    input_reference: str | None,
    latency_ms: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "required": required,
        "status": "succeeded",
        "request_id": request_id,
        "service_name": service_name,
        "endpoint_label": endpoint_label,
        "app_version": app_version,
        "model_name": model_name,
        "model_version": model_version,
        "artifact_sha256": artifact_sha256,
        "input_reference": input_reference,
        "latency_ms": latency_ms,
        "warnings": list(warnings or []),
        "error": None,
    }


def _validate_case_filename(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise SpecialistServiceError("unsafe_manifest", "manifest contains an unsafe file name")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SpecialistServiceError("unsafe_manifest", "manifest contains an unsafe file name")
    if len(parts) == 1:
        return name
    if len(parts) == 2 and parts[0] == "dicom_series":
        return name
    raise SpecialistServiceError("unsafe_manifest", "manifest contains an unsafe file name")


class CaseStorageClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("CASE_STORAGE_MODAL_URL") or "").rstrip("/")
        if not self.base_url:
            raise SpecialistServiceError("not_configured", "CASE_STORAGE_MODAL_URL is not set")
        self.timeout = timeout

    def healthz(self) -> dict[str, Any]:
        url = f"{self.base_url}-healthz.modal.run"
        data, _ = _http_json(url, timeout=self.timeout)
        if data.get("status") != "ok" or data.get("version") != "case-storage-modal-v0.5.0-alpha":
            raise SpecialistServiceError("contract_mismatch", "case storage health contract mismatch")
        return data

    def manifest(self, case_id: str, *, request_id: str, required: bool = True) -> SpecialistCall:
        if not re.fullmatch(r"[0-9a-f]{16}", case_id):
            raise SpecialistServiceError("invalid_case_id", "case_id must be exactly 16 lowercase hex characters")
        url = f"{self.base_url}-manifest.modal.run?{urllib_parse.urlencode({'case_id': case_id})}"
        data, elapsed = _http_json(url, timeout=self.timeout)
        if data.get("case_id") != case_id:
            raise SpecialistServiceError("contract_mismatch", "manifest case_id does not match request")
        sha256_full = str(data.get("sha256_full") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256_full) or not sha256_full.startswith(case_id):
            raise SpecialistServiceError("contract_mismatch", "manifest sha256_full is invalid")
        files = data.get("files")
        if not isinstance(files, list) or not files or len(files) != len(set(files)):
            raise SpecialistServiceError("unsafe_manifest", "manifest file list must be non-empty and unique")
        for name in files:
            _validate_case_filename(name)
        byte_counts = data.get("byte_counts")
        if not isinstance(byte_counts, dict) or set(byte_counts) != set(files):
            raise SpecialistServiceError("contract_mismatch", "manifest byte_counts must match files")
        if any(type(byte_counts[name]) is not int or byte_counts[name] < 0 for name in files):
            raise SpecialistServiceError("contract_mismatch", "manifest byte counts must be non-negative integers")
        version = str(data.get("app_version") or "")
        if version != "case-storage-modal-v0.5.0-alpha":
            raise SpecialistServiceError("contract_mismatch", f"unexpected case storage version {version!r}")
        manifest_sha = _canonical_sha256(data)
        return SpecialistCall(
            output={**data, "manifest_sha256": manifest_sha},
            receipt=_success_receipt(
                stage="case_storage",
                required=required,
                request_id=request_id,
                service_name="case-storage",
                endpoint_label=url.split("?")[0],
                app_version=version,
                model_name=None,
                model_version=None,
                artifact_sha256=manifest_sha,
                input_reference=f"case_id:{case_id};content_sha256:{sha256_full}",
                latency_ms=elapsed,
                warnings=[str(data.get("disclaimer") or "")],
            ),
        )

    def get_file(
        self,
        case_id: str,
        file_name: str,
        *,
        request_id: str,
        required: bool = True,
    ) -> SpecialistCall:
        if not re.fullmatch(r"[0-9a-f]{16}", case_id):
            raise SpecialistServiceError("invalid_case_id", "case_id must be exactly 16 lowercase hex characters")
        safe_name = _validate_case_filename(file_name)
        query = urllib_parse.urlencode({"case_id": case_id, "file": safe_name})
        url = f"{self.base_url}-get.modal.run?{query}"
        data, elapsed = _http_json(url, timeout=self.timeout)
        if data.get("case_id") != case_id or data.get("file") != safe_name:
            raise SpecialistServiceError("contract_mismatch", "case-storage file identity mismatch")
        if data.get("app_version") != "case-storage-modal-v0.5.0-alpha":
            raise SpecialistServiceError("contract_mismatch", "case-storage file version mismatch")
        encoded = data.get("bytes_b64")
        try:
            blob = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise SpecialistServiceError("contract_mismatch", "case-storage file body is invalid base64") from exc
        if type(data.get("byte_count")) is not int or data["byte_count"] != len(blob):
            raise SpecialistServiceError("contract_mismatch", "case-storage byte_count mismatch")
        blob_sha = hashlib.sha256(blob).hexdigest()
        return SpecialistCall(
            output={
                "case_id": case_id,
                "file": safe_name,
                "bytes": blob,
                "byte_count": len(blob),
                "sha256": blob_sha,
            },
            receipt=_success_receipt(
                stage="case_storage_file_retrieval",
                required=required,
                request_id=request_id,
                service_name="case-storage",
                endpoint_label=url.split("?")[0],
                app_version="case-storage-modal-v0.5.0-alpha",
                model_name=None,
                model_version=None,
                artifact_sha256=blob_sha,
                input_reference=f"case_id:{case_id};file:{safe_name}",
                latency_ms=elapsed,
                warnings=["deidentified_content_retrieval"],
            ),
        )


class PhikonClient:
    def __init__(self, embed_url: str | None = None, timeout: float = 300.0):
        self.embed_url = (embed_url or os.getenv("PHIKON_MODAL_EMBED_URL") or "").rstrip("/")
        if not self.embed_url:
            raise SpecialistServiceError("not_configured", "PHIKON_MODAL_EMBED_URL is not set")
        self.timeout = timeout

    def embed(self, image_bytes: bytes, *, request_id: str, required: bool = False) -> SpecialistCall:
        input_sha = hashlib.sha256(image_bytes).hexdigest()
        data, elapsed = _http_json(
            self.embed_url,
            payload={"inputs": [{"image_b64": base64.b64encode(image_bytes).decode("ascii")}]},
            timeout=self.timeout,
        )
        embedding = data.get("embedding")
        if embedding is None and isinstance(data.get("embeddings"), list) and data["embeddings"]:
            embedding = data["embeddings"][0]
        if not isinstance(embedding, list) or len(embedding) != 768:
            raise SpecialistServiceError("contract_mismatch", "Phikon embedding must contain exactly 768 values")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise SpecialistServiceError("contract_mismatch", "Phikon embedding contains a non-numeric value") from exc
        vector_sha = _canonical_sha256(vector)
        output = {key: value for key, value in data.items() if key not in {"embedding", "embeddings"}}
        output.update({"embedding_dim": 768, "vector_sha256": vector_sha})
        return SpecialistCall(
            output=output,
            receipt=_success_receipt(
                stage="phikon_embedding",
                required=required,
                request_id=request_id,
                service_name="phikon",
                endpoint_label=self.embed_url,
                app_version=str(data.get("app_version") or data.get("version") or "") or None,
                model_name=str(data.get("model") or data.get("model_name") or "phikon"),
                model_version=str(data.get("model_version") or "") or None,
                artifact_sha256=vector_sha,
                input_reference=f"sha256:{input_sha}",
                latency_ms=elapsed,
                warnings=["embedding_only_no_subtype_or_clinical_interpretation"],
            ),
        )


class Luna16Client:
    def __init__(self, detect_url: str | None = None, timeout: float = 600.0):
        self.detect_url = (detect_url or os.getenv("LUNA16_MODAL_DETECT_URL") or "").rstrip("/")
        if not self.detect_url:
            raise SpecialistServiceError("not_configured", "LUNA16_MODAL_DETECT_URL is not set")
        self.timeout = timeout

    def detect(self, case_id: str, *, request_id: str, top_n: int = 20, required: bool = True) -> SpecialistCall:
        if not re.fullmatch(r"[0-9a-f]{16}", case_id):
            raise SpecialistServiceError("invalid_case_id", "case_id must be exactly 16 lowercase hex characters")
        data, elapsed = _http_json(
            self.detect_url,
            payload={"case_id": case_id, "top_n": max(1, min(int(top_n), 100))},
            timeout=self.timeout,
        )
        if data.get("model_state") != "loaded_luna16_retinanet":
            raise SpecialistServiceError("contract_mismatch", "LUNA16 model_state is not loaded_luna16_retinanet")
        if data.get("bundle_version") != "0.6.9" or data.get("case_id") != case_id:
            raise SpecialistServiceError("contract_mismatch", "LUNA16 bundle or case_id mismatch")
        if not isinstance(data.get("detections"), list):
            raise SpecialistServiceError("contract_mismatch", "LUNA16 detections must be a list")
        return SpecialistCall(
            output=data,
            receipt=_success_receipt(
                stage="luna16_detection",
                required=required,
                request_id=request_id,
                service_name="luna16-infer",
                endpoint_label=self.detect_url,
                app_version=str(data.get("app_version") or "") or None,
                model_name=str(data.get("model_name") or "") or None,
                model_version="0.6.9",
                artifact_sha256=None,
                input_reference=f"case_id:{case_id};ingest:{data.get('ingest_source')}",
                latency_ms=elapsed,
                warnings=[str(data.get("disclaimer") or "")],
            ),
        )


class SLTherapyBridgeClient:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 60.0):
        self.url = (url or os.getenv("SL_THERAPY_BRIDGE_URL") or "").rstrip("/")
        self.key = key or os.getenv("ONCOLOGY_ARBITER_BRIDGE_KEY") or ""
        if not self.url:
            raise SpecialistServiceError("not_configured", "SL_THERAPY_BRIDGE_URL is not set")
        if not self.key:
            raise SpecialistServiceError("not_configured", "ONCOLOGY_ARBITER_BRIDGE_KEY is not set")
        self.timeout = timeout

    def run(self, payload: dict[str, Any], *, request_id: str, required: bool = True) -> SpecialistCall:
        input_sha = _canonical_sha256(payload)
        data, elapsed = _http_json(
            self.url,
            payload=payload,
            headers={"X-Oncology-Bridge-Key": self.key},
            timeout=self.timeout,
        )
        if data.get("status") != "succeeded" or data.get("bridge_version") != "v3.1":
            raise SpecialistServiceError("contract_mismatch", "SL bridge did not return succeeded v3.1 output")
        for field in ("sl_indicated_drugs", "policy_overlays", "gap_summary", "provenance"):
            if field not in data:
                raise SpecialistServiceError("contract_mismatch", f"SL bridge response missing {field}")
        return SpecialistCall(
            output=data,
            receipt=_success_receipt(
                stage="synthetic_lethality_therapy_bridge",
                required=required,
                request_id=request_id,
                service_name="crispro-backend-v2",
                endpoint_label=self.url,
                app_version="v3.1",
                model_name="SLTherapyBridge",
                model_version="v3.1",
                artifact_sha256=None,
                input_reference=f"sha256:{input_sha}",
                latency_ms=elapsed,
                warnings=["research_actionability_hypotheses_not_treatment_benefit"],
            ),
        )
