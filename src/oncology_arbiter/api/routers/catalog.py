"""Route module split out of api/app.py."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__
from oncology_arbiter.api import deps
from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.runtime import (
    _ARTIFACT_ROOTS,
    _elo_reason_for_modifier,
    _run_cbis_ddsm_probe_on_bytes,
)
from oncology_arbiter.api.schemas import *  # noqa: F403
from oncology_arbiter.api.services.helpers import failed_stage_receipt as _failed_stage_receipt
from oncology_arbiter.api.services.helpers import pipeline_status as _pipeline_status
from oncology_arbiter.api.services.helpers import skipped_stage_receipt as _skipped_stage_receipt
from oncology_arbiter.arbiter.manski import (
    RESEARCH_BLIND_INFERENCE_SCOPE,
    ManskiGateError,
)
from oncology_arbiter.auth import APIKey, require_api_key, require_scope

router = APIRouter()


def _contact_url() -> str:
    return os.environ.get("ONCOLOGY_ARBITER_CONTACT_URL", "https://crispro.ai/contact")

_to_schema_gate_report = deps.to_schema_gate_report
_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg
_score_arbiter = deps.score_arbiter
_is_env_true = deps.is_env_true
_demo_samples_dir = deps.demo_samples_dir
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report


def _app():
    """Late binding so tests can patch helpers on oncology_arbiter.api.app."""
    import oncology_arbiter.api.app as app_module
    return app_module


# ----------------------------------------------------------------------- #
# /v1/model-cards — index every model card shipped with the API

@router.get("/v1/model-cards", response_model=ModelCardsIndex)
def list_model_cards(
    tenant: APIKey = Depends(require_api_key),
) -> ModelCardsIndex:
    """PLAN.md §5.6: `Public model card + errata page` — served as JSON
    index so a client can enumerate what cards exist without a directory
    listing. Raw markdown is served by `/v1/artifacts/docs/{filename}`.
    """
    cards_dir = _ARTIFACT_ROOTS[ArtifactCategory.docs]
    cards: list[ModelCardSummary] = []
    if cards_dir.is_dir():
        for path in sorted(cards_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            first_h1 = ""
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    first_h1 = line[2:].strip()
                    break
            # A card is properly disclaimed if it either quotes the
            # RUO phrase verbatim OR references the RUO_DISCLAIMER
            # constant so a reader knows where the phrase lives. Both
            # patterns exist in this repo — accept either.
            ruo_ok = ("RESEARCH USE ONLY" in text) or ("RUO_DISCLAIMER" in text)
            cards.append(ModelCardSummary(
                slug=path.stem,
                title=first_h1 or path.stem,
                n_bytes=len(text.encode("utf-8")),
                honesty_markers={
                    "auroc_caveat_present": "AUROC" in text,
                    "ruo_disclaimer_present": ruo_ok,
                    "not_fda_cleared_note": "FDA" in text,
                },
            ))
    return ModelCardsIndex(
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        cards=cards,
    )

# ----------------------------------------------------------------------- #
# /v1/artifacts/{category}/{filename} — path-traversal-safe streamer.
# Mirrors org.backend/capabilities/progression_arbiter/router.py verbatim
# for the security model: category-whitelisted + relative_to() containment.

@router.get("/v1/artifacts/{category}/{filename}")
def stream_artifact(
    category: str,
    filename: str,
    tenant: APIKey = Depends(require_api_key),
):
    try:
        cat = ArtifactCategory(category)
    except ValueError:
        raise HTTPException(400, f"invalid category: {category}")
    category_dir = _ARTIFACT_ROOTS[cat].resolve()
    # Reject empty / suspicious filenames early
    if not filename or filename in {".", ".."} or "\x00" in filename:
        raise HTTPException(400, "invalid filename")
    candidate = (category_dir / filename).resolve()
    try:
        candidate.relative_to(category_dir)
    except ValueError:
        raise HTTPException(403, "directory traversal forbidden")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(404, f"artifact not found: {filename}")
    # Guess a sensible media type
    media_type = "text/markdown" if candidate.suffix == ".md" else (
        "application/json" if candidate.suffix == ".json" else (
            "application/sql" if candidate.suffix == ".sql" else "text/plain"
        )
    )
    return JSONResponse(
        status_code=200,
        content={
            "category": category,
            "filename": filename,
            "media_type": media_type,
            "n_bytes": candidate.stat().st_size,
            "content": candidate.read_text(encoding="utf-8", errors="replace"),
            "disclaimer": RUO_DISCLAIMER,
            "caveat": AUROC_CAVEAT,
        },
    )

# ----------------------------------------------------------------------- #
# /v1/demo/case — server-hosted sample case for first-time users
#
# v0.2.2: previously the SPA shipped a small canned pathology string but
# had no bundled DICOM, forcing users to hunt for one before they could
# exercise the pipeline. The demo endpoint returns a fully-formed case
# (real DICOM + pathology text + patient context) so a first-time user
# can click "Load demo case" and see the full workflow run.
#
# DICOM provenance: CBIS-DDSM (helloerikaaa/cbis-ddsm-r on HuggingFace,
# CC-BY-NC 4.0). Mass-Test_P_00016_LEFT_CC.dcm is the smallest of the
# 5 fixtures we already ship (~14 MB). No auth needed to fetch.
# Pathology text: synthetic luminal-A case (matches the frontend
# LUMINAL_A_EXAMPLE constant). Not a real patient.
#
# v0.2.2 fix (2026-07-06): PUBLIC endpoint (NO require_api_key). The
# whole point of the demo is to let an unauthenticated first-time
# visitor click "Load demo case" and see the pipeline work without
# first hunting for an API key. Locking this behind auth was
# UX-hostile *and* semantically wrong: it's a static server-hosted
# fixture (real CBIS-DDSM DICOM + synthetic luminal-A report +
# patient context) with no tenant data, no per-call cost beyond the
# one-time HF download that startup pre-warms into /tmp/oa-demo, and
# no scenario where knowing a tenant identity matters. Any endpoint
# that DOES touch tenant data (screening/biopsy/therapy/case_full)
# continues to require the header.

@router.get(
    "/v1/demo/case",
    response_model=DemoCaseResponse,
    summary="Return a fully-formed sample case for demoing the pipeline.",
)
def demo_case() -> DemoCaseResponse:
    from oncology_arbiter.api.demo_fixtures import DemoFixtureUnavailable, build_demo_case

    try:
        case = build_demo_case()
    except DemoFixtureUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Demo case unavailable: {e}. "
                "The server could not fetch the CBIS-DDSM fixture "
                "(helloerikaaa/cbis-ddsm-r on HuggingFace). "
                "This is transient — please retry in a moment."
            ),
        ) from e

    return DemoCaseResponse(
        dicom_bytes_b64=case.dicom_bytes_b64,
        dicom_source=case.dicom_source,
        dicom_sha256=case.dicom_sha256,
        dicom_size_bytes=case.dicom_size_bytes,
        report_text=case.report_text,
        patient_context=case.patient_context,
        warnings=case.warnings,
    )

# ----------------------------------------------------------------------- #
# /v1/demo/samples/{index,kind} — v0.3.0-alpha DEMO_MODE outputs.
#
# These serve pre-computed real inference outputs captured on our
# workers (see src/oncology_arbiter/api/static/demo_samples/*.json).
# Every sample envelope carries a `demo_provenance` sub-block with the
# DICOM sha256, worker id, weights used, latency, and a plain-English
# note explaining what was real vs. synthetic. When DEMO_MODE is off
# these endpoints still serve the JSON — the samples are useful
# reference outputs regardless of gate state.

@router.get("/v1/demo/samples")
def demo_samples_index():
    _sd = _demo_samples_dir()
    if _sd is None:
        return JSONResponse(
            status_code=200,
            content={
                "samples": [],
                "demo_mode": _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"),
                "contact_url": _contact_url(),
                "note": "demo_samples directory not present in this build",
            },
        )
    samples = []
    for p in sorted(_sd.glob("*.json")):
        samples.append({
            "kind": p.stem,
            "path": f"/v1/demo/samples/{p.stem}",
            "size_bytes": p.stat().st_size,
        })
    return JSONResponse(
        status_code=200,
        content={
            "samples": samples,
            "demo_mode": _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"),
            "contact_url": _contact_url(),
            "note": (
                "Each sample is a real /v1/... response captured on our "
                "workers with real weights loaded. See the "
                "`demo_provenance` sub-block in each envelope for the "
                "DICOM sha256, weights used, latency, and a plain-English "
                "note on what was real vs. synthetic."
            ),
        },
    )

@router.get("/v1/demo/samples/{kind}")
def demo_samples_read(kind: str):
    # kind must be a safe identifier — reject any traversal or dotfile.
    if not kind or not kind.replace("_", "").isalnum():
        raise HTTPException(400, f"invalid sample kind: {kind!r}")
    _sd = _demo_samples_dir()
    if _sd is None:
        raise HTTPException(404, "demo_samples not present in this build")
    candidate = (_sd / f"{kind}.json").resolve()
    try:
        candidate.relative_to(_sd.resolve())
    except ValueError:
        raise HTTPException(403, "directory traversal forbidden")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(
            404,
            (
                f"sample not found: {kind!r}. "
                "Available: GET /v1/demo/samples"
            ),
        )
    # Serve as JSON with cache headers so the SPA can grab all four
    # once and paint quickly.
    return FileResponse(
        candidate,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=300"},
    )


# ----------------------------------------------------------------------- #
# /v1/elo/rank — v0.3.0-alpha therapy candidate re-ranking
#
# PUBLIC endpoint (no require_api_key). Reasoning:
#   * Pure-CPU deterministic Python tournament; no Modal / GPU cost.
#   * On the public demo deploy the SPA's Elo panel must be usable by
#     an anonymous browser session, matching the pattern used by
#     /v1/demo/case and /v1/model-cards. Gating this behind an API
#     key would silently break the read-only demo UX.
#   * DEMO_MODE middleware still gates every other POST via its
#     _DEMO_MODE_POST_ALLOWLIST — /v1/elo/rank is the only allow-listed
#     entry.
