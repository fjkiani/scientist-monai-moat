"""Auto-extracted router: demo."""
from __future__ import annotations

import os


from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from oncology_arbiter.api import deps
from oncology_arbiter.api.schemas import DemoCaseResponse
from oncology_arbiter.api.services import runtime_hooks
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator

_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg
_score_arbiter = deps.score_arbiter
_to_schema_gate_report = deps.to_schema_gate_report
_is_env_true = deps.is_env_true
_demo_samples_dir = deps.demo_samples_dir
_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report
_medgemma_biopsy_parse = PathologyParseOrchestrator._parse_medgemma

# Patchable via oncology_arbiter.api.app.* in unit tests
_run_medsiglip_on_preprocessed = runtime_hooks.run_medsiglip_on_preprocessed
_run_cbis_ddsm_probe_on_bytes = runtime_hooks.run_cbis_ddsm_probe_on_bytes
_elo_reason_for_modifier = runtime_hooks.elo_reason_for_modifier
_run_clinicalbert_parse = runtime_hooks.run_clinicalbert_parse
_score_explicit_therapy_triage = runtime_hooks.score_explicit_therapy_triage


router = APIRouter()

@router.get(
    "/v1/demo/case",
    response_model=DemoCaseResponse,
    summary="Return a fully-formed sample case for demoing the pipeline.",
)
def demo_case() -> DemoCaseResponse:
    from ..demo_fixtures import DemoFixtureUnavailable, build_demo_case

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
# note explaining what was real vs. synthetic. These are archived RUO
# snapshots, not current inference-readiness claims; individual stages may
# carry TEMPLATE or RETIRED model states and must be interpreted as such.

@router.get("/v1/demo/samples")
def demo_samples_index():
    _sd = _demo_samples_dir()
    contact_url = os.environ.get(
        "ONCOLOGY_ARBITER_CONTACT_URL", "https://crispro.ai/contact"
    )
    if _sd is None:
        return JSONResponse(
            status_code=200,
            content={
                "samples": [],
                "demo_mode": _is_env_true("ONCOLOGY_ARBITER_DEMO_MODE"),
                "contact_url": contact_url,
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
            "contact_url": contact_url,
            "note": (
                "Archived research-use-only snapshots. Some specialist stages "
                "used real weights while arbiter stages may be TEMPLATE or "
                "RETIRED. Inspect each model_state, caveat, and demo_provenance "
                "block; these samples do not claim current inference readiness."
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
