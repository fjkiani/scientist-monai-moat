"""Auto-extracted router: artifacts."""
from __future__ import annotations

from pathlib import Path


from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.api.schemas import (
    ModelCardsIndex, ModelCardSummary, ArtifactCategory,
)
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.api import deps
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

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ARTIFACT_ROOTS: dict[ArtifactCategory, Path] = {
    ArtifactCategory.docs: _PROJECT_ROOT / "docs" / "model_cards",
    ArtifactCategory.reports: _PROJECT_ROOT / "artifacts" / "reports",
    ArtifactCategory.data: _PROJECT_ROOT / "artifacts" / "data",
    ArtifactCategory.models: _PROJECT_ROOT / "src" / "oncology_arbiter" / "arbiter" / "models",
}


router = APIRouter()

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
