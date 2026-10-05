"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__

from ..schemas import HealthResponse, ModelState
from .. import deps

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@router.get("/healthz", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    """/health endpoint - reports API status and available endpoints."""
    # `cancers` mirrors the surface `/v1/case/full?cancer=…` accepts.
    # `breast` is the flagship path (real preprocessing, arbiter, etc.);
    # `nsclc` is the LIDC-IDRI expansion track that worker-2 is wiring —
    # the endpoint currently returns a shape-only placeholder so the SPA
    # can already render a working NSCLC panel end-to-end.
    _demo_active = deps.is_env_true("ONCOLOGY_ARBITER_DEMO_MODE")
    _contact_url = deps.is_env_true("ONCOLOGY_ARBITER_CONTACT_URL") or "https://crispro.ai/contact"
    _sample_files: list[str] = []
    if _demo_active:
        _sd = deps.demo_samples_dir()
        if _sd is not None and _sd.exists():
            _sample_files = sorted(p.stem for p in _sd.glob("*.json"))
    endpoints = [
        "POST /v1/screening/analyze",
        "POST /v1/biopsy/analyze",
        "POST /v1/therapy/reason",
        "POST /v1/case/full",
        "POST /v1/elo/rank",
        # v0.4.0-alpha: standalone Co-Scientist loop (PLAN §5 PR #5)
        "POST /v1/offline_ranker/run",
        "POST /v1/tumor_board/dynamic",
        "GET  /v1/demo/case",
        "GET  /v1/model-cards",
        "GET  /v1/artifacts/{category}/{filename}",
        "GET  /health",
    ]
    if _demo_active:
        endpoints.append("GET  /v1/demo/samples")
        endpoints.append("GET  /v1/demo/samples/{kind}")
    return HealthResponse(
        status="ok",
        version=__version__,
        disclaimer=RUO_DISCLAIMER,
        caveat=AUROC_CAVEAT,
        endpoints=endpoints,
        demo_mode=_demo_active,
        contact_url=_contact_url if _demo_active else None,
        demo_samples=_sample_files,
        cancers={
            "breast": {
                "state": ModelState.CONFIGURED_UNVERIFIED.value,
                "case_full": True,
                "patient_probability_emitted": True,
                "endpoints": ["screening", "biopsy", "therapy", "tumor_board/dynamic", "case/full"],
                "notes": "Completion depends on successful specialist stage receipts; URL configuration alone is not inference readiness.",
            },
            "nsclc": {
                "state": ModelState.CONFIGURED_UNVERIFIED.value,
                "case_full": True,
                "patient_probability_emitted": True,
                "endpoints": ["case/full"],
                "notes": "Production flow requires verified case storage followed by LUNA16 RetinaNet bundle 0.6.9; no local series path or HU heuristic is accepted.",
            },
            "hgsoc": {
                "state": ModelState.RETIRED.value,
                # case_full is True because /v1/case/full genuinely accepts
                # cancer=hgsoc. It is NOT a claim that a scorer runs: the
                # two keys below carry that, so route reachability can no
                # longer be misread as scoring availability.
                "case_full": True,
                "patient_probability_emitted": False,
                "retired_scope": "ovarian_prognostic_arbiter_patient_level_scoring",
                "endpoints": ["tumor_board/dynamic", "tumor_board/bundle", "case/full"],
                "notes": "Route reachable; the ovarian prognostic arbiter is retired and emits no patient probability or risk bucket. Therapy and MedGemma stages remain independently receipt-gated.",
            },
        },
        models_loaded=deps.compute_models_loaded(),
    )
