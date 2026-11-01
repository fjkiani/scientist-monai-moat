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
        "POST /v1/screening/fusion",
        "POST /v1/biopsy/analyze",
        "POST /v1/therapy/reason",
        "POST /v1/case/full",
        "POST /v1/elo/rank",
        # v0.4.0-alpha: standalone Co-Scientist loop (PLAN §5 PR #5)
        "POST /v1/co-scientist/run",
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
                "state": ModelState.LOADED.value,
                "case_full": True,
                "patient_probability_emitted": True,
                "retired_scope": "ovarian_mortality_arbiter_v3_tcga_prognostic_only",
                "multivariate_platinum": {
                    "state": ModelState.LOADED.value,
                    "capability": "hgsoc-platinum-resistance-multivariate-v2",
                    "artifact": "ovarian_resistance_multivariate_v2.joblib",
                    "n_training": 75,
                    "note": (
                        "OOF multivariate does not beat univariate MFAP4 "
                        "0.6836 on this cohort; honesty_note in proof JSON."
                    ),
                },
                "mfap4_platinum": {
                    "state": ModelState.LOADED.value,
                    "capability": "hgsoc-mfap4-platinum",
                    "probe": "212713_at",
                    "auroc": 0.684,
                    "n_training": 75,
                    "endpoints": [
                        "GET /v1/hgsoc/mfap4/health",
                        "GET /v1/hgsoc/mfap4/metrics",
                        "POST /v1/hgsoc/mfap4/score",
                    ],
                },
                "endpoints": [
                    "POST /v1/case/full?cancer=hgsoc",
                    "POST /v1/tumor_board/dynamic",
                    "tumor_board/bundle",
                    "GET /v1/hgsoc/mfap4/health",
                    "GET /v1/hgsoc/mfap4/metrics",
                    "POST /v1/hgsoc/mfap4/score",
                ],
                "notes": (
                    "HGSOC case/full executes ovarian_resistance_multivariate_v2 "
                    "(GSE63885 honest n=75). TCGA ovarian mortality arbiter stays "
                    "retired. Univariate MFAP4 OOF remains stronger than "
                    "multivariate on this cohort. Retracted n=101 AUROC 0.7625 "
                    "forbidden. Breast paths are not used for ovarian requests."
                ),
            },
        },
        models_loaded=deps.compute_models_loaded(),
    )
