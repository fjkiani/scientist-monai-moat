"""ASGI entrypoint. Domain routes live in api/routers."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__
from oncology_arbiter.models.cbis_ddsm_probe import (
    EXPECTED_JOBLIB_SHA256 as CBIS_MEDSIGLIP_SHA256,
)

# models/cbis_ddsm_logreg_v2.joblib — pure-vision 1152-d, no assessment.
assert CBIS_MEDSIGLIP_SHA256 == (
    "e5f5b67caf8ba18006e3d582e91becf95fbb20b8b34417366c7ef07b906a211b"
)

from . import deps
from .http_stack import _lifespan, install_http_stack
from .routers import breast, hgsoc, immunotherapy, lung, nlp, ovarian, system, tumor_board
from .runtime import (
    _ARTIFACT_ROOTS,
    _elo_reason_for_modifier,
    _get_medsiglip,
    _run_cbis_ddsm_probe_on_bytes,
    _run_medsiglip_on_preprocessed,
)

_to_schema_gate_report = deps.to_schema_gate_report
_score_arbiter = deps.score_arbiter
_is_env_true = deps.is_env_true

_TRIAGE_REQUIRED_FIELDS = {
    "histology", "grade", "er_positive", "pr_positive", "her2_positive",
    "ki67_pct", "tumor_size_mm", "lymph_nodes_pos", "brca_pathogenic", "age_years",
}


def _score_explicit_therapy_triage(
    raw: dict[str, Any] | None,
):
    """Score an explicit therapy triage vector, or return None when none was asked for.

    A partial panel is scored and handed to the Manski gate. The caller gets a
    point estimate only when the panel is identified. Otherwise the gate
    returns HTTP 422 with the interval and the missing features.
    """
    if raw is None:
        return None
    from oncology_arbiter.api.schemas import TherapyTriageFeatures

    validated = TherapyTriageFeatures.model_validate(raw).model_dump()
    if all(validated[field] is None for field in _TRIAGE_REQUIRED_FIELDS):
        return None
    nodes = validated["lymph_nodes_pos"]
    features = {
        "histology": validated["histology"],
        "grade": None if validated["grade"] is None else str(validated["grade"]),
        "er_status_positive": validated["er_positive"],
        "pr_status_positive": validated["pr_positive"],
        "her2_status_positive": validated["her2_positive"],
        "ki67_norm": validated["ki67_pct"],
        "tumor_size_norm": validated["tumor_size_mm"],
        "node_status_positive": None if nodes is None else nodes > 0,
        "brca_status_known_pathogenic": validated["brca_pathogenic"],
        "age_at_diagnosis_norm": validated["age_years"],
    }
    return _score_arbiter(
        "therapy",
        features=features,
        stage="therapy_research_triage",
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncology-arbiter",
        version=__version__,
        description=RUO_DISCLAIMER + "\n\n" + AUROC_CAVEAT,
        lifespan=_lifespan,
    )
    install_http_stack(app)
    app.include_router(system.router)
    app.include_router(breast.router)
    app.include_router(lung.router)
    app.include_router(nlp.router)
    app.include_router(hgsoc.router)
    app.include_router(ovarian.router)
    app.include_router(immunotherapy.router)
    app.include_router(tumor_board.router)
    return app


app = create_app()
