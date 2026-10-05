"""Shared dependencies for API endpoints."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER

from .schemas import (
    ArbiterScore,
    GateReport,
    HonestyGateReport,
    ManskiBoundsBlock,
    ModelState,
    Provenance,
)


def to_schema_gate_report(runtime_gr: Any) -> GateReport | None:
    """Convert the runtime hai_def.GateReport dataclass into the pydantic
    schema GateReport, or None if the input is None.

    We do NOT rely on pydantic's model_validate over the dataclass because
    the runtime access_level is an Enum (`AccessLevel`) — we serialize its
    `.value` string so the schema's Literal validator accepts it, and so
    JSON output matches the wire contract.
    """
    if runtime_gr is None:
        return None
    return GateReport(
        repo_id=runtime_gr.repo_id,
        access_level=runtime_gr.access_level.value,
        status_code=runtime_gr.status_code,
        reason=runtime_gr.reason,
        has_token=runtime_gr.has_token,
        allowed=bool(runtime_gr.allowed),
    )


def envelope(
    request_id: str,
    model_state: ModelState = ModelState.PLACEHOLDER,
    model_name: str | None = None,
    gate_report: GateReport | None = None,
) -> dict[str, Any]:
    """Common envelope fields that must appear on every response body.

    `gate_report` is populated on `provenance.gate_report` when the endpoint
    ran a HAI-DEF preflight (either successfully or hit a gate). Callers on
    placeholder / pure-proxy paths pass None and it stays None on the wire.
    """
    return {
        "disclaimer": RUO_DISCLAIMER,
        "caveat": AUROC_CAVEAT,
        "provenance": Provenance(
            model_state=model_state,
            model_name=model_name,
            request_id=request_id,
            gate_report=gate_report,
        ),
        "honesty_gate": HonestyGateReport(
            seen_urls_count=0,
            evidence_kept=0,
            evidence_dropped=0,
        ),
        "evidence": [],
    }


def decode_bytes_arg(bytes_b64: str | None) -> bytes | None:
    """Decode base64-encoded bytes argument."""
    if bytes_b64 is None:
        return None
    try:
        return base64.b64decode(bytes_b64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 dicom_bytes: {e}") from e


def score_arbiter(
    name: str,
    features: dict[str, Any],
    *,
    stage: str | None = None,
) -> ArbiterScore:
    """Load the named L2 arbiter, score `features`, and apply the Manski gate.

    Wraps :func:`oncology_arbiter.arbiter.load_arbiter` and marshals the
    :class:`ArbiterResult` into the wire-level :class:`ArbiterScore` pydantic.

    This is the single choke point through which every stage's L2 point
    estimate reaches the wire, so it is where partial identification is
    enforced. When the assumption-free interval implied by the *unobserved*
    part of the declared feature schema is wider than
    :data:`MANSKI_MAX_WIDTH`, no point estimate is marshalled at all:
    :class:`ManskiGateError` propagates to the app-level handler and the caller
    receives 422 plus the raw interval.

    There is no bypass. This function takes no argument, reads no header and
    consults no environment variable that could release an unidentified point
    estimate on a public route. The research override is a separate route with
    its own privileged scope; see ``/v1/research/arbiter/identified_set``.

    Empty ``features={}`` is refused at this choke point: scoring an empty
    panel against ``n_training=0`` templates returns only the intercept
    constant (static sigmoid), which is not a clinical stage readout.
    """
    from oncology_arbiter.arbiter import load_arbiter
    from oncology_arbiter.arbiter.manski import ManskiBounds, enforce_manski_gate

    if not features:
        raise HTTPException(
            422,
            detail={
                "error": "empty_arbiter_features",
                "stage": stage or name,
                "detail": (
                    f"refusing to score {name!r} with features={{}}: empty panels "
                    "collapse to the intercept-only constant and are not a "
                    "clinical stage readout"
                ),
            },
        )

    if name == "biopsy":
        from oncology_arbiter.arbiter.stage_biopsy_wiring import (
            UnvalidatedLesionStratumError,
            enforce_validated_lesion_stratum,
        )

        try:
            enforce_validated_lesion_stratum(features)
        except UnvalidatedLesionStratumError as exc:
            raise HTTPException(422, detail=exc.detail()) from exc

    arb = load_arbiter(name)
    r = arb.score(features)
    bounds = ManskiBounds.from_arbiter(
        stage=stage or name,
        arbiter=arb,
        features=features,
        result=r,
    )
    provenance_warnings = enforce_manski_gate(bounds)
    return ArbiterScore(
        model_name=arb.model_name,
        p_positive=r.p_positive,
        logit=r.logit,
        risk_bucket=r.risk_bucket,  # type: ignore[arg-type]
        recommendation=r.recommendation,
        term_contributions=r.term_contributions,
        driving_feature=r.driving_feature,
        driving_feature_contribution=r.driving_feature_contribution,
        positive_class=arb.positive_class,
        n_training=arb.n_training,
        model_state=r.metadata["model_state"],  # type: ignore[arg-type]
        caveat=r.caveat,
        manski=ManskiBoundsBlock(**bounds.as_dict()),
        provenance_warnings=list(provenance_warnings),
    )


TRIAGE_REQUIRED_FIELDS = {
    "histology",
    "grade",
    "er_positive",
    "pr_positive",
    "her2_positive",
    "ki67_pct",
    "tumor_size_mm",
    "lymph_nodes_pos",
    "brca_pathogenic",
    "age_years",
}


def score_explicit_therapy_triage(
    raw: dict[str, Any] | None,
) -> ArbiterScore | None:
    """Score the explicit therapy triage vector, or return None if none was asked for.

    A *partial* panel is no longer silently dropped. Previously any missing
    field returned None with a "skipped" receipt, which hid from the caller
    both the fact that the arbiter could still have produced a number and how
    little that number would have been worth. The partial panel is now scored
    and handed to the Manski gate, so the caller either gets a point estimate
    that is genuinely identified, or a 422 carrying the interval and the list
    of features that would collapse it.
    """
    if raw is None:
        return None
    from oncology_arbiter.api.schemas import TherapyTriageFeatures

    validated = TherapyTriageFeatures.model_validate(raw).model_dump()
    if all(validated[field] is None for field in TRIAGE_REQUIRED_FIELDS):
        # Nothing at all was supplied: this is "no triage requested", not
        # "triage requested with an unobserved panel".
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
    return score_arbiter(
        "therapy",
        features=features,
        stage="therapy_research_triage",
    )


def is_env_true(name: str) -> bool:
    """Check if environment variable is truthy."""
    import os

    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def demo_samples_dir() -> Path | None:
    """Return demo samples directory if it exists."""
    import os

    demo_root = os.environ.get("ONCOLOGY_ARBITER_DEMO_SAMPLES_DIR")
    if demo_root:
        path = Path(demo_root)
        return path if path.exists() else None
    return None


def compute_models_loaded() -> dict[str, ModelState]:
    """Report configuration state without claiming successful inference readiness."""
    import os

    def configured(*names: str) -> ModelState:
        return (
            ModelState.CONFIGURED_UNVERIFIED
            if any(os.environ.get(name) for name in names)
            else ModelState.UNAVAILABLE
        )

    from oncology_arbiter.models.cbis_ddsm_probe import DEFAULT_MODEL_PATH as _CBIS_JOBLIB

    cbis_state = (
        ModelState.LOADED if Path(_CBIS_JOBLIB).exists() else ModelState.UNAVAILABLE
    )
    biopsy_probe_path = (
        Path(__file__).resolve().parents[2]
        / "arbiter"
        / "models"
        / "biopsy_probe_v0.json"
    )
    biopsy_probe_state = (
        ModelState.LOADED_BIOPSY_PROBE
        if biopsy_probe_path.exists()
        else ModelState.UNAVAILABLE
    )

    return {
        "case_storage": configured("CASE_STORAGE_MODAL_URL"),
        "medsiglip_448": configured("MODAL_MEDSIGLIP_URL"),
        "phikon": configured("PHIKON_MODAL_URL"),
        "clinicalbert": configured("CLINICALBERT_MODAL_URL"),
        "medgemma": configured("MEDGEMMA_MODAL_URL"),
        "cbis_ddsm_probe": cbis_state,
        "biopsy_probe_v1": biopsy_probe_state,
        "sl_therapy_bridge": configured("SL_THERAPY_BRIDGE_MODAL_URL"),
        "offline_ranker_r2_modal": configured("OFFLINE_RANKER_R2_MODAL_URL"),
        "luna16_case_storage": configured("LUNA16_CASE_STORAGE_MODAL_URL"),
    }
