"""Auto-extracted router: research."""
from __future__ import annotations



from fastapi import APIRouter, Depends, HTTPException

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_scope
from oncology_arbiter.api import deps
from oncology_arbiter.api.services import runtime_hooks
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.schemas import (
    ResearchIdentifiedSetRequest, ResearchIdentifiedSetResponse,
    ManskiBoundsBlock,
)
from oncology_arbiter.arbiter.manski import RESEARCH_BLIND_INFERENCE_SCOPE
from oncology_arbiter.observability import get_logger

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

_logger = get_logger()

router = APIRouter()

@router.post(
    "/v1/research/arbiter/identified_set",
    response_model=ResearchIdentifiedSetResponse,
)
def research_identified_set(
    req: ResearchIdentifiedSetRequest,
    tenant: APIKey = Depends(require_scope(RESEARCH_BLIND_INFERENCE_SCOPE)),
) -> ResearchIdentifiedSetResponse:
    """Return the Manski identified set for a partial panel, un-gated.

    This is a research instrument for measuring how much a missing
    covariate costs, not a clinical endpoint. When the panel is complete
    the response is identical in content to the public route; when it is
    not, the caller gets the interval that the public route converts into
    an HTTP 422.
    """
    from oncology_arbiter.arbiter import load_arbiter
    from oncology_arbiter.arbiter.manski import (
        MANSKI_ERROR_CODE,
        UNBOUNDED_ERROR_CODE,
        ManskiBounds,
        blind_inference_receipt,
        research_release,
    )

    request_id = new_request_id()
    try:
        arb = load_arbiter(req.arbiter)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, f"unknown arbiter {req.arbiter!r}: {exc}")

    result = arb.score(req.features)
    bounds = ManskiBounds.from_arbiter(
        stage=f"research_{req.arbiter}",
        arbiter=arb,
        features=req.features,
        result=result,
    )
    warnings = research_release(bounds)
    receipt = blind_inference_receipt(
        bounds,
        tenant_id=tenant.tenant_id,
        key_prefix=tenant.key_prefix,
        route="/v1/research/arbiter/identified_set",
        request_features=req.features,
    )
    error_code = (
        UNBOUNDED_ERROR_CODE
        if bounds.unbounded
        else (MANSKI_ERROR_CODE if not bounds.identified else None)
    )
    log_event(
        request_id,
        "/v1/research/arbiter/identified_set",
        model_state=str(result.metadata["model_state"]),
        tenant_id=tenant.tenant_id,
        extra={"blind_inference_receipt": receipt},
    )
    _logger.warning(
        "blind inference released stage=%s tenant=%s width=%.6f "
        "bounds=[%.6f,%.6f] public_route_would_have_returned=%s",
        bounds.stage, tenant.tenant_id, bounds.width,
        bounds.lower, bounds.upper, error_code,
    )
    return ResearchIdentifiedSetResponse(
        request_id=request_id,
        arbiter=req.arbiter,
        model_name=arb.model_name,
        point_estimate=result.p_positive,
        bounds=ManskiBoundsBlock(**bounds.as_dict()),
        would_have_been_rejected=error_code is not None,
        public_route_error_code=error_code,
        missing_features=list(bounds.missing_features),
        unobserved_unbounded=list(bounds.unobserved_unbounded),
        warnings=list(warnings),
        audit_receipt=receipt,
    )
