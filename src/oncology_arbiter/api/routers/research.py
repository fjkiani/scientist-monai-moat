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
from oncology_arbiter.observability import get_logger

router = APIRouter()

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
# /v1/research/arbiter/identified_set — the ONLY blind-inference surface
#
# The public gate has no bypass: no header, query parameter or environment
# variable releases an unidentified point estimate from /v1/screening,
# /v1/biopsy, /v1/therapy, /v1/tumor_board or /v1/case. Blind inference
# lives here instead, behind four separate locks:
#
#   1. a valid API key,
#   2. the privileged `research:blind_inference` scope — `require_scope`
#      deliberately does NOT honour ONCOLOGY_ARBITER_AUTH_MODE=off, so the
#      anonymous dev principal (which holds no scopes) is refused even in a
#      local process where every other route is wide open,
#   3. an explicit `acknowledge_not_for_clinical_use: true` in the body,
#   4. an audit receipt naming the tenant, written before the response is
#      marshalled, recording exactly what the public route would have
#      refused.
#
# The response returns the raw interval and the unobserved-feature list
# alongside the number, so the caller cannot receive a point estimate
# without simultaneously receiving the evidence that it is uninformative.

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
    get_logger().warning(
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
