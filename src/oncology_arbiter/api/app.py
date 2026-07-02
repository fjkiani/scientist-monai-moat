"""FastAPI app factory + routes.

Everything returned here today is a PLACEHOLDER. It's the *shape* of the
real response, not the real answer. We wire real models in Phase 2+.

Design invariants for these placeholders:
  1. Every response includes `disclaimer` (RUO) and `caveat` (AUROC)
     inline. If we ever accidentally strip them, tests fail loudly.
  2. Every response includes `provenance.model_state = PLACEHOLDER` so a
     downstream consumer cannot mistake a stub for a live inference.
  3. The mammography endpoint runs REAL preprocessing (readers, laterality,
     view, mask) even when the model is a placeholder — this way we get to
     exercise ~90% of the pipeline on real DICOMs end-to-end from HTTP.
     Only the classification score is placeholder.
  4. The `honesty_gate` field always reports {kept=0, dropped=0} on the
     placeholder path since no evidence was gathered.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER, __version__

from .audit import log_event, new_request_id
from .schemas import (
    ApiEnvelope,
    BiopsyReceptorPanel,
    BiopsyRequest,
    BiopsyResponse,
    EvidenceRecord,
    FullCaseRequest,
    FullCaseResponse,
    HealthResponse,
    HonestyGateReport,
    ModelState,
    Provenance,
    ScreeningRequest,
    ScreeningResponse,
    TherapyOption,
    TherapyRequest,
    TherapyResponse,
)


# --------------------------------------------------------------------------- #
# Helpers


def _envelope(request_id: str, model_state: ModelState = ModelState.PLACEHOLDER,
              model_name: str | None = None) -> dict[str, Any]:
    """Common envelope fields that must appear on every response body."""
    return {
        "disclaimer": RUO_DISCLAIMER,
        "caveat": AUROC_CAVEAT,
        "provenance": Provenance(
            model_state=model_state,
            model_name=model_name,
            request_id=request_id,
        ),
        "honesty_gate": HonestyGateReport(
            seen_urls_count=0, evidence_kept=0, evidence_dropped=0,
        ),
        "evidence": [],
    }


def _decode_bytes_arg(bytes_b64: str | None) -> bytes | None:
    if bytes_b64 is None:
        return None
    try:
        return base64.b64decode(bytes_b64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 dicom_bytes: {e}")


# --------------------------------------------------------------------------- #
# App factory


def create_app() -> FastAPI:
    app = FastAPI(
        title="oncology-arbiter",
        version=__version__,
        description=RUO_DISCLAIMER + "\n\n" + AUROC_CAVEAT,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            disclaimer=RUO_DISCLAIMER,
            caveat=AUROC_CAVEAT,
            endpoints=[
                "POST /v1/screening/analyze",
                "POST /v1/biopsy/analyze",
                "POST /v1/therapy/reason",
                "POST /v1/case/full",
                "GET /health",
            ],
            models_loaded={
                "monai_screening": ModelState.PLACEHOLDER,
                "medsiglip_biopsy": ModelState.PLACEHOLDER,
                "txgemma_therapy": ModelState.PLACEHOLDER,
                "co_scientist": ModelState.PLACEHOLDER,
            },
        )

    # ----------------------------------------------------------------------- #
    # /v1/screening/analyze — REAL preprocessing, placeholder classifier

    @app.post("/v1/screening/analyze", response_model=ScreeningResponse)
    def screening_analyze(req: ScreeningRequest) -> ScreeningResponse:
        request_id = new_request_id()

        if not req.dicom_url and not req.dicom_bytes_b64:
            raise HTTPException(400, "must provide dicom_url or dicom_bytes_b64")
        if req.dicom_url and req.dicom_bytes_b64:
            raise HTTPException(400, "provide exactly one of dicom_url or dicom_bytes_b64")

        raw_bytes = _decode_bytes_arg(req.dicom_bytes_b64)
        if raw_bytes is None and req.dicom_url is not None:
            # Placeholder: URL ingestion goes through our SSRF-guarded fetcher
            # in Phase 2. For now, refuse cleanly.
            log_event(request_id, "/v1/screening/analyze",
                      model_state="placeholder",
                      patient_id_hash=req.patient_id_hash,
                      extra={"reason": "url_ingestion_not_wired"})
            raise HTTPException(
                501, "dicom_url ingestion not yet wired (Phase 2). "
                "Send dicom_bytes_b64 for now.",
            )

        # Real preprocessing on the uploaded bytes.
        from oncology_arbiter.mammography import preprocess_mammogram
        assert raw_bytes is not None  # type narrowing
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
            tf.write(raw_bytes)
            tmp_path = tf.name
        try:
            result = preprocess_mammogram(
                tmp_path,
                laterality_hint=req.laterality_hint,
                view_hint=req.view_hint,
            )
        except Exception as e:
            log_event(request_id, "/v1/screening/analyze",
                      model_state="unavailable",
                      patient_id_hash=req.patient_id_hash,
                      extra={"error": str(e)[:200]})
            raise HTTPException(422, f"preprocessing failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        response = ScreeningResponse(
            **_envelope(request_id),
            laterality=result.metadata.laterality.value,
            view=result.metadata.view.value,
            orientation_flipped=result.metadata.orientation_flipped,
            breast_mask_coverage=float(result.breast_mask.mean()),
            findings=[],
            overall_score=None,   # placeholder — no classifier wired yet
        )
        log_event(request_id, "/v1/screening/analyze",
                  model_state="placeholder",
                  patient_id_hash=req.patient_id_hash,
                  extra={
                      "shape": [result.image.shape[0], result.image.shape[1]],
                      "laterality": result.metadata.laterality.value,
                      "view": result.metadata.view.value,
                      "mask_coverage": float(result.breast_mask.mean()),
                  })
        return response

    # ----------------------------------------------------------------------- #
    # /v1/biopsy/analyze — placeholder

    @app.post("/v1/biopsy/analyze", response_model=BiopsyResponse)
    def biopsy_analyze(req: BiopsyRequest) -> BiopsyResponse:
        request_id = new_request_id()
        if not req.wsi_url and not req.wsi_bytes_b64 and not req.report_text:
            raise HTTPException(400, "must provide wsi_url, wsi_bytes_b64, or report_text")
        log_event(request_id, "/v1/biopsy/analyze",
                  model_state="placeholder",
                  patient_id_hash=req.patient_id_hash,
                  extra={"has_wsi": bool(req.wsi_url or req.wsi_bytes_b64),
                         "has_report": bool(req.report_text)})
        return BiopsyResponse(
            **_envelope(request_id),
            subtype_prediction=None,
            receptor_panel=BiopsyReceptorPanel(),
            grade=None,
            confidence=None,
        )

    # ----------------------------------------------------------------------- #
    # /v1/therapy/reason — placeholder

    @app.post("/v1/therapy/reason", response_model=TherapyResponse)
    def therapy_reason(req: TherapyRequest) -> TherapyResponse:
        request_id = new_request_id()
        log_event(request_id, "/v1/therapy/reason",
                  model_state="placeholder",
                  patient_id_hash=None,
                  extra={"has_biopsy_input": req.biopsy_output is not None})
        return TherapyResponse(
            **_envelope(request_id),
            recommended_options=[],
            not_recommended=[],
        )

    # ----------------------------------------------------------------------- #
    # /v1/case/full — placeholder that chains sub-endpoints

    @app.post("/v1/case/full", response_model=FullCaseResponse)
    def case_full(req: FullCaseRequest) -> FullCaseResponse:
        request_id = new_request_id()
        screening: ScreeningResponse | None = None
        biopsy: BiopsyResponse | None = None
        therapy: TherapyResponse | None = None
        # If sub-inputs are provided, run them through the placeholder subroutes.
        if req.screening_input:
            screening = screening_analyze(req.screening_input)
        if req.biopsy_input:
            biopsy = biopsy_analyze(req.biopsy_input)
        therapy = therapy_reason(
            TherapyRequest(biopsy_output=biopsy, patient_context=req.therapy_context)
        )
        log_event(request_id, "/v1/case/full",
                  model_state="placeholder",
                  patient_id_hash=None,
                  extra={
                      "has_screening": screening is not None,
                      "has_biopsy": biopsy is not None,
                  })
        return FullCaseResponse(
            **_envelope(request_id),
            screening=screening,
            biopsy=biopsy,
            therapy=therapy,
            elo_ranked_hypotheses=[],  # placeholder — Co-Scientist Elo in Phase 3
        )

    return app
