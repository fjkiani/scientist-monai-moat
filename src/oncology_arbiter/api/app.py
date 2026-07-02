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
    ArbiterScore,
    ArtifactCategory,
    BiopsyReceptorPanel,
    BiopsyRequest,
    BiopsyResponse,
    EvidenceRecord,
    FullCaseRequest,
    FullCaseResponse,
    HealthResponse,
    HonestyGateReport,
    ModelCardsIndex,
    ModelCardSummary,
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


def _score_arbiter(name: str, features: dict[str, Any]) -> ArbiterScore:
    """Load the named L2 arbiter and score `features`.

    Wraps :func:`oncology_arbiter.arbiter.load_arbiter` and marshals the
    :class:`ArbiterResult` into the wire-level :class:`ArbiterScore` pydantic.
    """
    from oncology_arbiter.arbiter import load_arbiter
    arb = load_arbiter(name)
    r = arb.score(features)
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
    )


# --------------------------------------------------------------------------- #
# Artifact paths — mirrors progression_arbiter/router.py stream_artifact

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOTS: dict[ArtifactCategory, Path] = {
    ArtifactCategory.docs: _PROJECT_ROOT / "docs" / "model_cards",
    ArtifactCategory.reports: _PROJECT_ROOT / "artifacts" / "reports",
    ArtifactCategory.data: _PROJECT_ROOT / "artifacts" / "data",
    ArtifactCategory.models: _PROJECT_ROOT / "src" / "oncology_arbiter" / "arbiter" / "models",
}


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
                "GET  /v1/model-cards",
                "GET  /v1/artifacts/{category}/{filename}",
                "GET  /health",
            ],
            models_loaded={
                "monai_screening": ModelState.PLACEHOLDER,
                "medsiglip_biopsy": ModelState.PLACEHOLDER,
                "txgemma_therapy": ModelState.PLACEHOLDER,
                "co_scientist": ModelState.PLACEHOLDER,
                # The L3 arbiter templates ARE loaded (they're JSON on disk),
                # but they carry n_training=0 → treated as placeholder at the
                # health-check level per the PLAN.md honesty rules.
                "l3_arbiter": ModelState.PLACEHOLDER,
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

        # Score the L3 screening arbiter with a MINIMAL feature vector.
        # We have no BI-RADS from a real reader in Phase 1 — the classifier
        # isn't wired — so we deliberately submit the empty feature dict,
        # which by design falls through to intercept-only (i.e. base rate).
        # This proves the arbiter path is live end-to-end without pretending
        # to have information we don't. Phase 2 replaces the empty dict with
        # features extracted from L4a detector output.
        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("screening", features={})
        except Exception as e:
            log_event(request_id, "/v1/screening/analyze",
                      model_state="unavailable",
                      patient_id_hash=req.patient_id_hash,
                      extra={"arbiter_error": str(e)[:200]})
            arbiter_block = None

        response = ScreeningResponse(
            **_envelope(request_id),
            laterality=result.metadata.laterality.value,
            view=result.metadata.view.value,
            orientation_flipped=result.metadata.orientation_flipped,
            breast_mask_coverage=float(result.breast_mask.mean()),
            findings=[],
            overall_score=None,   # placeholder — no classifier wired yet
            arbiter_score=arbiter_block,
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
        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("biopsy", features={})
        except Exception:
            arbiter_block = None
        return BiopsyResponse(
            **_envelope(request_id),
            subtype_prediction=None,
            receptor_panel=BiopsyReceptorPanel(),
            grade=None,
            confidence=None,
            arbiter_score=arbiter_block,
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
        arbiter_block: ArbiterScore | None = None
        try:
            arbiter_block = _score_arbiter("therapy", features={})
        except Exception:
            arbiter_block = None
        return TherapyResponse(
            **_envelope(request_id),
            recommended_options=[],
            not_recommended=[],
            arbiter_score=arbiter_block,
        )

    # ----------------------------------------------------------------------- #
    # /v1/model-cards — index every model card shipped with the API

    @app.get("/v1/model-cards", response_model=ModelCardsIndex)
    def list_model_cards() -> ModelCardsIndex:
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

    @app.get("/v1/artifacts/{category}/{filename}")
    def stream_artifact(category: str, filename: str):
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
