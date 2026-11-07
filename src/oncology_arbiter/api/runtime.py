"""Shared API helpers extracted from app.py."""
from __future__ import annotations

import os
import tempfile
import time as _time
from pathlib import Path
from typing import Any

from oncology_arbiter.api.schemas import ArtifactCategory

_MEDSIGLIP_SINGLETON: Any = None


def _get_medsiglip() -> Any:
    """Lazy-construct the MedSigLIP client. Reuses one instance per process.

    Backend is chosen by ``MEDSIGLIP_BACKEND``:
      * ``modal`` → :class:`MedSigLipModalClient` (remote A10G, no local
        weight download, PNG/DICOM in-band via base64)
      * anything else (default ``local``) → local :class:`MedSigLip`

    The factory delegates so the choice can flip without touching this
    endpoint's contract; both backends produce the same
    :class:`MedSigLipResult` shape.
    """
    global _MEDSIGLIP_SINGLETON
    if _MEDSIGLIP_SINGLETON is None:
        from oncology_arbiter.models.medsiglip_modal_client import get_medsiglip_client
        _MEDSIGLIP_SINGLETON = get_medsiglip_client()
    return _MEDSIGLIP_SINGLETON


def _run_medsiglip_on_preprocessed(
    preprocess_result: Any,
    *,
    dicom_bytes: bytes | None = None,
) -> Any:
    """Run MedSigLIP on an already-preprocessed mammogram.

    Two paths depending on backend:

    * **Local backend** (:class:`oncology_arbiter.models.medsiglip.MedSigLip`):
      uses ``_preprocess_fn`` injection so the client's PIL conversion runs
      but no second DICOM I/O happens.
    * **Modal backend** (:class:`MedSigLipModalClient`): Modal does its own
      DICOM preprocess remotely — we need the raw DICOM bytes. We write
      them to a temp file and call ``.run(path)``. If ``dicom_bytes`` is
      not supplied, the fallback re-encodes the already-preprocessed
      float32 image to an 8-bit PNG so Modal's ``pixels_b64`` branch can
      handle it (loses some fidelity but preserves the honest path).

    Raises ``GatedAccessError`` on HAI-DEF gate denial. Callers MUST NOT
    catch this and silently fall back; the endpoint decides fallback
    policy explicitly via env flag.

    Phase 2 limitation (2026-07-03): the local branch mutates
    ``ms._preprocess_fn`` on the singleton for the duration of the call.
    That is safe under FastAPI's default single-threaded async request
    handling but is NOT safe under a threadpool. Phase 3 will either
    (a) construct a fresh MedSigLip per request (weights stay cached at
    class level so no re-download) or (b) thread the preprocess through
    the ``.run()`` signature directly.
    """
    ms = _get_medsiglip()
    # Late import avoids circulars and keeps modal client optional at load time.
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    if isinstance(ms, MedSigLipModalClient):
        if dicom_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
                tf.write(dicom_bytes)
                tmp_path = tf.name
            try:
                return ms.run(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        # No raw bytes: re-encode the already-normalised float32 image as
        # a temp PNG so Modal's pixels_b64 branch can consume it.
        import io as _io
        import numpy as _np
        from PIL import Image as _Image

        arr = _np.asarray(preprocess_result.image)
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8)
        buf = _io.BytesIO()
        _Image.fromarray(arr, mode="L").save(buf, format="PNG")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(buf.getvalue())
            tmp_path = tf.name
        try:
            return ms.run(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Local backend path (unchanged): inject the preprocessed image.
    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    ms._preprocess_fn = _inject
    return ms.run("(preprocessed)")


def _run_cbis_ddsm_probe_on_bytes(
    dicom_bytes: bytes,
    preprocess_result: Any = None,
    *,
    embedding: list[float] | None = None,
    dicom_path: str | Path | None = None,
    modal_client: Any | None = None,
) -> Any | None:
    """Run the trained CBIS-DDSM supervised probe on a mammogram.

    Scores ``models/cbis_ddsm_logreg_v2.joblib`` (sha256 ``e5f5b67caf8ba18006e3d582e91becf95fbb20b8b34417366c7ef07b906a211b``) over a
    1152-d MedSigLIP-448 embedding. Embedding sources (first hit wins):

      1. explicit ``embedding`` list (preferred — reuse the screening receipt)
      2. ``modal_client.embed_dicom(dicom_path)`` when both are supplied
      3. a fresh Modal embed of ``dicom_bytes`` via ``MedSigLipModalClient``

    Returns ``None`` when no Modal embedding path is available so callers can
    keep the zero-shot findings without poisoning the response.
    """
    from oncology_arbiter.models.cbis_ddsm_probe import CbisDdsmProbe
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    emb = embedding
    if emb is None and dicom_path is not None:
        client = modal_client if isinstance(modal_client, MedSigLipModalClient) else MedSigLipModalClient()
        emb = client.embed_dicom(dicom_path)
    if emb is None:
        client = modal_client if isinstance(modal_client, MedSigLipModalClient) else None
        if client is None:
            ms = _get_medsiglip()
            if isinstance(ms, MedSigLipModalClient):
                client = ms
            else:
                # Prefer an explicit Modal client over a local encoder that
                # may not expose 1152-d embeddings on this process.
                try:
                    client = MedSigLipModalClient()
                except Exception:
                    return None
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
            tf.write(dicom_bytes)
            tmp_path = tf.name
        try:
            emb = client.embed_dicom(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return CbisDdsmProbe.get().predict(emb)


# --------------------------------------------------------------------------- #
# Artifact paths — mirrors progression_arbiter/router.py stream_artifact

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOTS: dict[ArtifactCategory, Path] = {
    ArtifactCategory.docs: _PROJECT_ROOT / "docs" / "model_cards",
    ArtifactCategory.reports: _PROJECT_ROOT / "artifacts" / "reports",
    ArtifactCategory.data: _PROJECT_ROOT / "artifacts" / "data",
    ArtifactCategory.models: _PROJECT_ROOT / "src" / "oncology_arbiter" / "arbiter" / "models",
}


def _elo_reason_for_modifier(
    drug_id: str,
    delta: float,
    disease_context: dict,
) -> str:
    """Best-effort human-readable reason for the applied modifier delta.

    Kept deliberately terse — the SPA renders `rank_delta`, `applied_modifier`
    and `reason` side-by-side, so this text is a *hint*, not the source of
    truth.
    """
    if delta == 0.0:
        return "no modifier"

    d_low = drug_id.lower()
    ctx = disease_context or {}
    hrd = bool(ctx.get("hrd_positive")) or bool(ctx.get("brca_mutated"))
    pdl1 = ctx.get("pd_l1_cps")
    prior_lines = ctx.get("prior_lines")

    parp_names = ("olaparib", "niraparib", "rucaparib", "talazoparib")
    if any(n in d_low for n in parp_names) and hrd and delta > 0:
        return f"HRD+PARP boost ({delta:+.2f})"

    if "bevacizumab" in d_low and delta > 0:
        return f"Bevacizumab GOG-0218 posture ({delta:+.2f})"

    if any(n in d_low for n in ("pembrolizumab", "atezolizumab", "durvalumab", "nivolumab")):
        try:
            cps = float(pdl1) if pdl1 is not None else None
        except (TypeError, ValueError):
            cps = None
        if delta > 0 and cps is not None and cps >= 10:
            return f"PD-L1 CPS>=10 boost ({delta:+.2f})"
        if delta < 0 and cps is not None and cps < 10:
            return f"PD-L1 CPS<10 penalty ({delta:+.2f})"

    try:
        pl = int(prior_lines) if prior_lines is not None else None
    except (TypeError, ValueError):
        pl = None
    if delta < 0 and pl is not None and pl >= 3:
        return f"prior_lines={pl} penalty ({delta:+.2f})"

    if "ceralasertib" in d_low and delta != 0:
        return f"CAPRI ATR/PARP posture ({delta:+.2f})"

    if delta > 0:
        return f"manual boost ({delta:+.2f})"
    return f"manual penalty ({delta:+.2f})"


def _run_clinicalbert_parse(
    request_id: str,
    tenant_id: str | None,
    report_text: str | None,
    log_event_fn: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Dispatch the fine-tuned ClinicalBERT report parser.

    Pure report -> entities transform. Independent of CT / series_dir, so
    both the placeholder and the real-pipeline NSCLC branches call this to
    stamp ``parsed_report`` + ``parsed_report_provenance`` on their
    response. Failure modes:
      * backend unset or report_text empty  -> (None, None)   (silent skip)
      * modal / local client raises         -> (None, {source, error})
      * successful parse                    -> (parsed, provenance-with-source)
    Never fabricates a parse; the provenance dict always carries
    ``source`` (``clinicalbert_modal`` or ``clinicalbert_local``).
    """
    if not report_text:
        return None, None
    cbert_backend = os.environ.get("CLINICALBERT_BACKEND", "").lower()
    if cbert_backend not in ("modal", "local"):
        return None, None

    import time as _time  # local rebind for wall-clock timing
    t_cb0 = _time.perf_counter()
    parsed_dict: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    if cbert_backend == "modal":
        try:
            from oncology_arbiter.nlp.clinicalbert_modal_client import (
                ClinicalBertModalClient,
            )
            client = ClinicalBertModalClient()
            resp = client.parse(report_text)
            parsed_dict = resp.get("parsed") or {}
            provenance = {
                "provenance": resp.get("provenance"),
                "base_model": resp.get("base_model"),
                "training_seed": resp.get("training_seed"),
                "test_span_micro_f1": resp.get("test_span_micro_f1"),
                "test_micro_f1": resp.get("test_micro_f1"),
                "app_version": resp.get("app_version"),
                "n_tokens": resp.get("n_tokens"),
                "seconds": resp.get("seconds"),
                "n_entity_types": len(parsed_dict),
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
                "source": "clinicalbert_modal",
            }
        except Exception as exc:  # pylint: disable=broad-except
            try:
                log_event_fn(
                    request_id, "/v1/case/full",
                    model_state="clinicalbert_modal_error",
                    patient_id_hash=None,
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                    tenant_id=tenant_id,
                )
            except Exception:
                pass
            provenance = {
                "source": "clinicalbert_modal",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
            }
    else:  # local
        try:
            from oncology_arbiter.nlp.clinicalbert_local_client import (
                ClinicalBertLocalClient,
            )
            client = ClinicalBertLocalClient()
            resp = client.parse(report_text)
            parsed_dict = resp.get("parsed") or {}
            provenance = {
                "provenance": resp.get("provenance"),
                "base_model": resp.get("base_model"),
                "training_seed": resp.get("training_seed"),
                "test_span_micro_f1": resp.get("test_span_micro_f1"),
                "test_micro_f1": resp.get("test_micro_f1"),
                "app_version": resp.get("app_version"),
                "n_tokens": resp.get("n_tokens"),
                "seconds": resp.get("seconds"),
                "n_entity_types": len(parsed_dict),
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
                "source": "clinicalbert_local",
            }
        except Exception as exc:  # pylint: disable=broad-except
            try:
                log_event_fn(
                    request_id, "/v1/case/full",
                    model_state="clinicalbert_local_error",
                    patient_id_hash=None,
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                    tenant_id=tenant_id,
                )
            except Exception:
                pass
            provenance = {
                "source": "clinicalbert_local",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(_time.perf_counter() - t_cb0, 3),
            }
    return parsed_dict, provenance
