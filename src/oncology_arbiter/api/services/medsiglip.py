"""MedSigLIP runtime singleton + preprocessed-image runner."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from ..schemas import ModelState

_MEDSIGLIP_SINGLETON: Any = None


def get_medsiglip() -> Any:
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


def run_medsiglip_on_preprocessed(
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
    ms = get_medsiglip()
    # Late import avoids circulars and keeps modal client optional at load time.
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    if isinstance(ms, MedSigLipModalClient):
        from oncology_arbiter.models.medsiglip import (
            DEFAULT_ZERO_SHOT_LABELS,
            MEDSIGLIP_INPUT_RES,
            MEDSIGLIP_MAMMOGRAPHY_WARNING,
            MEDSIGLIP_REPO,
            MedSigLipResult,
        )
        import numpy as _np

        def _modal_run(path: str) -> MedSigLipResult:
            gate = ms.preflight()
            zs = ms.zero_shot_raw(path, labels=DEFAULT_ZERO_SHOT_LABELS)
            if zs.get("error"):
                raise RuntimeError(f"medsiglip_zero_shot_error:{zs.get('error')}")
            labels = zs.get("prompts") or []
            probs = zs.get("probs") or []
            if (
                not isinstance(labels, list)
                or not isinstance(probs, list)
                or not labels
                or len(labels) != len(probs)
            ):
                raise RuntimeError("medsiglip_contract_mismatch:zero_shot_shape")
            probs_f = [float(x) for x in probs]
            top_idx = max(range(len(probs_f)), key=lambda i: probs_f[i])
            emb = ms.embed_dicom(path)
            emb_arr = _np.asarray(emb, dtype=_np.float32)
            if emb_arr.shape != (1152,) or not _np.isfinite(emb_arr).all():
                raise RuntimeError("medsiglip_contract_mismatch:embed_vector")
            emb_sha = hashlib.sha256(emb_arr.tobytes()).hexdigest()
            return MedSigLipResult(
                source_path=str(path),
                labels=[str(x) for x in labels],
                probs=probs_f,
                top_label=str(labels[top_idx]),
                top_prob=float(probs_f[top_idx]),
                model_repo=MEDSIGLIP_REPO,
                model_state=ModelState.LOADED_MEDSIGLIP,
                input_resolution=MEDSIGLIP_INPUT_RES,
                logits_shape=(1, len(labels)),
                warnings=[MEDSIGLIP_MAMMOGRAPHY_WARNING],
                app_version=str(zs.get("app_version") or ""),
                inference_seconds=float(zs.get("seconds") or 0.0),
                prompts=[str(x) for x in labels],
                embedding_dim=int(emb_arr.shape[0]),
                embedding_sha256=emb_sha,
                gate_report=gate,
            )

        if dicom_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
                tf.write(dicom_bytes)
                tmp_path = tf.name
            try:
                return _modal_run(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        # No raw bytes: re-encode the already-normalised float32 image as
        # a temp PNG so Modal's pixels_b64 branch can consume it.
        import io as _io
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
            return _modal_run(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Local backend path (unchanged): inject the preprocessed image.
    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    ms._preprocess_fn = _inject
    return ms.run("(preprocessed)")



class MedSigLipRuntime:
    """Class facade over the module-level MedSigLIP helpers."""

    get_client = staticmethod(get_medsiglip)
    run_on_preprocessed = staticmethod(run_medsiglip_on_preprocessed)
