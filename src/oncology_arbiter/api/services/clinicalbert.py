"""ClinicalBERT report parse dispatcher (modal/local)."""
from __future__ import annotations

import os
from typing import Any

def run_clinicalbert_parse(
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



class ClinicalBertParseService:
    """Class facade for ClinicalBERT parse dispatch."""

    parse = staticmethod(run_clinicalbert_parse)
