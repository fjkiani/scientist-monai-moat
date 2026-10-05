"""Pathology report parsing orchestration.

Primary path: MedGemma structured JSON extraction (72h delivery).
Fallback: ClinicalBERT NER (only if ONCOLOGY_FORCE_CLINICALBERT_NER=1).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from ..schemas import BiopsyReceptorPanel, ExtendedReceptorField, ReportParseBlock
from ..services.helpers import failed_stage_receipt


class PathologyParseOrchestrator:
    """Orchestrates pathology report parsing with MedGemma primary and ClinicalBERT fallback."""

    @staticmethod
    def parse_biopsy_report(
        report_text: str, *, request_id: str
    ) -> tuple[ReportParseBlock | None, BiopsyReceptorPanel, int | None, dict[str, Any]]:
        """Parse biopsy report text.

        Returns:
            (parse_block, receptor_panel, grade, receipt)

        Primary: MedGemma structured JSON (unless ONCOLOGY_FORCE_CLINICALBERT_NER=1)
        Fallback: ClinicalBERT NER v0.5.2 sliding window
        """
        # Check environment override for ClinicalBERT
        force_clinicalbert = os.environ.get("ONCOLOGY_FORCE_CLINICALBERT_NER", "").strip() in {
            "1",
            "true",
            "TRUE",
            "yes",
        }

        if force_clinicalbert:
            return PathologyParseOrchestrator._parse_clinicalbert(report_text, request_id=request_id)
        return PathologyParseOrchestrator._parse_medgemma(report_text, request_id=request_id)

    @staticmethod
    def _parse_medgemma(
        report_text: str, *, request_id: str
    ) -> tuple[ReportParseBlock | None, BiopsyReceptorPanel, int | None, dict[str, Any]]:
        """Primary 72h path: MedGemma structured JSON extraction (RUO)."""
        from oncology_arbiter.nlp.medgemma_pathology_client import (
            MedGemmaPathologyError,
            extract_pathology,
        )

        input_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        try:
            response = extract_pathology(report_text)
            extraction = response.get("extraction") or {}
            er_status = str(extraction.get("er_status") or "UNKNOWN").upper()
            pr_status = str(extraction.get("pr_status") or "UNKNOWN").upper()
            her2_status_raw = str(extraction.get("her2_status") or "UNKNOWN").upper()
            her2_mapped = None
            if her2_status_raw in {"POSITIVE", "3+", "AMPLIFIED"}:
                her2_mapped = "positive"
            elif her2_status_raw in {"NEGATIVE", "0", "1+", "NON_AMPLIFIED"}:
                her2_mapped = "negative"
            elif her2_status_raw in {"EQUIVOCAL", "2+"}:
                her2_mapped = "equivocal"
            panel = BiopsyReceptorPanel(
                er_positive=True if er_status == "POSITIVE" else False if er_status == "NEGATIVE" else None,
                pr_positive=True if pr_status == "POSITIVE" else False if pr_status == "NEGATIVE" else None,
                her2_status=her2_mapped,
                ki67_percent=None,
                parse_state={
                    "er": "matched" if er_status != "UNKNOWN" else "no_match",
                    "pr": "matched" if pr_status != "UNKNOWN" else "no_match",
                    "her2": "matched" if her2_mapped is not None else "no_match",
                    "grade": "matched" if extraction.get("nottingham_grade") in {1, 2, 3} else "no_match",
                },
            )
            extended: dict[str, ExtendedReceptorField] = {}
            if extraction.get("tumor_size_mm") is not None:
                extended["tumor_size_mm"] = ExtendedReceptorField(
                    value=extraction.get("tumor_size_mm"),
                    match_state="matched",
                    matched_text=None,
                    confidence=0.0,
                    source="medgemma_pathology",
                )
            if extraction.get("er_percent") is not None:
                extended["er_percent"] = ExtendedReceptorField(
                    value=extraction.get("er_percent"),
                    match_state="matched",
                    matched_text=None,
                    confidence=0.0,
                    source="medgemma_pathology",
                )
            parsed_entities = {
                "ER_VALUE": {
                    "value": er_status.lower() if er_status != "UNKNOWN" else None,
                    "surface": er_status,
                },
                "PR_VALUE": {
                    "value": pr_status.lower() if pr_status != "UNKNOWN" else None,
                    "surface": pr_status,
                },
                "HER2_VALUE": {"value": her2_mapped, "surface": her2_status_raw},
                "GRADE": {
                    "value": extraction.get("nottingham_grade"),
                    "surface": extraction.get("nottingham_grade"),
                },
                "TUMOR_SIZE_MM": {
                    "value": extraction.get("tumor_size_mm"),
                    "surface": extraction.get("tumor_size_mm"),
                },
            }
            block = ReportParseBlock(
                parser_id="medgemma_1_5_4b_pathology_json_v1",
                fusion_mode="medgemma_pathology",
                per_field_confidence={},
                per_field_source={
                    "er": "medgemma_pathology" if er_status != "UNKNOWN" else "none",
                    "pr": "medgemma_pathology" if pr_status != "UNKNOWN" else "none",
                    "her2": "medgemma_pathology" if her2_mapped is not None else "none",
                    "grade": (
                        "medgemma_pathology"
                        if extraction.get("nottingham_grade") in {1, 2, 3}
                        else "none"
                    ),
                },
                extended_fields=extended,
                parsed_entities=parsed_entities,
                n_tokens=None,
                n_windows=None,
                window_tokens=None,
                overlap_tokens=None,
                window_aggregation=None,
                app_version="medgemma-pathology-json-v1",
                model_sha256=response.get("revision_sha256"),
                metrics_sha256=None,
            )
            receipt = {
                "stage": "medgemma_pathology_parse",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "medgemma",
                "endpoint_label": "medgemma-1-5-4b-chat",
                "app_version": "medgemma-pathology-json-v1",
                "model_name": response.get("model_id"),
                "model_version": response.get("revision_sha256"),
                "artifact_sha256": response.get("revision_sha256"),
                "input_reference": f"sha256:{input_sha}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "warnings": [str(response.get("disclaimer") or "")],
                "error": None,
                "honesty": "72h cannibalization: ClinicalBERT NER is not primary; MedGemma structured JSON is.",
            }
            grade = (
                extraction.get("nottingham_grade")
                if extraction.get("nottingham_grade") in {1, 2, 3}
                else None
            )
            return block, panel, grade, receipt
        except MedGemmaPathologyError as exc:
            return None, BiopsyReceptorPanel(), None, failed_stage_receipt(
                "medgemma_pathology_parse",
                True,
                request_id,
                "medgemma_pathology_failed",
                f"{type(exc).__name__}: {exc}",
                service_name="medgemma",
                input_reference=f"sha256:{input_sha}",
            )
        except Exception as exc:  # noqa: BLE001
            return None, BiopsyReceptorPanel(), None, failed_stage_receipt(
                "medgemma_pathology_parse",
                True,
                request_id,
                "medgemma_pathology_failed",
                f"{type(exc).__name__}: {exc}",
                service_name="medgemma",
                input_reference=f"sha256:{input_sha}",
            )

    @staticmethod
    def _parse_clinicalbert(
        report_text: str, *, request_id: str
    ) -> tuple[ReportParseBlock | None, BiopsyReceptorPanel, int | None, dict[str, Any]]:
        """Legacy ClinicalBERT NER (v0.5.2 sliding window).

        Only runs when ONCOLOGY_FORCE_CLINICALBERT_NER=1.
        Otherwise delegates to MedGemma (primary path).
        """
        from oncology_arbiter.nlp.clinicalbert_modal_client import (
            ClinicalBertModalClient,
            ClinicalBertModalError,
        )
        from oncology_arbiter.nlp.parser_acceptance_gate import (
            DEPLOYED_PARSER_METRICS,
            declared_deployment_info,
            evaluate_parser_acceptance,
        )

        input_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        started = time.perf_counter()

        # ---- Directive 3, layer 1: PRE-FLIGHT refusal -----------------------------
        # Refuse before the wire call, not after. Two reasons this ordering matters:
        #   (a) a transport error would otherwise pre-empt the gate, so the refusal
        #       reason would depend on network weather rather than on model fitness;
        #   (b) an ineligible parser must never receive patient report text at all.
        preflight = evaluate_parser_acceptance(declared_deployment_info())
        if not preflight.accepted:
            return None, BiopsyReceptorPanel(), None, failed_stage_receipt(
                "clinicalbert_pathology_parse",
                True,
                request_id,
                preflight.error_code,
                preflight.detail,
                service_name="clinicalbert",
                input_reference=f"sha256:{input_sha}",
            )

        try:
            client = ClinicalBertModalClient()
            response = client.parse(report_text)

            # ---- Directive 3, layer 2: post-parse re-validation ------------------
            # Defence in depth: the pre-flight trusts an operator-declared version.
            # This layer trusts only what the deployment actually said on the wire,
            # so a mis-declared env var cannot open the gate.
            # The parse call may have succeeded at the transport layer. That is not
            # evidence of clinical fitness. Refuse HERE -- after the wire call but
            # strictly before `parsed` is read into any receptor, grade, DSS,
            # subtype or therapy field -- so no parser-derived value can reach
            # downstream clinical logic. Measured motivation: the live v0.5.1
            # deployment returned HER2_VALUE="positive" for a documented HER2 1+
            # NEGATIVE report and KI67_PCT=67 for a documented 20%.
            observed_version = response.get("app_version")
            acceptance = evaluate_parser_acceptance(
                {
                    "app_version": observed_version,
                    **DEPLOYED_PARSER_METRICS.get(str(observed_version), {}),
                    **{
                        k: response[k]
                        for k in (
                            "test_span_micro_f1",
                            "test_micro_f1",
                            "micro_f1",
                        )
                        if k in response
                    },
                }
            )
            if not acceptance.accepted:
                return None, BiopsyReceptorPanel(), None, failed_stage_receipt(
                    "clinicalbert_pathology_parse",
                    True,
                    request_id,
                    acceptance.error_code,
                    acceptance.detail,
                    service_name="clinicalbert",
                    input_reference=f"sha256:{input_sha}",
                )

            parsed = response.get("parsed") or {}
            if not isinstance(parsed, dict):
                raise ClinicalBertModalError("clinicalbert_contract_mismatch:parsed")

            def value(entity: str) -> Any:
                item = parsed.get(entity)
                return item.get("value") if isinstance(item, dict) else None

            er_value, pr_value = value("ER_VALUE"), value("PR_VALUE")
            her2_value, grade_value, ki67_value = (
                value("HER2_VALUE"),
                value("GRADE"),
                value("KI67_PCT"),
            )
            panel = BiopsyReceptorPanel(
                er_positive=True if er_value == "positive" else False if er_value == "negative" else None,
                pr_positive=True if pr_value == "positive" else False if pr_value == "negative" else None,
                her2_status=(
                    her2_value
                    if her2_value in {"positive", "negative", "equivocal"}
                    else None
                ),
                ki67_percent=(
                    float(ki67_value)
                    if isinstance(ki67_value, (int, float)) and 0 <= float(ki67_value) <= 100
                    else None
                ),
                parse_state={
                    "er": "matched" if "ER_VALUE" in parsed else "no_match",
                    "pr": "matched" if "PR_VALUE" in parsed else "no_match",
                    "her2": "matched" if "HER2_VALUE" in parsed else "no_match",
                    "grade": "matched" if "GRADE" in parsed else "no_match",
                },
            )
            extended: dict[str, ExtendedReceptorField] = {}
            for entity, output_name in {
                "KI67_PCT": "ki67_pct",
                "TUMOR_SIZE_MM": "tumor_size_mm",
                "T_STAGE": "t_stage",
                "N_STAGE": "n_stage",
                "M_STAGE": "m_stage",
                "MARGIN": "margin",
                "LVI": "lvi",
            }.items():
                item = parsed.get(entity)
                if isinstance(item, dict):
                    extended[output_name] = ExtendedReceptorField(
                        value=item.get("value"),
                        match_state="matched",
                        matched_text=item.get("surface"),
                        confidence=0.0,
                        source="clinicalbert",
                    )
            block = ReportParseBlock(
                parser_id="clinicalbert_v0.5.2_sliding_window",
                fusion_mode="clinicalbert",
                per_field_confidence={},
                per_field_source={
                    key: "clinicalbert" if entity in parsed else "none"
                    for key, entity in {
                        "er": "ER_VALUE",
                        "pr": "PR_VALUE",
                        "her2": "HER2_VALUE",
                        "grade": "GRADE",
                    }.items()
                },
                extended_fields=extended,
                parsed_entities=parsed,
                n_tokens=response.get("n_tokens"),
                n_windows=response.get("n_windows"),
                window_tokens=response.get("window_tokens"),
                overlap_tokens=response.get("overlap_tokens"),
                window_aggregation=response.get("window_aggregation"),
                app_version=response.get("app_version"),
                model_sha256=response.get("model_sha256"),
                metrics_sha256=response.get("metrics_sha256"),
            )
            receipt = {
                "stage": "clinicalbert_pathology_parse",
                "required": True,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "clinicalbert",
                "endpoint_label": client.endpoints.parse,
                "app_version": response.get("app_version"),
                # Report the OBSERVED version, never a hardcoded expectation: a
                # receipt that asserts v0.5.2 while v0.5.1 served the request is a
                # falsified provenance record.
                "model_name": response.get("base_model"),
                "model_version": response.get("app_version"),
                "artifact_sha256": response.get("model_sha256"),
                "input_reference": f"sha256:{input_sha}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "warnings": [str(response.get("disclaimer") or "")],
                "error": None,
            }
            grade = int(grade_value) if grade_value in {1, 2, 3} else None
            return block, panel, grade, receipt
        except Exception as exc:
            return None, BiopsyReceptorPanel(), None, failed_stage_receipt(
                "clinicalbert_pathology_parse",
                True,
                request_id,
                getattr(exc, "code", "clinicalbert_failed"),
                f"{type(exc).__name__}: {exc}",
                service_name="clinicalbert",
                input_reference=f"sha256:{input_sha}",
            )
