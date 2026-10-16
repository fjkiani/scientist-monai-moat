"""Tumor board assembly: DSS, ovarian retirement, MedGemma co-scientist, dynamic board."""
from __future__ import annotations

import os
from typing import Any

from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.api.schemas import (
    ArbiterScore,
    BreastDssPrognosis,
    DynamicTumorBoardRequest,
    DynamicTumorBoardResponse,
    ModelState,
    ReportParseBlock,
)
from oncology_arbiter.api import deps
from oncology_arbiter.api.services.helpers import (
    failed_stage_receipt,
    skipped_stage_receipt,
    pipeline_status,
)

_clinicalbert_biopsy_parse = PathologyParseOrchestrator.parse_biopsy_report
_envelope = deps.envelope
_decode_bytes_arg = deps.decode_bytes_arg


def _score_explicit_therapy_triage(raw: dict[str, Any] | None) -> ArbiterScore | None:
    """Delegate to the AST-audited choke point on app (keeps Manski single path)."""
    from oncology_arbiter.api import app as app_module
    return app_module._score_explicit_therapy_triage(raw)


SUPPORTED_CANCERS = {"breast", "nsclc", "hgsoc"}


class TumorBoardAssembler:
    """Assemble specialist-stack tumor board responses with stage receipts."""

    SUPPORTED_CANCERS = SUPPORTED_CANCERS

    @staticmethod
    def breast_dss_from_features(features: Any) -> BreastDssPrognosis:
        from oncology_arbiter.models.breast_dss_arbiter import score_breast_dss

        result = score_breast_dss({
            "age": features.age,
            "tumor_size_mm": features.tumor_size_mm,
            "nodes_positive": features.nodes_positive,
            "grade": features.grade,
            "er_pos": features.er_positive,
            "pr_pos": features.pr_positive,
            "her2_pos": features.her2_positive,
        })
        return BreastDssPrognosis(
            model_name=result.model_name,
            disease_specific_mortality_score=result.score,
            logit=result.logit,
            term_contributions=result.term_contributions,
            artifact_sha256=result.artifact_sha256,
            n_training=result.n_training,
            events=result.events,
            oof_auroc=result.oof_auroc,
            caveats=list(result.caveats),
        )

    @staticmethod
    def ovarian_retirement(request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        block = {
            "model_name": "ovarian_mortality_arbiter_v3_tcga",
            "model_state": "retired",
            "endpoint": "2-year overall mortality",
            "retirement_reason": (
                "FIGO stage did not demonstrate incremental discrimination over age; "
                "the paired OOF AUROC increment was +0.02475247524752467 with an "
                "approximate 95% interval spanning zero."
            ),
            "retirement_evidence": {
                "cohort": "TCGA-OV",
                "n": 271,
                "age_only_oof_auroc": 0.658057110058832,
                "age_plus_figo_oof_auroc": 0.6828095853063567,
                "delta_oof_auroc": 0.02475247524752467,
                "delta_ci95_approx": [-0.041, 0.095],
                "external_validation": "no_feature_compatible_external_cohort_available",
            },
            "warning": "Retired research model; no patient probability, risk bucket, or driving feature is emitted.",
        }
        receipt = {
            "stage": "ovarian_prognostic_arbiter",
            "required": False,
            "status": "retired",
            "request_id": request_id,
            "service_name": "oncology-arbiter",
            "model_name": "ovarian_mortality_arbiter_v3_tcga",
            "model_version": "v3-retired",
            "warnings": ["retired_no_incremental_figo_discrimination_over_age"],
            "error": None,
        }
        return block, receipt

    @staticmethod
    def run_medgemma_co_scientist(
        *,
        context: dict[str, Any],
        request_id: str,
        requested: bool,
        has_patient_signal: bool,
        receipts: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not requested:
            receipts.append(skipped_stage_receipt(
                "medgemma_co_scientist", request_id, "Co-Scientist was not requested"
            ))
            return None
        if not has_patient_signal:
            receipts.append(skipped_stage_receipt(
                "medgemma_co_scientist", request_id,
                "No validated patient stage output was available for synthesis",
            ))
            return None
        if not os.environ.get("MEDGEMMA_MODAL_URL"):
            receipts.append(failed_stage_receipt(
                "medgemma_co_scientist", False, request_id,
                "medgemma_not_configured",
                "MEDGEMMA_MODAL_URL is required; alternate LLMs and deterministic Elo are not substitutes.",
                service_name="medgemma-1.5-4b",
            ))
            return None
        try:
            from oncology_arbiter.agents.supervisor import execute_stage
            from oncology_arbiter.models.medgemma_1_5_4b_wiring import (
                verify_artifact_identity,
            )

            identity = verify_artifact_identity()
            result = execute_stage(
                "case_full",
                context,
                n_hypotheses=6,
                n_evidence_top_k=3,
            )
            output = result.as_dict()
            if (
                result.model_state != "executed"
                or output.get("llm_provider") != "medgemma_modal"
                or output.get("llm_model") != identity["model_id"]
                or output.get("llm_app_version") != identity["app_version"]
                or not output.get("llm_request_ids")
            ):
                raise RuntimeError(
                    "MedGemma Co-Scientist did not complete with required provider provenance"
                )
            receipts.append({
                "stage": "medgemma_co_scientist",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "medgemma-1.5-4b",
                "endpoint_label": os.environ.get("MEDGEMMA_MODAL_URL"),
                "app_version": output["llm_app_version"],
                "model_name": output["llm_model"],
                "model_version": "1.5-4b-it",
                "artifact_sha256": identity["revision_sha256"],
                "latency_ms": round(float(output.get("llm_latency_s") or 0.0) * 1000.0, 3),
                "warnings": list(output.get("llm_honesty_warnings") or []),
                "error": None,
            })
            return output
        except Exception as exc:
            receipts.append(failed_stage_receipt(
                "medgemma_co_scientist", False, request_id,
                "medgemma_unavailable", f"{type(exc).__name__}: {exc}",
                service_name="medgemma-1.5-4b",
            ))
            return None

    @classmethod
    def assemble_dynamic(
        cls,
        req: DynamicTumorBoardRequest,
        *,
        request_id: str,
    ) -> DynamicTumorBoardResponse:
        receipts: list[dict[str, Any]] = []
        warnings: list[str] = []
        ovarian_arbiter: dict[str, Any] | None = None
        report_parse: ReportParseBlock | None = None
        phikon_embedding: dict[str, Any] | None = None
        case_manifest: dict[str, Any] | None = None
        dss_prognosis: BreastDssPrognosis | None = None
        therapy_bridge: dict[str, Any] | None = None
        therapy_triage: ArbiterScore | None = None

        if req.cancer == "hgsoc":
            ovarian_arbiter, retired_receipt = cls.ovarian_retirement(request_id)
            receipts.append(retired_receipt)
        else:
            receipts.append(skipped_stage_receipt(
                "ovarian_prognostic_arbiter", request_id, "not applicable to breast cancer"
            ))

        if req.case_id:
            try:
                from oncology_arbiter.models.specialist_clients import CaseStorageClient

                call = CaseStorageClient().manifest(req.case_id, request_id=request_id, required=True)
                case_manifest = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(failed_stage_receipt(
                    "case_storage", True, request_id,
                    getattr(exc, "code", "case_storage_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="case-storage", input_reference=f"case_id:{req.case_id}",
                ))
        else:
            receipts.append(skipped_stage_receipt(
                "case_storage", request_id, "no case_id supplied"
            ))

        if req.report_text:
            report_parse, _panel, _grade, receipt = _clinicalbert_biopsy_parse(
                req.report_text, request_id=request_id
            )
            receipts.append(receipt)
        else:
            receipts.append(skipped_stage_receipt(
                "clinicalbert_pathology_parse", request_id, "no report_text supplied"
            ))

        if req.pathology_image_b64:
            try:
                from oncology_arbiter.models.specialist_clients import PhikonClient

                image_bytes = _decode_bytes_arg(req.pathology_image_b64) or b""
                call = PhikonClient().embed(image_bytes, request_id=request_id, required=True)
                phikon_embedding = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(failed_stage_receipt(
                    "phikon_embedding", True, request_id,
                    getattr(exc, "code", "phikon_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="phikon",
                ))
        else:
            receipts.append(skipped_stage_receipt(
                "phikon_embedding", request_id, "no pathology image supplied"
            ))

        if req.cancer == "breast" and req.dss_features is not None:
            dss_prognosis = cls.breast_dss_from_features(req.dss_features)
            receipts.append({
                "stage": "breast_dss_prognosis",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "oncology-arbiter",
                "model_name": dss_prognosis.model_name,
                "model_version": "v3",
                "artifact_sha256": dss_prognosis.artifact_sha256,
                "input_reference": "explicit_complete_seven_feature_vector",
                "latency_ms": 0.0,
                "warnings": ["prognosis_under_metabric_treatment_mix_not_treatment_benefit"],
                "error": None,
            })
        else:
            reason = (
                "not applicable to HGSOC"
                if req.cancer == "hgsoc"
                else "complete explicit seven-feature vector not supplied"
            )
            receipts.append(skipped_stage_receipt(
                "breast_dss_prognosis", request_id, reason
            ))

        mutation_payload = [item.model_dump(exclude_none=True) for item in req.mutations]
        germline_payload = [item.model_dump(exclude_none=True) for item in req.germline_mutations]
        ranked_payload = [item.model_dump(exclude_none=True) for item in req.ranked_drugs]
        if mutation_payload or germline_payload:
            try:
                from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient

                call = SLTherapyBridgeClient().run({
                    "disease": "high-grade serous ovarian cancer" if req.cancer == "hgsoc" else "breast cancer",
                    "cancer_type": req.cancer,
                    "mutations": mutation_payload,
                    "germline_mutations": germline_payload,
                    "ranked_drugs": ranked_payload,
                    "rs_features": req.clinical_baseline,
                    "include_explanations": False,
                }, request_id=request_id, required=True)
                therapy_bridge = call.output
                receipts.append(call.receipt)
            except Exception as exc:
                receipts.append(failed_stage_receipt(
                    "synthetic_lethality_therapy_bridge", True, request_id,
                    getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                    service_name="crispro-backend-v2",
                ))
        else:
            receipts.append(skipped_stage_receipt(
                "synthetic_lethality_therapy_bridge", request_id,
                "no patient mutation profile supplied; no therapy recommendation computed",
            ))

        if req.cancer == "breast" and req.therapy_features is not None:
            therapy_triage = _score_explicit_therapy_triage(
                req.therapy_features.model_dump(),
            )
            if therapy_triage is not None:
                from oncology_arbiter.arbiter.stage_therapy_wiring import (
                    ARTIFACT_SHA256 as THERAPY_ARBITER_SHA256,
                )

                receipts.append({
                    "stage": "therapy_research_triage",
                    "required": False,
                    "status": "succeeded",
                    "request_id": request_id,
                    "service_name": "oncology-arbiter",
                    "model_name": therapy_triage.model_name,
                    "model_version": "v1",
                    "artifact_sha256": THERAPY_ARBITER_SHA256,
                    "input_reference": "explicit_complete_patient_features",
                    "latency_ms": 0.0,
                    "warnings": ["illustrative_research_triage_not_recommendation_or_treatment_benefit"],
                    "error": None,
                })
            else:
                receipts.append(skipped_stage_receipt(
                    "therapy_research_triage", request_id,
                    "complete explicit triage vector required; missing nodes never use unknown=0.5",
                ))
        else:
            receipts.append(skipped_stage_receipt(
                "therapy_research_triage", request_id,
                "not applicable to HGSOC" if req.cancer == "hgsoc" else "no explicit triage features supplied",
            ))

        has_signal = any((
            report_parse is not None,
            phikon_embedding is not None,
            case_manifest is not None,
            dss_prognosis is not None,
            therapy_bridge is not None,
            bool(req.clinical_baseline),
        ))
        co_scientist = cls.run_medgemma_co_scientist(
            context={
                "cancer": req.cancer,
                "clinical_baseline": req.clinical_baseline,
                "report_parse": report_parse.model_dump(mode="json") if report_parse else None,
                "phikon_embedding": phikon_embedding,
                "dss_prognosis": dss_prognosis.model_dump(mode="json") if dss_prognosis else None,
                "therapy_bridge": therapy_bridge,
            },
            request_id=request_id,
            requested=req.run_co_scientist,
            has_patient_signal=has_signal,
            receipts=receipts,
        )

        status = pipeline_status(receipts)
        state = ModelState.UNAVAILABLE if status == "failed_required_stage" else ModelState.LOADED
        env = _envelope(request_id, model_state=state, model_name="specialist-stack-composite")
        env["warnings"] = warnings
        return DynamicTumorBoardResponse(
            **env,
            pipeline_status=status,
            cancer=req.cancer,
            stage_receipts=receipts,
            ovarian_arbiter=ovarian_arbiter,
            report_parse=report_parse,
            phikon_embedding=phikon_embedding,
            case_manifest=case_manifest,
            dss_prognosis=dss_prognosis,
            therapy_bridge=therapy_bridge,
            therapy_triage=therapy_triage,
            co_scientist=co_scientist,
        )


# Module-level aliases for inspect/source gates that previously nested under create_app.
def _ovarian_retirement(request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return TumorBoardAssembler.ovarian_retirement(request_id)


def _run_medgemma_co_scientist(**kwargs: Any) -> dict[str, Any] | None:
    return TumorBoardAssembler.run_medgemma_co_scientist(**kwargs)


def _assemble_dynamic_tumor_board(
    req: DynamicTumorBoardRequest, *, request_id: str
) -> DynamicTumorBoardResponse:
    return TumorBoardAssembler.assemble_dynamic(req, request_id=request_id)


def _breast_dss_from_features(features: Any) -> BreastDssPrognosis:
    return TumorBoardAssembler.breast_dss_from_features(features)
