"""Auto-extracted router: therapy."""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any


from fastapi import APIRouter, Depends, HTTPException

from oncology_arbiter.api.audit import log_event, new_request_id
from oncology_arbiter.auth import APIKey, require_api_key
from oncology_arbiter.api import deps
from oncology_arbiter.api.services import runtime_hooks
from oncology_arbiter.api.services.helpers import (
    failed_stage_receipt as _failed_stage_receipt,
    skipped_stage_receipt as _skipped_stage_receipt,
    pipeline_status as _pipeline_status,
)
from oncology_arbiter.api.orchestrator.pathology_parse import PathologyParseOrchestrator
from oncology_arbiter.arbiter.manski import ManskiGateError
from oncology_arbiter.api.schemas import (
    ModelState, TherapyRequest, TherapyResponse, TherapyOption, ArbiterScore,
    EvidenceRecord, HonestyGateReport, BiopsyReceptorPanel,
)

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


router = APIRouter()

@router.post("/v1/therapy/reason", response_model=TherapyResponse)
def therapy_reason(
    req: TherapyRequest,
    tenant: APIKey = Depends(require_api_key),
) -> TherapyResponse:
    """Therapy reasoning surface.

    Contract (clinical-grade / no stubs):
    - The endpoint never returns an HTTP 200 "placeholder" body.
    - If the caller supplies no usable inputs, we fail closed (HTTP 400).
    - When optional specialists (TxGemma, rules-lite, PubMed) are disabled
      or cannot run, we emit explicit skipped/failed receipts and warnings
      rather than fabricating recommendations.
    """
    request_id = new_request_id()
    receipts: list[dict[str, Any]] = []
    warnings: list[str] = []
    if req.legacy_therapy_benefit_model_name is not None:
        warnings.append("deprecated_prognostic_model_alias_remapped")
    bridge_output: dict[str, Any] | None = None
    therapy_reasoner: dict[str, Any] | None = None
    recommended: list[TherapyOption] = []
    not_recommended: list[TherapyOption] = []
    evidence: list[EvidenceRecord] = []

    # Fail closed on empty input: no placeholders.
    if not (
        req.mutations
        or req.germline_mutations
        or req.biopsy_output is not None
        or req.therapy_features is not None
    ):
        raise HTTPException(
            400,
            "therapy requires at least one of: mutations, germline_mutations, biopsy_output, therapy_features",
        )

    def _extract_stage_tnm() -> str | None:
        """Try to recover a TNM-ish stage string for rules/TxGemma.

        Preference: explicit caller-provided `patient_context.genomic_markers`.
        Fallback: parsed report entities when biopsy_output.report_parse exists.
        """
        markers = (req.patient_context.genomic_markers or {}) if req.patient_context else {}
        for key in ("tnm_stage", "stage_tnm", "stage", "clinical_stage_tnm", "path_stage_tnm"):
            val = markers.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        bo = req.biopsy_output
        parsed = (bo.report_parse.parsed_entities if (bo and bo.report_parse) else {}) or {}
        for key in ("tnm_stage", "stage_tnm", "stage"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Try to compose from T/N/M fields if present.
        t = parsed.get("t_stage") or parsed.get("T_stage") or parsed.get("T")
        n = parsed.get("n_stage") or parsed.get("N_stage") or parsed.get("N")
        m = parsed.get("m_stage") or parsed.get("M_stage") or parsed.get("M")
        if isinstance(t, str) and isinstance(n, str) and isinstance(m, str):
            t, n, m = t.strip(), n.strip(), m.strip()
            if t and n and m and (t.upper().startswith("T") and n.upper().startswith("N") and m.upper().startswith("M")):
                return f"{t.upper()}{n.upper()}{m.upper()}"
        return None

    def _receptor_status_from_panel(panel: BiopsyReceptorPanel | None) -> dict[str, bool] | None:
        if panel is None:
            return None
        if panel.er_positive is None or panel.pr_positive is None or panel.her2_status is None:
            return None
        return {
            "ER": bool(panel.er_positive),
            "PR": bool(panel.pr_positive),
            "HER2": (panel.her2_status == "positive"),
        }

    def _receptor_status_from_therapy_features(raw: dict[str, Any] | None) -> dict[str, bool] | None:
        """Derive receptor booleans from explicit therapy_features when biopsy is missing/failed.

        This makes L4c durable: a caller who provides a complete explicit feature
        vector should not be blocked from rules-lite / TxGemma just because the
        biopsy leg didn't run.
        """
        if raw is None:
            return None
        try:
            from oncology_arbiter.api.schemas import TherapyTriageFeatures
            v = TherapyTriageFeatures.model_validate(raw).model_dump()
        except Exception:
            return None
        if v.get("er_positive") is None or v.get("pr_positive") is None or v.get("her2_positive") is None:
            return None
        return {
            "ER": bool(v["er_positive"]),
            "PR": bool(v["pr_positive"]),
            "HER2": bool(v["her2_positive"]),
        }

    def _grade_from_therapy_features(raw: dict[str, Any] | None) -> int | None:
        if raw is None:
            return None
        try:
            from oncology_arbiter.api.schemas import TherapyTriageFeatures
            v = TherapyTriageFeatures.model_validate(raw).model_dump()
        except Exception:
            return None
        g = v.get("grade")
        return int(g) if g in {1, 2, 3} else None

    # Common structured inputs for therapy specialists.
    biopsy_out = req.biopsy_output
    receptor_panel = req.receptors_override or (biopsy_out.receptor_panel if biopsy_out else None)
    receptor_status = _receptor_status_from_panel(receptor_panel) or _receptor_status_from_therapy_features(req.therapy_features)
    grade = (biopsy_out.grade if (biopsy_out and biopsy_out.grade is not None) else _grade_from_therapy_features(req.therapy_features))
    subtype = biopsy_out.subtype_prediction if biopsy_out else None
    stage_tnm = _extract_stage_tnm()
    menopausal = req.patient_context.menopausal_status if req.patient_context else None
    age = req.patient_context.age if req.patient_context else None

    if req.mutations or req.germline_mutations:
        try:
            from oncology_arbiter.models.specialist_clients import SLTherapyBridgeClient
            payload = {
                "disease": req.disease,
                "cancer_type": req.cancer_type,
                "mutations": req.mutations,
                "germline_mutations": req.germline_mutations,
                "ranked_drugs": req.ranked_drugs,
                "include_explanations": False,
            }
            call = SLTherapyBridgeClient().run(payload, request_id=request_id, required=True)
            bridge_output = call.output
            receipts.append(call.receipt)
            for item in bridge_output.get("sl_indicated_drugs") or []:
                if not isinstance(item, dict) or not item.get("drug_name"):
                    continue
                recommended.append(TherapyOption(
                    regimen=str(item["drug_name"]),
                    line_of_therapy=1,
                    rationale=str(item.get("sl_rationale") or "Synthetic-lethality actionability hypothesis"),
                ))
            for item in bridge_output.get("policy_overlays") or []:
                if not isinstance(item, dict) or item.get("policy_decision") != "DISALLOW":
                    continue
                drug = item.get("drug_name") or item.get("name")
                if drug:
                    not_recommended.append(TherapyOption(
                        regimen=str(drug), line_of_therapy=1,
                        rationale=str(item.get("policy_rationale_short") or "SL bridge policy DISALLOW"),
                    ))
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "synthetic_lethality_therapy_bridge", True, request_id,
                getattr(exc, "code", "sl_bridge_failed"), f"{type(exc).__name__}: {exc}",
                service_name="crispro-backend-v2",
            ))
            warnings.append("sl_bridge_required_stage_failed:no_therapy_recommendations_emitted")
    else:
        receipts.append(_skipped_stage_receipt(
            "synthetic_lethality_therapy_bridge", request_id,
            "no patient mutation profile supplied; no therapy recommendation computed",
        ))

    # TxGemma (optional, gated) — can be local (HAI-DEF) or Modal.
    enable_txgemma = _is_env_true("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA")
    if enable_txgemma:
        if receptor_status is None or grade is None or stage_tnm is None:
            receipts.append(_skipped_stage_receipt(
                "txgemma_therapy_reasoner",
                request_id,
                "txgemma skipped: receptor_status + grade + TNM stage required (supply receptors_override/biopsy_output + patient_context.genomic_markers.tnm_stage)",
            ))
        else:
            try:
                import importlib
                from oncology_arbiter.models.hai_def import GatedAccessError

                tx_modal_mod = importlib.import_module("oncology_arbiter.models.txgemma_modal_client")
                TxGemmaModalClient = tx_modal_mod.TxGemmaModalClient
                get_txgemma_client = tx_modal_mod.get_txgemma_client
                # NOTE: Avoid direct `import oncology_arbiter.models.txgemma_client` at module level
                # so `app.py` stays free of substitute-backend imports. Loaded dynamically here.
                tx_client_mod = importlib.import_module("oncology_arbiter.models.txgemma_client")
                TXGEMMA_HONESTY_WARNING = tx_client_mod.TXGEMMA_HONESTY_WARNING

                tx_client = get_txgemma_client()
                prompt = (
                    "You are a breast oncology treatment planner (RESEARCH USE ONLY).\n"
                    "Given these features, propose therapy options and briefly justify each.\n"
                    f"- Receptor status: ER={receptor_status['ER']} PR={receptor_status['PR']} HER2={receptor_status['HER2']}\n"
                    f"- Grade: {int(grade)}\n"
                    f"- Stage (TNM-ish): {stage_tnm}\n"
                    f"- Age: {age}\n"
                    f"- Menopausal status: {menopausal}\n"
                    f"- Biopsy subtype: {subtype}\n"
                    "Constraints: do not invent trial results; cite guideline sections if known; be explicit about uncertainty.\n"
                )
                t0 = time.perf_counter()
                model_name = None
                if isinstance(tx_client, TxGemmaModalClient):
                    max_new = int(os.environ.get("ONCOLOGY_ARBITER_TXGEMMA_MAX_NEW_TOKENS", "256"))
                    result = tx_client.reason(prompt, max_new_tokens=max_new, temperature=0.0)
                    therapy_reasoner = {
                        "backend": "modal",
                        "device": tx_client.device,
                        "text": result.text,
                        "model_repo": result.model_repo,
                        "app_version": result.app_version,
                        "honesty_warning": result.honesty_warning,
                    }
                    model_name = result.model_repo
                else:
                    # Local client is preflight-first; will raise if gated.
                    _ = tx_client.preflight()
                    result = tx_client.recommend_therapy(
                        receptor_status=receptor_status,
                        grade=int(grade),
                        stage=str(stage_tnm),
                        age=age,
                        menopausal_status=(
                            "postmenopausal" if menopausal == "post"
                            else ("premenopausal" if menopausal in {"pre", "peri"} else "unknown")
                        ) if menopausal else None,
                        subtype=subtype,
                    )
                    therapy_reasoner = {
                        "backend": "local",
                        "recommendations": list(result.recommendations),
                        "reasoning": result.reasoning,
                        "model_repo": result.model_name,
                        "honesty_warning": TXGEMMA_HONESTY_WARNING,
                    }
                    model_name = result.model_name
                latency_ms = (time.perf_counter() - t0) * 1000.0
                receipts.append({
                    "stage": "txgemma_therapy_reasoner",
                    "required": False,
                    "status": "succeeded",
                    "request_id": request_id,
                    "service_name": "txgemma",
                    "model_name": model_name or "google/txgemma-9b-chat",
                    "latency_ms": latency_ms,
                    "warnings": [TXGEMMA_HONESTY_WARNING],
                    "error": None,
                })
                warnings.append(TXGEMMA_HONESTY_WARNING)
            except GatedAccessError as exc:
                receipts.append(_failed_stage_receipt(
                    "txgemma_therapy_reasoner",
                    False,
                    request_id,
                    "txgemma_gated",
                    f"{exc.access_level.value}:{exc.reason}",
                    service_name="txgemma",
                ))
                warnings.append(f"txgemma_gated:{exc.access_level.value}")
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "txgemma_therapy_reasoner",
                    False,
                    request_id,
                    "txgemma_failed",
                    f"{type(exc).__name__}:{exc}",
                    service_name="txgemma",
                ))
                warnings.append(f"txgemma_failed:{type(exc).__name__}")
    else:
        receipts.append(_skipped_stage_receipt(
            "txgemma_therapy_reasoner",
            request_id,
            "txgemma disabled (set ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA=1 to attempt)",
        ))

    # NCCN-lite deterministic rules engine (optional; never silent fallback).
    enable_rules = _is_env_true("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY")
    rules_sha256: str | None = None
    rules_model_id: str | None = None
    branch_id: str | None = None
    if enable_rules:
        if receptor_status is None or grade is None or stage_tnm is None:
            receipts.append(_skipped_stage_receipt(
                "therapy_rules_lite",
                request_id,
                "rules-lite skipped: receptor_status + grade + TNM stage required",
            ))
        else:
            try:
                import importlib
                rules_mod = importlib.import_module("oncology_arbiter.models.therapy_rules_lite")
                THERAPY_RULES_PROXY_WARNING = rules_mod.THERAPY_RULES_PROXY_WARNING
                apply_nccn_lite_rules = rules_mod.apply_nccn_lite_rules
                t0 = time.perf_counter()
                meno = (
                    "postmenopausal" if menopausal == "post"
                    else ("premenopausal" if menopausal in {"pre", "peri"} else "unknown")
                ) if menopausal else None
                rules = apply_nccn_lite_rules(
                    receptor_status=receptor_status,
                    grade=int(grade),
                    stage=str(stage_tnm),
                    age=age,
                    menopausal_status=meno,
                    subtype=subtype,
                    strict=bool(req.strict),
                )
                rules_sha256 = rules.rules_sha256
                rules_model_id = rules.rules_model_id
                branch_id = rules.branch_id
                # Append recommendations; we keep them separate from SL bridge output.
                for opt in rules.recommended_options:
                    recommended.append(TherapyOption(
                        regimen=opt.name,
                        line_of_therapy=1,
                        rationale=f"{opt.rationale} (NCCN section {opt.nccn_section}; {opt.citation_url})",
                        evidence=[],
                    ))
                for opt in rules.not_recommended:
                    not_recommended.append(TherapyOption(
                        regimen=opt.name,
                        line_of_therapy=1,
                        rationale=f"{opt.rationale} (NCCN section {opt.nccn_section}; {opt.citation_url})",
                        evidence=[],
                    ))
                receipts.append({
                    "stage": "therapy_rules_lite",
                    "required": False,
                    "status": "succeeded",
                    "request_id": request_id,
                    "service_name": "oncology-arbiter",
                    "model_name": rules.model_name,
                    "input_reference": f"receptors+grade+stage:{branch_id}",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "warnings": [THERAPY_RULES_PROXY_WARNING],
                    "error": None,
                })
                warnings.append(THERAPY_RULES_PROXY_WARNING)
            except Exception as exc:
                receipts.append(_failed_stage_receipt(
                    "therapy_rules_lite",
                    False,
                    request_id,
                    "rules_lite_failed",
                    f"{type(exc).__name__}:{exc}",
                    service_name="oncology-arbiter",
                ))
                warnings.append(f"rules_lite_failed:{type(exc).__name__}")
    else:
        receipts.append(_skipped_stage_receipt(
            "therapy_rules_lite",
            request_id,
            "rules-lite disabled (set ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1 to enable)",
        ))

    # L2 evidence (PubMed) — optional, off by default for offline determinism.
    enable_l2_evidence = _is_env_true("ONCOLOGY_ARBITER_ENABLE_L2_EVIDENCE")
    if enable_l2_evidence and recommended:
        try:
            from oncology_arbiter.tools.pubmed_search import PubmedSearchTool
            from oncology_arbiter.tools.base import ToolCtx
            t0 = time.perf_counter()
            query = "breast cancer adjuvant therapy"
            if receptor_status is not None:
                query += f" ER {int(receptor_status['ER'])} PR {int(receptor_status['PR'])} HER2 {int(receptor_status['HER2'])}"
            if stage_tnm:
                query += f" {stage_tnm}"
            tool = PubmedSearchTool()
            # NOTE: sync route; executed in a worker thread, so asyncio.run is safe here.
            import asyncio
            ctx_dir = Path(tempfile.gettempdir()) / "oncology_arbiter_tool_runs"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            result = asyncio.run(tool.call(
                {"query": query, "max_results": 3, "sort": "relevance"},
                ToolCtx(artifacts_dir=ctx_dir),
            ))
            if getattr(result, "is_error", False):
                raise RuntimeError(getattr(result, "error_message", "pubmed_error"))
            payload = getattr(result, "content", {}) or {}
            urls: set[str] = set()
            for rec in (payload.get("results") or [])[:3]:
                url = rec.get("url")
                if not url:
                    continue
                urls.add(url)
                quoted = (rec.get("title") or "").strip()
                abstract = (rec.get("abstract") or "").strip()
                if abstract:
                    quoted = f"{quoted}\n\n{abstract[:800]}"
                evidence.append(EvidenceRecord(url=str(url), quoted_text=quoted or "PubMed record", source="pubmed"))
            receipts.append({
                "stage": "l2_pubmed_evidence",
                "required": False,
                "status": "succeeded",
                "request_id": request_id,
                "service_name": "ncbi_eutils",
                "model_name": "pubmed_search",
                "input_reference": f"query_sha256:{hashlib.sha256(query.encode('utf-8')).hexdigest()}",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "warnings": [],
                "error": None,
            })
        except Exception as exc:
            receipts.append(_failed_stage_receipt(
                "l2_pubmed_evidence",
                False,
                request_id,
                "pubmed_failed",
                f"{type(exc).__name__}:{exc}",
                service_name="ncbi_eutils",
            ))
            warnings.append(f"pubmed_failed:{type(exc).__name__}")
    else:
        receipts.append(_skipped_stage_receipt(
            "l2_pubmed_evidence",
            request_id,
            "L2 evidence disabled or no recommendations to ground (set ONCOLOGY_ARBITER_ENABLE_L2_EVIDENCE=1)",
        ))

    triage: ArbiterScore | None = None
    try:
        triage = _score_explicit_therapy_triage(
            req.therapy_features
        )
    except ManskiGateError:
        # Never downgrade a partial-identification block into a soft
        # optional-stage failure: the caller must see the 422 and the
        # interval, not a 200 with therapy_triage=null.
        raise
    except Exception as exc:
        receipts.append(_failed_stage_receipt(
            "therapy_research_triage", False, request_id, "invalid_triage_features",
            f"{type(exc).__name__}: {exc}", service_name="oncology-arbiter",
        ))
    if triage is None and not any(r["stage"] == "therapy_research_triage" for r in receipts):
        receipts.append(_skipped_stage_receipt(
            "therapy_research_triage", request_id,
            "no explicit triage vector supplied; nothing to identify",
        ))
    elif triage is not None:
        from oncology_arbiter.arbiter.stage_therapy_wiring import (
            ARTIFACT_SHA256 as THERAPY_ARBITER_SHA256,
        )

        receipts.append({
            "stage": "therapy_research_triage", "required": False, "status": "succeeded",
            "request_id": request_id, "service_name": "oncology-arbiter",
            "model_name": triage.model_name, "model_version": "v1",
            "artifact_sha256": THERAPY_ARBITER_SHA256,
            "input_reference": (
                "explicit_complete_patient_features" if not triage.provenance_warnings
                else "partially_unobserved_panel"
            ),
            "latency_ms": 0.0,
            "warnings": [
                "illustrative_research_triage_not_recommendation_or_treatment_benefit",
                *triage.provenance_warnings,
            ],
            "error": None,
        })

    status = _pipeline_status(receipts)
    # Choose a primary provenance state for the route.
    if therapy_reasoner is not None:
        state = (
            ModelState.LOADED_TXGEMMA_MODAL
            if therapy_reasoner.get("backend") == "modal"
            else ModelState.LOADED_TXGEMMA
        )
        model_name = str(therapy_reasoner.get("model_repo") or "google/txgemma-9b-chat")
    elif any(r.get("stage") == "therapy_rules_lite" and r.get("status") == "succeeded" for r in receipts):
        state = ModelState.LOADED_RULES_LITE
        model_name = rules_model_id or "nccn-lite-v0"
    elif bridge_output is not None:
        state = ModelState.LOADED_SL_THERAPY_BRIDGE
        model_name = "SLTherapyBridge-v3.1"
    else:
        state = ModelState.UNAVAILABLE
        model_name = None

    env = _envelope(request_id, model_state=state, model_name=model_name)
    env["warnings"] = warnings
    env["evidence"] = evidence
    env["honesty_gate"] = HonestyGateReport(
        seen_urls_count=len({e.url for e in evidence}),
        evidence_kept=len(evidence),
        evidence_dropped=0,
        hypotheses_dropped=0,
    )
    log_event(
        request_id, "/v1/therapy/reason", model_state=state.value,
        patient_id_hash=None,
        extra={"pipeline_status": status, "n_bridge_recommended": len(recommended)},
        tenant_id=tenant.tenant_id,
    )
    return TherapyResponse(
        **env,
        pipeline_status=status,
        stage_receipts=receipts,
        therapy_bridge={**(bridge_output or {}), **({"txgemma": therapy_reasoner} if therapy_reasoner else {})} or None,
        therapy_triage=triage,
        recommended_options=recommended,
        not_recommended=not_recommended,
        prognostic_model_executed=False,
        prognostic_score=None,
        arbiter_score=triage,
        rules_sha256=rules_sha256,
        rules_model_id=rules_model_id,
        branch_id=branch_id,
    )
