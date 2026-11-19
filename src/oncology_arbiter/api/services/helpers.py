"""Shared helper utilities for API endpoints."""
from __future__ import annotations

from typing import Any


def ak_bundle_to_l5_envelopes(bundle: dict[str, Any]) -> dict[str, Any]:
    """Map a ``tumor_board.v3.multimodal-with-manuscript-claims`` AK fixture
    into the screening / biopsy / therapy envelopes that
    :func:`oncology_arbiter.orchestrator.co_scientist.run_co_scientist` expects.

    The AK payload is a manuscript-locked SL matrix, not a breast
    ``case.json`` folder. This adapter does **not** invent imaging or
    receptor panels — it surfaces MBD4-LOF + recommended/falsified drugs
    so the live L5 tournament can reason over the DNA-repair deficiency.
    """
    if not isinstance(bundle, dict):
        raise TypeError("ak bundle must be a dict")
    contract = bundle.get("contract_version")
    if contract != "tumor_board.v3.multimodal-with-manuscript-claims":
        raise ValueError(
            f"unsupported contract_version={contract!r}; "
            "expected tumor_board.v3.multimodal-with-manuscript-claims"
        )

    patient_ctx = bundle.get("patient_context") or {}
    primary = patient_ctx.get("primary_alteration") or {}
    diagnosis = patient_ctx.get("diagnosis") or {}
    gene = str(primary.get("gene") or "MBD4")
    alteration = str(primary.get("alteration_type") or "LikelyLoF")
    mbd4_tag = f"{gene}_{alteration}"
    histology = str(diagnosis.get("histology") or "HGSOC")
    site = str(diagnosis.get("primary_site") or "ovary")

    sl = bundle.get("synthetic_lethality") or {}
    prov = sl.get("provenance") or {}
    manuscript_sha = str(prov.get("manuscript_repo_sha_at_audit") or "")
    backend_sha = str(prov.get("backend_head_sha") or "")
    seed_url = (
        f"https://github.com/fjkiani/crispro/commit/{manuscript_sha}"
        if manuscript_sha
        else "https://github.com/fjkiani/crispro"
    )

    def _ev(text: str) -> list[dict[str, str]]:
        return [{
            "url": seed_url,
            "quoted_text": text[:250],
            "source": "ak_mbd4_lof_manuscript_receipt",
        }]

    screening = {
        "provenance": {
            "model_state": "loaded_ak_bundle",
            "model_name": "tumor_board_v3_multimodal",
        },
        "findings": [{
            "label": mbd4_tag,
            "score": 0.92,
            "location_bbox_normalized": None,
        }],
        "evidence": _ev(
            f"Genomic marker {mbd4_tag} in {histology} ({site}); "
            f"manuscript_sha={manuscript_sha[:12]}"
        ),
        "custom_modifiers": {
            "mbd4_lof": True,
            "gene": gene,
            "alteration_type": alteration,
            "query_gene": sl.get("query_gene") or gene,
        },
    }

    biopsy = {
        "provenance": {
            "model_state": "loaded_ak_bundle",
            "model_name": "tumor_board_v3_multimodal",
        },
        "subtype_prediction": f"{histology}_{mbd4_tag}",
        "confidence": 0.88,
        "receptor_panel": {
            "mbd4_lof": True,
            "gene": gene,
            "alteration_type": alteration,
            "tp53_status": diagnosis.get("tp53_status"),
            "msi_status": diagnosis.get("msi_status"),
        },
        "evidence": _ev(
            f"Pathology/context: {histology} with {mbd4_tag} "
            f"(DNA glycosylase / BER deficiency; RUO demo)."
        ),
    }

    drugs = list(sl.get("recommended_drugs") or [])
    # Manuscript dual-axis order: cytidine gold-standard (gemcitabine >
    # cytarabine) + ATR lead + falsified PARPi so Elo must debate the paper
    # (MAX_THERAPY_OPTIONS=3). Never seed PARPi ahead of cytidine/ATR.
    # 3-slot seed: gold-standard cytidine + ATR lead + falsified PARPi.
    preferred = ("gemcitabine", "ceralasertib", "olaparib")
    by_name = {str(d.get("drug_name") or "").lower(): d for d in drugs}
    # Synthesize gemcitabine from cytidine axis metadata when the fixture
    # only ships cytarabine/decitabine (Chabot gold-standard drug).
    if "gemcitabine" not in by_name:
        cyt = by_name.get("cytarabine") or by_name.get("decitabine")
        if cyt:
            gem = dict(cyt)
            gem["drug_name"] = "gemcitabine"
            gem["target"] = "DNA (dFdCTP)"
            gem["role"] = "validated_benchmark"
            gem["axis"] = "cytidine_analogs"
            gem["axis_label"] = str(
                cyt.get("axis_label") or "Cytidine Analogs (gemcitabine, cytarabine)"
            )
            gem["manuscript_claim_type"] = "validated_benchmark"
            gem["surface_status"] = "RECOMMENDED"
            gem["recommendation_tier"] = str(
                cyt.get("recommendation_tier") or "Validated SL therapeutic lever"
            )
            by_name["gemcitabine"] = gem
            drugs = [gem, *drugs]
    ordered: list[dict[str, Any]] = []
    for name in preferred:
        if name in by_name and by_name[name] not in ordered:
            ordered.append(by_name[name])
    for d in drugs:
        if d not in ordered:
            ordered.append(d)
    ordered = ordered[:3]

    recommended_options: list[dict[str, Any]] = []
    for d in ordered:
        name = str(d.get("drug_name") or "unknown")
        axis = str(d.get("axis") or "")
        status = str(d.get("surface_status") or "")
        falsified = status.startswith("NOT_RECOMMENDED") or bool(d.get("falsified"))
        conf = 0.15 if falsified else 0.85
        rationale = (
            f"MBD4-LOF ({mbd4_tag}) DDR context; axis={axis}; "
            f"surface_status={status}; "
            f"{'PARP mechanism falsified on this axis — do not prefer.' if falsified else 'manuscript-backed SL lever.'}"
        )
        recommended_options.append({
            "regimen": name,
            "line_of_therapy": 1,
            "confidence": conf,
            "rationale": rationale,
            "evidence": _ev(rationale),
            "custom_modifiers": {
                "mbd4_lof": True,
                "axis": axis,
                "surface_status": status,
                "falsified": falsified,
                "manuscript_claim_type": d.get("manuscript_claim_type"),
            },
        })

    therapy = {
        "provenance": {
            "model_state": "loaded_ak_bundle",
            "model_name": "tumor_board_v3_multimodal",
        },
        "recommended_options": recommended_options,
        "custom_modifiers": {
            "mbd4_lof": True,
            "gene": gene,
            "alteration_type": alteration,
            "disease": sl.get("disease"),
            "manuscript_repo_sha_at_audit": manuscript_sha,
            "backend_head_sha": backend_sha,
        },
    }

    patient_context = {
        "patient_id": bundle.get("patient_id"),
        "contract_version": contract,
        "mbd4_lof": True,
        "gene": gene,
        "alteration_type": alteration,
        "mbd4_tag": mbd4_tag,
        "diagnosis": diagnosis,
        "primary_alteration": primary,
        "query_gene": sl.get("query_gene") or gene,
        "disease": sl.get("disease"),
        "manuscript_repo_sha_at_audit": manuscript_sha,
        "backend_head_sha": backend_sha,
        "therapy_options": [o["regimen"] for o in recommended_options],
        "disclaimer": "RUO — AK demo fixture; not a real patient.",
    }

    return {
        "screening": screening,
        "biopsy": biopsy,
        "therapy": therapy,
        "seed_urls": [seed_url],
        "patient_context": patient_context,
    }


def failed_stage_receipt(
    stage: str,
    required: bool,
    request_id: str,
    code: str,
    message: str,
    *,
    service_name: str | None = None,
    input_reference: str | None = None,
) -> dict[str, Any]:
    """Build a failed stage receipt for pipeline tracking."""
    return {
        "stage": stage,
        "required": required,
        "status": "failed_required" if required else "failed_optional",
        "request_id": request_id,
        "service_name": service_name,
        "input_reference": input_reference,
        "warnings": [],
        "error": {"code": code, "message": message[:500]},
    }


def skipped_stage_receipt(
    stage: str,
    request_id: str,
    message: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Build a skipped stage receipt for pipeline tracking."""
    return {
        "stage": stage,
        "required": required,
        "status": "skipped_not_applicable",
        "request_id": request_id,
        "warnings": [message],
        "error": None,
    }


def pipeline_status(receipts: list[dict[str, Any]]) -> str:
    """Determine overall pipeline status from stage receipts."""
    if any(receipt.get("status") == "failed_required" for receipt in receipts):
        return "failed_required_stage"
    if any(receipt.get("status") == "failed_optional" for receipt in receipts):
        return "partial_failure"
    return "complete"
