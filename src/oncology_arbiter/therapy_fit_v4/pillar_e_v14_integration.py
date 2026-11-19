"""Pillar E v14 mechanistic overlay for Therapy Fit v4 (RUO; SHA-locked artifacts)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oncology_arbiter.therapy_fit_v4.schemas import (
    PillarEV14DrugPrediction,
    PillarEV14HeadReceipt,
    PillarEV14PanelRow,
    PillarEV14Receipt,
    Tier2TranscriptomicInput,
)

_PILLAR_E_V14_DIR = Path(__file__).resolve().parent.parent / "ovarian" / "pillar_e_v14"
_RECEIPTS_DIR = _PILLAR_E_V14_DIR / "receipts"
_MODEL_JSON_PATH = _PILLAR_E_V14_DIR / "pillar_e_v14_model.json"
_TRANSFORMS_JSON_PATH = _PILLAR_E_V14_DIR / "v14_transforms_and_top4_heads.json"
_STANDALONE_PATH = _PILLAR_E_V14_DIR / "pillar_e_v14_orthogonal_standalone.py"

EXPECTED_PILLAR_E_V14_MODEL_SHA256 = (
    "43409b7c685f4c2b35289d4361f01b60ba9b38af68780cfbdde56ca11da0dbea"
)
EXPECTED_PILLAR_E_V14_STANDALONE_SHA256 = (
    "4b041da10268dffc2f53163dcd07333c68c7da4e3447e9730ce365000b7ee203"
)
EXPECTED_PILLAR_E_V14_TRANSFORMS_SHA256 = (
    "39957801616056886d8304c88a21093328e7028ec94baab2c18bbb7dfd766808"
)

SCORING_HEAD = "v14_sparse_all28_logistic_q75"
EXPLAIN_HEADS = (
    "v14_unified_all28_logistic_positive",
    "v14_unified_core8_logistic_positive",
)

_OVARIAN_CANCERS = frozenset(
    {"hgsoc", "ovarian", "ovarian_cancer", "ovarian_cancer_hgs"}
)

_MODULE_CACHE: Optional[Any] = None
_MODEL_CACHE: Optional[Dict[str, Any]] = None
_RECEIPTS_CACHE: Optional[Dict[str, Any]] = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pillar_e_v14_artifact_locks() -> Dict[str, str]:
    locks = {
        "pillar_e_v14_model.json": _sha256_file(_MODEL_JSON_PATH),
        "pillar_e_v14_orthogonal_standalone.py": _sha256_file(_STANDALONE_PATH),
        "v14_transforms_and_top4_heads.json": _sha256_file(_TRANSFORMS_JSON_PATH),
    }
    if locks["pillar_e_v14_model.json"] != EXPECTED_PILLAR_E_V14_MODEL_SHA256:
        raise RuntimeError(f"pillar_e_v14_model.json SHA mismatch: {locks['pillar_e_v14_model.json']}")
    if locks["pillar_e_v14_orthogonal_standalone.py"] != EXPECTED_PILLAR_E_V14_STANDALONE_SHA256:
        raise RuntimeError(
            f"pillar_e_v14_orthogonal_standalone.py SHA mismatch: {locks['pillar_e_v14_orthogonal_standalone.py']}"
        )
    if locks["v14_transforms_and_top4_heads.json"] != EXPECTED_PILLAR_E_V14_TRANSFORMS_SHA256:
        raise RuntimeError(
            f"v14_transforms_and_top4_heads.json SHA mismatch: {locks['v14_transforms_and_top4_heads.json']}"
        )
    return locks


def _load_v14_module() -> Any:
    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE
    verify_pillar_e_v14_artifact_locks()
    mod_name = "pillar_e_v14_orthogonal_standalone_prod"
    spec = importlib.util.spec_from_file_location(mod_name, _STANDALONE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load pillar_e_v14 standalone module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULE_CACHE = mod
    return mod


def _load_pillar_model() -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        pillar_e = _load_v14_module()
        _MODEL_CACHE = pillar_e.load_model(_MODEL_JSON_PATH)
    return _MODEL_CACHE


def _tier2_gene_value(gene: str, tier2: Tier2TranscriptomicInput, transforms: Dict[str, Any]) -> Optional[float]:
    idx = None
    pillar_e = _load_v14_module()
    try:
        idx = pillar_e.GENES.index(gene)
    except ValueError:
        return None
    center = transforms["expression_center"][idx]
    scale = transforms["expression_scale"][idx]
    if gene == "VDR":
        if tier2.vdr_expr is not None:
            return float(tier2.vdr_expr)
        if tier2.z_VDR is not None:
            return center + float(tier2.z_VDR) * scale
    elif gene == "MFAP4" and tier2.mfap4_expr is not None:
        return float(tier2.mfap4_expr)
    elif gene == "EPHA2" and tier2.epha2_expr is not None:
        return float(tier2.epha2_expr)
    elif gene == "SRC" and tier2.z_SRC is not None:
        return center + float(tier2.z_SRC) * scale
    elif gene == "YES1" and tier2.z_YES1 is not None:
        return center + float(tier2.z_YES1) * scale
    elif gene == "PARP1":
        if tier2.parp1_expr_log2tpm is not None:
            return float(tier2.parp1_expr_log2tpm)
        if tier2.parp1_rna_z is not None:
            return center + float(tier2.parp1_rna_z) * scale
    return None


def build_molecular_inputs_v14(
    tier2: Tier2TranscriptomicInput,
    post_h_clean: float,
    post_s_sen: float,
    *,
    strict: bool = True,
    menu: str = "Full28",
) -> Dict[str, float]:
    """Fail-closed: all 23 expression genes must be finite (no population-center imputation)."""
    model = _load_pillar_model()
    if menu not in model["transforms"]:
        raise ValueError(f"Unknown Pillar E v14 transform menu: {menu}")
    transforms = model["transforms"][menu]
    overrides = dict(tier2.pillar_e_molecular or {})
    pillar_e = _load_v14_module()

    molecular: Dict[str, float] = {
        "H_clean": float(post_h_clean),
        "S_sen": float(post_s_sen),
    }
    missing: List[str] = []
    for gene in pillar_e.GENES:
        if gene in overrides:
            try:
                molecular[gene] = float(overrides[gene])
            except (TypeError, ValueError):
                missing.append(gene)
            continue
        val = _tier2_gene_value(gene, tier2, transforms)
        if val is None or val != val:  # NaN
            missing.append(gene)
        else:
            molecular[gene] = float(val)

    if missing:
        msg = f"Pillar E v14 missing or non-finite genes: {sorted(missing)}"
        if strict:
            raise ValueError(msg)
        for gene in missing:
            idx = pillar_e.GENES.index(gene)
            molecular[gene] = float(transforms["expression_center"][idx])
    return molecular


def build_adc_surface_from_tier2(tier2: Tier2TranscriptomicInput) -> Dict[str, float]:
    """Build antigen + payload surface vector for ADC scoring head (RUO)."""
    model = _load_pillar_model()
    sp = model.get("surface_payload") or {}
    overrides = dict(tier2.pillar_e_molecular or {})
    surface: Dict[str, float] = {}

    antigens = sp.get("antigens") or []
    a_centers = sp.get("antigen_center") or []
    for i, gene in enumerate(antigens):
        if gene in overrides:
            surface[gene] = float(overrides[gene])
        elif gene == "CEACAM5" and tier2.ceacam5_expr is not None:
            surface[gene] = float(tier2.ceacam5_expr)
        elif gene == "EPHA2" and tier2.epha2_expr is not None:
            surface[gene] = float(tier2.epha2_expr)
        elif i < len(a_centers):
            surface[gene] = float(a_centers[i])

    payload_genes = sp.get("payload_genes") or []
    p_centers = sp.get("payload_center") or []
    for i, gene in enumerate(payload_genes):
        if gene in overrides:
            surface[gene] = float(overrides[gene])
        elif gene in ("TOP1", "SLFN11") and tier2.top1_slfn11_score is not None:
            surface[gene] = float(tier2.top1_slfn11_score)
        elif i < len(p_centers):
            surface[gene] = float(p_centers[i])
    return surface


def molecular_from_expression_centers(menu: str = "Full28") -> Dict[str, float]:
    """Test helper: build a complete gene panel at training expression centers (RUO harness only)."""
    model = _load_pillar_model()
    transforms = model["transforms"][menu]
    pillar_e = _load_v14_module()
    return {g: float(transforms["expression_center"][i]) for i, g in enumerate(pillar_e.GENES)}


def _load_receipt_bundle() -> Dict[str, Any]:
    global _RECEIPTS_CACHE
    if _RECEIPTS_CACHE is not None:
        return _RECEIPTS_CACHE

    def _read_csv(name: str, q_col: str, bh_method: str) -> List[Dict[str, Any]]:
        path = _RECEIPTS_DIR / name
        rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                q_raw = row.get(q_col) or ""
                try:
                    q = float(q_raw)
                except ValueError:
                    continue
                rows.append(
                    {
                        "drug": row.get("drug") or row.get("compound_id") or "",
                        "q_value": q,
                        "bh_method": bh_method,
                        "passes_q05": q < 0.05,
                        "passes_q10": q < 0.10,
                        "assay": row.get("assay"),
                        "target_gene": row.get("target_gene"),
                    }
                )
        q05 = sum(1 for r in rows if r["passes_q05"])
        q10 = sum(1 for r in rows if r["passes_q10"])
        return {
            "source_csv": str(path.name),
            "bh_method": bh_method,
            "n_rows": len(rows),
            "n_q_lt_0_05": q05,
            "n_q_lt_0_10": q10,
            "min_q": min((r["q_value"] for r in rows), default=None),
            "rows": rows,
        }

    selective_path = _RECEIPTS_DIR / "worker1_v14_selective_receipt.json"
    selective = json.loads(selective_path.read_text(encoding="utf-8")) if selective_path.is_file() else {}

    adc_rows: List[Dict[str, Any]] = []
    adc_path = _RECEIPTS_DIR / "adc_2gate_capture_summary.csv"
    with adc_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            adc_rows.append(dict(row))

    _RECEIPTS_CACHE = {
        "door2_panel": _read_csv(
            "top25_door2_dedup.csv",
            "q_rho_S_sen_BH_dedup",
            "rho_S_sen_OR_BH_dedup_not_selective",
        ),
        "door1_panel": _read_csv(
            "top15_door1_dedup.csv",
            "q_rho_DDR_BH_dedup",
            "rho_DDR_BH_dedup",
        ),
        "selective_bh_receipt": selective,
        "adc_2gate_summary": adc_rows,
        "biomni_claim_refuted": {
            "claim": "10 Door2 drugs at selective BH q<0.05",
            "actual_selective_BH_q05": selective.get("per_unique_drug_BH_q05", 0),
            "actual_selective_min_q": selective.get("per_unique_drug_min_q"),
        },
    }
    return _RECEIPTS_CACHE


def _rows_to_predictions(rows: List[Dict[str, Any]]) -> List[PillarEV14DrugPrediction]:
    out: List[PillarEV14DrugPrediction] = []
    for r in rows:
        warnings = list(r.get("warnings_and_flags") or [])
        fc = r.get("feature_contributions")
        if isinstance(fc, list):
            fc = {"items": fc}
        adc = r.get("adc_surface_payload_recommendations")
        if isinstance(adc, dict):
            adc = [adc]
        elif adc is not None and not isinstance(adc, list):
            adc = [{"value": adc}]
        out.append(
            PillarEV14DrugPrediction(
                drug=str(r["drug"]),
                score=float(r["score"]),
                probability=float(r["probability"]),
                probability_target=str(r.get("probability_target") or ""),
                active_mechanistic_gate=r.get("active_mechanistic_gate"),
                warnings=warnings,
                feature_contributions=fc,
                adc_surface_payload_recommendations=adc,
            )
        )
    return out


def _head_predictions(
    molecular: Dict[str, float],
    head: str,
    model: Dict[str, Any],
    *,
    explain: bool,
    surface: Optional[Dict[str, float]] = None,
) -> PillarEV14HeadReceipt:
    pillar_e = _load_v14_module()
    menu = model["models"][head].get("menu", "Full28")
    drugs = model["training_menus"].get(menu, model["menus"].get(menu))
    rows = pillar_e.predict(
        molecular,
        drugs=drugs,
        head=head,
        model=model,
        explain=explain,
        surface=surface,
    )
    return PillarEV14HeadReceipt(
        head_id=head,
        menu=str(menu),
        explain=explain,
        predictions=_rows_to_predictions(rows),
    )


def build_pillar_e_v14_receipt(
    cancer_type: str,
    tier2: Tier2TranscriptomicInput,
    post_h_clean: Optional[float],
    post_s_sen: Optional[float],
    artifact_locks: Optional[Dict[str, str]] = None,
    adc_surface: Optional[Dict[str, float]] = None,
    *,
    strict_molecular: bool = False,
) -> PillarEV14Receipt:
    locks = artifact_locks or verify_pillar_e_v14_artifact_locks()
    receipts = _load_receipt_bundle()

    base = PillarEV14Receipt(
        enabled=False,
        model_sha256=locks["pillar_e_v14_model.json"],
        standalone_sha256=locks["pillar_e_v14_orthogonal_standalone.py"],
        transforms_sha256=locks["v14_transforms_and_top4_heads.json"],
        selective_bh_receipt=receipts["selective_bh_receipt"],
        door2_pharmacogenomic_panel=[
            PillarEV14PanelRow(**{k: row[k] for k in PillarEV14PanelRow.model_fields if k in row})
            for row in receipts["door2_panel"]["rows"][:25]
        ],
        door1_pharmacogenomic_panel=[
            PillarEV14PanelRow(**{k: row[k] for k in PillarEV14PanelRow.model_fields if k in row})
            for row in receipts["door1_panel"]["rows"][:15]
        ],
        door2_panel_summary={
            "bh_method": receipts["door2_panel"]["bh_method"],
            "n_q_lt_0_05": receipts["door2_panel"]["n_q_lt_0_05"],
            "n_q_lt_0_10": receipts["door2_panel"]["n_q_lt_0_10"],
            "min_q": receipts["door2_panel"]["min_q"],
        },
        door1_panel_summary={
            "bh_method": receipts["door1_panel"]["bh_method"],
            "n_q_lt_0_05": receipts["door1_panel"]["n_q_lt_0_05"],
            "n_q_lt_0_10": receipts["door1_panel"]["n_q_lt_0_10"],
            "min_q": receipts["door1_panel"]["min_q"],
        },
        adc_2gate_flags=receipts["adc_2gate_summary"],
        biomni_selective_claim_refuted=receipts["biomni_claim_refuted"],
        ruo_note=(
            "Pillar E v14 mechanistic overlay (RUO). Selective BH (rho_S_sen/H_clean qualifier) "
            "is separate from OR-BH dedup columns in door2 CSV receipts."
        ),
    )

    cancer = (cancer_type or "").strip().lower()
    if cancer not in _OVARIAN_CANCERS:
        base.skip_reason = f"cancer_type={cancer_type!r} outside ovarian/HGSOC scope"
        return base
    if post_h_clean is None or post_s_sen is None:
        base.skip_reason = "post_H_clean and post_S_sen required for Pillar E v14"
        return base

    model = _load_pillar_model()
    menu = model["models"][SCORING_HEAD].get("menu", "Full28")
    molecular = build_molecular_inputs_v14(
        tier2,
        float(post_h_clean),
        float(post_s_sen),
        strict=strict_molecular,
        menu=menu,
    )

    heads: List[PillarEV14HeadReceipt] = []
  # scoring head
    if SCORING_HEAD in model.get("models", {}):
        heads.append(_head_predictions(molecular, SCORING_HEAD, model, explain=False, surface=adc_surface))
    for head in EXPLAIN_HEADS:
        if head in model.get("models", {}):
            heads.append(_head_predictions(molecular, head, model, explain=True, surface=adc_surface))

    return PillarEV14Receipt(
        enabled=True,
        model_sha256=locks["pillar_e_v14_model.json"],
        standalone_sha256=locks["pillar_e_v14_orthogonal_standalone.py"],
        transforms_sha256=locks["v14_transforms_and_top4_heads.json"],
        heads=heads,
        molecular_summary={
            "H_clean": molecular["H_clean"],
            "S_sen": molecular["S_sen"],
            "buffer": molecular["S_sen"] - molecular["H_clean"],
        },
        selective_bh_receipt=receipts["selective_bh_receipt"],
        door2_pharmacogenomic_panel=base.door2_pharmacogenomic_panel,
        door1_pharmacogenomic_panel=base.door1_pharmacogenomic_panel,
        door2_panel_summary=base.door2_panel_summary,
        door1_panel_summary=base.door1_panel_summary,
        adc_2gate_flags=base.adc_2gate_flags,
        biomni_selective_claim_refuted=base.biomni_selective_claim_refuted,
        ruo_note=base.ruo_note,
    )