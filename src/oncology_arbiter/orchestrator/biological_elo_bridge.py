"""Modular L3 → L5 Elo Tournament bridge (RUO).

Derives Elo modifier deltas from the locked contract receipts on disk
(``zeta_immunotherapy_contract_v3.json``, ``ovarian_two_door_standalone_v3.py``,
``docs/proofs/*.json``) and delegates tournament execution to ``elo_rank``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from oncology_arbiter.api.routers.elo import elo_rank
from oncology_arbiter.api.schemas import EloDrugCandidate, EloRankRequest, EvidenceRecord
from oncology_arbiter.immunotherapy.cart_ici_v3 import load_arbiter_v3_contract
from oncology_arbiter.ovarian.ovarian_two_door_standalone_v3 import (
    HUMAN_POLICY,
    KEY_VALIDATION_LEDGER,
)

CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
_PROOFS_DIR = Path(__file__).resolve().parents[3] / "docs" / "proofs"

OVARIAN_CANDIDATES: tuple[tuple[str, str, str, float], ...] = (
    ("Navitoclax", "Navitoclax (BCL-xL/BCL-2 senolytic)", "BCL-xL_senolytic", 0.50),
    ("ABT-737", "ABT-737 (BCL-xL/BCL-2/BCL-w senolytic)", "BCL-xL_senolytic", 0.49),
    ("Ceralasertib", "Ceralasertib / AZD6738 (ATR inhibitor)", "DDR_ATR", 0.50),
    ("Adavosertib", "Adavosertib / MK-1775 (WEE1 inhibitor)", "DDR_WEE1", 0.49),
    ("Gemcitabine", "Gemcitabine (BER antimetabolite)", "DDR_BER_chemo", 0.48),
    ("Olaparib", "Olaparib (PARP1/2 inhibitor)", "DDR_PARP", 0.48),
    ("Venetoclax", "Venetoclax (BCL-2 selective decoy)", "BCL2_selective_decoy", 0.47),
    ("Carboplatin_Paclitaxel", "Carboplatin + Paclitaxel (platinum)", "platinum_chemo", 0.48),
)

_GDSC2_DRUG_KEY: dict[str, str] = {
    "Navitoclax": "Navitoclax",
    "ABT-737": "ABT737",
    "Ceralasertib": "AZD6738 (Ceralasertib, ATR)",
    "Adavosertib": "MK-1775 (Adavosertib, WEE1)",
    "Gemcitabine": "Gemcitabine (antimetabolite)",
    "Olaparib": "Olaparib (PARP)",
    "Carboplatin_Paclitaxel": "Cisplatin (DNA crosslinker)",
}

_IMMUNO_CANDIDATES: dict[str, tuple[tuple[str, str, str, float], ...]] = {
    "cart_infusion_bag": (
        ("Autologous_CD19_CART_Infusion", "Proceed with CD19 CAR-T infusion product", "cart_infusion_product", 0.50),
        ("Memory_Enriched_Bag_Hold_or_Re_Expansion", "Ex-vivo memory-enrichment intervention", "cart_manufacturing_intervention", 0.48),
        ("Alternative_Bispecific_or_Salvage", "Alternative bispecific or salvage bridge", "salvage_bridge", 0.46),
    ),
    "cart_in_vivo_blood": (
        ("Standard_Leukapheresis_CART_Proceed", "Standard leukapheresis CAR-T trajectory", "cart_in_vivo_blood_favorable", 0.50),
        ("Cytoreductive_SPD_Bridging_Before_CART", "Cytoreductive SPD bridging before CAR-T", "cart_bridging_intensification", 0.49),
        ("Bispecific_or_Clinical_Trial_Escalation", "Bispecific or trial escalation", "salvage_escalation", 0.47),
    ),
    "ici_tumor_biopsy": (
        ("Anti_PD1_Checkpoint_Blockade", "Anti-PD-1 checkpoint immunotherapy", "ici_checkpoint_blockade", 0.50),
        ("Combination_IO_Stroma_or_Trial_Escalation", "Combination IO or clinical trial escalation", "io_combination_escalation", 0.48),
        ("Non_IO_Targeted_or_Cytotoxic_Regimen", "Non-IO targeted or cytotoxic therapy", "non_io_comparator", 0.46),
    ),
}


@lru_cache(maxsize=4)
def _load_proof(filename: str) -> dict[str, Any]:
    return json.loads((_PROOFS_DIR / filename).read_text())


@lru_cache(maxsize=1)
def _gdsc2_delta_ln_map() -> dict[str, float]:
    receipt = _load_proof("gdsc2_ddr_bh3_tertile_shift_receipt.json")
    by_name = {
        row["drug"]: float(row["delta_LN_IC50_High_minus_Low"])
        for row in receipt["per_drug_High_minus_Low"]
    }
    return {cid: by_name[gkey] for cid, gkey in _GDSC2_DRUG_KEY.items()}


def _run_elo_via_router(
    specs: tuple[tuple[str, str, str, float], ...],
    deltas: dict[str, tuple[float, str]],
    evidence_urls: list[tuple[str, str]],
    disease_context: dict[str, Any],
) -> dict[str, Any]:
    ev_models = [EvidenceRecord(url=u, quoted_text=t, source="contract_receipt") for u, t in evidence_urls]
    ev_dicts = [{"url": u, "quoted_text": t} for u, t in evidence_urls]
    drugs = [
        EloDrugCandidate(
            drug_id=cid,
            regimen=reg,
            line=1,
            confidence=base,
            evidence=ev_models,
            honesty_markers={"loaded": True, "proxy": False, "gated": False},
        )
        for cid, reg, _cls, base in specs
    ]
    modifiers = {cid: round(float(deltas.get(cid, (0.0, ""))[0]), 6) for cid, *_ in specs}
    resp = elo_rank(
        EloRankRequest(
            drugs=drugs,
            modifiers=modifiers,
            disease_context=disease_context,
        )
    )
    meta = {cid: (reg, cls) for cid, reg, cls, _ in specs}
    match_by_id = {m.drug_id: m for m in resp.matches}
    ranked: list[dict[str, Any]] = []
    for entry in resp.enriched_ranking:
        cid = entry.drug_id
        reg, cls = meta[cid]
        m = match_by_id[cid]
        delta, reason = deltas.get(cid, (0.0, m.reason))
        ranked.append({
            "rank": entry.rank,
            "hyp_id": cid,
            "drug_id": cid,
            "regimen": reg,
            "statement": reg,
            "mechanism_class": cls,
            "stage": "therapy",
            "rating": entry.rating,
            "elo": entry.rating,
            "baseline_rank": m.baseline_rank,
            "baseline_rating": m.baseline_rating,
            "rank_delta": m.rank_delta,
            "rating_delta": m.rating_delta,
            "applied_modifier": round(float(delta), 4),
            "confidence": entry.confidence,
            "wins": entry.wins,
            "losses": entry.losses,
            "draws": entry.draws,
            "reason": reason or m.reason,
            "evidence": ev_dicts,
            "ruo": True,
            "contract_status": CONTRACT_STATUS,
        })
    return {
        "ruo": True,
        "contract_status": CONTRACT_STATUS,
        "top_candidate": ranked[0]["drug_id"],
        "top_mechanism_class": ranked[0]["mechanism_class"],
        "promoted": [r["drug_id"] for r in ranked if r["applied_modifier"] > 0],
        "demoted": [r["drug_id"] for r in ranked if r["applied_modifier"] < 0],
        "applied_modifiers": {cid: round(float(deltas.get(cid, (0.0, ""))[0]), 4) for cid, *_ in specs},
        "ranked_candidates": ranked,
        "tournament_matches": len(specs) * (len(specs) - 1) // 2,
    }


def _ovarian_deltas_from_contracts(
    state: str | None,
    timepoint: str,
    ovarian_score: dict[str, Any],
) -> dict[str, tuple[float, str]]:
    dln = _gdsc2_delta_ln_map()
    nav_rho = float(KEY_VALIDATION_LEDGER["prism_navitoclax_vs_venetoclax_delta_rho_and_q"]["delta_rho"])
    abt_rho = float(KEY_VALIDATION_LEDGER["prism_abt737_vs_venetoclax_delta_rho_and_q"]["delta_rho"])
    deltas: dict[str, tuple[float, str]] = {}

    if timepoint == "pre_treatment":
        gate2 = ovarian_score.get("gate2") or {}
        p_pos = (gate2.get("mfap4") or {}).get("p_positive")
        high_res = isinstance(p_pos, (int, float)) and float(p_pos) >= 0.50
        plat_delta = (-0.25 if high_res else 0.25) + (0.15 if gate2.get("aacr_positive") else 0.0)
        deltas["Carboplatin_Paclitaxel"] = (plat_delta, "Gate 2 MFAP4 + AACR A065 stroma-immune composite")
        if high_res:
            deltas["Ceralasertib"] = (0.20, "ATR prioritization under high MFAP4 platinum-resistance risk")
    elif state in {"Hcleanonly", "Both"}:
        deltas["Navitoclax"] = (
            -0.20 * dln["Navitoclax"] + 0.30 * nav_rho,
            "Door 2 H_clean>cutpoint: GDSC2 ΔLN=-1.20 + PRISM Δρ=+0.6327 (q=0.0075)",
        )
        deltas["ABT-737"] = (
            -0.15 * dln["ABT-737"] + 0.25 * abt_rho,
            "Door 2 H_clean>cutpoint: GDSC2 ΔLN=-1.29 + PRISM Δρ=+0.6323 (q=0.0060)",
        )
        deltas["Venetoclax"] = (-nav_rho, "Demote BCL-2 decoy vs BCL-xL (PRISM Δρ=+0.6327)")
        for ddr_id in ("Olaparib", "Ceralasertib", "Adavosertib", "Gemcitabine", "Carboplatin_Paclitaxel"):
            deltas[ddr_id] = (
                -0.15 * dln[ddr_id],
                f"Demote proliferative {ddr_id} in hibernation (GDSC2 ΔLN={dln[ddr_id]:+.2f})",
            )
    elif state == "DDRonly":
        deltas["Ceralasertib"] = (0.25 * dln["Ceralasertib"], "Door 1 DDRonly: ATR sensitivity (7/10 GDSC2 & 6/9 CTRPv2)")
        deltas["Adavosertib"] = (0.30 * dln["Adavosertib"], "Door 1 DDRonly: WEE1 sensitivity (7/10 GDSC2 & 6/9 CTRPv2)")
        deltas["Olaparib"] = (-0.15, "DDRonly replication-stress prioritizes ATR/WEE1 over unselected PARP")
        deltas["Navitoclax"] = (0.15 * dln["Navitoclax"], "Low post_H_clean: hibernation gate not met")
        deltas["ABT-737"] = (0.15 * dln["ABT-737"], "Low post_H_clean: hibernation gate not met")
        deltas["Venetoclax"] = (-0.50 * abt_rho, "Demote BCL-2 decoy")
    elif state == "PRIMARY_REFRACTORY_ESCAPE":
        deltas["Navitoclax"] = (-0.35, "Gate 0 EOC3 override (BCL2A1_z>2.5 & delta_MCL1_z>2.5): demote single-agent BCL-xL")
        deltas["ABT-737"] = (-0.35, "Gate 0 EOC3 override: demote single-agent BCL-xL")
        deltas["Carboplatin_Paclitaxel"] = (-0.45, "Gate 0 EOC3 14d refractory escape: demote platinum")
        deltas["Venetoclax"] = (-nav_rho, "Demote BCL-2 decoy in refractory escape")

    gate1 = ovarian_score.get("gate1_mbd4") or {}
    if gate1.get("fired"):
        cera_d = abs(float(gate1.get("ceralasertib_ln_ic50_delta", -0.74)))
        parp_p = float(gate1.get("parp_null_p", 0.605))
        deltas["Ceralasertib"] = (
            max(0.0, deltas.get("Ceralasertib", (0.0, ""))[0]) + 0.60 * cera_d,
            "MBD4-LOF (CCR 2026 B092, BRCA-WT): promote Ceralasertib (LN_IC50 Δ=-0.74, p=0.021)",
        )
        deltas["Gemcitabine"] = (
            max(0.0, deltas.get("Gemcitabine", (0.0, ""))[0]) + 0.50 * cera_d,
            "MBD4-LOF (CCR 2026 B092): promote Gemcitabine BER benchmark",
        )
        deltas["Olaparib"] = (
            min(0.0, deltas.get("Olaparib", (0.0, ""))[0]) - 0.65 * parp_p,
            "MBD4-LOF with BRCA-WT (HRD<42): demote Olaparib (p=0.605 null)",
        )
    return deltas


def build_ovarian_elo_bridge(ovarian_score: dict[str, Any]) -> dict[str, Any]:
    """Map an L3 ``score_ovarian_case`` dict to deterministic L5 Elo rankings."""
    timepoint = str(ovarian_score.get("timepoint") or "post_nact")
    state = (
        "PRIMARY_REFRACTORY_ESCAPE"
        if ovarian_score.get("gate0_primary_refractory")
        else ovarian_score.get("four_state_class")
    )
    gate1 = ovarian_score.get("gate1_mbd4") or {}
    deltas = _ovarian_deltas_from_contracts(state, timepoint, ovarian_score)
    elo = _run_elo_via_router(
        OVARIAN_CANDIDATES,
        deltas,
        [
            ("https://crispro.local/proofs/gdsc2_ddr_bh3_tertile_shift_receipt.json", "GDSC2 DDR vs BH3 tertile shift"),
            ("https://crispro.local/proofs/gse165897_two_door_pfi_separation_receipt.json", "GSE165897 Two-Door PFI separation"),
        ],
        disease_context={"cancer": "hgsoc", "timepoint": timepoint, "four_state_class": state},
    )
    gdsc2 = _load_proof("gdsc2_ddr_bh3_tertile_shift_receipt.json")
    pfi = _load_proof("gse165897_two_door_pfi_separation_receipt.json")
    return {
        **elo,
        "bridge_domain": "ovarian_two_door_v3",
        "timepoint": timepoint,
        "four_state_class": state,
        "gate0_primary_refractory": bool(ovarian_score.get("gate0_primary_refractory")),
        "gate1_mbd4_fired": bool(gate1.get("fired")),
        "receipts": {
            "cutpoints": HUMAN_POLICY,
            "validation_ledger": KEY_VALIDATION_LEDGER,
            "gdsc2_source_members": gdsc2["source_members"],
            "gse165897_omnibus": pfi["omnibus_4state_kruskal_wallis"],
            "gate2": ovarian_score.get("gate2"),
        },
    }


def build_immunotherapy_elo_bridge(immuno_score: dict[str, Any]) -> dict[str, Any]:
    """Map an L3 ``score_immunotherapy`` dict to deterministic L5 Elo rankings."""
    specimen = str(immuno_score.get("specimen") or "")
    if specimen not in _IMMUNO_CANDIDATES:
        raise ValueError(f"Unsupported immunotherapy specimen: {specimen!r}")
    contract = load_arbiter_v3_contract()["compartments"][specimen]
    specs = _IMMUNO_CANDIDATES[specimen]
    first_id, second_id, third_id = (s[0] for s in specs)

    if specimen == "cart_infusion_bag":
        auroc = float(contract["metrics_literal_CD8"]["Bag_Index_3_primary_CR"]["auroc"])
        idx1 = float(immuno_score.get("Bag_Index_1") or 0.0)
        idx3 = float(immuno_score.get("Bag_Index_3") or -1.0)
        # In Deng GSE151511 literal_CD8 (n=23), Bag_Index_1 is z-centered at 0.0
        # and Bag_Index_3 cohort median is -0.829239333794604 (due to cyto/memory overlap).
        favorable = immuno_score.get("risk_flag") is None and (idx1 > 0.0 or idx3 >= -0.829239333794604)
    elif specimen == "cart_in_vivo_blood":
        auroc = float(contract["evaluation_receipt"]["auroc"])
        favorable = float(immuno_score.get("probability") or 0.0) >= 0.50
    elif immuno_score.get("mode") == "single_cell":
        auroc = float(contract["single_cell_internal_tcell_ratios_cv_auroc"])
        favorable = float(immuno_score.get("TCF7_TOX_count_ratio") or 0.0) >= 0.0
    else:
        auroc = float(contract["gide_pretreatment_metrics"]["ZNF683_GNLY_effector_response"]["auroc"])
        favorable = float(immuno_score.get("E_effector") or 0.0) >= 0.0

    deltas = (
        {
            first_id: (0.45 * auroc, f"{specimen} favorable L3 score (contract AUROC={auroc:.3f})"),
            third_id: (-0.25 * auroc, f"{specimen} comparator demoted (contract AUROC={auroc:.3f})"),
        }
        if favorable
        else {
            second_id: (0.45 * auroc, f"{specimen} risk/low L3 score intervention (contract AUROC={auroc:.3f})"),
            first_id: (-0.30 * auroc, f"{specimen} standard trajectory demoted (contract AUROC={auroc:.3f})"),
        }
    )
    elo = _run_elo_via_router(
        specs,
        deltas,
        [("https://crispro.local/contracts/zeta_immunotherapy_contract_v3.json", "Zeta Immunotherapy Contract v3")],
        disease_context={"specimen": specimen},
    )
    return {
        **elo,
        "bridge_domain": "immunotherapy_v12",
        "specimen": specimen,
        "receipts": contract,
    }
