"""Modular L3 → L5 Elo Tournament bridge (RUO).

Reads validation receipts directly from the locked contracts on disk
(``zeta_immunotherapy_contract_v3.json``, ``ovarian_two_door_standalone_v3.py``,
``docs/proofs/*.json``) and delegates pairwise ranking to ``rank_hypotheses``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from oncology_arbiter.immunotherapy.cart_ici_v3 import load_arbiter_v3_contract
from oncology_arbiter.orchestrator.co_scientist.models import Hypothesis
from oncology_arbiter.orchestrator.co_scientist.rank import rank_hypotheses
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

_OVARIAN_DELTAS: dict[str, dict[str, tuple[float, str]]] = {
    "Hcleanonly": {
        "Navitoclax": (0.42, "Door 2 H_clean>cutpoint: promote BCL-xL senolytic (PRISM Δρ=+0.6327 q=0.0075, BCL2L1 CRISPR ρ=-0.6547 q=2.5e-5)"),
        "ABT-737": (0.36, "Door 2 H_clean>cutpoint: promote BCL-xL senolytic (PRISM Δρ=+0.6323 q=0.0060)"),
        "Venetoclax": (-0.42, "Demote BCL-2 decoy (PRISM matched-donor Δρ=+0.6323/+0.6327 vs BCL-xL)"),
        "Olaparib": (-0.30, "Demote DDR/PARP in hibernation (0/8 GDSC2 & 0/5 CTRPv2 DDR activity)"),
        "Ceralasertib": (-0.22, "Demote proliferative ATR in hibernation (0/8 GDSC2 & 0/5 CTRPv2 DDR activity)"),
        "Adavosertib": (-0.24, "Demote proliferative WEE1 in hibernation"),
        "Gemcitabine": (-0.28, "Demote proliferative antimetabolite in hibernation"),
        "Carboplatin_Paclitaxel": (-0.32, "Demote proliferative platinum in hibernation"),
    },
    "DDRonly": {
        "Ceralasertib": (0.42, "Door 1 DDRonly: promote Ceralasertib (ATR; 7/10 GDSC2 & 6/9 CTRPv2 DDR activity)"),
        "Adavosertib": (0.35, "Door 1 DDRonly: promote Adavosertib (WEE1; 7/10 GDSC2 & 6/9 CTRPv2 DDR activity)"),
        "Olaparib": (-0.15, "DDRonly replication-stress prioritizes ATR/WEE1 over unselected PARP"),
        "Navitoclax": (-0.18, "Low post_H_clean: hibernation senolytic gate not met"),
        "ABT-737": (-0.20, "Low post_H_clean: hibernation senolytic gate not met"),
        "Venetoclax": (-0.30, "Demote BCL-2 decoy"),
    },
    "PRIMARY_REFRACTORY_ESCAPE": {
        "Navitoclax": (-0.35, "Gate 0 EOC3 override (post_BCL2A1_z>2.5 & delta_MCL1_z>2.5): demote single-agent BCL-xL"),
        "ABT-737": (-0.35, "Gate 0 EOC3 override: demote single-agent BCL-xL"),
        "Carboplatin_Paclitaxel": (-0.45, "Gate 0 EOC3 14d refractory escape: demote platinum"),
        "Venetoclax": (-0.42, "Demote BCL-2 decoy in refractory escape"),
    },
}
_OVARIAN_DELTAS["Both"] = _OVARIAN_DELTAS["Hcleanonly"]

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


def _run_elo(
    specs: tuple[tuple[str, str, str, float], ...],
    deltas: dict[str, tuple[float, str]],
    evidence_urls: list[str],
) -> dict[str, Any]:
    ev = [{"url": u} for u in evidence_urls]
    markers = {"loaded": True, "proxy": False, "gated": False}
    base_hyps = [
        Hypothesis(hyp_id=cid, stage="therapy", statement=reg, confidence=base, evidence=ev, honesty_markers=markers)
        for cid, reg, _cls, base in specs
    ]
    mod_hyps = [
        Hypothesis(
            hyp_id=cid,
            stage="therapy",
            statement=reg,
            confidence=base + deltas.get(cid, (0.0, ""))[0],
            evidence=ev,
            honesty_markers=markers,
        )
        for cid, reg, _cls, base in specs
    ]
    base_map = {
        e.hypothesis.hyp_id: (idx, round(e.rating, 4))
        for idx, e in enumerate(rank_hypotheses(base_hyps, mode="offline"), start=1)
    }
    meta = {cid: (reg, cls) for cid, reg, cls, _ in specs}
    transcript: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    for idx, entry in enumerate(rank_hypotheses(mod_hyps, mode="offline", transcript=transcript), start=1):
        cid = entry.hypothesis.hyp_id
        reg, cls = meta[cid]
        b_rank, b_rating = base_map[cid]
        delta, reason = deltas.get(cid, (0.0, "baseline"))
        ranked.append({
            "rank": idx,
            "hyp_id": cid,
            "drug_id": cid,
            "regimen": reg,
            "statement": reg,
            "mechanism_class": cls,
            "stage": "therapy",
            "rating": round(entry.rating, 4),
            "elo": round(entry.rating, 4),
            "baseline_rank": b_rank,
            "baseline_rating": b_rating,
            "rank_delta": b_rank - idx,
            "rating_delta": round(entry.rating - b_rating, 4),
            "applied_modifier": round(delta, 4),
            "confidence": round(max(0.0, min(1.0, entry.hypothesis.confidence)), 4),
            "wins": entry.wins,
            "losses": entry.losses,
            "draws": entry.draws,
            "reason": reason,
            "evidence": ev,
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
        "applied_modifiers": {cid: round(deltas.get(cid, (0.0, ""))[0], 4) for cid, *_ in specs},
        "ranked_candidates": ranked,
        "tournament_matches": len(transcript),
    }


def build_ovarian_elo_bridge(ovarian_score: dict[str, Any]) -> dict[str, Any]:
    """Map an L3 ``score_ovarian_case`` dict to deterministic L5 Elo rankings."""
    timepoint = str(ovarian_score.get("timepoint") or "post_nact")
    state = "PRIMARY_REFRACTORY_ESCAPE" if ovarian_score.get("gate0_primary_refractory") else ovarian_score.get("four_state_class")
    deltas: dict[str, tuple[float, str]] = dict(_OVARIAN_DELTAS.get(str(state), {}))

    if timepoint == "pre_treatment":
        gate2 = ovarian_score.get("gate2") or {}
        p_pos = (gate2.get("mfap4") or {}).get("p_positive")
        high_res = isinstance(p_pos, (int, float)) and float(p_pos) >= 0.50
        plat_delta = (-0.25 if high_res else 0.25) + (0.15 if gate2.get("aacr_positive") else 0.0)
        deltas["Carboplatin_Paclitaxel"] = (plat_delta, "Pre-treatment Gate 2 MFAP4 + AACR A065 stroma-immune composite")
        if high_res:
            deltas["Ceralasertib"] = (0.20, "ATR prioritization under elevated stromal MFAP4 resistance risk")

    gate1 = ovarian_score.get("gate1_mbd4") or {}
    if gate1.get("fired"):
        deltas["Ceralasertib"] = (max(0.0, deltas.get("Ceralasertib", (0.0, ""))[0]) + 0.44, "MBD4-LOF (CCR 2026 B092, BRCA-WT): promote Ceralasertib (LN_IC50 Δ=-0.74, p=0.021)")
        deltas["Gemcitabine"] = (max(0.0, deltas.get("Gemcitabine", (0.0, ""))[0]) + 0.38, "MBD4-LOF (CCR 2026 B092): promote Gemcitabine BER benchmark")
        deltas["Olaparib"] = (min(0.0, deltas.get("Olaparib", (0.0, ""))[0]) - 0.40, "MBD4-LOF with BRCA-WT (HRD<42): demote Olaparib (p=0.605 null)")

    elo = _run_elo(
        OVARIAN_CANDIDATES,
        deltas,
        [
            "docs/proofs/gdsc2_ddr_bh3_tertile_shift_receipt.json",
            "docs/proofs/gse165897_two_door_pfi_separation_receipt.json",
        ],
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
        favorable = immuno_score.get("risk_flag") is None and float(immuno_score.get("Bag_Index_3") or 0.0) >= 0.0
    elif specimen == "cart_in_vivo_blood":
        favorable = float(immuno_score.get("probability") or 0.0) >= 0.50
    elif immuno_score.get("mode") == "single_cell":
        favorable = float(immuno_score.get("TCF7_TOX_count_ratio") or 0.0) >= 0.0
    else:
        favorable = float(immuno_score.get("E_effector") or 0.0) >= 0.0

    deltas = (
        {first_id: (0.38, f"{specimen} favorable L3 score"), third_id: (-0.22, f"{specimen} favorable L3 score")}
        if favorable
        else {second_id: (0.36, f"{specimen} risk/low L3 score"), first_id: (-0.25, f"{specimen} risk/low L3 score")}
    )
    elo = _run_elo(specs, deltas, ["src/oncology_arbiter/arbiter/models/zeta_immunotherapy_contract_v3.json"])
    return {
        **elo,
        "bridge_domain": "immunotherapy_v12",
        "specimen": specimen,
        "receipts": contract,
    }
