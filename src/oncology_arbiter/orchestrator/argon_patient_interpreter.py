"""Deterministic Argon Patient Meaning Interpreter (L3 Arbiter + L5 Elo → Clinical Meaning).

Turns every scored patient (Ovarian Two-Door v3 or Immunotherapy v12) into a
canonical Argon input envelope and produces a deterministic ``argon_output``
explaining what the patient's biomarker state, gate triggers, and L5 Elo
ranking mean biologically (RUO, ``temperature=0.0``).
"""
from __future__ import annotations

import json
import os
from typing import Any

from oncology_arbiter.orchestrator.co_scientist.json_util import chat_json_with_retry

CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
ARGON_MODEL_ID = "gemini-4-argon"
ARGON_TEMPERATURE = 0.0

ARGON_INTERPRETER_SYSTEM = (
    "You are Gemini 4 Argon operating at temperature=0.0 as the deterministic "
    "clinical-meaning interpreter for the Oncology Arbiter (RESEARCH USE ONLY). "
    "Given PATIENT_INPUT_JSON containing L3 arbiter scores, cutpoints, and L5 "
    "Elo rankings, return ONLY one JSON object with keys:\n"
    '{"biological_state_meaning": "...", "gate_and_compartment_meaning": "...", '
    '"elo_ranking_meaning": "...", "actionable_summary_ruo": "..."}\n'
    "Every claim must cite the exact numbers in PATIENT_INPUT_JSON. First character '{'."
)


def build_argon_patient_input(scored_body: dict[str, Any]) -> dict[str, Any]:
    """Turn a scored ovarian or immunotherapy response into a deterministic Argon input."""
    bridge = scored_body.get("elo_bridge") or {}
    ranked = bridge.get("ranked_candidates") or []
    top_3 = [
        {
            "rank": r.get("rank"),
            "drug_id": r.get("drug_id"),
            "elo": r.get("elo"),
            "applied_modifier": r.get("applied_modifier"),
            "reason": r.get("reason"),
        }
        for r in ranked[:3]
    ]
    if "four_state_class" in scored_body or scored_body.get("model_name") == "ovarian_two_door_v3":
        domain = "ovarian_two_door_v3"
        sub = scored_body.get("timepoint") or "post_nact"
        l3_state = {
            "timepoint": sub,
            "four_state_class": scored_body.get("four_state_class"),
            "decision": scored_body.get("decision"),
            "cutpoints": scored_body.get("cutpoints"),
            "scores": scored_body.get("scores"),
            "gate0_primary_refractory": scored_body.get("gate0_primary_refractory"),
            "gate1_mbd4": scored_body.get("gate1_mbd4"),
            "gate2": scored_body.get("gate2"),
            "recommend": scored_body.get("recommend"),
            "contraindicated": scored_body.get("contraindicated"),
        }
    else:
        domain = "immunotherapy_v12"
        sub = scored_body.get("specimen") or "unknown"
        l3_state = {
            k: v
            for k, v in scored_body.items()
            if k not in {"elo_bridge", "argon_interpretation", "ruo_disclaimer"}
        }

    patient_id = str(
        scored_body.get("patient_id")
        or f"{sub}:{scored_body.get('mode') or scored_body.get('bag_gate') or 'case'}"
    )
    envelope = {
        "patient_id": patient_id,
        "domain": domain,
        "timepoint_or_specimen": sub,
        "l3_arbiter_state": l3_state,
        "l5_elo_summary": {
            "top_candidate": bridge.get("top_candidate"),
            "top_mechanism_class": bridge.get("top_mechanism_class"),
            "promoted": bridge.get("promoted"),
            "demoted": bridge.get("demoted"),
            "door2_subtype_routing": bridge.get("door2_subtype_routing"),
            "top_3_ranked": top_3,
        },
    }
    envelope["prompt"] = (
        "PATIENT_INPUT_JSON:\n"
        + json.dumps(envelope, sort_keys=True, ensure_ascii=False)
        + "\n\nInterpret what this patient's scores and L5 Elo ranking mean (JSON):"
    )
    return envelope


def _ovarian_deterministic_meaning(
    scored_body: dict[str, Any],
    l5: dict[str, Any],
) -> dict[str, str]:
    tp = scored_body.get("timepoint") or "post_nact"
    top_drug = l5.get("top_candidate") or "None"
    promoted = ", ".join(l5.get("promoted") or []) or "none"
    demoted = ", ".join(l5.get("demoted") or []) or "none"

    if tp == "pre_treatment":
        g2 = scored_body.get("gate2") or {}
        mfap4 = g2.get("mfap4") or {}
        p_pos = mfap4.get("p_positive")
        expr = g2.get("mfap4_expr")
        aacr = bool(g2.get("aacr_positive"))
        bio = (
            f"Pre-treatment HGSOC biopsy scored on GSE63885 MFAP4 (212713_at={expr}, "
            f"p_platinum_resistant={p_pos}) and AACR A065 stroma-immune rule "
            f"(z(FAP)<0 & z(CXCL10)>0 -> aacr_positive={aacr})."
        )
        gate_msg = (
            "Gate 2 (Day-0 Stroma-Immune Arbiter) evaluated stromal fibrosis vs CXCL10 "
            "chemokine infiltration (+8.86m GSE32062 RMST; TCGA OS HR=0.593)."
        )
    else:
        scores = scored_body.get("scores") or {}
        cuts = scored_body.get("cutpoints") or {}
        h_val = scores.get("post_H_clean")
        d_val = scores.get("post_DDR")
        s_val = scores.get("post_S_sen")
        h_cut = cuts.get("post_H_clean")
        d_cut = cuts.get("post_DDR")
        cls = scored_body.get("four_state_class")
        dec = scored_body.get("decision")
        bio = (
            f"Post-NACT residual tumor has post_H_clean={h_val:.4f} (cutpoint>{h_cut:.4f}, "
            f"ties low) and post_DDR={d_val:.6f} (cutpoint>{d_cut:.6f}), with senolytic "
            f"index post_S_sen={s_val if s_val is not None else 'N/A'} -> state={cls} ({dec})."
        )
        if scored_body.get("gate0_primary_refractory"):
            gate_msg = (
                f"Gate 0 PRIMARY_REFRACTORY_ESCAPE fired (post_BCL2A1_z={scores.get('post_BCL2A1_z'):.3f}>2.5 "
                f"and delta_MCL1_z={scores.get('delta_MCL1_z'):.3f}>2.5 / post_EMT_z={scores.get('post_EMT_z'):.3f}>1.0), "
                "overriding geometric H_clean to flag rapid refractory escape (EOC3 14d archetype)."
            )
        elif (scored_body.get("gate1_mbd4") or {}).get("fired"):
            gate_msg = (
                "Gate 1 CCR 2026 B092 MBD4-LOF fired (TP53-mut, BRCA-WT, HRD<42): "
                "base-excision repair deficiency creates ATR/BER synthetic lethality "
                "(Ceralasertib ΔLN_IC50=-0.74, p=0.021) while PARP monotherapy is null (p=0.605)."
            )
        elif cls in {"Hcleanonly", "Both"}:
            sub_route = l5.get("door2_subtype_routing") or {}
            sub_name = sub_route.get("subtype") or "Subtype_2A_BCLxL_Classical"
            buf = sub_route.get("S_sen_minus_H_clean_buffer")
            buf_str = f"{buf:+.4f}" if isinstance(buf, (int, float)) else "N/A"
            if sub_route.get("subtype_2B_triggered"):
                sub_detail = (
                    f"Exploratory Subtype 2B routing triggered ({sub_name}, buffer S_sen-H_clean={buf_str}; "
                    f"triggers={sub_route.get('triggers')}): highlight Subtype 2B (Dasatinib + "
                    "Doxercalciferol/Paricalcitol; VDR 3.369 vs 1.427, EPHA2 7.566 vs 6.227, exact P=0.04444) "
                    "alongside BCL-xL (Navitoclax/ABT-737)."
                )
            else:
                sub_detail = (
                    f"Exploratory Subtype 2A routing ({sub_name}, buffer S_sen-H_clean={buf_str} >= 0): "
                    "classical BCL-xL senolytic vulnerability (Navitoclax/ABT-737) with v9 DoseResponse_Consensus_Z "
                    "co-winners Dasatinib, Doxercalciferol, Azithromycin (all 5 BH q=0.047242) + Vismodegib (q=0.035734) "
                    "and Dasatinib+Navitoclax composite (rho=+0.5279, q=0.00060)."
                )
            gate_msg = (
                "Gate 3 Door 2 (Hibernation / Senolytic) fired: therapy-induced quiescent persister "
                "state is pan-resistant to proliferative DDR/chemo (0/8 GDSC2 & 0/5 CTRPv2) but "
                "dependent on BCL2L1/BCL-xL (CRISPR rho=-0.6547, q=2.5e-5; 7-gene CRISPR vulnerability "
                "rho=-0.6645, q=0.00012; CLPP rho=-0.336, P=0.01680). "
                + sub_detail
            )
        elif cls == "DDRonly":
            gate_msg = (
                "Gate 3 Door 1 (DDR Proliferative) fired: actively cycling residual disease with "
                "elevated replication-stress DDR proxy (7/10 GDSC2 & 6/9 CTRPv2 DDR-sensitive; "
                "GSE165897 Door 1 median PFI 50.5d vs Door 2 203.5d, exact MWU p=0.02857)."
            )
        else:
            gate_msg = (
                "Gate 3 Chemo-Cleared Remission (Neither): both post_DDR and post_H_clean sit "
                "at or below cutpoints (GSE165897 Neither median PFI 490.0d)."
            )

    elo_msg = (
        f"L5 Elo ranks {top_drug} #1 ({l5.get('top_mechanism_class')}); "
        f"promoted=[{promoted}], demoted=[{demoted}]."
    )
    summary = f"{bio} {elo_msg} (RUO exploratory prioritization)."
    return {
        "biological_state_meaning": bio,
        "gate_and_compartment_meaning": gate_msg,
        "elo_ranking_meaning": elo_msg,
        "actionable_summary_ruo": summary,
    }


def _immunotherapy_deterministic_meaning(
    scored_body: dict[str, Any],
    l5: dict[str, Any],
) -> dict[str, str]:
    spec = scored_body.get("specimen")
    top_drug = l5.get("top_candidate") or "None"
    promoted = ", ".join(l5.get("promoted") or []) or "none"
    demoted = ", ".join(l5.get("demoted") or []) or "none"

    if spec == "cart_infusion_bag":
        idx3 = scored_body.get("Bag_Index_3")
        idx1 = scored_body.get("Bag_Index_1")
        low_c = scored_body.get("Bag_low_cyto")
        rflag = scored_body.get("risk_flag")
        bio = (
            f"Ex-vivo CAR-T infusion product (Deng GSE151511 v12, gate={scored_body.get('bag_gate')}) "
            f"scored Bag_Index_3={idx3:.4f}, Bag_Index_1={idx1:.4f}, Bag_low_cyto={low_c:.4f}, "
            f"risk_flag={rflag}."
        )
        gate_msg = (
            "Premature ex-vivo cytotoxic differentiation (z(f_cyto_marker)>0) flags "
            "EX_VIVO_CYTOTOXIC_DIFFERENTIATION_RISK; memory-preserved bags drive 3-mo CR "
            "(Bag_Index_3 AUROC=0.881, q=0.00831, global q=0.0191)."
        )
    elif spec == "cart_in_vivo_blood":
        prob = scored_body.get("probability")
        bio = (
            f"In-vivo leukapheresis PBMC (GSE290722 v9 6-feature Ridge, C=1.0) "
            f"predicted response probability={prob:.6f} (intercept={scored_body.get('intercept_standardized'):.6f})."
        )
        gate_msg = (
            "Combines cytotoxic-weighted ZNF683 (+0.4094, 50/50 folds) and GNLY (+0.3531, 49/50 folds) "
            "loads against baseline tumor SPD (-0.6149) and prior regimens (-1.1621) "
            "(pooled OOF AUROC=0.782, within-fold=0.823, Brier=0.199)."
        )
    else:
        mode = scored_body.get("mode") or "bulk"
        if mode == "single_cell":
            ratio = scored_body.get("TCF7_TOX_count_ratio")
            bio = (
                f"Single-cell T-cell biopsy (GSE120575 mode) scored "
                f"TCF7_TOX_count_ratio={ratio:.4f} via log2((n_TCF7+0.5)/(n_TOX+0.5))."
            )
            gate_msg = (
                "Positive log2(TCF7/TOX) indicates stem-like progenitor dominance over "
                "terminal exhaustion (internal CV AUROC=0.877)."
            )
        else:
            eff = scored_body.get("E_effector")
            triad = scored_body.get("S_triad")
            stem = scored_body.get("S_stem_eff")
            bio = (
                f"Bulk melanoma RNA biopsy ({scored_body.get('context')}) scored "
                f"E_effector={eff:.4f}, S_triad={triad:.4f}, S_stem_eff={stem:.4f}."
            )
            gate_msg = (
                "ZNF683/GNLY effector score captures tissue-resident cytolytic infiltration "
                "(Gide response AUROC=0.804 [q=0.00565], OS HR=0.370 [q=0.00667]; "
                "Riaz on-tx AUROC=0.700; TCGA-SKCM adj OS HR=0.718, +8.18m RMST)."
            )

    elo_msg = (
        f"L5 Elo ranks {top_drug} #1 ({l5.get('top_mechanism_class')}); "
        f"promoted=[{promoted}], demoted=[{demoted}]."
    )
    return {
        "biological_state_meaning": bio,
        "gate_and_compartment_meaning": gate_msg,
        "elo_ranking_meaning": elo_msg,
        "actionable_summary_ruo": f"{bio} {elo_msg} (RUO exploratory prioritization).",
    }


def interpret_patient_with_argon(
    scored_body: dict[str, Any],
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Produce deterministic Argon input + output for a scored patient (temperature=0.0)."""
    argon_input = build_argon_patient_input(scored_body)
    l5 = argon_input["l5_elo_summary"]
    if argon_input["domain"] == "ovarian_two_door_v3":
        meaning = _ovarian_deterministic_meaning(scored_body, l5)
    else:
        meaning = _immunotherapy_deterministic_meaning(scored_body, l5)

    engine_mode = "deterministic_argon_contract_engine"
    request_id = None

    active_llm = llm
    if active_llm is None and os.environ.get("ARGON_PATIENT_LIVE", "").strip() in {"1", "true", "TRUE"}:
        try:
            from oncology_arbiter.models.gemini_argon_client import GeminiArgonClient

            active_llm = GeminiArgonClient(deterministic=True)
        except Exception:
            active_llm = None

    if active_llm is not None:
        try:
            resp, parsed = chat_json_with_retry(
                active_llm,
                system=ARGON_INTERPRETER_SYSTEM,
                user=argon_input["prompt"],
                max_tokens=900,
                temperature=ARGON_TEMPERATURE,
            )
            if isinstance(parsed, dict) and parsed.get("biological_state_meaning"):
                for key in (
                    "biological_state_meaning",
                    "gate_and_compartment_meaning",
                    "elo_ranking_meaning",
                    "actionable_summary_ruo",
                ):
                    if parsed.get(key):
                        meaning[key] = str(parsed[key])
                engine_mode = "live_gemini_argon"
                request_id = getattr(resp, "request_id", None)
        except Exception:
            pass

    return {
        "ruo": True,
        "contract_status": CONTRACT_STATUS,
        "model": ARGON_MODEL_ID,
        "temperature": ARGON_TEMPERATURE,
        "deterministic": True,
        "mode": engine_mode,
        "request_id": request_id,
        "argon_input": {k: v for k, v in argon_input.items() if k != "prompt"},
        "argon_prompt": argon_input["prompt"],
        "argon_output": meaning,
    }
