"""TCGA-COADREAD MSS portfolio routing."""
from __future__ import annotations

import csv
import math
from typing import Any, Mapping

import numpy as np

from .paths import _CASPRO_ROOT
from .patient import _clip

def route_485_tcga_coadread_mss_patients(
    W_MoA: np.ndarray,
    decoded_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the learned 42-trial 8D matrix (+ comparators) to all 485 TCGA-COADREAD MSS CRC patients."""
    v2_csv = _CASPRO_ROOT / "src/data/pharma/crc-ranker-v1/crc_ranker_results_v2_gated.csv"
    with open(v2_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    decoded_42 = list(decoded_bundle["decoded_42_trials"])
    decoded_comp = dict(decoded_bundle["decoded_comparator_trials"])

    # Pre-extract unit vectors for all 42 registry entries
    t42_unit = np.array([d["learned_8d_vector_unit"] for d in decoded_42], dtype=float)  # (42, 8)
    t42_keys = [d["unique_key"] for d in decoded_42]
    t42_names = [f"[{d['registry_index']:02d}] {d['id']} — {d['trial_name']} ({d['program']})" for d in decoded_42]

    # Eligibility / clinical applicability mask for 1L/2L+ MSS CRC routing across the 42 registry entries
    # (Excludes GBM dataset analyses EM-02-A/B/C, MSI-H-only KEYNOTE-177 for non-POLE MSS, and EU-hold NCT06131840 unless raw-scored)
    crc_routable_42_mask = np.ones(42, dtype=bool)
    for i, d in enumerate(decoded_42):
        if d["entry_type"] == "DATASET_ANALYSIS":
            crc_routable_42_mask[i] = False
        elif d["unique_key"] == "NCT03050814_HIST":
            # Duplicate design precedent of row #11 QUILT-2.004
            crc_routable_42_mask[i] = False

    # Build the Precision MSS CRC Portfolio (combining the key registry programs + adjudicated comparators):
    #   1. STC-1010 BreAK CRC-001 [NCT06934538] (Vaccine + mFOLFOX6 ± Bev)
    #   2. Daraxonrasib RAS(ON) GI Platform [NCT06445062] (Pan-RAS(ON) + mFOLFOX6 + Bev)
    #   3. BREAKWATER [NCT04607421] (Encorafenib + Cetuximab + mFOLFOX6 for BRAF V600E)
    #   4. PRIME / PEAK [NCT00364013 / NCT00819780] (Panitumumab + FOLFOX for RAS/RAF-WT)
    #   5. ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316] (Berzosertib/Ceralasertib/Adavosertib for RS-High/HRD-High)
    #   6. CEACAM5 TOP1 ADC [NCT05464030 PROCEADE-CRC-01 / NCT06131840 SGN-CEACAM5C] (Low-efflux TOP1-sensitive)
    #   7. AtezoTRIBE / MEDITREME [NCT03721653 / NCT03202758] (Chemo + IO)
    #   8. TRIBE2 / BECOME / NO16966 [NCT02339116] (mFOLFOX6 + Bevacizumab SOC backbone)

    reg_42_top1_counts: dict[str, int] = {name: 0 for name in t42_names}
    reg_42_top3_counts: dict[str, int] = {name: 0 for name in t42_names}
    reg_42_mean_scores = np.zeros(42, dtype=float)

    portfolio_counts: dict[str, int] = {
        "STC-1010 BreAK CRC-001 [NCT06934538]": 0,
        "Daraxonrasib RAS(ON) [NCT06445062]": 0,
        "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]": 0,
        "CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]": 0,
        "BREAKWATER Encorafenib+Cetux+FOLFOX [NCT04607421]": 0,
        "PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]": 0,
    }
    stratum_routing_breakdown: dict[str, dict[str, int]] = {}
    patient_routing_records: list[dict[str, Any]] = []

    # Precompute unit vectors for the 6 strategic programs
    vec_stc1010 = np.array(decoded_42[38]["learned_8d_vector_unit"], dtype=float)
    vec_darax = np.array(decoded_comp["NCT06445062"]["learned_8d_vector_unit"], dtype=float)
    vec_atr = np.array(decoded_42[0]["learned_8d_vector_unit"], dtype=float)
    vec_ceacam5 = np.array(decoded_42[6]["learned_8d_vector_unit"], dtype=float)
    vec_breakwater = np.array(decoded_comp["NCT04607421"]["learned_8d_vector_unit"], dtype=float)
    vec_prime = np.array(decoded_comp["NCT00364013"]["learned_8d_vector_unit"], dtype=float)

    for r in rows:
        pid = r["patient_id"]
        tmb = float(r["tmb"]) if r.get("tmb") not in ("", "NA", "nan") else 3.3
        fga = float(r["fraction_genome_altered"]) if r.get("fraction_genome_altered") not in ("", "NA", "nan") else 0.2671
        aneu = float(r["aneuploidy_score"]) if r.get("aneuploidy_score") not in ("", "NA", "nan") else 13.0
        efflux = float(r["p_efflux"]) if r.get("p_efflux") not in ("", "NA", "nan") else 0.18
        rss = float(r["p_rss"]) if r.get("p_rss") not in ("", "NA", "nan") else 0.10
        hrd_str = r.get("hrd_Constantin") or "HRD-Intermediate"
        kras = r.get("kras_mut") in ("1", "True", "true")
        nras = r.get("nras_mut") in ("1", "True", "true")
        braf = r.get("braf_mut") in ("1", "True", "true")
        ras_raf_status = r.get("ras_raf_status") or "TRIPLE_WT"
        cms = r.get("cms_label") or "UNLABELED"
        imm_z = float(r["immune_score_rna"]) if r.get("immune_score_rna") not in ("", "NA", "nan") else -0.5481
        gate2_hot = r.get("gate2_immune_hot") in ("1", "True", "true")
        is_pole = tmb >= 25.0

        # Construct signed 8D patient state vector p_i in [-1, +1]^8
        # d1_ddr_rs: HRD-High (0.35) or high replication stress (rss >= 0.15 / fga >= 0.375)
        d1_raw = (
            (0.65 if hrd_str == "HRD-High" else (0.10 if hrd_str == "HRD-Intermediate" else -0.35))
            + (0.55 if rss >= 0.15 else (fga - 0.2671) * 1.2)
            - (0.35 if efflux > 0.22 else 0.0)
        )
        d1 = _clip(d1_raw, -1.0, 1.0)
        d2 = -0.25 if d1 > 0.35 else 0.10
        # d3_rtk_egfr_addiction: +0.90 in Triple-WT, -1.00 in KRAS/NRAS/BRAF-mutant!
        d3 = -1.00 if (kras or nras or braf) else 0.90
        # d4_ras_raf_oncogenic: +0.95 in BRAF-mut, +0.90 in KRAS/NRAS-mut, -0.75 in Triple-WT
        d4 = 0.95 if braf else (0.90 if (kras or nras) else -0.75)
        # d5_io_effector: positive in Gate-2 Immune-Hot (imm_z >= -0.0257) or POLE (tmb >= 25), negative in Gate-2 Cold
        if is_pole:
            d5 = 0.98
        elif gate2_hot:
            d5 = _clip(0.55 + 0.45 * math.tanh((imm_z - (-0.0257)) / 0.40), 0.35, 0.95)
        else:
            d5 = _clip(-0.45 + 0.40 * math.tanh((imm_z - (-0.0257)) / 0.50), -0.95, -0.15)
        # d6_stroma_myeloid_barrier: elevated in CMS4 or deep immune cold, negative in Gate-2 Hot / CMS1-2
        if cms == "CMS4":
            d6 = 0.75
        elif not gate2_hot and imm_z < -0.3839:
            d6 = 0.55
        elif gate2_hot:
            d6 = -0.40
        else:
            d6 = 0.20
        # d7_surface_adc_payload: high when efflux is low (<= 0.15, TOP1-sensitive) and non-DDR-high
        d7 = 0.85 if (efflux <= 0.15 and d1 < 0.45) else _clip((0.18 - efflux) * 3.5, -0.65, 0.45)
        # d8_angio_metabolic_safety
        d8 = 0.45

        p_i = np.array([d1, d2, d3, d4, d5, d6, d7, d8], dtype=float)

        # 1. Score all 42 registry entries via dot(p_i, t_j) / ||t_j||_2
        scores_42 = t42_unit @ p_i
        reg_42_mean_scores += scores_42

        # Apply clinical eligibility & trial-status weighting for 42-trial CRC ranking:
        # Active/recruiting & primary-program trials (BREAK_CRC_001 NCT06934538, PROCEADE-CRC-01 NCT05464030,
        # ATR_DDR GLB trials) compete directly via dot(p_i, t_j) / ||t_j||_2, while terminated/failed trials,
        # refractory/adjuvant IO_APPENDIX trials, and HISTORICAL_BENCHMARK control arms carry their trial-status offset.
        masked_scores_42 = scores_42.copy()
        for idx_t, d_entry in enumerate(decoded_42):
            if not crc_routable_42_mask[idx_t]:
                masked_scores_42[idx_t] = -99.0
            elif d_entry["id"] == "NCT02563002" and not is_pole:
                # KEYNOTE-177 is MSI-H/hypermutator only
                masked_scores_42[idx_t] = -99.0
            elif d_entry["id"] == "NCT06131840":
                # EU clinical hold (D8 Systemic) — secondary to active PROCEADE-CRC-01 unless TOP1/CEACAM5 Rescue
                masked_scores_42[idx_t] -= 0.12
            elif d_entry["program"] in {"IO_CORE", "IO_APPENDIX"}:
                if not (gate2_hot or is_pole):
                    masked_scores_42[idx_t] -= 0.45
                else:
                    # Completed/terminated comparator/appendix IO trials rank behind recruiting 1L BreAK CRC-001 (NCT06934538)
                    masked_scores_42[idx_t] -= 0.22
            elif d_entry["program"] == "HISTORICAL_BENCHMARK":
                if gate2_hot or is_pole or hrd_str == "HRD-High" or rss >= 0.15 or efflux <= 0.15:
                    masked_scores_42[idx_t] -= 0.35
            elif d_entry["program"] == "BREAK_CRC_001":
                if not (gate2_hot or is_pole):
                    # In Gate-2 Cold non-POLE MSS CRC, direct vaccination requires Stratum 4 stromal/hepatic conversion first
                    masked_scores_42[idx_t] -= 0.35

        order_42 = np.argsort(-masked_scores_42)
        top1_idx = int(order_42[0])
        reg_42_top1_counts[t42_names[top1_idx]] += 1
        for k_idx in order_42[:3]:
            reg_42_top3_counts[t42_names[int(k_idx)]] += 1

        # 2. Score the 6-Program Precision MSS CRC Routing Table (42-Trial + Comparator Synthesis)
        s_stc = float(np.dot(p_i, vec_stc1010))
        s_darax = float(np.dot(p_i, vec_darax))
        s_atr = float(np.dot(p_i, vec_atr))
        s_cea = float(np.dot(p_i, vec_ceacam5))
        s_bw = float(np.dot(p_i, vec_breakwater))
        s_prime = float(np.dot(p_i, vec_prime))

        # Biological routing across the 6 precision programs:
        # - Gate-2 Immune-Hot (n=152) or POLE (n=6): STC-1010 BreAK CRC-001 [NCT06934538] (or BREAKWATER if BRAF V600E Cold)
        # - Gate-2 Cold BRAF-mutant (n=11 incl RAS+BRAF comut): BREAKWATER [NCT04607421]
        # - Gate-2 Cold KRAS/NRAS-mutant (n=152):
        #     * If HRD-High or High-RSS (rss >= 0.15): ATR_DDR Rescue [NCT02595892 / NCT03462342]
        #     * Else: Daraxonrasib RAS(ON) [NCT06445062]
        # - Gate-2 Cold Triple-WT (n=169):
        #     * If HRD-High or High-RSS (rss >= 0.15): ATR_DDR Rescue [NCT02595892 / NCT03462342]
        #     * Else if Low-Efflux TOP1-sensitive (efflux <= 0.15): CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]
        #     * Else: PRIME / PEAK Anti-EGFR + FOLFOX [NCT00364013 / NCT00819780]
        if gate2_hot or is_pole:
            assigned_prog = "STC-1010 BreAK CRC-001 [NCT06934538]"
        elif braf:
            assigned_prog = "BREAKWATER Encorafenib+Cetux+FOLFOX [NCT04607421]"
        elif kras or nras:
            if hrd_str == "HRD-High" or rss >= 0.15:
                assigned_prog = "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]"
            else:
                assigned_prog = "Daraxonrasib RAS(ON) [NCT06445062]"
        else:
            if hrd_str == "HRD-High" or rss >= 0.15:
                assigned_prog = "ATR_DDR Rescue [NCT02595892 / NCT03462342 / NCT03579316]"
            elif efflux <= 0.15:
                assigned_prog = "CEACAM5 TOP1 ADC [NCT05464030 / NCT06131840]"
            else:
                assigned_prog = "PRIME / PEAK Anti-EGFR+FOLFOX [NCT00364013 / NCT00819780]"

        portfolio_counts[assigned_prog] += 1
        stratum_key = f"{'Gate2_Hot_or_POLE' if (gate2_hot or is_pole) else 'Gate2_Cold'} | {ras_raf_status}"
        stratum_routing_breakdown.setdefault(stratum_key, {})
        stratum_routing_breakdown[stratum_key][assigned_prog] = (
            stratum_routing_breakdown[stratum_key].get(assigned_prog, 0) + 1
        )

        patient_routing_records.append({
            "patient_id": pid,
            "ras_raf_status": ras_raf_status,
            "gate2_immune_hot": gate2_hot,
            "is_pole": is_pole,
            "hrd_status": hrd_str,
            "p_efflux": round(efflux, 4),
            "p_rss": round(rss, 4),
            "p_signed_8d": [round(float(x), 4) for x in p_i],
            "top1_registry_42_trial": t42_keys[top1_idx],
            "assigned_precision_program": assigned_prog,
            "program_scores": {
                "STC_1010_NCT06934538": round(s_stc, 4),
                "Daraxonrasib_NCT06445062": round(s_darax, 4),
                "ATR_DDR_NCT02595892": round(s_atr, 4),
                "CEACAM5_ADC_NCT05464030": round(s_cea, 4),
                "BREAKWATER_NCT04607421": round(s_bw, 4),
                "PRIME_PEAK_NCT00364013": round(s_prime, 4),
            },
        })

    reg_42_mean_scores /= max(1, len(rows))
    table_42_summary: list[dict[str, Any]] = []
    for i, d in enumerate(decoded_42):
        name_k = t42_names[i]
        c1 = reg_42_top1_counts[name_k]
        c3 = reg_42_top3_counts[name_k]
        table_42_summary.append({
            "registry_index": d["registry_index"],
            "id": d["id"],
            "unique_key": d["unique_key"],
            "trial_name": d["trial_name"],
            "program": d["program"],
            "routable_in_mss_crc": bool(crc_routable_42_mask[i]),
            "rank1_patient_count": int(c1),
            "rank1_pct_of_485": round(100.0 * c1 / len(rows), 2),
            "top3_patient_count": int(c3),
            "top3_pct_of_485": round(100.0 * c3 / len(rows), 2),
            "mean_signed_fit_across_485": round(float(reg_42_mean_scores[i]), 4),
            "primary_positive_axis": d["primary_positive_axis"],
            "primary_repelled_axis": d["primary_repelled_resistance_axis"],
        })

    return {
        "cohort": "TCGA-COADREAD MSS CRC (n=485)",
        "n_patients": len(rows),
        "precision_portfolio_routing_summary": {
            k: {"count": v, "pct": round(100.0 * v / len(rows), 2)}
            for k, v in portfolio_counts.items()
        },
        "stratum_routing_breakdown": stratum_routing_breakdown,
        "registry_42_trial_routing_table": table_42_summary,
        "sample_patient_routings_first_10": patient_routing_records[:10],
    }
