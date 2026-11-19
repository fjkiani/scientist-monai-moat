"""Deterministic 6-Domain STC-1010 MSS CRC Biomarker Lock v2 & 4-Stratum Regimen Engine.

Implements:
  1. Exact 6-Domain gene-level / module-level / precomputed-domain scoring
     using frozen ``TCGA`` (``n=592`` all-tumor, ``n=490/492`` MSS) and ``GSE39582``
     (``n=566`` all-tumor, ``n=444/444`` MSS) ``ddof=0`` population references from
     ``stc1010_standalone_v2.py``.
  2. Integration with the 8D ``G(p, t) * PATH A`` CRC trial ranker across all 8
     CRC trials (``STC_1010``, ``MOSAIC``, ``N0147``, ``PRIME``, ``PEAK_FOLFIRI``,
     ``PACCE``, ``HORIZON_III``, ``VELOUR``).
  3. Deterministic 4-Stratum ``BreAK CRC-001`` Regimen Engineering Plan:
     - ``STRATUM_1_DIRECT_VACCINE_PERMISSIVE``
     - ``STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN`` (+ ``B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED``)
     - ``STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE`` (+ ``KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47``)
     - ``STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION``

Every output enforces ``ruo: True`` and
``contract_status: "exploratory_research_rationale"``.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping

from oncology_arbiter.arbiter.pathway_8d_ranker import (
    CRC_TRIAL_VECTORS_8D,
    ContraindicationContext,
    TCGA_CRC_ANEUPLOIDY_MEDIAN,
    TCGA_CRC_FGA_MEDIAN,
    TCGA_CRC_TMB_MEDIAN,
    build_crc_patient_vector_8d,
    rank_trials_path_a,
)
from oncology_arbiter.crc.stc1010_standalone_v2 import (
    CLASS_MODULE_MEAN,
    CLASS_MODULE_SD,
    CLASSIFIER_NESTED_CV_METRICS,
    COEF,
    DOMAINS,
    FEATURES,
    GENES,
    GENOTYPE_CONTRASTS,
    INTERCEPT,
    MODULES,
    MSS_MODULE_QUANTILES,
    REF,
    cox_results,
    prevalence_results,
    rmst_results,
    score as score_standalone_v2,
)

CONTRACT_STATUS_V2 = "exploratory_research_rationale"
RUO_DISCLAIMER_V2 = (
    "RESEARCH USE ONLY (RUO). Retrospective two-cohort (N=934 MSS CRC) biomarker "
    "stratification and regimen rationale; not a validated clinical selection rule "
    "or treatment order."
)

STRATUM_1_DIRECT_VACCINE_PERMISSIVE = "STRATUM_1_DIRECT_VACCINE_PERMISSIVE"
STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN = "STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN"
STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE = "STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE"
STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION = (
    "STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION"
)

B2M_LOSS_FLAG = "B2M_STRUCTURAL_LOSS_NK_INNATE_REQUIRED"
KRAS_G12D_CD47_FLAG = "KRAS_G12D_LOW_CD47_NO_EMPIRIC_ANTI_CD47"


class Stc1010InputError(ValueError):
    """Raised when required gene expression or domain inputs are missing or invalid."""

    def __init__(self, missing: list[str], detail: str | None = None) -> None:
        self.missing = missing
        msg = detail or ("missing_required_inputs:" + ",".join(missing))
        super().__init__(msg)


def _as_finite_float(val: Any, key: str) -> float:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        raise Stc1010InputError([key])
    f = float(val)
    if not math.isfinite(f):
        raise Stc1010InputError([key])
    return f


def _sigmoid(logit: float) -> float:
    e = math.exp(-abs(logit))
    return 1.0 / (1.0 + e) if logit >= 0.0 else e / (1.0 + e)


def _compute_from_gene_z(
    gene_z: Mapping[str, Any],
    reference: str,
    mss: bool,
) -> dict[str, Any]:
    missing = [g for g in GENES if g not in gene_z or gene_z[g] is None]
    if missing:
        raise Stc1010InputError(missing)
    z: dict[str, float] = {g: _as_finite_float(gene_z[g], g) for g in GENES}
    _, _, mm, ms, dm, ds, apm_cut, tgfb_cut = REF[reference]
    modules = {k: sum(z[g] * w for g, w in weights) for k, weights in MODULES}
    fv = [
        (modules[k] - a) / b
        for k, a, b in zip(FEATURES, CLASS_MODULE_MEAN, CLASS_MODULE_SD)
    ]
    logit = INTERCEPT + sum(w * v for w, v in zip(COEF, fv))
    probability = _sigmoid(logit)
    mz = {k: (modules[k] - a) / b for k, a, b in zip(FEATURES, mm, ms)}
    raw = (
        modules["CXCR3_Trafficking"] - modules["TGFB_CAF_Barrier"],
        modules["APM_TAP_Score"],
        -modules["VEGF_TOX_Exhaustion"],
        modules["S_triad"],
        -(mz["KRAS_IRF2_CXCL3_Axis"] + mz["SPP1_Core_Barrier"]) / 2.0,
        logit,
    )
    dz = tuple((v - a) / b for v, a, b in zip(raw, dm, ds))
    apm = all(
        v >= cut
        for v, cut in zip((raw[1], z["TAP1"], z["TAP2"], z["PSMB8"]), apm_cut)
    )
    aacr = z["FAP"] < 0.0 and z["CXCL10"] > 0.0
    hot = probability >= 0.5
    gates = {
        "CMS1_Like_MSS": hot if mss else False,
        "Trafficking_Permissive_Primed_MSS": (hot and raw[0] > 0.0 and apm) if mss else False,
        "AACR_A065_Positive_MSS": aacr if mss else False,
        "AACR_A065_APM_Intact_MSS": (aacr and apm) if mss else False,
        "D1_Plus_D2_Permissive_MSS": (raw[0] > 0.0 and raw[1] > 0.0) if mss else False,
        "v1_lowTGFB_gate_comparison": (
            hot and modules["TGFB_CAF_Barrier"] < tgfb_cut and apm
        )
        if mss
        else False,
    }
    return {
        **modules,
        **dict(zip(DOMAINS, raw)),
        "classifier_score": probability,
        "classifier_logit": logit,
        "D1_D2_Vaccine_Effector_Index": sum(dz[:2]) / 2.0,
        "D1_D2_D3_D4_Immune_Competence_Index": sum(dz[:4]) / 4.0,
        "STC1010_Readiness_Score": sum(dz) / 6.0,
        "APM_intact": apm,
        "z_FAP": z["FAP"],
        "z_CXCL10": z["CXCL10"],
        "z_TAP1": z["TAP1"],
        "z_TAP2": z["TAP2"],
        "z_PSMB8": z["PSMB8"],
        "z_B2M": z["B2M"],
        **gates,
    }


def _resolve_domain_and_module_scores(
    payload: Mapping[str, Any],
    reference: str,
    mss: bool,
) -> dict[str, Any]:
    """Resolve 6-domain and 10-module scores from log_expression, gene_z, or precomputed domains."""
    if not payload:
        raise Stc1010InputError(["log_expression", "gene_z", "domains"])

    log_expr = payload.get("log_expression")
    gene_z = payload.get("gene_z")
    domains_in = payload.get("domains")

    # Path 1: Full 42-gene log2 expression profile
    if log_expr is not None:
        if not isinstance(log_expr, Mapping) or len(log_expr) == 0:
            raise Stc1010InputError(["log_expression"])
        missing = [g for g in GENES if g not in log_expr or log_expr[g] is None]
        if missing:
            raise Stc1010InputError(missing)
        clean_expr: dict[str, float] = {
            g: _as_finite_float(log_expr[g], g) for g in GENES
        }
        scored = score_standalone_v2(clean_expr, reference, mss=mss)
        gm, gs, *_ = REF[reference]
        z_map = {
            g: (clean_expr[g] - m) / s for g, m, s in zip(GENES, gm, gs)
        }
        scored.update(
            {
                "z_FAP": z_map["FAP"],
                "z_CXCL10": z_map["CXCL10"],
                "z_TAP1": z_map["TAP1"],
                "z_TAP2": z_map["TAP2"],
                "z_PSMB8": z_map["PSMB8"],
                "z_B2M": z_map["B2M"],
                "input_mode": "log_expression_42_genes",
            }
        )
        return scored

    # Path 2: Pre-standardized 42-gene z-score profile
    if gene_z is not None:
        if not isinstance(gene_z, Mapping) or len(gene_z) == 0:
            raise Stc1010InputError(["gene_z"])
        scored = _compute_from_gene_z(gene_z, reference, mss=mss)
        scored["input_mode"] = "gene_z_42_genes"
        return scored

    # Path 3: Precomputed domain / module scores (either inside `domains` or top-level)
    src: Mapping[str, Any] = (
        domains_in if isinstance(domains_in, Mapping) and len(domains_in) > 0 else payload
    )

    # Alias resolution for D1..D6
    d1_raw = src.get("D1_trafficking", src.get("D1"))
    d2_raw = src.get("D2_presentation", src.get("D2", src.get("APM_TAP_Score")))
    d3_raw = src.get("D3_effector", src.get("D3"))
    d4_raw = src.get("D4_triad", src.get("D4", src.get("S_triad")))
    d5_raw = src.get("D5_myeloid", src.get("D5"))
    d6_raw = src.get("D6_CMS1like", src.get("D6", src.get("classifier_logit")))

    # Allow computing D1..D5 from module scores if modules are provided
    _, _, mm, ms, dm, ds, apm_cut, tgfb_cut = REF[reference]
    mod_mean_map = dict(zip(FEATURES, mm))
    mod_sd_map = dict(zip(FEATURES, ms))

    cxcr3_mod = src.get("CXCR3_Trafficking")
    tgfb_mod = src.get("TGFB_CAF_Barrier")
    if d1_raw is None and cxcr3_mod is not None and tgfb_mod is not None:
        d1_raw = _as_finite_float(cxcr3_mod, "CXCR3_Trafficking") - _as_finite_float(
            tgfb_mod, "TGFB_CAF_Barrier"
        )

    vegf_tox_mod = src.get("VEGF_TOX_Exhaustion")
    if d3_raw is None and vegf_tox_mod is not None:
        d3_raw = -_as_finite_float(vegf_tox_mod, "VEGF_TOX_Exhaustion")

    kras_axis_mod = src.get("KRAS_IRF2_CXCL3_Axis")
    spp1_core_mod = src.get("SPP1_Core_Barrier")
    if d5_raw is None and kras_axis_mod is not None and spp1_core_mod is not None:
        zk = (
            _as_finite_float(kras_axis_mod, "KRAS_IRF2_CXCL3_Axis")
            - mod_mean_map["KRAS_IRF2_CXCL3_Axis"]
        ) / mod_sd_map["KRAS_IRF2_CXCL3_Axis"]
        zs = (
            _as_finite_float(spp1_core_mod, "SPP1_Core_Barrier")
            - mod_mean_map["SPP1_Core_Barrier"]
        ) / mod_sd_map["SPP1_Core_Barrier"]
        d5_raw = -0.5 * (zk + zs)

    required_domains = {
        "D1_trafficking": d1_raw,
        "D2_presentation": d2_raw,
        "D3_effector": d3_raw,
        "D4_triad": d4_raw,
        "D5_myeloid": d5_raw,
        "D6_CMS1like": d6_raw,
    }
    missing_doms = [k for k, v in required_domains.items() if v is None]
    if missing_doms:
        raise Stc1010InputError(missing_doms)

    raw_tuple = tuple(
        _as_finite_float(required_domains[k], k) for k in DOMAINS
    )
    dz = tuple((v - a) / b for v, a, b in zip(raw_tuple, dm, ds))

    d12 = (
        _as_finite_float(src["D1_D2_Vaccine_Effector_Index"], "D1_D2_Vaccine_Effector_Index")
        if src.get("D1_D2_Vaccine_Effector_Index") is not None
        else (dz[0] + dz[1]) / 2.0
    )
    d1234 = (
        _as_finite_float(
            src["D1_D2_D3_D4_Immune_Competence_Index"],
            "D1_D2_D3_D4_Immune_Competence_Index",
        )
        if src.get("D1_D2_D3_D4_Immune_Competence_Index") is not None
        else sum(dz[:4]) / 4.0
    )
    readiness = (
        _as_finite_float(src["STC1010_Readiness_Score"], "STC1010_Readiness_Score")
        if src.get("STC1010_Readiness_Score") is not None
        else sum(dz) / 6.0
    )

    logit = raw_tuple[5]
    prob = (
        _as_finite_float(src["classifier_score"], "classifier_score")
        if src.get("classifier_score") is not None
        else _sigmoid(logit)
    )

    # Resolve APM_Intact from explicit flag or TAP1/TAP2/PSMB8 + D2_presentation vs MSS Q20
    z_tap1 = (
        _as_finite_float(src["z_TAP1"], "z_TAP1")
        if src.get("z_TAP1") is not None
        else None
    )
    z_tap2 = (
        _as_finite_float(src["z_TAP2"], "z_TAP2")
        if src.get("z_TAP2") is not None
        else None
    )
    z_psmb8 = (
        _as_finite_float(src["z_PSMB8"], "z_PSMB8")
        if src.get("z_PSMB8") is not None
        else None
    )
    z_b2m = (
        _as_finite_float(src["z_B2M"], "z_B2M")
        if src.get("z_B2M") is not None
        else None
    )

    explicit_apm = src.get("APM_intact", src.get("apm_intact"))
    if explicit_apm is not None:
        if not isinstance(explicit_apm, bool):
            raise Stc1010InputError(["apm_intact"])
        apm_intact = explicit_apm
    elif z_tap1 is not None and z_tap2 is not None and z_psmb8 is not None:
        apm_intact = bool(
            raw_tuple[1] >= apm_cut[0]
            and z_tap1 >= apm_cut[1]
            and z_tap2 >= apm_cut[2]
            and z_psmb8 >= apm_cut[3]
        )
    else:
        apm_intact = bool(raw_tuple[1] >= apm_cut[0])

    z_fap = (
        _as_finite_float(src["z_FAP"], "z_FAP")
        if src.get("z_FAP") is not None
        else None
    )
    z_cxcl10 = (
        _as_finite_float(src["z_CXCL10"], "z_CXCL10")
        if src.get("z_CXCL10") is not None
        else None
    )
    explicit_aacr = src.get("AACR_A065_Positive_MSS", src.get("aacr_a065_positive"))
    if explicit_aacr is not None:
        if not isinstance(explicit_aacr, bool):
            raise Stc1010InputError(["aacr_a065_positive"])
        aacr_pos = explicit_aacr
    elif z_fap is not None and z_cxcl10 is not None:
        aacr_pos = bool(z_fap < 0.0 and z_cxcl10 > 0.0)
    else:
        aacr_pos = False

    tgfb_val = (
        _as_finite_float(tgfb_mod, "TGFB_CAF_Barrier")
        if tgfb_mod is not None
        else None
    )
    vegf_tox_val = (
        _as_finite_float(vegf_tox_mod, "VEGF_TOX_Exhaustion")
        if vegf_tox_mod is not None
        else -raw_tuple[2]
    )
    treg_val = (
        _as_finite_float(src["Treg_Load"], "Treg_Load")
        if src.get("Treg_Load") is not None
        else None
    )
    spp1_val = (
        _as_finite_float(spp1_core_mod, "SPP1_Core_Barrier")
        if spp1_core_mod is not None
        else None
    )
    cxcr3_val = (
        _as_finite_float(cxcr3_mod, "CXCR3_Trafficking")
        if cxcr3_mod is not None
        else None
    )
    kras_axis_val = (
        _as_finite_float(kras_axis_mod, "KRAS_IRF2_CXCL3_Axis")
        if kras_axis_mod is not None
        else None
    )

    hot = prob >= 0.5
    return {
        "TGFB_CAF_Barrier": tgfb_val,
        "CXCR3_Trafficking": cxcr3_val,
        "APM_TAP_Score": raw_tuple[1],
        "VEGF_TOX_Exhaustion": vegf_tox_val,
        "Treg_Load": treg_val,
        "S_triad": raw_tuple[3],
        "KRAS_IRF2_CXCL3_Axis": kras_axis_val,
        "SPP1_Core_Barrier": spp1_val,
        **dict(zip(DOMAINS, raw_tuple)),
        "classifier_score": prob,
        "classifier_logit": logit,
        "D1_D2_Vaccine_Effector_Index": d12,
        "D1_D2_D3_D4_Immune_Competence_Index": d1234,
        "STC1010_Readiness_Score": readiness,
        "APM_intact": apm_intact,
        "CMS1_Like_MSS": bool(hot) if mss else False,
        "Trafficking_Permissive_Primed_MSS": bool(hot and raw_tuple[0] > 0.0 and apm_intact)
        if mss
        else False,
        "AACR_A065_Positive_MSS": bool(aacr_pos) if mss else False,
        "AACR_A065_APM_Intact_MSS": bool(aacr_pos and apm_intact) if mss else False,
        "D1_Plus_D2_Permissive_MSS": bool(raw_tuple[0] > 0.0 and raw_tuple[1] > 0.0)
        if mss
        else False,
        "v1_lowTGFB_gate_comparison": bool(
            hot and tgfb_val is not None and tgfb_val < tgfb_cut and apm_intact
        )
        if mss
        else False,
        "z_FAP": z_fap,
        "z_CXCL10": z_cxcl10,
        "z_TAP1": z_tap1,
        "z_TAP2": z_tap2,
        "z_PSMB8": z_psmb8,
        "z_B2M": z_b2m,
        "input_mode": "precomputed_domains",
    }


def _evaluate_4_stratum_break_crc001_plan(
    scored: Mapping[str, Any],
    *,
    reference: str,
    liver_metastasis: bool,
    b2m_truncating_lof: bool,
    kras_g12d: bool,
) -> dict[str, Any]:
    """Evaluate the deterministic 4-Stratum BreAK CRC-001 Regimen Engineering Plan."""
    q75_map = MSS_MODULE_QUANTILES[reference]["0.75"]
    q20_map = MSS_MODULE_QUANTILES[reference]["0.2"]

    d1 = float(scored["D1_trafficking"])
    apm_intact = bool(scored["APM_intact"])

    tgfb_val = scored.get("TGFB_CAF_Barrier")
    vegf_tox_val = scored.get("VEGF_TOX_Exhaustion")
    treg_val = scored.get("Treg_Load")
    spp1_val = scored.get("SPP1_Core_Barrier")

    tgfb_high_q75 = bool(
        tgfb_val is not None and float(tgfb_val) > float(q75_map["TGFB_CAF_Barrier"])
    )
    vegf_tox_high_q75 = bool(
        vegf_tox_val is not None
        and float(vegf_tox_val) > float(q75_map["VEGF_TOX_Exhaustion"])
    )
    treg_high_q75 = bool(
        treg_val is not None and float(treg_val) > float(q75_map["Treg_Load"])
    )
    spp1_high_q75 = bool(
        spp1_val is not None and float(spp1_val) > float(q75_map["SPP1_Core_Barrier"])
    )

    z_b2m = scored.get("z_B2M")
    b2m_expression_intact = (
        True
        if z_b2m is None
        else bool(float(z_b2m) >= float(q20_map.get("z_B2M", -0.843303127027565)))
    )
    b2m_intact = bool(not b2m_truncating_lof and b2m_expression_intact)

    # Evaluate each of the 4 strata boolean triggers
    stratum_4_triggered = bool(
        d1 <= 0.0 or tgfb_high_q75 or liver_metastasis
    )
    stratum_2_triggered = bool(
        d1 > 0.0 and (not apm_intact or b2m_truncating_lof)
    )
    stratum_3_triggered = bool(
        vegf_tox_high_q75 or treg_high_q75 or spp1_high_q75
    )
    stratum_1_triggered = bool(
        d1 > 0.0 and apm_intact and not stratum_4_triggered and not b2m_truncating_lof
    )

    # Primary stratum priority:
    # 1. If D1 <= 0 or TGFB > Q75 or liver_metastasis -> STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION
    # 2. Else if D1 > 0 and (not APM_Intact or B2M LOF) -> STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN
    # 3. Else if Stratum 3 barrier (VEGF_TOX > Q75 or Treg > Q75 or SPP1 > Q75) -> STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE
    # 4. Else -> STRATUM_1_DIRECT_VACCINE_PERMISSIVE
    if stratum_4_triggered:
        primary_stratum = STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION
    elif stratum_2_triggered:
        primary_stratum = STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN
    elif stratum_3_triggered:
        primary_stratum = STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE
    else:
        primary_stratum = STRATUM_1_DIRECT_VACCINE_PERMISSIVE

    active_strata: list[str] = []
    if stratum_1_triggered:
        active_strata.append(STRATUM_1_DIRECT_VACCINE_PERMISSIVE)
    if stratum_2_triggered:
        active_strata.append(STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN)
    if stratum_3_triggered:
        active_strata.append(STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE)
    if stratum_4_triggered:
        active_strata.append(STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION)

    flags: list[str] = []
    if b2m_truncating_lof:
        flags.append(B2M_LOSS_FLAG)
    if kras_g12d:
        flags.append(KRAS_G12D_CD47_FLAG)
    if bool(scored.get("AACR_A065_Positive_MSS")):
        flags.append("AACR_A065_LOW_FAP_HIGH_CXCL10_PERMISSIVE")
    if bool(scored.get("D1_Plus_D2_Permissive_MSS")):
        flags.append("D1_PLUS_D2_VACCINE_EFFECTOR_PERMISSIVE")

    mandates: list[str] = []
    if primary_stratum == STRATUM_1_DIRECT_VACCINE_PERMISSIVE:
        mandates.append(
            "DIRECT_STC1010_VACCINATION_ELIGIBLE: Chemokine trafficking exceeds CAF barrier (D1 > 0) and MHC-I/TAP antigen presentation is intact (APM_Intact == True)."
        )
    if stratum_2_triggered:
        if b2m_truncating_lof:
            mandates.append(
                f"{B2M_LOSS_FLAG}: Structural B2M loss abolishes canonical MHC-I peptide presentation; redirect away from CD8 MHC-I-restricted monotherapy toward NK/innate or MHC-II-engaging combinations."
            )
        else:
            mandates.append(
                "FOLFOX_ICD_TAP_PRIMING_LEAD_IN: D1 > 0 (CXCR3 trafficking permissive) with B2M intact, but TAP1/TAP2/PSMB8/APM_TAP_Score < MSS Q20; administer 2-cycle mFOLFOX6 oxaliplatin ICD + IFN-gamma induction lead-in before STC-1010 intradermal priming."
            )
    if stratum_3_triggered:
        reasons_s3 = []
        if vegf_tox_high_q75:
            reasons_s3.append("VEGF_TOX_Exhaustion > Q75")
        if treg_high_q75:
            reasons_s3.append("Treg_Load > Q75")
        if spp1_high_q75:
            reasons_s3.append("SPP1_Core_Barrier > Q75")
        mandates.append(
            f"BEVACIZUMAB_AND_LOW_DOSE_CYCLOPHOSPHAMIDE_MANDATE ({', '.join(reasons_s3)}): Require Bevacizumab (anti-VEGF endothelial/TOX normalization) + low-dose metronomic Cyclophosphamide (FOXP3/IL2RA Treg depletion) + myeloid conditioning."
        )
    if kras_g12d:
        mandates.append(
            f"{KRAS_G12D_CD47_FLAG}: TCGA/GSE39582 KRAS G12D MSS CRC exhibits significantly LOWER CD47 expression (TCGA raw Cliff's delta = -0.295, q = 0.0304; joint CD8+PTPRC residualized delta = -0.265, q = 0.0460); do not add empiric anti-CD47 blockade without direct tumor CD47 IHC/RNA confirmation."
        )
    if stratum_4_triggered:
        reasons_s4 = []
        if d1 <= 0.0:
            reasons_s4.append("D1_trafficking <= 0")
        if tgfb_high_q75:
            reasons_s4.append("TGFB_CAF_Barrier > Q75")
        if liver_metastasis:
            reasons_s4.append("liver_metastasis == True")
        mandates.append(
            f"STROMAL_TGFB_AND_HEPATIC_CONVERSION_REQUIRED ({', '.join(reasons_s4)}): Hold naked vaccine priming; deploy stromal TGF-beta/FAP barrier conversion and/or hepatic-directed chemo-anti-VEGF conditioning prior to STC-1010."
        )

    return {
        "ruo": True,
        "contract_status": CONTRACT_STATUS_V2,
        "primary_stratum": primary_stratum,
        "active_strata": active_strata,
        "stratum_triggers": {
            STRATUM_1_DIRECT_VACCINE_PERMISSIVE: stratum_1_triggered,
            STRATUM_2_FOLFOX_TAP_PRIMING_LEAD_IN: stratum_2_triggered,
            STRATUM_3_BEVACIZUMAB_AND_CY_MANDATE: stratum_3_triggered,
            STRATUM_4_STROMAL_TGFB_AND_HEPATIC_CONVERSION: stratum_4_triggered,
        },
        "barrier_flags": {
            "tgfb_caf_barrier_gt_q75": tgfb_high_q75,
            "vegf_tox_exhaustion_gt_q75": vegf_tox_high_q75,
            "treg_load_gt_q75": treg_high_q75,
            "spp1_core_barrier_gt_q75": spp1_high_q75,
            "b2m_intact": b2m_intact,
            "b2m_truncating_lof": b2m_truncating_lof,
            "liver_metastasis": liver_metastasis,
            "kras_g12d": kras_g12d,
        },
        "flags": flags,
        "regimen_mandates": mandates,
    }


def score_stc1010_crc_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Score an MSS CRC case under ``stc1010_mss_crc_biomarker_lock_v2`` + 8D ``PATH A``."""
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise Stc1010InputError(["log_expression", "gene_z", "domains"])

    reference = str(payload.get("reference") or "TCGA").strip()
    if reference not in REF:
        raise Stc1010InputError(
            ["reference"], detail=f"Invalid reference={reference!r}; expected 'TCGA' or 'GSE39582'."
        )

    mss_raw = payload.get("mss", True)
    if not isinstance(mss_raw, bool):
        raise Stc1010InputError(["mss"])
    mss = bool(mss_raw)

    scored = _resolve_domain_and_module_scores(payload, reference, mss)

    kras_mut = bool(payload.get("kras_mut", False) or payload.get("kras_g12d", False))
    kras_g12d = bool(payload.get("kras_g12d", False))
    nras_mut = bool(payload.get("nras_mut", False))
    braf_mut = bool(payload.get("braf_mut", False) or payload.get("braf_v600e", False))
    b2m_truncating_lof = bool(payload.get("b2m_truncating_lof", False))
    liver_metastasis = bool(payload.get("liver_metastasis", False))

    hrd_proxy = str(payload.get("hrd_proxy") or "HRD-Intermediate")
    brca_somatic = bool(payload.get("brca_somatic", False))
    tmb = (
        _as_finite_float(payload["tmb"], "tmb")
        if payload.get("tmb") is not None
        else TCGA_CRC_TMB_MEDIAN
    )
    aneuploidy_score = (
        _as_finite_float(payload["aneuploidy_score"], "aneuploidy_score")
        if payload.get("aneuploidy_score") is not None
        else TCGA_CRC_ANEUPLOIDY_MEDIAN
    )
    fraction_genome_altered = (
        _as_finite_float(payload["fraction_genome_altered"], "fraction_genome_altered")
        if payload.get("fraction_genome_altered") is not None
        else TCGA_CRC_FGA_MEDIAN
    )
    mapk_convention = str(payload.get("mapk_convention") or "f07")

    regimen_plan = _evaluate_4_stratum_break_crc001_plan(
        scored,
        reference=reference,
        liver_metastasis=liver_metastasis,
        b2m_truncating_lof=b2m_truncating_lof,
        kras_g12d=kras_g12d,
    )

    d1_val = float(scored["D1_trafficking"])
    d2_val = float(scored["D2_presentation"])
    d3_val = float(scored["D3_effector"])
    d4_val = float(scored["D4_triad"])
    d5_val = float(scored["D5_myeloid"])
    d6_val = float(scored["D6_CMS1like"])
    d12_val = float(scored["D1_D2_Vaccine_Effector_Index"])
    d1234_val = float(scored["D1_D2_D3_D4_Immune_Competence_Index"])
    apm_intact = bool(scored["APM_intact"])
    aacr_pos = bool(scored["AACR_A065_Positive_MSS"])

    # Build 8D patient vector using 6-domain v2 D1 + APM_Intact -> io mapping
    patient_vector_8d = build_crc_patient_vector_8d(
        hrd_proxy=hrd_proxy,
        brca_somatic=brca_somatic,
        tmb=tmb,
        aneuploidy_score=aneuploidy_score,
        fraction_genome_altered=fraction_genome_altered,
        kras_mut=kras_mut,
        nras_mut=nras_mut,
        braf_mut=braf_mut,
        d1_trafficking=d1_val,
        d2_presentation=d2_val,
        d3_effector=d3_val,
        d4_triad=d4_val,
        d5_myeloid=d5_val,
        d6_cms1like=d6_val,
        d1_d2_vaccine_effector_index=d12_val,
        apm_intact=apm_intact,
        mode="stc1010_v2_6domain",
        mapk_convention=mapk_convention,
    )

    ctx = ContraindicationContext(
        kras_mut=kras_mut,
        nras_mut=nras_mut,
        braf_mut=braf_mut,
        mss=mss,
        liver_met=liver_metastasis,
        tgfb_barrier_high=bool(regimen_plan["barrier_flags"]["tgfb_caf_barrier_gt_q75"]),
        gate2_immune_hot=bool(d1_val > 0.0 and apm_intact),
        d1_trafficking=d1_val,
        apm_intact=apm_intact,
        b2m_truncating_lof=b2m_truncating_lof,
    )

    ranked_trials = rank_trials_path_a(
        patient_vector_8d,
        CRC_TRIAL_VECTORS_8D,
        ctx=ctx,
        apply_gate=True,
        use_axis_mask_for_ranking=bool(payload.get("use_axis_mask_for_ranking", False)),
    )

    return {
        "ruo": True,
        "ruo_disclaimer": RUO_DISCLAIMER_V2,
        "contract_status": CONTRACT_STATUS_V2,
        "model_name": "stc1010_mss_crc_biomarker_lock_v2",
        "model_state": "loaded",
        "patient_id": payload.get("patient_id"),
        "reference": reference,
        "input_mode": scored["input_mode"],
        "mss": mss,
        "domains": {
            "D1_trafficking": d1_val,
            "D2_presentation": d2_val,
            "D3_effector": d3_val,
            "D4_triad": d4_val,
            "D5_myeloid": d5_val,
            "D6_CMS1like": d6_val,
            "D1_D2_Vaccine_Effector_Index": d12_val,
            "D1_D2_D3_D4_Immune_Competence_Index": d1234_val,
            "STC1010_Readiness_Score": float(scored["STC1010_Readiness_Score"]),
            "classifier_score": float(scored["classifier_score"]),
            "classifier_logit": float(scored["classifier_logit"]),
        },
        "modules": {
            k: scored.get(k)
            for k in FEATURES
            if scored.get(k) is not None
        },
        "gates": {
            "APM_Intact": apm_intact,
            "AACR_A065_Positive": aacr_pos,
            "AACR_A065_APM_Intact_MSS": bool(scored["AACR_A065_APM_Intact_MSS"]),
            "CMS1_Like_MSS": bool(scored["CMS1_Like_MSS"]),
            "Trafficking_Permissive_Primed_MSS": bool(
                scored["Trafficking_Permissive_Primed_MSS"]
            ),
            "D1_Plus_D2_Permissive_MSS": bool(scored["D1_Plus_D2_Permissive_MSS"]),
        },
        "pathway_8d": {
            "axes": ("ddr", "mapk", "pi3k", "vegf", "her2", "io", "efflux", "rss"),
            "patient_vector_8d": list(patient_vector_8d),
            "top_trial": ranked_trials[0].trial_id,
            "stc1010_rank": next(
                r.rank for r in ranked_trials if r.trial_id == "STC_1010"
            ),
            "stc1010_gated_fit": next(
                r.gated_fit for r in ranked_trials if r.trial_id == "STC_1010"
            ),
            "ranked_trials": [asdict(r) for r in ranked_trials],
        },
        "break_crc001_regimen_plan": regimen_plan,
        "locked_evidence_summary": {
            "cohorts": "TCGA-COADREAD (490/492 MSS, 592 all-tumor) + GSE39582 (444/444 MSS, 566 all-tumor) = 934 MSS CRC",
            "classifier_nested_cv": {
                k: dict(v) for k, v in CLASSIFIER_NESTED_CV_METRICS.items()
            },
            "genotype_contrasts": {
                k: {sk: dict(sv) for sk, sv in v.items()}
                for k, v in GENOTYPE_CONTRASTS.items()
            },
            "n_prevalence_rows": len(prevalence_results()),
            "n_cox_rows": len(cox_results()),
            "n_rmst_rows": len(rmst_results()),
        },
    }
