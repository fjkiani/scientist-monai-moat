"""Deterministic 4-gate ovarian scorer.

Constants come from ``ovarian_two_door_standalone_v3.py`` (archive member of
``MASTER_OVARIAN_HIBERNATION_V1_TO_V6.tar.gz``, sha256
``f980a2ff7f1c8a680cc0e3dafbbce0ec07a16711a5483bd620513fa7bb90a011``).
Cutpoints use strict ``>`` because ``HUMAN_POLICY['four_state_rule']`` says ties low.
"""
from __future__ import annotations

import math
from typing import Any

from oncology_arbiter.ovarian.ovarian_two_door_standalone_v3 import (
    HUMAN_POLICY,
    KEY_VALIDATION_LEDGER,
)

DDR_CUT = float(HUMAN_POLICY["post_DDR_cutpoint"])
H_CUT = float(HUMAN_POLICY["post_H_clean_cutpoint"])
RUO = (
    "RESEARCH USE ONLY. Two-door class is not a treatment order and is not "
    "an FDA-cleared diagnostic."
)


class OvarianInputError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("missing_required_inputs:" + ",".join(missing))


def _need(payload: dict[str, Any], keys: list[str]) -> dict[str, float]:
    missing = [key for key in keys if payload.get(key) is None]
    if missing:
        raise OvarianInputError(missing)
    out: dict[str, float] = {}
    for key in keys:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OvarianInputError([key])
        number = float(value)
        if not math.isfinite(number):
            raise OvarianInputError([key])
        out[key] = number
    return out


def _geometric_state(post_ddr: float, post_h: float) -> str:
    ddr_high = post_ddr > DDR_CUT
    h_high = post_h > H_CUT
    if ddr_high and h_high:
        return "Both"
    if ddr_high:
        return "DDRonly"
    if h_high:
        return "Hcleanonly"
    return "Neither"


def _door_for(state: str) -> dict[str, Any]:
    if state == "Neither":
        return {
            "decision": "CHEMO_CLEARED_REMISSION",
            "recommend": [],
            "contraindicated": [],
            "examples": ["EOC136", "EOC372"],
        }
    if state == "DDRonly":
        return {
            "decision": "DOOR_1_DDR_PROLIFERATIVE",
            "recommend": ["Ceralasertib", "Adavosertib"],
            "contraindicated": [],
            "examples": ["EOC87", "EOC349", "EOC1005", "EOC733"],
        }
    return {
        "decision": "DOOR_2_HIBERNATION_SENOLYTIC",
        "recommend": ["Navitoclax", "ABT-737"],
        "contraindicated": ["Venetoclax"],
        "examples": ["EOC540", "EOC443", "EOC227", "EOC153"],
    }


def _gate0(values: dict[str, float]) -> bool:
    return values["post_BCL2A1_z"] > 2.5 and (
        values["delta_MCL1_z"] > 2.5 or values["post_EMT_z"] > 1.0
    )


def _gate1(payload: dict[str, Any]) -> dict[str, Any] | None:
    hgvs = str(payload.get("mbd4_hgvs_c") or "")
    lof = bool(payload.get("mbd4_lof")) or ("1293delA" in hgvs) or (
        str(payload.get("mbd4_consequence") or "").lower() == "truncating"
    )
    if not lof:
        return None
    required = ["tp53_mut", "brca_wt", "hrd"]
    if any(payload.get(key) is None for key in required):
        raise OvarianInputError([key for key in required if payload.get(key) is None])
    hrd = float(payload["hrd"])
    if not (bool(payload["tp53_mut"]) and bool(payload["brca_wt"]) and hrd < 42.0):
        return {
            "fired": False,
            "reason": "MBD4 LOF present but TP53/BRCA/HRD rule not met",
            "hrd": hrd,
        }
    return {
        "fired": True,
        "rule": "CCR_2026_B092_MBD4_LOF",
        "parp_monotherapy": "demoted",
        "parp_null_p": 0.605,
        "recommend": ["Ceralasertib", "Gemcitabine"],
        "ceralasertib_ln_ic50_delta": -0.74,
        "ceralasertib_p": 0.021,
    }


def _resolve_post_scores(payload: dict[str, Any]) -> dict[str, float]:
    keys = ["post_DDR", "post_BCL2A1_z", "delta_MCL1_z", "post_EMT_z"]
    if payload.get("post_H_clean") is None:
        z_keys = ["z_stress_34_clean", "z_proliferation_106"]
        if any(payload.get(key) is None for key in z_keys):
            raise OvarianInputError(["post_H_clean"])
        z = _need(payload, z_keys)
        payload = dict(payload)
        payload["post_H_clean"] = z["z_stress_34_clean"] - z["z_proliferation_106"]
    values = _need(payload, ["post_H_clean", *keys])
    if payload.get("post_S_sen") is None and all(
        payload.get(key) is not None for key in ("z_BCL2L1", "z_PMAIP1", "z_MCL1")
    ):
        z = _need(payload, ["z_BCL2L1", "z_PMAIP1", "z_MCL1"])
        values["post_S_sen"] = (
            values["post_H_clean"] + z["z_BCL2L1"] + z["z_PMAIP1"] - z["z_MCL1"]
        )
    elif payload.get("post_S_sen") is not None:
        values["post_S_sen"] = _need(payload, ["post_S_sen"])["post_S_sen"]
    return values


def _gate2(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("z_FAP", "z_CXCL10", "mfap4_expr") if payload.get(key) is None]
    if missing:
        raise OvarianInputError(missing)
    z = _need(payload, ["z_FAP", "z_CXCL10", "mfap4_expr"])
    aacr_positive = z["z_FAP"] < 0.0 and z["z_CXCL10"] > 0.0
    mfap4_block: dict[str, Any] | None = None
    try:
        from oncology_arbiter.api.deps import score_arbiter

        scored = score_arbiter(
            "mfap4",
            {"mfap4_expr": z["mfap4_expr"]},
            stage="hgsoc-mfap4-platinum",
            require_complete=True,
        )
        mfap4_block = scored.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 — surface the locked model, do not invent a score
        mfap4_block = {"model_state": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    tcga = KEY_VALIDATION_LEDGER["tcga_aacr_os_hr_and_rmst"]
    gse = KEY_VALIDATION_LEDGER["gse32062_aacr_rmst_60m"]
    return {
        "gate": 2,
        "timepoint": "pre_treatment",
        "aacr_positive": aacr_positive,
        "rule": "z(FAP)<0 and z(CXCL10)>0",
        "mfap4_expr": z["mfap4_expr"],
        "mfap4": mfap4_block,
        "ledger": {
            "tcga_os_hr": tcga["HR"],
            "tcga_p": tcga["p"],
            "tcga_rmst_months": tcga["RMST_difference_months"],
            "gse32062_rmst_months": gse["difference_months"],
            "gse32062_rmst_ci": gse["ci"],
        },
    }


def score_ovarian_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Score one case. Missing numbers raise ``OvarianInputError``."""
    timepoint = str(payload.get("timepoint") or "post_nact")
    if timepoint not in {"pre_treatment", "post_nact"}:
        raise OvarianInputError(["timepoint"])
    result: dict[str, Any] = {
        "ruo": True,
        "ruo_disclaimer": RUO,
        "contract_status": "evidence_bound_exploratory_prioritization",
        "timepoint": timepoint,
        "patient_id": payload.get("patient_id"),
        "cutpoints": {"post_DDR": DDR_CUT, "post_H_clean": H_CUT, "ties": "low"},
        "model_name": "ovarian_two_door_v3",
        "model_state": "loaded",
    }
    if timepoint == "pre_treatment":
        result["gate2"] = _gate2(payload)
        result["four_state_class"] = None
        result["decision"] = "PRETREATMENT_STROMA_IMMUNE"
        return result

    values = _resolve_post_scores(payload)
    geometric = _geometric_state(values["post_DDR"], values["post_H_clean"])
    refractory = _gate0(values)
    if refractory:
        four_state = "PRIMARY_REFRACTORY_ESCAPE"
        decision = {
            "decision": "PRIMARY_REFRACTORY_ESCAPE",
            "recommend": [],
            "contraindicated": [],
            "archetype": "EOC3",
            "geometric_four_state": geometric,
        }
    else:
        four_state = geometric
        decision = _door_for(geometric)
        decision["geometric_four_state"] = geometric
    gate1 = _gate1(payload)
    result.update(
        {
            "scores": values,
            "gate0_primary_refractory": refractory,
            "gate1_mbd4": gate1,
            "gate3": decision,
            "four_state_class": four_state,
            "decision": decision["decision"],
            "recommend": list(decision["recommend"]),
            "contraindicated": list(decision["contraindicated"]),
        }
    )
    if gate1 and gate1.get("fired"):
        result["recommend"] = list(dict.fromkeys([*gate1["recommend"], *result["recommend"]]))
        result["parp_monotherapy"] = "demoted"
    return result
