"""Three-compartment immunotherapy scorer.

Coefficients and transforms are loaded from the archive JSON contracts, not
from rounded prose:

* ``cart_hobit_gnly_clinical_contract_v1.json``
* ``zeta_immunotherapy_external_validation_v3.json``
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONTRACTS = Path(__file__).resolve().parent / "contracts"
_ARBITER_MODELS = Path(__file__).resolve().parents[1] / "arbiter" / "models"
CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
RUO = (
    "RESEARCH USE ONLY. Compartment scores are not a treatment order and are "
    "not an FDA-cleared diagnostic."
)


class ImmunoInputError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("missing_required_inputs:" + ",".join(missing))


@lru_cache(maxsize=1)
def load_arbiter_v3_contract() -> dict[str, Any]:
    path = _ARBITER_MODELS / "zeta_immunotherapy_contract_v3.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_cart_contract() -> dict[str, Any]:
    path = _CONTRACTS / "cart_hobit_gnly_clinical_contract_v1.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_v3_contract() -> dict[str, Any]:
    path = _CONTRACTS / "zeta_immunotherapy_external_validation_v3.json"
    return json.loads(path.read_text())


def _finite(payload: dict[str, Any], key: str) -> float:
    if key not in payload or payload[key] is None:
        raise ImmunoInputError([key])
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImmunoInputError([key])
    number = float(value)
    if not math.isfinite(number):
        raise ImmunoInputError([key])
    return number


def _z(value: float, spec: dict[str, float]) -> float:
    sd = float(spec["sd"])
    if sd == 0.0:
        return 0.0
    return (value - float(spec["mean"])) / sd


def score_infusion_bag(payload: dict[str, Any]) -> dict[str, Any]:
    bag_contract = load_arbiter_v3_contract()["compartments"]["cart_infusion_bag"]
    gate = str(payload.get("bag_gate") or "literal_CD8")
    transforms = bag_contract["transforms"]
    if gate not in transforms:
        raise ImmunoInputError(["bag_gate"])
    need = ["n_CD8", "M_ratio", "M_expr", "f_cyto_marker", "n_memory", "n_cyto", "n_exhausted"]
    missing = [key for key in need if key not in payload or payload[key] is None]
    if missing:
        raise ImmunoInputError(missing)
    n_cd8 = _finite(payload, "n_CD8")
    if n_cd8 <= 0:
        raise ImmunoInputError(["n_CD8"])
    values = {key: _finite(payload, key) for key in need if key != "n_CD8"}
    spec = transforms[gate]
    z_ratio = _z(values["M_ratio"], spec["M_ratio"])
    z_cyto = _z(values["f_cyto_marker"], spec["f_cyto_marker"])
    index3 = math.log2(
        (values["n_memory"] + 0.5) / (values["n_cyto"] + values["n_exhausted"] + 0.5)
    )
    high_cyto = z_cyto > 0.0
    return {
        "ruo": True,
        "ruo_disclaimer": RUO,
        "contract_status": CONTRACT_STATUS,
        "specimen": "cart_infusion_bag",
        "model_state": "loaded",
        "bag_gate": gate,
        "Bag_Index_1": 0.5 * (z_ratio - z_cyto),
        "Bag_Index_2": 0.5 * (_z(values["M_expr"], spec["M_expr"]) - z_cyto),
        "Bag_Index_3": index3,
        "Bag_low_cyto": 1.0 - values["f_cyto_marker"],
        "flag": bag_contract["legacy_risk_flag_alias"] if high_cyto else None,
        "risk_flag": bag_contract["risk_flag_high_f_cyto_marker"] if high_cyto else None,
    }


_BLOOD_ALIASES = {
    "log_baseline_spd": ("log_baseline_spd", "log1p_spd"),
    "prior_autologous_transplant": ("prior_autologous_transplant", "prior_auto_sct"),
}


def score_in_vivo_blood(payload: dict[str, Any]) -> dict[str, Any]:
    contract = load_arbiter_v3_contract()["compartments"]["cart_in_vivo_blood"]
    names = list(contract["feature_order"])
    prep = contract["preprocessing"]
    normalized = dict(payload)
    for canonical, aliases in _BLOOD_ALIASES.items():
        if normalized.get(canonical) is None:
            for alias in aliases:
                if normalized.get(alias) is not None:
                    normalized[canonical] = normalized[alias]
                    break
    missing = [name for name in names if name not in normalized or normalized[name] is None]
    if missing:
        raise ImmunoInputError(missing)
    values: list[float] = []
    for name in names:
        raw = normalized[name]
        if isinstance(raw, bool):
            raw = int(raw)
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ImmunoInputError([name])
        value = float(raw)
        if name.startswith("load_") and value < 0:
            raise ImmunoInputError([name])
        if name == "log_baseline_spd" and value <= 0:
            raise ImmunoInputError([name])
        if name == "prior_regimens" and (value < 0 or value != int(value)):
            raise ImmunoInputError([name])
        if name in {"cyto_absent", "prior_autologous_transplant"} and value not in (0.0, 1.0):
            raise ImmunoInputError([name])
        values.append(value)
    if values[names.index("cyto_absent")] == 1.0 and any(
        values[names.index(gene)] != 0.0 for gene in ("load_ZNF683", "load_GNLY")
    ):
        raise ImmunoInputError(["load_ZNF683", "load_GNLY"])
    standardized = [
        (value - float(prep["mean"][index])) / float(prep["scale"][index])
        for index, value in enumerate(values)
    ]
    coefs = list(contract["coefficients_standardized_list"])
    linear = float(contract["intercept_standardized"]) + sum(
        float(beta) * zee
        for beta, zee in zip(coefs, standardized, strict=True)
    )
    if linear >= 0:
        probability = 1.0 / (1.0 + math.exp(-linear))
    else:
        exp_linear = math.exp(linear)
        probability = exp_linear / (1.0 + exp_linear)
    return {
        "ruo": True,
        "ruo_disclaimer": RUO,
        "contract_status": CONTRACT_STATUS,
        "specimen": "cart_in_vivo_blood",
        "model_state": "loaded",
        "model_name": "cart_hobit_gnly_clinical_contract_v1",
        "selected_C": contract["selected_C"],
        "intercept_standardized": contract["intercept_standardized"],
        "coefficients_standardized": coefs,
        "probability": probability,
        "feature_order": names,
    }


def _gene_z(expression: dict[str, Any], spec: dict[str, Any], gene: str) -> float:
    if gene not in expression or expression[gene] is None:
        raise ImmunoInputError([gene])
    value = expression[gene]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImmunoInputError([gene])
    number = float(value)
    if not math.isfinite(number):
        raise ImmunoInputError([gene])
    return _z(number, spec[gene])


def _population_scale(context: str) -> dict[str, Any]:
    ici_contract = load_arbiter_v3_contract()["compartments"]["ici_tumor_biopsy"]
    if context not in ici_contract["melanoma_transforms"]:
        raise ImmunoInputError(["context"])
    population_key = context.replace("|", "_")
    scales = ici_contract["cohort_population_transforms"]
    block = scales.get(population_key, {}).get("population_transforms")
    if not isinstance(block, dict):
        raise ImmunoInputError(["context"])
    return block


def score_ici_biopsy(payload: dict[str, Any]) -> dict[str, Any]:
    ici_contract = load_arbiter_v3_contract()["compartments"]["ici_tumor_biopsy"]
    mode = str(payload.get("mode") or "bulk")
    if mode == "single_cell":
        n_tcf7 = _finite(payload, "n_TCF7_positive")
        n_tox = _finite(payload, "n_TOX_positive")
        return {
            "ruo": True,
            "ruo_disclaimer": RUO,
            "contract_status": CONTRACT_STATUS,
            "specimen": "ici_tumor_biopsy",
            "mode": "single_cell",
            "model_state": "loaded",
            "TCF7_TOX_count_ratio": math.log2((n_tcf7 + 0.5) / (n_tox + 0.5)),
            "formula": "log2((n_TCF7_positive+0.5)/(n_TOX_positive+0.5))",
        }
    if mode != "bulk":
        raise ImmunoInputError(["mode"])
    context = str(payload.get("context") or "")
    spec = _population_scale(context)
    expression = payload.get("expression")
    if not isinstance(expression, dict):
        raise ImmunoInputError(["expression"])
    triad = ici_contract["S_triad"]
    genes = ["TCF7", "ZNF683", "GNLY", *triad["positive"], *triad["negative"]]
    ordered: list[str] = []
    for gene in genes:
        if gene not in ordered:
            ordered.append(gene)
    missing_scale = [gene for gene in ordered if gene not in spec]
    if missing_scale:
        raise ImmunoInputError(missing_scale)
    zz = {gene: _gene_z(expression, spec, gene) for gene in ordered}
    effector = 0.5 * (zz["ZNF683"] + zz["GNLY"])
    positive = [zz[gene] for gene in triad["positive"]]
    negative = [zz[gene] for gene in triad["negative"]]
    return {
        "ruo": True,
        "ruo_disclaimer": RUO,
        "contract_status": CONTRACT_STATUS,
        "specimen": "ici_tumor_biopsy",
        "mode": "bulk",
        "model_state": "loaded",
        "context": context,
        "E_effector": effector,
        "S_stem_eff": (zz["TCF7"] + zz["ZNF683"] + zz["GNLY"]) / 3.0,
        "S_stem_eff_TCF7_half": 0.5 * zz["TCF7"] + 0.5 * effector,
        "S_triad": (sum(positive) / len(positive)) - (sum(negative) / len(negative)),
        "S_triad_positive": list(triad["positive"]),
        "S_triad_negative": list(triad["negative"]),
    }


def score_immunotherapy(payload: dict[str, Any]) -> dict[str, Any]:
    specimen = payload.get("specimen")
    if specimen == "cart_infusion_bag":
        return score_infusion_bag(payload)
    if specimen == "cart_in_vivo_blood":
        return score_in_vivo_blood(payload)
    if specimen == "ici_tumor_biopsy":
        return score_ici_biopsy(payload)
    raise ImmunoInputError(["specimen"])
