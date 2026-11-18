"""Signed 8D bilinear trial decoder package (constants separated from logic)."""
from __future__ import annotations

from .constants import (
    CONTRACT_STATUS,
    MOA_CLASSES_9_EXT,
    MOA_CLASSES_15,
    MOA_CLASSES_24,
    RUO_DISCLAIMER,
    SIGNED_AXES_8D,
)
from .exceptions import BilinearDecoderInputError
from .patient import build_signed_8d_patient_vector
from .decoder_io import (
    bilinear_contract_path,
    contract_moa_classes,
    load_bilinear_contract,
    load_bilinear_contract_prefer_v2,
)
from .scoring import signed_cosine_similarity, signed_fit_batch, signed_fit_score
from .registry import COMPARATOR_MOA_SPECS, RECOGNIZED_CLINICAL_FEATURE_KEYS, REGISTRY_42_MOA_SPECS
from .trials import rank_42_trials_bilinear_ml, score_bilinear_ml_loto
from .cohorts import (
    ResponderCohortBlock,
    SubgroupContrastBlock,
    load_all_responder_cohorts,
    load_published_subgroup_contrasts,
)
from .fit import (
    build_biological_anchor_prior_w_moa,
    compose_w_moa_v2_from_v1_matrix,
    fit_bilinear_w_moa,
)
from .loto import (
    decode_all_trials_with_w_moa,
    run_loto_and_heldout_validation,
    run_loto_and_heldout_validation_v2,
)
from .routing import route_485_tcga_coadread_mss_patients
from .pipeline import export_bilinear_decoder_v2_contract, run_full_8d_bilinear_pipeline_and_export

__all__ = [
    "BilinearDecoderInputError",
    "CONTRACT_STATUS",
    "COMPARATOR_MOA_SPECS",
    "MOA_CLASSES_9_EXT",
    "MOA_CLASSES_15",
    "MOA_CLASSES_24",
    "RECOGNIZED_CLINICAL_FEATURE_KEYS",
    "REGISTRY_42_MOA_SPECS",
    "ResponderCohortBlock",
    "RUO_DISCLAIMER",
    "SIGNED_AXES_8D",
    "SubgroupContrastBlock",
    "bilinear_contract_path",
    "build_biological_anchor_prior_w_moa",
    "compose_w_moa_v2_from_v1_matrix",
    "export_bilinear_decoder_v2_contract",
    "build_signed_8d_patient_vector",
    "contract_moa_classes",
    "decode_all_trials_with_w_moa",
    "fit_bilinear_w_moa",
    "load_all_responder_cohorts",
    "load_bilinear_contract",
    "load_bilinear_contract_prefer_v2",
    "load_published_subgroup_contrasts",
    "rank_42_trials_bilinear_ml",
    "route_485_tcga_coadread_mss_patients",
    "run_full_8d_bilinear_pipeline_and_export",
    "run_loto_and_heldout_validation",
    "run_loto_and_heldout_validation_v2",
    "score_bilinear_ml_loto",
    "signed_cosine_similarity",
    "signed_fit_batch",
    "signed_fit_score",
]
