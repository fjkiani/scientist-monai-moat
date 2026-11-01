"""Product routing for histology / tissue microscopy (WSI patches).

Primary product path: Owkin Phikon embedding + ``phikon_probe_wiring`` (NCT-CRC
nine-class head, ``models/phikon_probe_v1.joblib``).

Secondary research path: MedSigLIP + ``biopsy_probe_v1_wiring`` (BACH three-class
head). Small-cohort research probe — not the product tissue classifier.
"""
from __future__ import annotations

from oncology_arbiter.models import phikon_probe_wiring
from oncology_arbiter.models.biopsy_probe_v1_wiring import CAPABILITY as BIOPSY_CAPABILITY

PRIMARY_CAPABILITY = phikon_probe_wiring.CAPABILITY
PRIMARY_WIRING_MODULE = "oncology_arbiter.models.phikon_probe_wiring"
PRIMARY_ARTIFACT_PATH = phikon_probe_wiring.ARTIFACT_PATH
PRIMARY_ARTIFACT_SHA256 = phikon_probe_wiring.ARTIFACT_SHA256

SECONDARY_RESEARCH_CAPABILITY = BIOPSY_CAPABILITY

PRODUCT_TISSUE_HONESTY_NOTE = (
    "product_tissue_microscopy:primary=phikon+NCT-CRC_probe_v1;"
    "secondary_research=biopsy-probe_v1_MedSigLIP_BACH;"
    "subtype_prediction_when_present_is_BACH_research_only"
)


def product_tissue_loader_module() -> str:
    """Import path of the identity-locked product tissue probe wiring."""
    return PRIMARY_WIRING_MODULE


def product_tissue_model_name() -> str:
    """Stable product label for provenance when Phikon+probe succeeded."""
    prefix = PRIMARY_ARTIFACT_SHA256[:12]
    return f"owkin/phikon+{phikon_probe_wiring.ARTIFACT_FILENAME}@sha256:{prefix}"


__all__ = [
    "PRIMARY_ARTIFACT_PATH",
    "PRIMARY_ARTIFACT_SHA256",
    "PRIMARY_CAPABILITY",
    "PRIMARY_WIRING_MODULE",
    "PRODUCT_TISSUE_HONESTY_NOTE",
    "SECONDARY_RESEARCH_CAPABILITY",
    "product_tissue_loader_module",
    "product_tissue_model_name",
]
