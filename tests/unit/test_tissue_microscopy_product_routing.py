"""Product tissue microscopy routing: Phikon primary, biopsy-probe secondary."""
from __future__ import annotations

import importlib

from oncology_arbiter.models import phikon_probe_wiring, tissue_microscopy_product


def test_product_tissue_loader_is_phikon_wiring() -> None:
    assert tissue_microscopy_product.PRIMARY_CAPABILITY == "phikon"
    assert tissue_microscopy_product.product_tissue_loader_module() == (
        "oncology_arbiter.models.phikon_probe_wiring"
    )
    mod = importlib.import_module(tissue_microscopy_product.product_tissue_loader_module())
    assert mod is phikon_probe_wiring
    assert mod.ARTIFACT_SHA256 == tissue_microscopy_product.PRIMARY_ARTIFACT_SHA256
    assert mod.ARTIFACT_SHA256.startswith("7c77d075")
    assert tissue_microscopy_product.SECONDARY_RESEARCH_CAPABILITY == "biopsy-probe"
    assert "primary=phikon" in tissue_microscopy_product.PRODUCT_TISSUE_HONESTY_NOTE
    assert "secondary_research=biopsy-probe" in tissue_microscopy_product.PRODUCT_TISSUE_HONESTY_NOTE
