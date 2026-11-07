"""ClinicalBERT acceptance gate is retired. MedGemma owns pathology extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from oncology_arbiter.nlp.parser_acceptance_gate import (
    ClinicalBertAcceptanceRetired,
    evaluate_parser_acceptance,
)


def test_parser_gate_does_not_import_clinicalbert_app_version() -> None:
    source = Path(evaluate_parser_acceptance.__code__.co_filename).read_text()
    assert "EXPECTED_APP_VERSION" not in source
    assert "0.08089260808926081" not in source
    assert "clinicalbert_modal_client" not in source


def test_parser_gate_refuses_clinicalbert_acceptance() -> None:
    with pytest.raises(ClinicalBertAcceptanceRetired):
        evaluate_parser_acceptance({"app_version": "clinicalbert-modal-v0.5.2-sliding-window"})
