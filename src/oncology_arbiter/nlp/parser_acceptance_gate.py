"""ClinicalBERT parser acceptance is retired.

Pathology text extraction is MedGemma structured JSON
(oncology_arbiter.nlp.medgemma_pathology_client.extract_pathology).
This module does not import a ClinicalBERT app version and does not score NER F1.
"""
from __future__ import annotations


class ClinicalBertAcceptanceRetired(RuntimeError):
    """Raised if a caller still asks this module to accept a ClinicalBERT deployment."""


def evaluate_parser_acceptance(*_args: object, **_kwargs: object) -> None:
    raise ClinicalBertAcceptanceRetired(
        "ClinicalBERT parser acceptance is retired; pathology extraction uses MedGemma"
    )
