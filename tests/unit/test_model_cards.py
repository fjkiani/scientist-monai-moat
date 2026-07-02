"""Structural tests for oncology-arbiter model cards.

These tests do NOT run any model. They gate the *content* of the four
model-card markdown files that live under ``docs/model_cards/``, ensuring
each card:

- names the correct HuggingFace repo id,
- cites the correct arxiv key publication,
- states the correct model version / creation date,
- carries the ``mammography imagery is NOT`` disclosure where applicable,
- carries the ``google/medsiglip-448`` breast-cancer AUC row verbatim
  in the MedSigLIP card,
- carries the correct ``MedQA (4-op)`` scores verbatim in the two
  MedGemma cards,
- explicitly clarifies that the pathology "Invasive Breast Cancer" row
  is NOT mammography,
- references the honesty invariant symbols (``RUO_DISCLAIMER``,
  ``AUROC_CAVEAT``, ``ModelState``) so future edits cannot silently
  strip them.

If any of these tests fail, we have either shipped a stale card or
introduced a factual drift. Fix the card, don't relax the test.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "model_cards"

MEDSIGLIP = DOCS / "medsiglip_448.md"
MEDGEMMA_1_5 = DOCS / "medgemma_1_5_4b.md"
MEDGEMMA_27 = DOCS / "medgemma_1_27b.md"
SIGLIP_PROXY = DOCS / "siglip_base_patch16_224.md"
ERRATA = DOCS / "errata.md"


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [MEDSIGLIP, MEDGEMMA_1_5, MEDGEMMA_27, SIGLIP_PROXY, ERRATA],
    ids=lambda p: p.name,
)
def test_card_exists(path: pathlib.Path) -> None:
    assert path.exists(), f"Missing model card: {path}"
    assert path.stat().st_size > 500, f"Card is suspiciously small: {path}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def medsiglip_text() -> str:
    return MEDSIGLIP.read_text()


@pytest.fixture(scope="module")
def medgemma_1_5_text() -> str:
    return MEDGEMMA_1_5.read_text()


@pytest.fixture(scope="module")
def medgemma_27_text() -> str:
    return MEDGEMMA_27.read_text()


@pytest.fixture(scope="module")
def siglip_proxy_text() -> str:
    return SIGLIP_PROXY.read_text()


# ---------------------------------------------------------------------------
# MedSigLIP card
# ---------------------------------------------------------------------------


def test_medsiglip_repo_id_is_correct(medsiglip_text: str) -> None:
    assert "google/medsiglip-448" in medsiglip_text


def test_medsiglip_arxiv_publication(medsiglip_text: str) -> None:
    assert "arXiv:2507.05201" in medsiglip_text or "2507.05201" in medsiglip_text


def test_medsiglip_version(medsiglip_text: str) -> None:
    assert "1.0.0" in medsiglip_text
    assert "July 9, 2025" in medsiglip_text


def test_medsiglip_resolution(medsiglip_text: str) -> None:
    assert re.search(r"448\s*[×x]\s*448", medsiglip_text) is not None
    assert "64 text tokens" in medsiglip_text or "64 tokens" in medsiglip_text


def test_medsiglip_400M_encoders(medsiglip_text: str) -> None:
    # Vision + text encoders both 400M
    assert medsiglip_text.count("400M") >= 2


def test_medsiglip_breast_cancer_row_verbatim(medsiglip_text: str) -> None:
    """The 0.933 zero-shot / 0.930 linear probe row on histopathology
    invasive breast cancer must appear verbatim, together with n=5000
    and 3 classes so nobody misreads it as a mammography number."""
    assert "0.933" in medsiglip_text
    assert "0.930" in medsiglip_text
    assert "5,000" in medsiglip_text
    assert "Invasive Breast Cancer" in medsiglip_text or "Invasive breast cancer" in medsiglip_text


def test_medsiglip_disclaims_mammography(medsiglip_text: str) -> None:
    """Must explicitly say mammography is NOT in the training data
    AND that the 0.933 breast AUC is pathology, not mammography."""
    lowered = medsiglip_text.lower()
    assert "mammography" in lowered
    # Some form of "not in training data" disclosure
    assert re.search(
        r"(mammograph\w+ (imag\w+|is|are)\s+not|not\s+listed\s+in\s+the\s+training)",
        lowered,
    ), "MedSigLIP card must state mammography is NOT in training data"
    # Must explicitly say pathology, not mammography
    assert re.search(r"histopatholog\w+\s+patches", lowered) or "pathology patches" in lowered


def test_medsiglip_references_honesty_symbols(medsiglip_text: str) -> None:
    assert "RUO_DISCLAIMER" in medsiglip_text
    assert "AUROC_CAVEAT" in medsiglip_text
    assert "ModelState" in medsiglip_text


# ---------------------------------------------------------------------------
# MedGemma 1.5 4B card
# ---------------------------------------------------------------------------


def test_medgemma_1_5_repo_id(medgemma_1_5_text: str) -> None:
    assert "google/medgemma-1.5-4b-it" in medgemma_1_5_text


def test_medgemma_1_5_arxiv(medgemma_1_5_text: str) -> None:
    assert "2507.05201" in medgemma_1_5_text


def test_medgemma_1_5_version(medgemma_1_5_text: str) -> None:
    assert "1.5.0" in medgemma_1_5_text
    assert "Jan 13, 2026" in medgemma_1_5_text
    assert "May 20, 2025" in medgemma_1_5_text  # initial release
    assert "July 9, 2025" in medgemma_1_5_text  # EOI bug fix


def test_medgemma_1_5_medqa_score_verbatim(medgemma_1_5_text: str) -> None:
    """MedQA (4-op) MedGemma 1.5 4B score = 69.1 per card."""
    assert "MedQA" in medgemma_1_5_text
    assert "69.1" in medgemma_1_5_text


def test_medgemma_1_5_disclaims_mammography(medgemma_1_5_text: str) -> None:
    lowered = medgemma_1_5_text.lower()
    assert "mammography" in lowered
    assert re.search(
        r"(mammograph\w+ (imag\w+|is|are)\s+not|not\s+listed\s+in\s+the\s+training)",
        lowered,
    )


def test_medgemma_1_5_references_honesty_symbols(medgemma_1_5_text: str) -> None:
    assert "RUO_DISCLAIMER" in medgemma_1_5_text
    assert "AUROC_CAVEAT" in medgemma_1_5_text
    assert "ModelState" in medgemma_1_5_text


# ---------------------------------------------------------------------------
# MedGemma 1 27B card
# ---------------------------------------------------------------------------


def test_medgemma_27_repo_id(medgemma_27_text: str) -> None:
    assert "google/medgemma-27b-it" in medgemma_27_text


def test_medgemma_27_medqa_score_verbatim(medgemma_27_text: str) -> None:
    """MedQA (4-op) MedGemma 1 27B score = 85.3 per card."""
    assert "MedQA" in medgemma_27_text
    assert "85.3" in medgemma_27_text


def test_medgemma_27_disclaims_mammography(medgemma_27_text: str) -> None:
    lowered = medgemma_27_text.lower()
    assert "mammography" in lowered
    assert "not listed" in lowered


def test_medgemma_27_references_honesty_symbols(medgemma_27_text: str) -> None:
    assert "RUO_DISCLAIMER" in medgemma_27_text
    assert "AUROC_CAVEAT" in medgemma_27_text


# ---------------------------------------------------------------------------
# SigLIP proxy card
# ---------------------------------------------------------------------------


def test_siglip_proxy_repo_id(siglip_proxy_text: str) -> None:
    assert "google/siglip-base-patch16-224" in siglip_proxy_text


def test_siglip_proxy_is_ungated(siglip_proxy_text: str) -> None:
    """The whole point of the proxy is that it is ungated."""
    lowered = siglip_proxy_text.lower()
    assert "ungated" in lowered


def test_siglip_proxy_declares_not_medical(siglip_proxy_text: str) -> None:
    """Proxy card MUST warn that it is not a medical model."""
    lowered = siglip_proxy_text.lower()
    assert "not a medical model" in lowered or "no medical imagery" in lowered or "general-domain" in lowered


def test_siglip_proxy_bans_medsiglip_substitution_claim(siglip_proxy_text: str) -> None:
    """Must state that proxy output MUST NOT be reported as MedSigLIP output."""
    # Match across newlines because the negative directive typically spans lines
    assert re.search(
        r"do\s+not\s+report\s+proxy.*medsiglip|"
        r"not.*medsiglip.*(output|mammography\s+performance)|"
        r"MUST\s+NOT.*medsiglip",
        siglip_proxy_text,
        re.IGNORECASE | re.DOTALL,
    ) is not None


def test_siglip_proxy_references_proxy_state(siglip_proxy_text: str) -> None:
    """Proxy card must reference the audit envelope ModelState symbol."""
    assert "PROXY_SIGLIP" in siglip_proxy_text or "ModelState.PROXY_SIGLIP" in siglip_proxy_text


# ---------------------------------------------------------------------------
# Cross-card invariants
# ---------------------------------------------------------------------------


def test_no_card_claims_mammography_auc(
    medsiglip_text: str,
    medgemma_1_5_text: str,
    medgemma_27_text: str,
    siglip_proxy_text: str,
) -> None:
    """No card may claim a mammography-specific AUC. If Google publishes
    such a number in a future card update, this test breaks intentionally
    so we can add the correct citation."""
    forbidden = re.compile(
        r"mammograph\w+\s+(AUC|AUROC|zero.?shot)\s*(?:of\s*|=\s*)?0\.\d+",
        re.IGNORECASE,
    )
    for name, text in [
        ("medsiglip", medsiglip_text),
        ("medgemma_1_5", medgemma_1_5_text),
        ("medgemma_27", medgemma_27_text),
        ("siglip_proxy", siglip_proxy_text),
    ]:
        assert forbidden.search(text) is None, (
            f"{name} card contains an unattributed mammography AUC — "
            "either cite it from the source model card explicitly or remove it"
        )


def test_all_cards_reference_hai_def_gating(
    medsiglip_text: str,
    medgemma_1_5_text: str,
    medgemma_27_text: str,
) -> None:
    """All three Google-Health cards must document HAI-DEF terms + gated access."""
    for name, text in [
        ("medsiglip", medsiglip_text),
        ("medgemma_1_5", medgemma_1_5_text),
        ("medgemma_27", medgemma_27_text),
    ]:
        assert "HAI-DEF" in text or "Health AI Developer Foundations" in text, (
            f"{name} card must reference HAI-DEF terms"
        )
        assert "gated" in text.lower(), f"{name} card must note the gated access model"


def test_errata_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errata file must document one entry per card at minimum."""
    text = ERRATA.read_text()
    for slug in ["medsiglip_448", "medgemma_1_5_4b", "medgemma_1_27b", "siglip_base_patch16_224"]:
        assert slug in text, f"Errata missing entry for {slug}"
    # Schema fence
    assert "### YYYY-MM-DD" in text or "## Entries" in text
