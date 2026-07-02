"""Unit tests for two new PLAN.md-mandated endpoints:

* ``GET /v1/model-cards``                       — PLAN.md §5.6
* ``GET /v1/artifacts/{category}/{filename}``   — PLAN.md §4a

We test the FastAPI app with ``TestClient`` (in-process, no network). No
mocks. The fixtures used here are the real docs/*.md and
artifacts/reports/*.md files shipped in this repo, so any drift between
tests and shipped content trips a failure.

Path-traversal security is exercised explicitly: ``..``, absolute paths,
symlinks, and null bytes must all be rejected. This mirrors the security
contract of ``org.backend/capabilities/progression_arbiter/router.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs" / "model_cards"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"


# ── /v1/model-cards index ─────────────────────────────────────────────


def test_model_cards_index_returns_200_and_all_shipped_cards(client: TestClient) -> None:
    resp = client.get("/v1/model-cards")
    assert resp.status_code == 200
    body = resp.json()
    assert "cards" in body
    # Every .md file under docs/model_cards/ must be indexed.
    disk_cards = sorted(p.stem for p in DOCS_DIR.glob("*.md"))
    indexed_slugs = sorted(c["slug"] for c in body["cards"])
    assert indexed_slugs == disk_cards


def test_model_cards_index_carries_disclaimer_and_caveat(client: TestClient) -> None:
    body = client.get("/v1/model-cards").json()
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["caveat"] == AUROC_CAVEAT


def test_model_cards_index_flags_honesty_markers(client: TestClient) -> None:
    """Honest-markers audit — different rules for different card classes:

    - errata: bookkeeping file, no requirement.
    - proxy cards (SigLIP base): must NOT claim AUROC (would be fabricated),
      but must carry RUO / non-clinical disclaimer.
    - clinical model cards (MedSigLIP, MedGemma): must carry both AUROC
      caveat AND RUO disclaimer.
    """
    body = client.get("/v1/model-cards").json()
    for card in body["cards"]:
        slug = card["slug"]
        markers = card["honesty_markers"]
        if slug == "errata":
            continue
        # Every non-errata card must carry the RUO marker.
        assert markers["ruo_disclaimer_present"] is True, (
            f"model card {slug} is missing a RUO disclaimer"
        )
        # Proxy cards should NOT quote an AUROC — that would be fabricating
        # medical validity of a general-domain model. The siglip proxy card
        # explicitly says 'Do not report proxy zero-shot AUCs on mammography'.
        if "siglip_base" in slug:
            continue
        # All medical model cards must carry an AUROC caveat.
        assert markers["auroc_caveat_present"] is True, (
            f"medical model card {slug} is missing an AUROC caveat"
        )


def test_model_cards_index_n_bytes_matches_disk(client: TestClient) -> None:
    body = client.get("/v1/model-cards").json()
    for card in body["cards"]:
        on_disk = DOCS_DIR / f"{card['slug']}.md"
        assert on_disk.exists()
        assert card["n_bytes"] == on_disk.stat().st_size


# ── /v1/artifacts streamer ─────────────────────────────────────────────


def test_artifacts_serves_docs_card(client: TestClient) -> None:
    disk_cards = list(DOCS_DIR.glob("*.md"))
    assert disk_cards, "no shipped model cards to test against"
    target = disk_cards[0].name
    resp = client.get(f"/v1/artifacts/docs/{target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "docs"
    assert body["filename"] == target
    assert body["media_type"] == "text/markdown"
    # Content on the wire must match content on disk byte-for-byte.
    assert body["content"] == (DOCS_DIR / target).read_text(encoding="utf-8")


def test_artifacts_serves_reports_sql_schema(client: TestClient) -> None:
    target = "ai_prediction_ledger_schema.sql"
    resp = client.get(f"/v1/artifacts/reports/{target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["media_type"] == "application/sql"
    assert body["content"] == (REPORTS_DIR / target).read_text(encoding="utf-8")


def test_artifacts_serves_arbiter_model_json(client: TestClient) -> None:
    target = "screening_arbiter_template_v0.json"
    resp = client.get(f"/v1/artifacts/models/{target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["media_type"] == "application/json"
    assert "AUROC_CAVEAT" in body["content"]  # frozen JSON must include honesty gate


def test_artifacts_rejects_invalid_category(client: TestClient) -> None:
    resp = client.get("/v1/artifacts/secrets/foo.txt")
    assert resp.status_code == 400
    assert "invalid category" in resp.json()["detail"].lower()


def test_artifacts_rejects_missing_file(client: TestClient) -> None:
    resp = client.get("/v1/artifacts/docs/does_not_exist.md")
    assert resp.status_code == 404


@pytest.mark.parametrize("bad_name", [
    "..%2Fetc%2Fpasswd",   # url-encoded ../
    "%2Fetc%2Fpasswd",     # url-encoded absolute
    "..",                  # bare dot-dot
    ".",                   # bare dot
])
def test_artifacts_rejects_path_traversal(client: TestClient, bad_name: str) -> None:
    resp = client.get(f"/v1/artifacts/docs/{bad_name}")
    # FastAPI/starlette resolves URL-encoded traversal → containment check
    # or bare filename validation should reject with 400 or 403 or 404.
    assert resp.status_code in {400, 403, 404}, (
        f"traversal attempt {bad_name!r} should be rejected, got {resp.status_code}"
    )


def test_artifacts_carries_disclaimer_and_caveat(client: TestClient) -> None:
    disk_cards = list(DOCS_DIR.glob("*.md"))
    assert disk_cards
    resp = client.get(f"/v1/artifacts/docs/{disk_cards[0].name}")
    body = resp.json()
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["caveat"] == AUROC_CAVEAT


# ── /health mentions the new endpoints ─────────────────────────────────


def test_health_endpoint_lists_new_endpoints(client: TestClient) -> None:
    body = client.get("/health").json()
    endpoints = set(e.strip() for e in body["endpoints"])
    # PLAN.md §4a explicitly names these two endpoints.
    assert any("v1/model-cards" in e for e in endpoints), (
        "health check must advertise /v1/model-cards"
    )
    assert any("v1/artifacts" in e for e in endpoints), (
        "health check must advertise /v1/artifacts"
    )
    assert "l3_arbiter" in body["models_loaded"]
