"""Live integration test for the Modal-backed Phikon pathology client.

Skipped unless ``PHIKON_MODAL_EMBED_URL`` is set. Uses 5 real H&E tissue
patches (NCT-CRC-HE-100K, CC-BY-4.0, Zenodo DOI 10.5281/zenodo.1214456;
see tests/fixtures/pathology/PROVENANCE.json) across 5 distinct real
tissue classes -- not a synthetic image, not a degenerate solid-color PNG.

This file tests the *deployed endpoint's I/O contract* with a handful of
real images (dimension, determinism, non-degeneracy). It is deliberately
separate from the large-scale population validation
(phikon_nct_crc_population_validation.json), which embeds the full
107,180-image dataset locally and fits/evaluates a real classifier -- that
is a different artifact answering a different question (does the backbone
carry population-level signal) from this one (is the live endpoint healthy
and contract-correct).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

LIVE = pytest.mark.skipif(
    not os.environ.get("PHIKON_MODAL_EMBED_URL"),
    reason="PHIKON_MODAL_EMBED_URL not set (skipping live Modal test)",
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pathology"
TUM_TILE = FIXTURE_ROOT / "nct_crc_he100k_tum.png"
NORM_TILE = FIXTURE_ROOT / "nct_crc_he100k_norm.png"
STR_TILE = FIXTURE_ROOT / "nct_crc_he100k_str.png"
LYM_TILE = FIXTURE_ROOT / "nct_crc_he100k_lym.png"
BACK_TILE = FIXTURE_ROOT / "nct_crc_he100k_back.png"


def _skip_if_missing(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"fixture missing: {p}")


@LIVE
def test_embed_real_tumor_tile_returns_768_dim() -> None:
    from oncology_arbiter.models.specialist_clients import PhikonClient

    _skip_if_missing(TUM_TILE)
    client = PhikonClient()
    call = client.embed(TUM_TILE.read_bytes(), request_id="pytest-tum")
    assert call.output["embedding_dim"] == 768
    assert call.receipt["status"] == "succeeded"
    assert call.receipt["service_name"] == "phikon"


@LIVE
def test_embed_is_deterministic_across_repeated_calls() -> None:
    """Same real image, two separate network calls -> byte-identical
    embedding (via vector_sha256), not just matching dimension."""
    from oncology_arbiter.models.specialist_clients import PhikonClient

    _skip_if_missing(NORM_TILE)
    client = PhikonClient()
    raw = NORM_TILE.read_bytes()
    call1 = client.embed(raw, request_id="pytest-determinism-1")
    call2 = client.embed(raw, request_id="pytest-determinism-2")
    assert call1.output["vector_sha256"] == call2.output["vector_sha256"], (
        "identical input image produced different embeddings on repeated calls"
    )


@LIVE
def test_embeddings_of_genuinely_different_tissue_are_not_degenerate() -> None:
    """Mathematical interrogation, not a dimension-only smoke check: feed
    5 real images from 5 distinct, verified-different tissue classes
    (tumor epithelium, normal mucosa, stroma, lymphocytes, background) and
    require the embeddings actually differ from each other. A model that
    silently returns a constant/degenerate vector for every input would
    pass a dimension check but fail this -- which is exactly the kind of
    "returns 200 but the content is wrong" failure mode this audit is
    built to catch (the medsiglip endpoint audit found exactly this
    pattern: HTTP 200 on corrupted/degenerate input).

    ``PhikonClient.embed()`` deliberately strips the raw vector from its
    returned ``output`` (it keeps only ``embedding_dim``/``vector_sha256``
    in the audit receipt), so this test calls the raw endpoint directly
    (same payload contract the client uses internally) to get real float
    values to compute actual pairwise distances on -- a quantitative
    check, not just a hash-equality check.
    """
    import math

    from oncology_arbiter.models.specialist_clients import _http_json

    tiles = {"TUM": TUM_TILE, "NORM": NORM_TILE, "STR": STR_TILE, "LYM": LYM_TILE, "BACK": BACK_TILE}
    _skip_if_missing(*tiles.values())
    import base64

    embed_url = os.environ["PHIKON_MODAL_EMBED_URL"].rstrip("/")
    embeddings: dict[str, list[float]] = {}
    for name, path in tiles.items():
        raw = path.read_bytes()
        data, _ = _http_json(
            embed_url,
            payload={"inputs": [{"image_b64": base64.b64encode(raw).decode("ascii")}]},
            timeout=60.0,
        )
        vec = data.get("embedding")
        if vec is None and isinstance(data.get("embeddings"), list):
            vec = data["embeddings"][0]
        assert isinstance(vec, list) and len(vec) == 768, f"{name}: unexpected embedding shape"
        embeddings[name] = [float(v) for v in vec]

    def _euclidean(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    names = list(embeddings)
    pair_distances = {
        f"{names[i]}-{names[j]}": _euclidean(embeddings[names[i]], embeddings[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    min_pair, min_dist = min(pair_distances.items(), key=lambda kv: kv[1])
    max_pair, max_dist = max(pair_distances.items(), key=lambda kv: kv[1])
    assert min_dist > 1e-6, (
        f"embeddings for visually distinct real tissue images are near-identical "
        f"(smallest pairwise distance {min_dist:.2e} for {min_pair}) -- endpoint may be "
        "returning a degenerate/constant vector regardless of input"
    )
    assert max_dist > 0.01, (
        f"largest pairwise distance across 5 distinct real tissue classes is only "
        f"{max_dist:.4f} ({max_pair}) -- too small to represent genuinely different "
        "tissue types; treat as a possible defect, not dismiss as expected behavior"
    )


@LIVE
def test_receipt_carries_verifiable_link_to_exact_input_bytes() -> None:
    """Confirms the client-side receipt actually carries a hash tied to the
    real bytes we sent -- i.e. the audit trail (not just the model output)
    is trustworthy and reproducible, not just a status string."""
    from oncology_arbiter.models.specialist_clients import PhikonClient

    _skip_if_missing(STR_TILE)
    raw = STR_TILE.read_bytes()
    expected_sha = hashlib.sha256(raw).hexdigest()
    client = PhikonClient()
    call = client.embed(raw, request_id="pytest-input-sha")
    assert call.receipt.get("input_reference"), "receipt missing input_reference"
    assert expected_sha in str(call.receipt["input_reference"]), (
        f"receipt input_reference does not embed the real sha256 of the bytes sent "
        f"({expected_sha}); receipt={call.receipt['input_reference']!r}"
    )
