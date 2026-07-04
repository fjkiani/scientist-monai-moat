"""Unit tests for the NSCLC lung pipeline (placeholder-grade).

Covers:
    - HU thresholds are pinned to the values the model card advertises
    - `_diameter_to_logit` interpolates correctly across all anchors
    - `_diameter_bucket` respects Fleischner-style boundaries
    - `_isotropic_diameter_mm` matches the sphere-volume formula
    - `run_lung_heuristic` on a synthetic HU cube produces the expected
      lung fraction and finds a single planted "nodule"
    - `score_nsclc` derives the expected risk bucket from a
      `LungHeuristicOutput` fixture (matches previous LIDC pilot)
    - `NsclcArbiterFeatures.from_lung_output` round-trips
    - `score_nsclc_therapy` returns citation-carrying options for every
      bucket and the mass addendum kicks in for `>30 mm` HIGH

These are pure-Python tests — no DICOM fixture required — so they run
in every CI environment. Real-DICOM regression is left to the
`tests/data/test_lidc_real_ct.py` file, which is skipped when the LIDC
cohort is not mounted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from oncology_arbiter.lung.arbiter import (
    ArbiterScore,
    BUCKET_LOW_MAX_MM,
    BUCKET_MID_MAX_MM,
    DIAMETER_LOGIT_ANCHORS,
    NsclcArbiterFeatures,
    _diameter_bucket,
    _diameter_to_logit,
    _sigmoid,
    score_nsclc,
)
from oncology_arbiter.lung.pipeline import (
    BODY_HU_MIN,
    DEFAULT_DILATE_ITER,
    DEFAULT_MAX_VOXELS,
    DEFAULT_MIN_VOXELS,
    DEFAULT_TOP_N,
    LUNG_HU_MAX,
    LungHeuristicOutput,
    NODULE_HU_MAX,
    NODULE_HU_MIN,
    NoduleCandidate,
    _isotropic_diameter_mm,
    lung_mask_from_hu,
    nodule_candidate_blobs,
    run_lung_heuristic,
    summarize_candidates,
)
from oncology_arbiter.models.nccn_nsclc_rules import (
    FLEISCHNER_2017_DOI,
    FLEISCHNER_2017_URL,
    NCCN_NSCLC_URL,
    NCCN_NSCLC_VERSION,
    NSCLC_RULES_PROXY_WARNING,
    NsclcTherapyRulesResult,
    score_nsclc_therapy,
)


# --------------------------------------------------------------------------- #
# HU threshold pins


def test_hu_thresholds_are_pinned_to_model_card_values():
    """The model card advertises these HU thresholds — regressions here
    would silently change every downstream response."""
    assert LUNG_HU_MAX == -500.0
    assert BODY_HU_MIN == -900.0
    assert NODULE_HU_MIN == -300.0
    assert NODULE_HU_MAX == 200.0


def test_default_blob_bounds_match_model_card():
    """Default voxel-count filter bounds are the ones the model card
    documents. Change them here → change the model card."""
    assert DEFAULT_MIN_VOXELS == 8
    assert DEFAULT_MAX_VOXELS == 20_000
    assert DEFAULT_DILATE_ITER == 3
    assert DEFAULT_TOP_N == 10


# --------------------------------------------------------------------------- #
# Diameter → logit / bucket


def test_diameter_logit_anchors_are_pinned():
    """The Fleischner anchors are advertised in the model card."""
    assert DIAMETER_LOGIT_ANCHORS == [
        (0.0, -4.0),
        (4.0, -1.5),
        (8.0, 0.0),
        (15.0, 1.5),
        (30.0, 3.0),
        (60.0, 4.5),
    ]


@pytest.mark.parametrize(
    "d_mm, expected",
    [
        (0.0, -4.0),
        (2.0, -2.75),          # midpoint between (0,-4) and (4,-1.5)
        (4.0, -1.5),
        (8.0, 0.0),
        (15.0, 1.5),
        (30.0, 3.0),
        (60.0, 4.5),
        (100.0, 4.5),           # clamp above last anchor
    ],
)
def test_diameter_to_logit_interpolates_correctly(d_mm, expected):
    got = _diameter_to_logit(d_mm)
    assert got == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "d_mm, expected_bucket",
    [
        (0.0, "NEGATIVE"),
        (2.0, "LOW"),
        (BUCKET_LOW_MAX_MM - 0.1, "LOW"),
        (BUCKET_LOW_MAX_MM, "MID"),
        (BUCKET_MID_MAX_MM - 0.1, "MID"),
        (BUCKET_MID_MAX_MM, "HIGH"),
        (35.0, "HIGH"),
        (60.0, "HIGH"),
    ],
)
def test_diameter_bucket_boundaries(d_mm, expected_bucket):
    assert _diameter_bucket(d_mm) == expected_bucket


def test_sigmoid_matches_math_reference():
    for x in [-4.0, -1.5, 0.0, 1.5, 3.0, 4.5]:
        assert _sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)), rel=1e-9)


# --------------------------------------------------------------------------- #
# Isotropic diameter formula


def test_isotropic_diameter_mm_matches_sphere_formula():
    # 1 mm³ voxels, 1000 voxels → V = 1000 mm³ → r = (3*1000/4π)^(1/3) ≈ 6.20 mm
    d = _isotropic_diameter_mm(1000, (1.0, 1.0, 1.0))
    expected = 2.0 * (3.0 * 1000.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    assert d == pytest.approx(expected, rel=1e-9)


def test_isotropic_diameter_mm_zero_when_empty():
    assert _isotropic_diameter_mm(0, (1.0, 1.0, 1.0)) == 0.0


def test_isotropic_diameter_mm_respects_anisotropic_spacing():
    # 100 voxels with (dz=2.5, dy=0.7, dx=0.7): V = 100 * 2.5 * 0.49 = 122.5 mm³
    d = _isotropic_diameter_mm(100, (2.5, 0.7, 0.7))
    expected = 2.0 * (3.0 * 122.5 / (4.0 * math.pi)) ** (1.0 / 3.0)
    assert d == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic HU cube


def _make_synthetic_lung_cube(
    shape=(24, 64, 64),
    body_hu=-50.0,
    lung_hu=-800.0,
    nodule_hu=50.0,
):
    """Build a small HU volume that has:
        outer 4-voxel shell    : gantry air (-1200)
        body silhouette        : soft tissue (body_hu)
        interior lung region   : aerated lung (lung_hu)
        one solid nodule       : 5x5x5 cube of nodule_hu inside the lung
    """
    Z, Y, X = shape
    vol = np.full(shape, -1200.0, dtype=np.float32)
    # body silhouette (soft tissue)
    vol[2:Z - 2, 4:Y - 4, 4:X - 4] = body_hu
    # lung region inside body
    vol[4:Z - 4, 10:Y - 10, 10:X - 10] = lung_hu
    # solid nodule in the middle of the lung
    zc, yc, xc = Z // 2, Y // 2, X // 2
    vol[zc - 2:zc + 3, yc - 2:yc + 3, xc - 2:xc + 3] = nodule_hu
    return vol


def test_run_lung_heuristic_on_synthetic_cube_finds_the_planted_nodule():
    vol = _make_synthetic_lung_cube()
    out = run_lung_heuristic(vol, spacing_mm=(1.0, 1.0, 1.0), top_n=10)
    Z, Y, X = vol.shape
    center = (Z / 2.0, Y / 2.0, X / 2.0)
    assert out.lung_voxel_fraction > 0.05
    assert out.n_candidates_total >= 1
    assert out.max_diameter_mm > 0.0
    # Find the candidate whose centroid is closest to the volume center;
    # the planted 5x5x5 nodule (125 voxels, d ≈ 6.2 mm at 1mm spacing) sits there.
    def _dist(c):
        z, y, x = c.centroid_zyx_vox
        return ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2) ** 0.5
    nodule = min(out.candidates, key=_dist)
    assert _dist(nodule) < 2.0, f"expected central nodule, got centroid {nodule.centroid_zyx_vox}"
    assert 100 <= nodule.voxel_count <= 200
    assert 4.0 < nodule.diameter_mm < 10.0
    assert nodule.mean_hu > 0.0  # solid nodule vs soft-tissue shell (~-50)


def test_run_lung_heuristic_synthetic_cube_returns_at_most_top_n():
    vol = _make_synthetic_lung_cube()
    out = run_lung_heuristic(vol, spacing_mm=(1.0, 1.0, 1.0), top_n=3)
    assert len(out.candidates) <= 3


def test_lung_mask_from_hu_is_boolean_with_matching_shape():
    vol = _make_synthetic_lung_cube()
    mask = lung_mask_from_hu(vol)
    assert mask.shape == vol.shape
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_summarize_candidates_top_n_capping():
    """Ensure the top_n cap is applied AFTER voxel-count filtering."""
    # Build a synthetic label volume with 5 blobs of sizes {10, 30, 50, 15, 100}
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    counts_planted = {1: 10, 2: 30, 3: 50, 4: 15, 5: 100}
    lin = labels.ravel()
    off = 0
    for lbl, n in counts_planted.items():
        lin[off:off + n] = lbl
        off += n
    labels = lin.reshape((20, 20, 20))
    volume = np.zeros_like(labels, dtype=np.float32)

    cands, n_total, n_kept = summarize_candidates(
        labels, volume, spacing_mm=(1.0, 1.0, 1.0),
        min_voxels=8, max_voxels=200, top_n=3,
    )
    assert n_total == 5
    assert n_kept == 5
    assert len(cands) == 3
    # Top by voxel count desc:
    assert [c.voxel_count for c in cands] == [100, 50, 30]


# --------------------------------------------------------------------------- #
# Arbiter


def test_score_nsclc_negative_bucket_when_no_diameter():
    feats = NsclcArbiterFeatures(max_diameter_mm=0.0, n_candidates=0)
    s = score_nsclc(feats)
    assert s.risk_bucket == "NEGATIVE"
    assert s.max_diameter_mm == 0.0
    assert s.prob < 0.5


@pytest.mark.parametrize(
    "d_mm, expected_bucket",
    [
        (3.0, "LOW"),
        (7.0, "MID"),
        (15.0, "HIGH"),
        (35.4, "HIGH"),   # matches LIDC-IDRI-0001 pilot value
    ],
)
def test_score_nsclc_bucket_matches_diameter(d_mm, expected_bucket):
    feats = NsclcArbiterFeatures(max_diameter_mm=d_mm, n_candidates=1)
    s = score_nsclc(feats)
    assert s.risk_bucket == expected_bucket


def test_score_nsclc_count_bonus_kicks_in_at_high_counts():
    lo = score_nsclc(NsclcArbiterFeatures(max_diameter_mm=6.0, n_candidates=2))
    mid = score_nsclc(NsclcArbiterFeatures(max_diameter_mm=6.0, n_candidates=3))
    hi = score_nsclc(NsclcArbiterFeatures(max_diameter_mm=6.0, n_candidates=6))
    assert lo.logit < mid.logit < hi.logit


def test_score_nsclc_mass_flag_when_diameter_over_30mm():
    s = score_nsclc(NsclcArbiterFeatures(max_diameter_mm=35.4, n_candidates=1))
    assert s.driving_feature == "mass_diameter_gt_30mm"
    assert s.risk_bucket == "HIGH"


def test_arbiter_features_from_lung_output_round_trip():
    out = LungHeuristicOutput(
        lung_voxel_fraction=0.21,
        n_candidates_total=42,
        n_candidates_kept=10,
        candidates=[],
        max_diameter_mm=12.3,
        spacing_mm=(2.5, 0.7, 0.7),
    )
    feats = NsclcArbiterFeatures.from_lung_output(out)
    assert feats.max_diameter_mm == 12.3
    assert feats.n_candidates == 0
    assert feats.lung_voxel_fraction == 0.21


# --------------------------------------------------------------------------- #
# NCCN-NSCLC-lite therapy rules


def test_nsclc_rules_return_recommendations_for_every_bucket():
    for bucket in ("NEGATIVE", "LOW", "MID", "HIGH"):
        r = score_nsclc_therapy(bucket, max_diameter_mm=10.0)
        assert isinstance(r, NsclcTherapyRulesResult)
        assert r.risk_bucket == bucket
        assert len(r.recommended_options) >= 1
        # every option must carry a citation URL
        for opt in r.recommended_options:
            assert opt.citation_url.startswith("http")
        # warning names the proxy status
        assert any("rules-lite" in w or "proxy" in w.lower() for w in r.warnings)


def test_nsclc_rules_mass_addendum_kicks_in_for_gt_30mm_high():
    r = score_nsclc_therapy(
        "HIGH", max_diameter_mm=35.4, driving_feature="mass_diameter_gt_30mm"
    )
    names = [o.name for o in r.recommended_options]
    assert any("mass" in n.lower() for n in names), names


def test_nsclc_rules_mass_addendum_not_present_for_high_with_moderate_diameter():
    r = score_nsclc_therapy(
        "HIGH", max_diameter_mm=10.0, driving_feature="max_diameter_mm"
    )
    names = [o.name for o in r.recommended_options]
    assert not any("mass workup" in n.lower() for n in names), names


def test_nsclc_rules_unknown_bucket_defaults_to_negative():
    r = score_nsclc_therapy("UNKNOWN", max_diameter_mm=0.0)
    assert r.risk_bucket == "NEGATIVE"


def test_nsclc_rules_citations_point_at_nccn_or_fleischner():
    r = score_nsclc_therapy("HIGH", max_diameter_mm=35.4)
    urls = {o.citation_url for o in r.recommended_options}
    assert any(NCCN_NSCLC_URL in u for u in urls)
    # NCCN version pin
    assert r.nccn_version == NCCN_NSCLC_VERSION
    # Fleischner citation string is present
    assert FLEISCHNER_2017_DOI in r.dataset_citation


# --------------------------------------------------------------------------- #
# LIDC dataset provenance


def test_lidc_dataset_provenance_carries_verified_dois():
    from oncology_arbiter.data.lidc_idri import (
        LIDC_TCIA_DOI,
        LIDC_MEDICAL_PHYSICS_DOI,
        LIDC_LICENSE,
        LIDC_N_PATIENTS,
        LIDC_N_IMAGES,
        dataset_provenance,
    )
    d = dataset_provenance()
    # DOIs verified against TCIA on 2026-07-04
    assert LIDC_TCIA_DOI == "10.7937/K9/TCIA.2015.LO9QL9SX"
    assert LIDC_MEDICAL_PHYSICS_DOI == "10.1118/1.3528204"
    assert LIDC_LICENSE == "CC-BY-3.0"
    assert LIDC_N_PATIENTS == 1010
    assert LIDC_N_IMAGES == 244_527
    assert d["tcia_doi"] == LIDC_TCIA_DOI
    assert LIDC_TCIA_DOI in d["citation"]
    assert LIDC_MEDICAL_PHYSICS_DOI in d["citation"]


def test_lidc_cohort_not_found_when_root_missing(tmp_path, monkeypatch):
    """Missing cohort must raise LidcCohortNotFound, not fabricate cases."""
    from oncology_arbiter.data.lidc_idri import (
        LidcCohortNotFound,
        list_lidc_series,
        resolve_series_dir,
    )
    monkeypatch.setenv("ONCOLOGY_ARBITER_LIDC_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(LidcCohortNotFound):
        list_lidc_series()
    with pytest.raises(LidcCohortNotFound):
        resolve_series_dir("LIDC-IDRI-0001")
