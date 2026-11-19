#!/usr/bin/env python
"""Prove that removed tests were migrated, not deleted.

The audit's charge is that tests were deleted to make a red suite go green. The
defence has to be falsifiable, so this script does not accept the mapping below
on trust. For every removed test it independently checks:

* the old test really is gone from the working tree (an entry claiming a
  migration for a test that still exists is a bookkeeping error);
* the named replacement really exists, by collecting node ids from pytest
  rather than grepping for a function name that might live in a skipped file;
* the replacement actually passes, by running it;
* and where the claim is "capability retired" rather than "behaviour replaced",
  that the capability is absent from BOTH the public API surface and the model
  registry — because a capability that is merely unrouted is still shippable.

Emits ``retired_test_migration.json`` with the columns the auditor specified:
old_test, retired_behavior, replacement_test, new_invariant, replacement_result.
Exits non-zero if any row fails verification.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BASE = "715aca9c8bac"

# --------------------------------------------------------------------------- #
# Retirement claims. Each names the substitute that was removed, the symbols
# that must no longer exist in the API surface, and the registry values that
# must be unconstructible.

RETIREMENTS: dict[str, dict[str, Any]] = {
    "siglip_proxy": {
        "substitute": "open-domain SigLIP standing in for MedSigLIP",
        "api_symbols_banned": ["siglip_baseline", "SigLipBaseline", "proxy_siglip"],
        "registry_values_banned": ["proxy_siglip"],
        "invariant_tests": [
            "tests/models/test_hai_def.py::test_model_state_enum_cannot_express_the_retired_siglip_proxy",
            "tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible",
            "tests/unit/test_screening_production_contract.py::test_app_module_imports_no_substitute_backend",
        ],
    },
    "regex_report_parser": {
        "substitute": "regex report parser standing in for ClinicalBERT NER",
        "api_symbols_banned": ["proxy_regex_v0", "fused_regex_clinicalbert"],
        "registry_values_banned": ["proxy_regex_v0", "fused_regex_clinicalbert"],
        "invariant_tests": [
            "tests/unit/test_production_specialist_contracts.py::test_biopsy_clinicalbert_failure_has_no_regex_fallback",
            "tests/unit/test_production_specialist_contracts.py::test_biopsy_source_calls_only_clinicalbert_for_report_parsing",
        ],
    },
    "rules_lite_therapy": {
        "substitute": "NCCN-lite static rules standing in for the SL bridge / TxGemma",
        "api_symbols_banned": ["proxy_rules_lite"],
        "registry_values_banned": ["proxy_rules_lite"],
        "invariant_tests": [
            "tests/unit/test_production_specialist_contracts.py::test_therapy_endpoint_uses_bridge_as_only_recommendation_source",
            "tests/unit/test_production_specialist_contracts.py::test_full_case_source_contains_no_local_ct_or_nccn_substitution",
        ],
    },
    "lung_hu_heuristic": {
        "substitute": "Hounsfield-unit threshold heuristic standing in for LUNA16",
        "api_symbols_banned": [
            "run_lung_heuristic", "proxy_lung_heuristic", "LungNoduleDetector",
        ],
        "registry_values_banned": ["proxy_lung_heuristic"],
        "invariant_tests": [
            "tests/unit/test_production_specialist_contracts.py::test_full_nsclc_requires_case_id_and_never_substitutes_heuristic",
            "tests/unit/test_production_specialist_contracts.py::test_full_case_source_contains_no_local_ct_or_nccn_substitution",
            "tests/unit/test_production_specialist_contracts.py::test_luna16_requires_case_id_loaded_state_and_bundle",
        ],
    },
    "monai_screening_heuristic": {
        "substitute": "MONAI intensity heuristic standing in for a screening detector",
        "api_symbols_banned": ["proxy_monai_heuristic"],
        "registry_values_banned": ["proxy_monai_heuristic"],
        "invariant_tests": [
            "tests/unit/test_screening_production_contract.py::test_success_requires_1152_dimensional_production_receipt",
            "tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible",
        ],
    },
    "midpoint_unknown_imputation": {
        "substitute": "unknown boolean imputed at 0.5 (the midpoint of the encoding)",
        "api_symbols_banned": [],
        "registry_values_banned": [],
        "invariant_tests": [
            "tests/unit/test_arbiter_missingness_encoding.py::test_missing_bool_encoding_constant_is_zero_not_half",
            "tests/unit/test_arbiter_missingness_encoding.py::test_midpoint_inflation_is_no_longer_reachable",
            "tests/unit/test_arbiter_missingness_encoding.py::test_shipped_templates_declare_no_nonzero_unknown_level",
            "tests/unit/test_arbiter_missingness_encoding.py::test_missing_bool_is_indistinguishable_from_observed_false_in_logit",
            "tests/unit/test_arbiter_missingness_encoding.py::test_prevalence_valued_encoding_would_still_overshoot_the_base_rate",
        ],
    },
    "template_arbiter_on_free_text": {
        "substitute": "template L2 arbiter scored from free text / inferred / empty inputs",
        "api_symbols_banned": [],
        "registry_values_banned": [],
        "invariant_tests": [
            "tests/unit/test_production_specialist_contracts.py::test_only_arbiter_call_site_requires_the_complete_explicit_vector",
            "tests/unit/test_production_specialist_contracts.py::test_screening_emits_no_template_arbiter_block",
            "tests/unit/test_production_specialist_contracts.py::test_biopsy_endpoint_emits_no_template_arbiter_score",
            "tests/unit/test_production_specialist_contracts.py::test_case_full_emits_no_template_arbiters_from_free_text",
        ],
    },
    "co_scientist_proxy": {
        "substitute": "deterministic proxy standing in for a live Co-Scientist LLM",
        "api_symbols_banned": ["proxy_co_scientist"],
        "registry_values_banned": [],
        "invariant_tests": [
            "tests/unit/test_production_specialist_contracts.py::test_configured_medgemma_is_terminal_and_surfaces_provenance",
        ],
    },
    "ovarian_patient_scoring": {
        "substitute": "patient-level ovarian mortality probability",
        "api_symbols_banned": [],
        "registry_values_banned": [],
        "invariant_tests": [
            "tests/unit/test_ovarian_retirement_validation.py::test_retirement_source_cannot_emit_patient_probability",
            "tests/unit/test_production_specialist_contracts.py::test_hgsoc_dynamic_response_has_no_patient_score",
        ],
    },
}

# --------------------------------------------------------------------------- #
# old_test -> (retirement_key | None, retired_behavior, replacement nodeids,
#              new_invariant)
#
# A row with retirement_key set is a capability retirement: the behaviour is
# gone from the product, so the replacement proves absence rather than
# equivalence. A row with retirement_key None is a behaviour replacement and
# must name a replacement that asserts the *new* contract.

M: dict[str, tuple[str | None, str, list[str], str]] = {}


def _bulk(keys: list[str], retirement: str, behavior: str, invariant: str) -> None:
    for k in keys:
        M[k] = (retirement, behavior, RETIREMENTS[retirement]["invariant_tests"],
                invariant)


_bulk(
    ["test_baseline_calls_processor_with_sigLIP_padding",
     "test_baseline_close_drops_model",
     "test_baseline_defaults_to_ungated_proxy_repo",
     "test_baseline_load_is_idempotent",
     "test_baseline_rejects_single_label",
     "test_baseline_run_returns_proxy_siglip_state",
     "test_default_labels_are_two_neutral_prompts",
     "test_siglip_proxy_architecture_is_vit_b_16",
     "test_siglip_proxy_input_resolution_is_224",
     "test_siglip_proxy_license_is_apache",
     "test_siglip_proxy_repo_verbatim",
     "test_proxy_warning_cites_pathology_anchor",
     "test_proxy_warning_forbids_reporting_as_medsiglip",
     "test_proxy_warning_names_apache_or_general_domain",
     "test_proxy_warning_references_mammography_absence",
     "test_proxy_warning_references_proxy_and_medsiglip",
     "test_model_state_enum_has_proxy_siglip_variant",
     "test_screening_siglip_proxy_beats_medsiglip",
     "test_screening_siglip_proxy_beats_monai",
     "test_real_siglip_smoke_on_cbis_dicom",
     "test_real_siglip_two_fixtures_produce_distinct_logits",
     "test_to_pil_clips_out_of_range_values",
     "test_to_pil_grayscale_to_rgb",
     "test_to_pil_rejects_3d_input",
     "test_medsiglip_disabled_proxy_still_works",
     "test_medsiglip_gate_denied_does_not_fall_back_to_proxy_by_default",
     "test_medsiglip_gate_denied_with_proxy_enabled_falls_back_with_both_warnings",
     "test_medsiglip_allowed_takes_precedence_over_proxy"],
    "siglip_proxy",
    "asserted the behaviour of an open-domain SigLIP proxy substituted for "
    "MedSigLIP, including its warning text and its precedence over other "
    "backends",
    "the proxy is deleted from the codebase, cannot be imported by the API "
    "module, and `proxy_siglip` is not a constructible ModelState; a screening "
    "success now requires a 1152-dimensional MedSigLIP receipt",
)

_bulk(
    ["test_biopsy_report_parser_always_proxy_regex",
     "test_biopsy_luminal_a_receptor_panel_and_grade_populated",
     "test_biopsy_tnbc_receptor_panel_all_negative",
     "test_biopsy_her2_equivocal_marked_ambiguous",
     "test_biopsy_her2_positive_case",
     "test_biopsy_no_match_report_returns_all_no_match",
     "test_biopsy_no_report_text_returns_empty_panel_but_valid_response",
     "test_biopsy_needs_wsi_or_report",
     "test_biopsy_wired_returns_subtype"],
    "regex_report_parser",
    "asserted that a regex parser populated the receptor panel and that its "
    "output was reported as a parsed report",
    "report parsing is ClinicalBERT-only; a ClinicalBERT failure is terminal "
    "for the parsing stage and no regex path can produce a panel",
)

_bulk(
    ["test_therapy_rules_lite_fallback",
     "test_therapy_rules_lite_hr_positive",
     "test_therapy_txgemma_gated_falls_through_to_rules",
     "test_therapy_txgemma_gated_no_rules_stays_placeholder",
     "test_therapy_txgemma_gated_response_has_structured_gate_report",
     "test_therapy_txgemma_gated_still_falls_through_to_rules_proxy",
     "test_therapy_txgemma_wins",
     "test_therapy_rules_proxy_has_no_gate_report",
     "test_therapy_reason_rules_lite_surfaces_sha_and_branch",
     "test_therapy_reason_fingerprint_is_stable_across_calls",
     "test_therapy_reason_placeholder_leaves_fingerprint_fields_null",
     "test_therapy_reason_strict_rejects_bad_stage_with_400",
     "test_therapy_reason_non_strict_still_serves_on_bad_stage",
     "test_therapy_default_placeholder",
     "test_therapy_placeholder_when_flags_off",
     "test_therapy_placeholder_has_no_gate_report"],
    "rules_lite_therapy",
    "asserted that a static NCCN-lite rules table produced therapy "
    "recommendations when TxGemma was gated or absent",
    "the live SL bridge is the only recommendation source; when it is "
    "unavailable the stage fails required and no options are emitted",
)

_bulk(
    ["test_nsclc_always_proxy_lung_heuristic",
     "test_nsclc_flagged_at_least_as_proxy",
     "test_nsclc_returns_placeholder_envelope",
     "test_case_full_nsclc_placeholder_when_gate_off",
     "test_case_full_nsclc_placeholder_when_no_ct_input",
     "test_case_full_nsclc_real_pipeline_400_on_missing_dir",
     "test_case_full_nsclc_real_pipeline_on_synthetic_ct"],
    "lung_hu_heuristic",
    "asserted that a Hounsfield-unit threshold heuristic produced nodule "
    "findings when LUNA16 was unavailable, and that a local series directory "
    "could be read directly",
    "NSCLC requires a verified case manifest and a loaded LUNA16 bundle; "
    "ALLOW_SERIES_DIR stays off and the heuristic symbols are banned from the "
    "case_full AST",
)

_bulk(
    ["test_monai_detector_off_by_default",
     "test_monai_detector_on_produces_bbox_findings",
     "test_monai_detector_promotes_placeholder_to_proxy",
     "test_monai_honesty_warning_surfaced",
     "test_screening_monai_heuristic"],
    "monai_screening_heuristic",
    "asserted that a MONAI intensity heuristic promoted a placeholder "
    "screening response to a proxy result carrying bounding boxes",
    "`proxy_monai_heuristic` is not a constructible ModelState and a screening "
    "success requires a production MedSigLIP embedding receipt",
)

_bulk(
    ["test_bool_unknown_encodes_as_half"],
    "midpoint_unknown_imputation",
    "asserted that an unobserved boolean was encoded at 0.5, the midpoint of "
    "the {false: 0.0, true: 1.0} encoding",
    "an unobserved boolean contributes exactly zero log-odds; the midpoint "
    "encoding is unreachable, and on the screening template it would have "
    "injected +1.15 log-odds and inflated the base rate by 26.78%",
)

_bulk(
    ["test_screening_endpoint_returns_arbiter_score",
     "test_biopsy_endpoint_returns_arbiter_score",
     "test_therapy_endpoint_returns_arbiter_score",
     "test_case_full_returns_biopsy_and_therapy_arbiters",
     "test_biopsy_arbiter_sum_of_terms_matches_logit",
     "test_case_full_no_confirmed_falls_back_to_biopsy",
     "test_case_full_receptors_confirmed_drives_therapy_branch",
     # This one pinned the purest form of the defect: it asserted the screening
     # arbiter block was present *and* documented in its own docstring that the
     # template has n_training = 0 and empty features, so p_positive "falls back
     # to the intercept". It was asserting that a constant be emitted as a
     # patient probability.
     "test_medsiglip_score_present_alongside_arbiter_block"],
    "template_arbiter_on_free_text",
    "asserted that screening, biopsy and case_full emitted an L2 template "
    "arbiter score derived from free text or from another stage's inferred "
    "output",
    "only an explicit, complete, caller-supplied feature vector produces a "
    "score; screening and biopsy emit arbiter_score = null",
)

_bulk(
    ["test_co_scientist_default_placeholder",
     "test_co_scientist_enabled_reports_proxy_co_scientist"],
    "co_scientist_proxy",
    "asserted that a deterministic proxy satisfied the Co-Scientist stage",
    "a configured MedGemma failure is terminal and surfaces its provenance "
    "rather than degrading to a deterministic stand-in",
)

# --------------------------------------------------------------------------- #
# tests/nlp/test_biopsy_report_parse_wire.py::TestReportParseBlockRegex
#
# This class pinned the retired regex parser AS A SUCCESS SOURCE: it asserted
# parser_id == "proxy_regex_v0" and fusion_mode == "regex" on the wire. The
# whole class is a capability retirement, not a behaviour replacement -- there
# is no "equivalent" assertion to migrate, because a successful regex parse is
# exactly the thing that must no longer be reachable.
_bulk(
    ["test_full_luminal_a_report",
     "test_report_with_no_signal_returns_no_match_block",
     "test_warnings_still_carry_matched_count"],
    "regex_report_parser",
    "asserted on the WIRE that a regex parser was an accepted success source: "
    "parser_id == 'proxy_regex_v0', fusion_mode == 'regex', a populated "
    "receptor panel from free text, and a 'matched count' warning describing "
    "how many regexes fired",
    "the wire contract now forbids a regex parse block entirely: an "
    "unparseable report yields report_parse = null, pipeline_status = "
    "'failed_required_stage', a failed_required ClinicalBERT receipt, grade "
    "and every receptor field null, and no receptor_panel_source warning",
)
# The nlp-file replacements are asserted in the same file the old class lived
# in, so the migration is verifiable in place rather than by cross-reference.
for _k in ("test_full_luminal_a_report",
           "test_report_with_no_signal_returns_no_match_block",
           "test_warnings_still_carry_matched_count"):
    _rk, _beh, _inv, _new = M[_k]
    M[_k] = (
        _rk, _beh,
        ["tests/nlp/test_biopsy_report_parse_wire.py::TestBiopsyRequestShape::"
         "test_unparseable_report_yields_no_parse_block_and_a_failed_stage",
         "tests/nlp/test_biopsy_report_parse_wire.py::TestBiopsyRequestShape::"
         "test_no_report_text_returns_no_parse_block",
         *_inv],
        _new,
    )

# --------------------------------------------------------------------------- #
# tests/data/test_api_real_dicom.py -- placeholder-era wire assertions.
#
# These are behaviour replacements: the endpoints still exist, but "placeholder"
# is no longer an honest answer once a required stage has been attempted and
# failed. Each replacement asserts the NEW state, and asserts that the old
# placeholder wording is absent.
M["test_biopsy_placeholder_returns_shape"] = (
    None,
    "asserted that /v1/biopsy/analyze returned a PLACEHOLDER-shaped response "
    "for free text, which reported 'nothing was attempted' even though the "
    "ClinicalBERT required stage had been attempted and had failed",
    ["tests/data/test_api_real_dicom.py::test_biopsy_free_text_yields_no_regex_parse"],
    "free text yields model_state != placeholder, pipeline_status == "
    "'failed_required_stage', report_parse is None, grade and every receptor "
    "field None, and arbiter_score None; app.py now promotes PLACEHOLDER to "
    "UNAVAILABLE whenever any receipt is failed_required, matching what "
    "/v1/therapy/reason already did in the identical situation",
)
M["test_screening_url_not_yet_wired"] = (
    None,
    "asserted HTTP 501 'not yet wired' for a screening URL input, i.e. that "
    "the capability was merely deferred and would arrive later",
    ["tests/data/test_api_real_dicom.py::test_screening_url_is_refused_not_deferred"],
    "the URL input is refused with HTTP 400, the string 'not yet wired' is "
    "absent from the body, and no overall_score or medsiglip key is emitted -- "
    "a refusal, not a promise",
)
M["test_therapy_placeholder_returns_empty_options"] = (
    None,
    "asserted that /v1/therapy/reason returned empty option lists in a "
    "PLACEHOLDER state, which conflated 'no bridge configured' with "
    "'bridge ran and found nothing'",
    ["tests/data/test_api_real_dicom.py::test_therapy_without_bridge_is_unavailable_not_placeholder"],
    "the response reports model_state == 'unavailable' (not placeholder), "
    "empty option lists, therapy_bridge is None and prognostic_model_executed "
    "is False, so an absent dependency is distinguishable from a negative "
    "result",
)

# Behaviour replacements (not capability retirements).
M["test_therapy_arbiter_suppressed_when_nodes_missing"] = (
    None,
    "asserted that an unobserved lymph-node status returned therapy_triage = "
    "null with a skipped receipt, which hid how much the missing covariate "
    "cost",
    ["tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_gates_when_nodes_missing"],
    "an unobserved node status returns HTTP 422 ManskiBoundsExceeded carrying "
    "the identified set [0.168682, 0.623634], width 0.454952 -- nearly half "
    "the probability scale from one missing field",
)
M["test_dry_run_reports_no_changes_when_repo_is_current"] = (
    None,
    "asserted that a dry run reported no changes, which was vacuous: the "
    "assertion held whether or not the ledger was current",
    ["tests/unit/test_progress_ledger.py::test_committed_ledger_is_current_without_regenerating_it",
     "tests/unit/test_progress_ledger.py::test_check_fails_on_a_stale_ledger",
     "tests/unit/test_progress_ledger.py::test_ledger_path_override_cannot_be_used_to_write"],
    "--check exits 1 on drift against the committed ledger, and --ledger-path "
    "exits 2 unless combined with --check or --dry-run so it cannot be used to "
    "write somewhere unexpected",
)
M["test_status_is_one_of_the_two_tiers"] = (
    None,
    "asserted a two-tier ledger status vocabulary",
    ["tests/unit/test_progress_ledger.py::test_status_is_one_of_the_three_tiers"],
    "the vocabulary is three-tier so a partially-delivered item cannot be "
    "rounded up to done",
)
M["test_health_includes_all_expected_slots"] = (
    None,
    "asserted the health payload advertised every model slot as present",
    ["tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible"],
    "no retired substitute state can appear in the health payload; the "
    "advertised slots are constrained by the registry rather than by a "
    "hand-maintained list",
)
M["test_render_env_reports_realistic_state"] = (
    None,
    "asserted the deployed environment reported a 'realistic' mixture of "
    "placeholder and proxy states",
    ["tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible"],
    "proxy states are not expressible, so there is no realistic-proxy mixture "
    "left to assert",
)
for _k, _behavior in [
    ("test_all_backends_disabled_returns_placeholder",
     "asserted a placeholder envelope when every backend was disabled"),
    ("test_screening_default_placeholder",
     "asserted screening defaulted to a placeholder score"),
    ("test_biopsy_classifier_default_placeholder",
     "asserted biopsy defaulted to a placeholder classifier state"),
    ("test_biopsy_placeholder_when_flag_off",
     "asserted a placeholder biopsy response when the flag was off"),
    ("test_biopsy_placeholder_has_no_gate_report",
     "asserted the placeholder biopsy response carried no gate report"),
    ("test_biopsy_gated_state_when_forbidden",
     "asserted a gated biopsy state when access was forbidden"),
    ("test_biopsy_gated_response_has_structured_gate_report",
     "asserted the gated biopsy response carried a structured gate report"),
    ("test_gated_response_warning_encodes_level_and_reason",
     "asserted the gated warning string encoded level and reason"),
    ("test_l3_arbiter_always_template",
     "asserted the L3 arbiter was permanently in template state"),
    ("test_screening_medsiglip_wins",
     "asserted MedSigLIP won a precedence contest against proxies"),
    ("test_biopsy_medsiglip_probe_on",
     "asserted a MedSigLIP probe flag changed the biopsy state"),
    ("test_model_state_loaded_medsiglip_is_distinct",
     "asserted LOADED_MEDSIGLIP was distinct from the proxy state"),
    ("test_screening_analyze_returns_two_zero_shot_findings",
     "asserted two zero-shot findings were returned"),
    ("test_screening_analyze_uses_modal_and_reports_loaded_medsiglip",
     "asserted the Modal path reported LOADED_MEDSIGLIP"),
    ("test_screening_analyze_wires_cbis_ddsm_probe_when_enabled",
     "asserted a CBIS-DDSM probe was wired in when enabled"),
    ("test_screening_analyze_preprocessing_still_runs_on_real_dicom",
     "asserted preprocessing ran on a real DICOM"),
]:
    M[_k] = (
        None, _behavior,
        ["tests/unit/test_screening_production_contract.py::test_success_requires_1152_dimensional_production_receipt",
         "tests/unit/test_screening_production_contract.py::test_zero_shot_probs_are_reported_as_independent_sigmoid_scores",
         "tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible"],
        "the precedence ladder is gone: there is exactly one success path, it "
        "requires a 1152-dimensional production embedding receipt, and the "
        "zero-shot outputs are labelled independent uncalibrated sigmoids "
        "rather than a normalised distribution",
    )
for _k, _behavior in [
    ("test_therapy_override_her2_positive_wins",
     "asserted a receptor override beat the biopsy panel"),
    ("test_therapy_override_none_fields_treated_as_negative",
     "asserted null override fields were treated as negative -- an imputation"),
    ("test_therapy_override_absent_falls_back_to_biopsy_panel",
     "asserted an absent override fell back to the biopsy panel"),
    ("test_therapy_uses_receptors_override_over_biopsy_output",
     "asserted the override took precedence over biopsy output"),
    ("test_therapy_arbiter_bucket_matches_recommendation",
     "asserted the risk bucket matched the recommendation on an inferred panel"),
]:
    M[_k] = (
        None, _behavior,
        ["tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_scores_only_from_explicit_inputs",
         "tests/unit/test_api_arbiter_wiring.py::test_therapy_bucket_matches_recommendation",
         "tests/unit/test_production_specialist_contracts.py::test_therapy_triage_contrasts_are_mathematically_exact"],
        "a null field is unobserved, never negative: it either closes the "
        "interval when supplied or trips the Manski gate when not",
    )
M["test_client"] = (
    None,
    "a fixture-shaped helper collected as a test; it asserted nothing",
    ["tests/unit/test_manski_gate_no_public_bypass.py::test_enforce_manski_gate_accepts_no_bypass_argument"],
    "no vacuous test remains in the suite; helpers are fixtures",
)


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout


def _removed_tests() -> set[str]:
    diff = _sh(["git", "diff", f"{BASE}..HEAD", "--", "tests/"])
    removed = {m.group(1) for m in re.finditer(r"^-\s*def (test_\w+)", diff, re.M)}
    added = {m.group(1) for m in re.finditer(r"^\+\s*def (test_\w+)", diff, re.M)}
    return removed - added


def _collect_nodeids(env: dict[str, str]) -> set[str]:
    out = subprocess.run(
        # Collect over the WHOLE tests/ tree, not just unit+regression.
        # pyproject declares testpaths = ["tests"]; collecting a narrower
        # scope here silently reports "replacement test does not exist" for
        # any replacement that lives in tests/models, tests/data, tests/nlp
        # or tests/integration. That is a false negative in the receipt
        # generator itself, so the scope must match the declared suite.
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=REPO, capture_output=True, text=True, env=env,
    ).stdout
    return {line.split(" ")[0].strip() for line in out.splitlines() if "::" in line}


def main(out: Path) -> int:
    import os
    env = dict(os.environ, PYTHONPATH="src")

    removed = _removed_tests()
    # Tests removed in the uncommitted working tree too.
    wt = _sh(["git", "diff", "--", "tests/"])
    removed |= {m.group(1) for m in re.finditer(r"^-\s*def (test_\w+)", wt, re.M)}
    removed -= {m.group(1) for m in re.finditer(r"^\+\s*def (test_\w+)", wt, re.M)}

    nodeids = _collect_nodeids(env)
    live_names = {n.split("::")[-1].split("[")[0] for n in nodeids}

    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    unmapped = sorted(removed - set(M))
    for name in unmapped:
        problems.append(f"UNMAPPED removed test: {name}")

    # Verify every retirement claim against the API surface and the registry.
    from oncology_arbiter.api.schemas import RETIRED_MODEL_STATE_VALUES, ModelState
    app_src = (REPO / "src/oncology_arbiter/api/app.py").read_text()
    constructible = {m.value for m in ModelState}
    retirement_status: dict[str, Any] = {}
    for key, spec in RETIREMENTS.items():
        api_leaks = [s for s in spec["api_symbols_banned"] if s in app_src]
        reg_leaks = [
            v for v in spec["registry_values_banned"] if v in constructible
        ]
        not_declared = [
            v for v in spec["registry_values_banned"]
            if v not in RETIRED_MODEL_STATE_VALUES
        ]
        retirement_status[key] = {
            "substitute": spec["substitute"],
            "absent_from_api": not api_leaks,
            "api_leaks": api_leaks,
            "absent_from_registry": not reg_leaks,
            "registry_leaks": reg_leaks,
            "declared_retired_in_registry": not not_declared,
        }
        if api_leaks:
            problems.append(f"{key}: still referenced in api/app.py: {api_leaks}")
        if reg_leaks:
            problems.append(f"{key}: still constructible ModelState: {reg_leaks}")
        if not_declared:
            problems.append(
                f"{key}: {not_declared} absent from RETIRED_MODEL_STATE_VALUES, so "
                f"nothing stops it being re-added"
            )

    # Resolve and run every distinct replacement nodeid once.
    wanted: set[str] = set()
    for _, (_, _, repls, _) in M.items():
        wanted.update(repls)
    missing = sorted(n for n in wanted if n not in nodeids)
    for n in missing:
        problems.append(f"replacement test does not exist: {n}")
    runnable = sorted(wanted & nodeids)

    result_by_node: dict[str, str] = {}
    if runnable:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
             "no:cacheprovider", *runnable],
            cwd=REPO, capture_output=True, text=True, env=env,
        )
        failed = {
            line.split(" ")[1] for line in proc.stdout.splitlines()
            if line.startswith("FAILED ")
        }
        for n in runnable:
            result_by_node[n] = "failed" if n in failed else "passed"
        if proc.returncode != 0:
            problems.append(
                f"replacement suite exit={proc.returncode}; failures={sorted(failed)}"
            )
    for n in missing:
        result_by_node[n] = "MISSING"

    for old in sorted(M):
        key, behavior, repls, invariant = M[old]
        still_present = old in live_names
        if still_present:
            problems.append(
                f"{old} is listed as migrated but still exists in the suite"
            )
        results = {n: result_by_node.get(n, "not_run") for n in repls}
        row = {
            "old_test": old,
            "retired_behavior": behavior,
            "replacement_test": repls,
            "new_invariant": invariant,
            "replacement_result": (
                "passed" if results and all(v == "passed" for v in results.values())
                else "FAILED"
            ),
            "replacement_result_detail": results,
            "migration_kind": (
                "capability_retired" if key else "behaviour_replaced"
            ),
            "capability_retired": key,
            "old_test_absent_from_suite": not still_present,
        }
        if key:
            rs = retirement_status[key]
            row["retired_from_api"] = rs["absent_from_api"]
            row["retired_from_registry"] = rs["absent_from_registry"]
        rows.append(row)

    receipt = {
        "receipt_type": "retired_test_migration",
        "base_commit": BASE,
        "n_removed_tests": len(removed),
        "n_rows": len(rows),
        "n_unmapped": len(unmapped),
        "unmapped": unmapped,
        "retirement_status": retirement_status,
        "verification": {
            "old_tests_confirmed_absent": all(
                r["old_test_absent_from_suite"] for r in rows),
            "all_replacements_exist": not missing,
            "all_replacements_pass": all(
                r["replacement_result"] == "passed" for r in rows),
            "missing_replacements": missing,
        },
        "problems": problems,
        "verdict": "MIGRATED" if not problems else "INCOMPLETE",
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({
        "verdict": receipt["verdict"],
        "n_removed_tests": len(removed),
        "n_rows": len(rows),
        "n_unmapped": len(unmapped),
        "unmapped": unmapped[:15],
        "n_problems": len(problems),
        "problems": problems[:15],
        "out": str(out),
    }, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "artifacts/audit/retired_test_migration.json")
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main(a.out))
