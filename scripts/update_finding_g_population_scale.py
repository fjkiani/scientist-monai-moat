#!/usr/bin/env python3
"""Rewrite Finding G in gate_violation_bf5e54f8.json with the completed
population-scale LUNA16 subset0 result (v2 full 89-series run + FROC/CPM),
replacing the earlier n=1-pilot-plus-header-scan placeholder. Applies the
identical edit to both the canonical and in-repo copies and verifies they
are byte-identical afterward (established pattern for Findings K/L).
"""
import hashlib
import json
from datetime import datetime, timezone

CANONICAL = "/mnt/results/gate_violation_bf5e54f8.json"
IN_REPO = "/workspace/smm/artifacts/audit/gate_violation_bf5e54f8.json"

with open("/workspace/smm/docs/proofs/luna16_froc_subset0_v2_metrics.json") as f:
    metrics = json.load(f)

pop = metrics["population_failure_analysis"]
froc = metrics["froc_evaluation"]
ok55 = froc["conditional_on_successful_processing__ok55"]
all89 = froc["intent_to_treat__all89"]
logit = pop["logistic_regression_analysis"]

new_finding_g = {
    "relationship_to_finding_F": (
        "Direct mechanistic consequence of the SAME missing-resample defect "
        "documented in evidence key F_luna16_infer_missing_resample_preprocessing_defect. "
        "Finding F showed the deployed endpoint feeds native-spacing volumes straight "
        "into the RetinaNet with no Spacingd resample, causing SILENT spatial "
        "mis-calibration for mildly-off-spec volumes. This finding G shows the severe "
        "end of the same defect: a large, population-measured fraction of real LUNA16 "
        "series crash the endpoint outright with an unhandled server-side exception, "
        "rather than merely mis-calibrating."
    ),
    "discovery_context": (
        "Surfaced when the first full 89-series subset0 FROC background run "
        "(scripts/run_luna16_froc_subset0.py) crashed the ENTIRE batch on the 4th "
        "series with an uncaught HTTPError, by original design ('record and re-raise: "
        "no silent skip'). Per standing instruction, this anomaly was mathematically "
        "interrogated rather than dismissed as a fluke or silently retried away."
    ),
    "failing_series_n1_pilot": {
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987",
        "shape_dhw": [538, 512, 512],
        "spacing_xyz_mm": [0.7421879768371582, 0.7421879768371582, 0.625],
        "raw_volume_bytes": 282066944,
        "z_extent_mm": 336.25,
        "note": "This n=1 pilot case is one of the 34 failures counted in the population-scale result below; retained here for historical continuity with the original discovery.",
    },
    "reproducibility_check_n1_pilot": {
        "method": "Re-uploaded the identical 538-slice volume as a fresh case (new case_id=0326b58ee90697ca, upload=31.1s) and called /luna16-detect on it a second time in isolation, independent of the crashed batch run, with no other change to inputs.",
        "result": "FAILED again after 185.7s wall: SpecialistServiceError HTTP 500: Internal Server Error (byte-identical error string to the first, in-batch failure).",
        "conclusion": "Deterministic / reproducible for this series, NOT a transient blip.",
    },
    "SUPERSEDED_original_hypothesis_correction": {
        "original_v1_claim": (
            "population_risk_sizing (v1, header-scan-only, no live calls): "
            "'Up to 20/89 (22%) subset0 series share the D>300 characteristic of the "
            "confirmed failure and may exhibit the same crash.' This was a prediction "
            "from a single n=1 failing case plus 3 succeeding comparators, extrapolated "
            "via a single D>300 proxy threshold -- explicitly labeled at the time as "
            "'the exact D threshold is NOT assumed.'"
        ),
        "what_the_full_population_run_actually_found": (
            "The v1 hypothesis was DIRECTIONALLY CORRECT but MATERIALLY INCOMPLETE. "
            "D>300 is real and strong (90.0% failure rate in that stratum vs 23.2% "
            "baseline -- a ~4x risk multiplier, logistic-regression p=1.3e-05), but it "
            "is NOT a clean deterministic threshold: (a) there is a substantial 23.2% "
            "baseline failure rate even among normal-sized (D<=300) series -- 5x higher "
            "than the near-zero rate the v1 pilot's 3 small comparators implied; "
            "(b) the D distributions of ok vs. failed series overlap across a wide "
            "127-325-slice zone, so no single D cutoff correctly separates them; "
            "(c) a stratified logistic regression shows D alone already achieves "
            "AUC=0.811 (a strong, continuous, monotonic risk relationship, not a step "
            "function), and adding spacing_z on top of D does NOT significantly "
            "improve the fit (likelihood-ratio test joint-vs-D-only: chi2=1.91, "
            "p=0.167) -- spacing_z's univariate association with failure (AUC=0.797 "
            "alone) is explained by its correlation with D, not independent "
            "information. The true root cause remains an unhandled server-side "
            "exception whose exact trigger is not fully explained by slice count or "
            "spacing alone; this was disclosed honestly rather than forcing a clean "
            "single-variable story onto a messier reality."
        ),
    },
    "population_scale_empirical_result": {
        "method": (
            "Full resilient re-run of ALL 89/89 subset0 series against the live "
            "luna16-infer endpoint (job run_id=luna16_froc_subset0_run_v2; resumed "
            "cleanly from a v1 checkpoint at series 5 after the original crash-on-series-4; "
            "wall=9019s for series 5-89). scripts/run_luna16_froc_subset0.py was changed "
            "from 'log the real error then re-raise (crash the whole batch)' to 'log the "
            "real error (including failed_stage, shape_dhw, spacing_xyz where available) "
            "then continue to the next series' -- pipeline resilience, NOT "
            "mocking-to-pass: every failure's exact exception string and volume geometry "
            "remain in artifacts/luna16_froc_subset0/processed_seriesuids.json and are "
            "counted as 'failed' (never silently dropped) in run.log's own summary and in "
            "the FROC computed from results.csv. D/spacing extracted independently this "
            "session from the raw .mhd headers of all 89 series (no third-party ITK/DICOM "
            "dependency), reproducibly via scripts/luna16_population_failure_analysis.py."
        ),
        "run_log_summary": pop["run_log_summary"],
        "overall_failure_rate": pop["overall_failure_rate"],
        "D_stratified_failure_rates": pop["D_stratified"],
        "D_distribution": pop["D_distribution"],
        "spacing_z_distribution_mm": pop["spacing_z_distribution_mm"],
        "stratified_logistic_regression": {
            "note": logit["note"],
            "auc_D_only": logit["model_D_only"]["auc"],
            "auc_spacing_z_only": logit["model_spacing_z_only"]["auc"],
            "auc_D_plus_spacing_z": logit["model_D_plus_spacing_z"]["auc"],
            "likelihood_ratio_test_joint_vs_D_only_p_value": logit["likelihood_ratio_test_joint_vs_D_only"]["p_value"],
            "conclusion": "spacing_z adds no significant discrimination beyond D alone (p=0.167); D is a strong (AUC=0.81) continuous risk factor, not a redundant confound and not a clean deterministic cutoff.",
        },
        "timing_signature": pop["timing"],
        "distinct_H_W_across_all_89_series": pop["distinct_H_W_across_all_89_series"],
    },
    "froc_cpm_evaluation": {
        "evaluator_provenance": froc["evaluator_provenance"],
        "conditional_on_successful_processing__ok55": {
            "n_series": ok55["n_series"],
            "n_nodules_official_annotations": ok55["n_nodules_official_annotations"],
            "true_positives": ok55["true_positives"],
            "false_positives": ok55["false_positives"],
            "false_negatives": ok55["false_negatives"],
            "cpm_bootstrap_mean": ok55["cpm_bootstrap_mean"],
            "cpm_95ci_joint": ok55["cpm_95ci_joint"],
            "sensitivity_at_official_fp_points": ok55["sensitivity_at_official_fp_points"],
        },
        "intent_to_treat__all89": {
            "n_series": all89["n_series"],
            "n_nodules_official_annotations": all89["n_nodules_official_annotations"],
            "true_positives": all89["true_positives"],
            "false_positives": all89["false_positives"],
            "false_negatives": all89["false_negatives"],
            "cpm_bootstrap_mean": all89["cpm_bootstrap_mean"],
            "cpm_95ci_joint": all89["cpm_95ci_joint"],
            "sensitivity_at_official_fp_points": all89["sensitivity_at_official_fp_points"],
        },
        "full_receipt": "docs/proofs/luna16_froc_subset0_v2_metrics.json",
    },
    "ship_gate_screening": "BELOW_SHIP_GATE_SCREENING",
    "ship_gate_note": metrics["ship_gate_note"],
    "status": "CONFIRMED_REPRODUCIBLE_POPULATION_SCALE_MEASURED_38pct_FAILURE_RATE_FROC_COMPUTED",
}

_note_template = (
    "Finding G REWRITTEN (not appended) now that luna16_froc_subset0_run_v2 (the "
    "full 89-series resilient re-run) has completed: 55 ok / 34 failed / 89 total "
    "(38.2 pct failure rate), materially higher than the original v1 pilot's ~22 pct "
    "D>300-only estimate. The v1 D>300 hypothesis is explicitly retained and marked "
    "SUPERSEDED-but-directionally-correct rather than deleted, per the standing "
    "instruction to mathematically interrogate anomalies rather than silently "
    "revise them away. A stratified logistic regression (D alone AUC=0.811, "
    "p=1.3e-05; D+spacing_z joint not significantly better, LR-test p=0.167) shows "
    "D is a strong continuous risk factor, not a clean threshold and not confounded "
    "by spacing_z. FROC/CPM computed with the official (mechanically-ported, diff-verified) "
    "LUNA16 evaluator plus a from-scratch joint bootstrap CI on the CPM scalar itself "
    "(not the naive average-of-marginal-per-point-CIs shortcut): CPM_ok55(conditional)={cpm_ok55:.3f} "
    "[{lb_ok55:.3f}, {ub_ok55:.3f}], CPM_all89(intent-to-treat)={cpm_all89:.3f} [{lb_all89:.3f}, {ub_all89:.3f}]. "
    "Ship-gate BELOW_SHIP_GATE_SCREENING applied; endpoint's raw 38.2 pct failure rate is "
    "independently disqualifying for the word 'screening' regardless of CPM."
)

update_log_entry = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "session": "real-tests-and-dataset-gathering (Wave-1) -- Alpha audit follow-up",
    "updated_evidence_keys": ["G_luna16_large_volume_input_causes_deterministic_server_500"],
    "note": _note_template.format(
        cpm_ok55=ok55["cpm_bootstrap_mean"], lb_ok55=ok55["cpm_95ci_joint"][0], ub_ok55=ok55["cpm_95ci_joint"][1],
        cpm_all89=all89["cpm_bootstrap_mean"], lb_all89=all89["cpm_95ci_joint"][0], ub_all89=all89["cpm_95ci_joint"][1],
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


generated_utc_fixed = datetime.now(timezone.utc).isoformat()

for path in (CANONICAL, IN_REPO):
    with open(path) as f:
        d = json.load(f)
    d["evidence"]["G_luna16_large_volume_input_causes_deterministic_server_500"] = new_finding_g
    d["update_log"].append(update_log_entry)
    d["generated_utc"] = generated_utc_fixed
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")

sha_canonical = sha256(CANONICAL)
sha_in_repo = sha256(IN_REPO)
print("canonical sha256:", sha_canonical)
print("in_repo   sha256:", sha_in_repo)
assert sha_canonical == sha_in_repo, "MISMATCH: canonical and in-repo copies diverged after write!"
print("OK: canonical and in-repo copies are byte-identical after Finding G rewrite.")
