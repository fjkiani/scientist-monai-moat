#!/usr/bin/env python3
"""Assemble the single consolidated Finding-G-v2 metrics receipt for
`docs/proofs/luna16_froc_subset0_v2_metrics.json`, combining:
  1. The population-scale D/spacing_z failure root-cause analysis
     (docs/proofs/luna16_froc_subset0_v2_population_analysis.json)
  2. The official-evaluator FROC/CPM result, both on the 55
     successfully-processed series (conditional-on-no-pipeline-crash) and
     on all 89 series (intent-to-treat, 34 failures scored as zero
     detections -- because that is what actually happened when the
     endpoint 500'd)
  3. File-level SHA256 provenance for the inputs that fed the evaluator

This script does not compute anything new; it only reads and re-packages
already-computed, already-verified artifacts so there is one canonical
JSON to cite from the Finding G rewrite.
"""
import hashlib
import json


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open("docs/proofs/luna16_froc_subset0_v2_population_analysis.json") as f:
        population = json.load(f)
    with open("artifacts/luna16_froc_subset0/froc_eval_ok55/cpm_with_joint_ci.json") as f:
        cpm_ok55 = json.load(f)
    with open("artifacts/luna16_froc_subset0/froc_eval_all89/cpm_with_joint_ci.json") as f:
        cpm_all89 = json.load(f)

    def read_cad_analysis(path):
        with open(path) as f:
            text = f.read()
        out = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("True positives:"):
                out["true_positives"] = int(line.split(":")[1])
            elif line.startswith("False positives:"):
                out["false_positives"] = int(line.split(":")[1])
            elif line.startswith("False negatives:"):
                out["false_negatives"] = int(line.split(":")[1])
            elif line.startswith("Total number of nodules:"):
                out["total_number_of_nodules"] = int(line.split(":")[1])
            elif line.startswith("Sensitivity:"):
                out["raw_sensitivity_all_operating_points"] = float(line.split(":")[1])
            elif line.startswith("Average number of candidates per scan:"):
                out["avg_candidates_per_scan"] = float(line.split(":")[1])
        return out

    cad_ok55 = read_cad_analysis("artifacts/luna16_froc_subset0/froc_eval_ok55/CADAnalysis.txt")
    cad_all89 = read_cad_analysis("artifacts/luna16_froc_subset0/froc_eval_all89/CADAnalysis.txt")

    provenance_files = {
        "results_csv": "artifacts/luna16_froc_subset0/results.csv",
        "processed_seriesuids_json": "artifacts/luna16_froc_subset0/processed_seriesuids.json",
        "seriesuids_ok55_csv": "artifacts/luna16_froc_subset0/seriesuids_ok55.csv",
        "seriesuids_all89_csv": "artifacts/luna16_froc_subset0/seriesuids_all89.csv",
        "luna16_annotations_csv": "vendor/luna16_evaluation/annotations/annotations.csv",
        "luna16_annotations_excluded_csv": "vendor/luna16_evaluation/annotations/annotations_excluded.csv",
    }
    sha256_receipts = {k: sha256(v) for k, v in provenance_files.items()}

    receipt = {
        "finding": "G_v2",
        "endpoint": "luna16-infer",
        "cohort": "LUNA16 subset0, 89 CT series (all of subset0, not a further subsample)",
        "run_id": "luna16_froc_subset0_run_v2",
        "population_failure_analysis": population,
        "froc_evaluation": {
            "evaluator_provenance": (
                "Official LUNA16 grand-challenge evaluation script "
                "(evaluationScript.zip, MD5 02680d438a80dc26eeff0c12e1942642, "
                "Zenodo 10.5281/zenodo.3723295, CC-BY-4.0), mechanically "
                "ported Python2->3 with zero changes to matching/FROC logic "
                "(vendor/luna16_evaluation/py3_port/PORT_NOTES.md documents "
                "every diff). CPM (mean sensitivity at the 7 official "
                "FP/scan operating points [0.125,0.25,0.5,1,2,4,8]) is not "
                "printed by the vendored script itself; computed here via "
                "scripts/luna16_compute_cpm_with_ci.py, which drives the "
                "unmodified vendored evaluateCAD()/computeFROC_bootstrap() "
                "and only adds a side-channel capture of the per-replicate "
                "interpolated sensitivity matrix so the CPM's 95% CI can be "
                "computed as a proper joint statistic (all 7 points on a "
                "replicate share the same resampled cases and are "
                "correlated) rather than as the average of 7 independent "
                "marginal per-point CIs."
            ),
            "conditional_on_successful_processing__ok55": {
                "description": "Scored on the 55/89 series the endpoint actually returned detections for. Isolates core detector quality from the pipeline-crash defect.",
                "n_series": 55, "n_nodules_official_annotations": cad_ok55["total_number_of_nodules"],
                "true_positives": cad_ok55["true_positives"], "false_positives": cad_ok55["false_positives"],
                "false_negatives": cad_ok55["false_negatives"], "avg_candidates_per_scan": cad_ok55["avg_candidates_per_scan"],
                "cpm_bootstrap_mean": cpm_ok55["cpm_bootstrap_mean"],
                "cpm_95ci_joint": [cpm_ok55["cpm_95ci_lower_joint"], cpm_ok55["cpm_95ci_upper_joint"]],
                "sensitivity_at_official_fp_points": dict(zip(cpm_ok55["official_fp_per_scan_points"], cpm_ok55["per_point_sensitivity_bootstrap_mean"])),
            },
            "intent_to_treat__all89": {
                "description": "Scored on all 89/89 subset0 series; the 34 series the endpoint 500'd on contribute zero detections (because that is literally what a caller received). This is the honest 'as-deployed today' number.",
                "n_series": 89, "n_nodules_official_annotations": cad_all89["total_number_of_nodules"],
                "true_positives": cad_all89["true_positives"], "false_positives": cad_all89["false_positives"],
                "false_negatives": cad_all89["false_negatives"], "avg_candidates_per_scan": cad_all89["avg_candidates_per_scan"],
                "cpm_bootstrap_mean": cpm_all89["cpm_bootstrap_mean"],
                "cpm_95ci_joint": [cpm_all89["cpm_95ci_lower_joint"], cpm_all89["cpm_95ci_upper_joint"]],
                "sensitivity_at_official_fp_points": dict(zip(cpm_all89["official_fp_per_scan_points"], cpm_all89["per_point_sensitivity_bootstrap_mean"])),
            },
        },
        "ship_gate_screening": "BELOW_SHIP_GATE_SCREENING",
        "ship_gate_note": (
            "CPM (LUNA16's own headline metric) is not an AUC and the >=0.85 "
            "AUC screening gate does not directly apply to it in the same "
            "units; regardless, CPM_ok55=%.3f and CPM_all89=%.3f are both far "
            "below any plausible screening-grade sensitivity/FP-rate "
            "trade-off, and the endpoint's 38.2%% raw failure rate on normal "
            "clinical CT geometry alone is independently disqualifying for "
            "the word 'screening' regardless of detector CPM." % (
                cpm_ok55["cpm_bootstrap_mean"], cpm_all89["cpm_bootstrap_mean"],
            )
        ),
        "sha256_provenance": sha256_receipts,
    }

    with open("docs/proofs/luna16_froc_subset0_v2_metrics.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2)[:4000])


if __name__ == "__main__":
    main()
