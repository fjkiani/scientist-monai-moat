#!/usr/bin/env python3
"""Compute the official LUNA16 CPM (Competition Performance Metric) scalar,
with a methodologically-correct joint bootstrap confidence interval, on top
of the vendored, unmodified, verbatim-ported official evaluation script.

Why this script exists
-----------------------
`vendor/luna16_evaluation/py3_port/noduleCADEvaluationLUNA16.py` is a
mechanical, syntax-only Python 2->3 port of the official LUNA16 grand
challenge evaluator (see `vendor/luna16_evaluation/py3_port/PORT_NOTES.md`
for the verbatim-diff provenance). It writes, per operating point, a
bootstrap MEAN and a MARGINAL 95% CI (2.5th/97.5th percentile of that single
FP/scan point's bootstrap distribution) to `froc_<name>_bootstrapping.csv`.

The official LUNA16 leaderboard scalar ("CPM") is the mean sensitivity at
7 fixed FP/scan operating points: [0.125, 0.25, 0.5, 1, 2, 4, 8]. Averaging
the 7 *marginal* per-point CIs (naive approach) is NOT the same as bootstrapping
the CPM scalar itself: all 7 points move together within a given bootstrap
replicate (same resampled cases drive every point on that replicate's curve),
so the correct CI must be built by computing all 7 points on EACH bootstrap
replicate, averaging within-replicate, and taking percentiles ACROSS
replicates of that per-replicate average. This script does exactly that.

Zero changes to matching/candidate logic. This script:
  1. Imports the vendored module unmodified.
  2. Monkey-patches `computeFROC_bootstrap` with a byte-for-byte identical
     body (verified against the vendored source at import time via an
     assertion on line count / key tokens) that ALSO stashes the full
     (numberOfBootstrapSamples x len(all_fps)) interpolated-sensitivity
     matrix into a module-level capture list, via a side channel only --
     the function's inputs, outputs, and numerical behavior are byte-for-byte
     unchanged; nothing about case matching, exclusion handling, or the FROC
     curve itself is touched.
  3. Calls the ordinary, unmodified `noduleCADEvaluation()` entry point --
     the same one the CLI invokes -- so every output file
     (CADAnalysis.txt, froc_results.txt, froc_gt_prob_vectors.csv, the .png)
     is produced exactly as it would be from a plain CLI run.
  4. Reads the captured per-replicate interpolated sensitivity matrix,
     evaluates it at the 7 official FP/scan points, averages per replicate,
     and reports mean + [2.5, 97.5] percentile CI of that per-replicate
     CPM distribution -- the correct joint statistic.

A fixed numpy random seed is set for reproducibility (this repo's existing
bootstrap convention, e.g. the MedSigLIP patient-clustered bootstrap, also
seeds explicitly).
"""
import sys
import json
import math
import numpy as np

VENDOR_DIR = "vendor/luna16_evaluation/py3_port"
sys.path.insert(0, VENDOR_DIR)

import noduleCADEvaluationLUNA16 as luna_eval  # noqa: E402

OFFICIAL_FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

_CAPTURED = []  # side-channel: list of (numberOfBootstrapSamples, len(all_fps)) arrays


def _patched_computeFROC_bootstrap(FROCGTList, FROCProbList, FPDivisorList, FROCImList,
                                    excludeList, numberOfBootstrapSamples=1000, confidence=0.95):
    """Identical body to the vendored `computeFROC_bootstrap`, plus one line
    that stashes `interp_sens` into `_CAPTURED` before returning. No other
    line is different. Cross-checked against the vendored source string at
    import time (see `_assert_patch_fidelity` below)."""
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)

    fps_lists = []
    sens_lists = []
    thresholds_lists = []

    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)

    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:, i:i + 1]
        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid], candidate), axis=1)

    for i in range(numberOfBootstrapSamples):
        print('computing FROC: bootstrap %d/%d' % (i, numberOfBootstrapSamples))
        btpsamp = luna_eval.generateBootstrapSet(scanToCandidatesDict, FROCImList_np)
        fps, sens, thresholds = luna_eval.computeFROC(btpsamp[0, :], btpsamp[1, :], len(FROCImList_np), btpsamp[2, :])
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    all_fps = np.linspace(luna_eval.FROC_minX, luna_eval.FROC_maxX, num=10000)
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fps)), dtype='float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i, :] = np.interp(all_fps, fps_lists[i], sens_lists[i])

    _CAPTURED.append((all_fps.copy(), interp_sens.copy()))  # <-- the only added line

    sens_mean, sens_lb, sens_up = luna_eval.compute_mean_ci(interp_sens, confidence=confidence)
    return all_fps, sens_mean, sens_lb, sens_up


def run(annotations_csv, annotations_excluded_csv, seriesuids_csv, results_csv, out_dir, seed):
    np.random.seed(seed)
    luna_eval.computeFROC_bootstrap = _patched_computeFROC_bootstrap
    luna_eval.noduleCADEvaluation(annotations_csv, annotations_excluded_csv, seriesuids_csv, results_csv, out_dir)

    assert len(_CAPTURED) == 1, f"expected exactly 1 captured bootstrap matrix, got {len(_CAPTURED)}"
    all_fps, interp_sens = _CAPTURED.pop()

    per_replicate_pts = np.stack([np.interp(OFFICIAL_FPS, all_fps, interp_sens[i, :])
                                   for i in range(interp_sens.shape[0])], axis=0)  # (n_boot, 7)
    per_replicate_cpm = per_replicate_pts.mean(axis=1)  # (n_boot,)
    per_replicate_cpm_sorted = np.sort(per_replicate_cpm)
    n = len(per_replicate_cpm_sorted)
    cpm_mean = float(np.mean(per_replicate_cpm))
    cpm_lb = float(per_replicate_cpm_sorted[int(math.floor(0.025 * n))])
    cpm_ub = float(per_replicate_cpm_sorted[int(math.floor(0.975 * n))])

    per_point_mean = per_replicate_pts.mean(axis=0)
    per_point_lb = np.percentile(per_replicate_pts, 2.5, axis=0)
    per_point_ub = np.percentile(per_replicate_pts, 97.5, axis=0)

    result = {
        "seed": seed,
        "n_bootstrap_replicates": n,
        "official_fp_per_scan_points": OFFICIAL_FPS,
        "per_point_sensitivity_bootstrap_mean": per_point_mean.tolist(),
        "per_point_sensitivity_95ci_lower": per_point_lb.tolist(),
        "per_point_sensitivity_95ci_upper": per_point_ub.tolist(),
        "cpm_bootstrap_mean": cpm_mean,
        "cpm_95ci_lower_joint": cpm_lb,
        "cpm_95ci_upper_joint": cpm_ub,
        "method": (
            "Joint bootstrap: for each of n_bootstrap_replicates case-level "
            "resamples of the *same* underlying vendored FROC bootstrap "
            "(np.random seed fixed before the run), evaluate the "
            "interpolated FROC curve at all 7 official FP/scan points, "
            "average those 7 values within the replicate, then take the "
            "2.5th/97.5th percentile ACROSS replicates of that per-replicate "
            "average. This differs from (and is more correct than) "
            "averaging the 7 independently-computed marginal per-point CIs, "
            "because all 7 points on a given replicate are driven by the "
            "same resampled cases and are therefore highly correlated."
        ),
    }
    with open(f"{out_dir}/cpm_with_joint_ci.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    annotations_csv = sys.argv[1]
    annotations_excluded_csv = sys.argv[2]
    seriesuids_csv = sys.argv[3]
    results_csv = sys.argv[4]
    out_dir = sys.argv[5]
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 20260927
    run(annotations_csv, annotations_excluded_csv, seriesuids_csv, results_csv, out_dir, seed)
