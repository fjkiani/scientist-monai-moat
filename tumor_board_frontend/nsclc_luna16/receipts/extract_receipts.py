"""Extract NSCLC receipts from the live E2E run into flat JSON tables
that the frontend can render without further processing.

Produces:
  01_monai_bundle_card.json     — bundle version/license/metrics/preprocessing
  02_detections.json            — per-detection score, center_mm, diameter,
                                  in_lung_parenchyma boolean
  03_lung_heuristic.json        — risk_score, risk_bucket, driving_feature,
                                  max_diameter_mm, logit, feature contributions
  04_nccn_lite_rules.json       — recommended + not_recommended options with
                                  NCCN section + citation_url
  05_elo_transcript.json        — baseline vs enriched with per-drug reason
  99_transcript_summary.json    — six-column table for the audit-trail card
"""
from __future__ import annotations

import json
import pathlib

E2E = pathlib.Path(__file__).parents[2] / "e2e_run" / "nsclc_case"
OUT = pathlib.Path(__file__).parent


def _load(name: str) -> dict:
    return json.loads((E2E / name).read_text())


case = _load("02_case_full.json")
n = case["nsclc"]
luna16 = n["luna16"]
cs = _load("03_co_scientist_run.json")
elo = _load("04_elo_rank.json")
summary = _load("summary.json")

# ---- 01 MONAI bundle card ---------------------------------------------
bundle_card = {
    "bundle_id": "monai/lung_nodule_ct_detection",
    "bundle_version": luna16["bundle_version"],
    "bundle_license": "Apache-2.0",
    "bundle_source": "https://monai.io/model-zoo.html",
    "task": "3D lung-nodule detection (RetinaNet)",
    "training_data_note": "LIDC-IDRI + trainer-provided negatives; nodules 3–30 mm.",
    "reported_metrics": {
        "mAP_IoU_0.10_0.50_step_0.05_MaxDet_100": summary.get("monai_mAP_reported"),
        "mAR": summary.get("monai_mAR_reported"),
        "source": summary.get("monai_metrics_source"),
    },
    "preprocessing": {
        "hu_clamp": [-1024, 300],
        "target_spacing_mm_zyx": [1.25, 0.703, 0.703],
        "roi_size": [192, 192, 80],
        "nms_thresh": 0.22,
        "score_thresh": 0.02,
        "note": (
            "The 41.5-mm detection here is OUTSIDE the training-distribution "
            "size range (3–30 mm). The detector still spatially localizes "
            "the planted mass but reports low confidence (0.042). The "
            "lung heuristic's diameter measure is what drives the HIGH "
            "risk-bucket, not the RetinaNet score. Both are surfaced."
        ),
    },
    "invocation": {
        "model_state": n["model_state"],  # loaded_luna16_retinanet
        "model_name": n["model_name"],
        "request_id": case.get("provenance", {}).get("request_id"),
        "inference_seconds": luna16["inference_seconds"],
        "n_detections": luna16["n_detections"],
        "top_score": luna16["top_score"],
    },
}
(OUT / "01_monai_bundle_card.json").write_text(json.dumps(bundle_card, indent=2))

# ---- 02 Detections ----------------------------------------------------
det_rows = []
for i, d in enumerate(luna16.get("detections", []) or []):
    det_rows.append({
        "det_id": i,
        "score": d["score"],
        "center_x_mm": d["center_x_mm"],
        "center_y_mm": d["center_y_mm"],
        "center_z_mm": d["center_z_mm"],
        "diameter_mm": d["diameter_mm"],
        "in_lung_parenchyma": d.get("in_lung_parenchyma"),
        "in_distribution": 3.0 <= d["diameter_mm"] <= 30.0,
    })
(OUT / "02_detections.json").write_text(json.dumps({
    "detections": det_rows,
    "note": "in_distribution=false → detection larger than LUNA16 training range.",
}, indent=2))

# ---- 03 Lung heuristic --------------------------------------------------
heur = {
    "model_state_before_upgrade": "proxy_lung_heuristic",
    "risk_score": n.get("risk_score"),
    "risk_bucket": n.get("risk_bucket"),
    "logit": n.get("logit"),
    "driving_feature": n.get("driving_feature"),
    "max_diameter_mm": n.get("max_diameter_mm"),
    "lung_voxel_fraction": n.get("lung_voxel_fraction"),
    "n_candidates_considered": n.get("n_candidates"),
    "note": (
        "Heuristic bucketing on Fleischner-inspired thresholds. NOT a "
        "trained ML classifier. Diameter > 30 mm → HIGH bucket → mass branch."
    ),
}
(OUT / "03_lung_heuristic.json").write_text(json.dumps(heur, indent=2))

# ---- 04 NCCN-lite rules ------------------------------------------------
nccn = {
    "model_state": "proxy_rules_lite",
    "rules_source": "NCCN NSCL v5.2026 + Fleischner 2017 Table 1",
    "warnings_verbatim": n.get("warnings", []),
    "recommended_options": [
        {
            "name": t.get("name"),
            "category": t.get("category"),
            "nccn_section": t.get("nccn_section"),
            "rationale": t.get("rationale"),
            "citation_url": t.get("citation_url"),
        }
        for t in (n.get("therapy_recommended") or [])
    ],
    "not_recommended": [
        {
            "name": t.get("name"),
            "category": t.get("category"),
            "nccn_section": t.get("nccn_section"),
            "rationale": t.get("rationale"),
            "citation_url": t.get("citation_url"),
        }
        for t in (n.get("therapy_not_recommended") or [])
    ],
}
(OUT / "04_nccn_lite_rules.json").write_text(json.dumps(nccn, indent=2))

# ---- 05 Elo transcript --------------------------------------------------
elo_out = {
    "model_state": "proxy_co_scientist",
    "seed": elo.get("seed") or 20260619,
    "k_factor": elo.get("k_factor") or 16,
    "disease_context": elo.get("disease_context"),
    "applied_modifiers": elo.get("applied_modifiers"),
    "baseline_ranking": elo.get("baseline_ranking"),
    "enriched_ranking": elo.get("enriched_ranking"),
    "matches": elo.get("matches"),
    "note": (
        "Deterministic Elo audit-sort over the NCCN-lite rule outputs. "
        "The rank delta on PET/CT (baseline #3 → enriched #2) is driven "
        "by the +0.10 mass-branch modifier applied because risk_bucket=HIGH. "
        "The modifier is a visible, labeled disease-context weight."
    ),
}
(OUT / "05_elo_transcript.json").write_text(json.dumps(elo_out, indent=2))

# ---- 99 Transcript summary --------------------------------------------
transcript_rows = [
    json.loads(line)
    for line in (E2E / "transcript.jsonl").read_text().strip().splitlines()
]
(OUT / "99_transcript_summary.json").write_text(json.dumps({
    "case_id": "SYN-0001",
    "case_type": "synthetic phantom CT (not a real patient)",
    "case_role": "demonstrates the arbiter reasoning end-to-end from CT bytes to Elo",
    "monai_bundle": "monai/lung_nodule_ct_detection@0.6.9 (Apache-2.0)",
    "n_calls": len(transcript_rows),
    "all_2xx": all(200 <= r["status"] < 300 for r in transcript_rows),
    "calls": transcript_rows,
}, indent=2))

print(f"[nsclc receipts] wrote {len(list(OUT.iterdir()))} files to {OUT}")
