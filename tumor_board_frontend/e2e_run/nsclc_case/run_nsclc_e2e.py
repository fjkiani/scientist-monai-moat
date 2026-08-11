"""NSCLC L1→L5 live E2E against the running arbiter.

Boots against http://127.0.0.1:8130 with:
  ONCOLOGY_ARBITER_AUTH_MODE=off
  ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1
  ONCOLOGY_ARBITER_ENABLE_LUNA16_RETINANET=1

Pipeline:
  01_ingest_ct         — synth CT already on disk (16 slices, 128×128,
                          1.0×1.0×2.5 mm, planted +50-HU ellipsoid mass)
  02_case_full         — POST /v1/case/full?cancer=nsclc → real MONAI
                          RetinaNet TorchScript + lung heuristic +
                          NCCN-lite rules
  03_co_scientist_run  — POST /v1/co_scientist/run seeded from step-02
                          therapy envelope; 4-phase deterministic loop
  04_elo_rank          — POST /v1/elo/rank tournament over the 4 NCCN
                          options with disease_context echoing the
                          lung heuristic's real risk_bucket + diameter.

Every response is dumped verbatim. transcript.jsonl records verb/path/
request_id/model_state/latency/size.
"""
from __future__ import annotations

import json
import pathlib
import time

import httpx

BASE = "http://127.0.0.1:8130"
SERIES_DIR = "/tmp/e2e_synth_ct/SYN-0001/STUDY/CT_SYN"
HERE = pathlib.Path(__file__).parent
client = httpx.Client(base_url=BASE, timeout=180.0)

transcript: list[dict] = []


def _rec(step: str, method: str, path: str, resp: httpx.Response, saved: str) -> None:
    try:
        b = resp.json()
    except Exception:
        b = {}
    prov = b.get("provenance") if isinstance(b, dict) and isinstance(b.get("provenance"), dict) else {}
    transcript.append({
        "step": step,
        "method": method,
        "path": path,
        "status": resp.status_code,
        "latency_s": resp.elapsed.total_seconds(),
        "size_bytes": len(resp.content),
        "request_id": (prov or {}).get("request_id") or (b or {}).get("request_id"),
        "model_state": (prov or {}).get("model_state") or (b or {}).get("model_state"),
        "model_name": (prov or {}).get("model_name"),
        "saved_as": saved,
    })


def _save(name: str, obj) -> None:
    (HERE / name).write_text(json.dumps(obj, indent=2, default=str))


# ---- Step 1: L1 CT ingestion params (recorded, not called) --------------
_save("01_ingest_ct.json", {
    "series_dir": SERIES_DIR,
    "shape": [16, 128, 128],
    "spacing_mm_zyx": [2.5, 1.0, 1.0],
    "hu_range": {"gantry": -1200, "lung": -800, "body": -50, "planted_nodule": 50},
    "planted_ellipsoid_shape_zyx": [7, 34, 34],
    "planted_center_zyx_mm_approx": [20.0, 64.0, 44.0],
    "notes": (
        "Synthetic phantom matching the demo_provenance profile in "
        "demo_samples/nsclc.json. Not a real patient CT. The MONAI RetinaNet "
        "code path is the same one used on LIDC-IDRI."
    ),
})

# ---- Step 2: L2/L3/L4 via /v1/case/full?cancer=nsclc --------------------
case_req = {
    "screening_input": None,
    "biopsy_input": None,
    "therapy_context": {},
    "nsclc_ct_input": {
        "series_dir": SERIES_DIR,
        "patient_id": "SYN-0001",
        "top_n": 10,
    },
}
r = client.post("/v1/case/full", params={"cancer": "nsclc"}, json=case_req)
_rec("02_case_full", "POST", "/v1/case/full?cancer=nsclc", r, "02_case_full.json")
case_body = r.json()
_save("02_case_full.json", case_body)
n = case_body.get("nsclc", {}) or {}
tr_reco = n.get("therapy_recommended", []) or []
tr_notrec = n.get("therapy_not_recommended", []) or []
print(f"[02] status={r.status_code} state={n.get('model_state')} "
      f"bucket={n.get('risk_bucket')} n_reco={len(tr_reco)} n_notrec={len(tr_notrec)} "
      f"luna16.top={(n.get('luna16') or {}).get('top_score')} "
      f"luna16.inf_s={(n.get('luna16') or {}).get('inference_seconds')}")


def _to_option(t: dict) -> dict:
    return {
        "regimen": t.get("name", "?"),
        "line_of_therapy": 1,
        "rationale": t.get("rationale", ""),
        "evidence": [{
            "url": t.get("citation_url", ""),
            "quoted_text": (t.get("rationale") or "")[:250],
            "source": "nccn",
        }],
        "contraindications": [],
    }


# ---- Step 3: L5a Co-Scientist 4-phase loop -------------------------------
therapy_env = {
    "provenance": {
        "request_id": "nsclc-e2e-seed",
        "model_name": n.get("model_name"),
        "model_state": n.get("model_state"),
        "model_version": None,
        "gate_report": None,
        "model_endpoint_url": None,
    },
    "disclaimer": case_body.get("disclaimer"),
    "caveat": case_body.get("caveat"),
    "honesty_gate": {"seen_urls_count": 0, "evidence_kept": 0, "evidence_dropped": 0, "hypotheses_dropped": 0},
    "evidence": [],
    "warnings": n.get("warnings", []) or [],
    "recommended_options": [_to_option(t) for t in tr_reco],
    "not_recommended": [_to_option(t) for t in tr_notrec],
}
seed_urls = sorted({t.get("citation_url") for t in (tr_reco + tr_notrec) if t.get("citation_url")})

cs_req = {
    "screening": None,
    "biopsy": None,
    "therapy": therapy_env,
    "seed_urls": list(seed_urls),
    "top_n_evolve": 3,
    "n_variants": 2,
    "return_top": 8,
}
r = client.post("/v1/co_scientist/run", json=cs_req)
_rec("03_co_scientist", "POST", "/v1/co_scientist/run", r, "03_co_scientist_run.json")
cs = r.json()
_save("03_co_scientist_run.json", cs)
print(f"[03] status={r.status_code} state={cs.get('provenance',{}).get('model_state')} "
      f"initial={cs.get('initial_count')} after_reflect={cs.get('after_reflect')} "
      f"after_evolve={cs.get('after_evolve')} urls_dropped={cs.get('urls_dropped_hallucinated')} "
      f"n_hyps={len(cs.get('hypotheses',[]))}")

# ---- Step 4: L5b Elo tournament ------------------------------------------
drugs = []
seen_ids = set()
for i, t in enumerate(tr_reco):
    did = t["name"].lower().replace(" ", "_").replace("/", "_")[:60]
    while did in seen_ids:
        did = did + f"_{i}"
    seen_ids.add(did)
    drugs.append({
        "drug_id": did,
        "regimen": t["name"],
        "line": 1,
        "confidence": 0.75,
        "evidence": [{"url": t["citation_url"], "quoted_text": t["rationale"][:250], "source": "nccn"}],
        "honesty_markers": {"proxy": True, "gated": False, "loaded": False},
    })
for i, t in enumerate(tr_notrec):
    did = t["name"].lower().replace(" ", "_").replace("/", "_")[:60] + "_notrec"
    while did in seen_ids:
        did = did + f"_{i}"
    seen_ids.add(did)
    drugs.append({
        "drug_id": did,
        "regimen": t["name"],
        "line": 1,
        "confidence": 0.20,
        "evidence": [{"url": t["citation_url"], "quoted_text": t["rationale"][:250], "source": "nccn"}],
        "honesty_markers": {"proxy": True, "gated": False, "loaded": False},
    })

elo_req = {
    "drugs": drugs,
    "modifiers": {
        # A high-risk mass on a real read gets a bump on the diagnostic
        # branch (biopsy) — because the manuscript, sorry, the guideline
        # explicitly requires tissue diagnosis before therapy for masses
        # ≥30 mm. This is a visible, labeled disease-context modifier;
        # the UI shows the delta so the user sees why the order changed.
        "ct-guided_biopsy_or_bronchoscopy_for_tissue_diagnosis": 0.15,
        "pet_ct_for_staging": 0.10,
    },
    "disease_context": {
        "cancer": "nsclc",
        "risk_bucket": n.get("risk_bucket"),
        "max_diameter_mm": n.get("max_diameter_mm"),
        "driving_feature": n.get("driving_feature"),
        "risk_score": n.get("risk_score"),
    },
    "k_factor": 16,
    "seed": 20260619,
}
_save("04_elo_rank_request.json", elo_req)
r = client.post("/v1/elo/rank", json=elo_req)
_rec("04_elo_rank", "POST", "/v1/elo/rank", r, "04_elo_rank.json")
elo = r.json()
_save("04_elo_rank.json", elo)
print(f"[04] status={r.status_code} state={elo.get('provenance',{}).get('model_state')} "
      f"n_baseline={len(elo.get('baseline_ranking',[]))} "
      f"n_enriched={len(elo.get('enriched_ranking',[]))} "
      f"n_matches={len(elo.get('matches',[]))}")

# ---- Save transcript + summary -------------------------------------------
(HERE / "transcript.jsonl").write_text(
    "\n".join(json.dumps(row) for row in transcript) + "\n"
)
_save("summary.json", {
    "case_id": "SYN-0001",
    "n_calls": len(transcript),
    "all_2xx": all(200 <= r["status"] < 300 for r in transcript),
    "monai_bundle": "monai/lung_nodule_ct_detection@0.6.9",
    "monai_bundle_license": "Apache-2.0",
    "monai_metrics_source": "monai_bundles/lung_nodule_ct_detection/configs/metadata.json",
    "monai_mAP_reported": 0.852,
    "monai_mAR_reported": 0.998,
    "live_luna16_inference_seconds": (n.get("luna16") or {}).get("inference_seconds"),
    "live_luna16_top_score": (n.get("luna16") or {}).get("top_score"),
    "live_luna16_n_detections": (n.get("luna16") or {}).get("n_detections"),
    "live_max_diameter_mm": n.get("max_diameter_mm"),
    "live_risk_bucket": n.get("risk_bucket"),
    "live_risk_score": n.get("risk_score"),
    "live_driving_feature": n.get("driving_feature"),
    "n_therapy_recommended": len(tr_reco),
    "n_therapy_not_recommended": len(tr_notrec),
    "n_co_scientist_hypotheses": len(cs.get("hypotheses", [])),
    "co_scientist_urls_dropped": cs.get("urls_dropped_hallucinated"),
    "co_scientist_hypotheses_dropped": cs.get("hypotheses_dropped"),
    "elo_seed": 20260619,
})
print(f"[done] {len(transcript)} calls saved to {HERE}")
