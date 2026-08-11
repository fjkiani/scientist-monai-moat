"""AK MBD4-LOF ingestion + audit-and-surface transcript.

This is the *honest* AK flow. The arbiter did NOT generate the manuscript's
p-values, effect sizes, or the PARP falsification. Those come from the
`crispro-backend-v2 @ bfd6d11f...` bundle SHA-locked to manuscript repo SHA
`d33f6403...`. What the arbiter DOES do:

  01_get_sample        — GET  /v1/demo/samples/ak_mbd4_lof_case
                          → fetch the pre-packaged bundle payload
  02_validate_bundle   — POST /v1/tumor_board/bundle
                          → contract validation against TumorBoardBundle
                          → checks 6-row evidence_matrix, 4 stress_test
                          + 1 axis_partner + 1 falsification_arm auxiliary
                          → returns bundle_sha256 + request_id as audit
                          receipt of record
  03_elo_audit_sort    — POST /v1/elo/rank
                          → runs the 7 AK drugs through Elo with visible,
                          labeled manuscript-derived modifiers
                          (ceralasertib +0.20 lead, olaparib/talazoparib
                          −0.30 falsified). This is an *audit sort*
                          not a discovery ranking.

Nothing in this pipeline discovers new biology. Everything the arbiter
contributes here is verifiable: (a) contract enforcement, (b) URL honesty
gate (dropping non-seed URLs), (c) audit-of-record bundle SHA, (d)
deterministic Elo ordering with the manuscript-derived modifiers surfaced
as labeled weights.
"""
from __future__ import annotations

import json
import pathlib
import time

import httpx

BASE = "http://127.0.0.1:8130"
HERE = pathlib.Path(__file__).parent
client = httpx.Client(base_url=BASE, timeout=60.0)

transcript: list[dict] = []


def _rec(step: str, method: str, path: str, resp: httpx.Response, saved: str) -> None:
    try:
        b = resp.json()
    except Exception:
        b = {}
    prov = (
        b.get("provenance")
        if isinstance(b, dict) and isinstance(b.get("provenance"), dict)
        else {}
    )
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


# ---- Step 1: fetch the packaged sample ----------------------------------
r = client.get("/v1/demo/samples/ak_mbd4_lof_case")
_rec("01_get_sample", "GET", "/v1/demo/samples/ak_mbd4_lof_case", r, "01_sample.json")
sample = r.json()
_save("01_sample.json", sample)
bundle = sample.get("payload", sample)  # the demo samples endpoint returns {payload: ..., meta: ...}
_sl_prov = (bundle.get("synthetic_lethality") or {}).get("provenance", {})
print(f"[01] status={r.status_code} contract={bundle.get('contract_version')} "
      f"patient={bundle.get('patient_id')} "
      f"backend_sha={_sl_prov.get('backend_head_sha')} "
      f"manuscript_sha={_sl_prov.get('manuscript_repo_sha_at_audit')}")

# ---- Step 2: validate the bundle (arbiter's real contribution) ----------
# The endpoint accepts either the full bundle envelope or the payload
# subtree. Post the payload as-is; validator rejects if contract missing.
r = client.post("/v1/tumor_board/bundle", json=bundle)
_rec("02_validate_bundle", "POST", "/v1/tumor_board/bundle", r, "02_validate_bundle.json")
val = r.json()
_save("02_validate_bundle.json", val)
prov = val.get("provenance", {}) or {}
print(f"[02] status={r.status_code} state={prov.get('model_state')} "
      f"request_id={prov.get('request_id')} "
      f"bundle_sha256={val.get('bundle_sha256') or val.get('audit',{}).get('bundle_sha256')}")

# ---- Step 3: audit sort the 7 recommended drugs -------------------------
# Manuscript-derived modifiers, labeled so the UI shows them as visible
# weights, not silent priors.
sl = (bundle.get("synthetic_lethality") or {})
reco = sl.get("recommended_drugs") or []
if not reco:
    # fall back to whatever field the sample uses
    reco = bundle.get("recommended_drugs", []) or []

MODIFIERS = {
    "ceralasertib_atr_wee1": (0.20, "manuscript_lead_axis_boost"),
    "adavosertib_atr_wee1": (0.10, "manuscript_supporting_axis_partner"),
    "berzosertib_atr_wee1": (0.05, "manuscript_supporting_axis"),
    "olaparib_parp_inhibitors": (-0.30, "manuscript_falsified_penalty"),
    "talazoparib_parp_inhibitors": (-0.30, "manuscript_falsified_penalty"),
}

em_rows = ((bundle.get("synthetic_lethality") or {}).get("provenance") or {}) \
    .get("evidence_matrix", {}).get("rows", [])
axis_row = {r.get("axis"): r for r in em_rows}


def _row_evidence(axis: str) -> list[dict]:
    """Pull one anchor URL per row (gdsc + top auxiliary) for the audit."""
    row = axis_row.get(axis) or {}
    out: list[dict] = []
    g = row.get("gdsc") or {}
    if g.get("p_value") is not None:
        pmids = g.get("pmids") or []
        out.append({
            "url": pmids[0] if pmids else "",
            "quoted_text": (g.get("summary") or "")[:250],
            "source": "manuscript_receipt.gdsc",
        })
    for ae in (row.get("auxiliary_evidence") or []):
        if ae.get("p_value") is None:
            continue
        pmids = ae.get("pmids") or []
        out.append({
            "url": pmids[0] if pmids else "",
            "quoted_text": (ae.get("summary") or "")[:250],
            "source": f"manuscript_receipt.{ae.get('modality','aux')}",
        })
    return out


drugs = []
seed_urls: set[str] = set()
for d in reco:
    drug_name = d.get("drug_name") or ""  # sample uses drug_name
    axis = d.get("axis") or ""
    surface_status = d.get("surface_status") or ""
    did = f"{drug_name}_{axis}".lower().replace(" ", "_")[:60]
    ev_out = _row_evidence(axis)
    for ev in ev_out:
        if ev["url"]:
            seed_urls.add(ev["url"])
    if not ev_out:
        # Every drug must carry at least one evidence row for the schema.
        # Fall back to the axis row's mechanism string.
        row = axis_row.get(axis) or {}
        ev_out = [{
            "url": "",
            "quoted_text": (row.get("mechanism") or "manuscript_receipt")[:250],
            "source": "manuscript_receipt.mechanism",
        }]
    confidence = 0.85 if surface_status == "RECOMMENDED" else 0.15
    drugs.append({
        "drug_id": did,
        "regimen": drug_name,
        "line": 1,
        "confidence": confidence,
        "evidence": ev_out,
        "honesty_markers": {"proxy": False, "gated": False, "loaded": True},
    })

elo_req = {
    "drugs": drugs,
    "modifiers": {k: v[0] for k, v in MODIFIERS.items()},
    "disease_context": {
        "cancer": "ampullary_adenocarcinoma_hgsoc_like",
        "biomarker": "MBD4_LOF",
        "manuscript_repo_sha_at_audit": bundle.get("manuscript_repo_sha_at_audit"),
        "backend_head_sha": bundle.get("crispro_backend_head_sha") or bundle.get("backend_head_sha"),
    },
    "k_factor": 16,
    "seed": 20260619,
}
_save("03_elo_audit_sort_request.json", elo_req)
r = client.post("/v1/elo/rank", json=elo_req)
_rec("03_elo_audit_sort", "POST", "/v1/elo/rank", r, "03_elo_audit_sort.json")
elo = r.json()
_save("03_elo_audit_sort.json", elo)
print(f"[03] status={r.status_code} n_drugs={len(drugs)} "
      f"n_baseline={len(elo.get('baseline_ranking',[]))} "
      f"n_enriched={len(elo.get('enriched_ranking',[]))}")

# ---- Save transcript + summary -------------------------------------------
(HERE / "transcript.jsonl").write_text(
    "\n".join(json.dumps(r) for r in transcript) + "\n"
)
# Pull the manuscript anchors from the bundle for the summary card
em_axes = [row.get("axis") or row.get("evidence_axis") for row in bundle.get("evidence_matrix", []) or []]

_save("summary.json", {
    "case_id": bundle.get("patient_id"),
    "n_calls": len(transcript),
    "all_2xx": all(200 <= r["status"] < 300 for r in transcript),
    "contract_version": bundle.get("contract_version"),
    "manuscript_repo_sha_at_audit": _sl_prov.get("manuscript_repo_sha_at_audit"),
    "backend_head_sha": _sl_prov.get("backend_head_sha"),
    "backend_branch": _sl_prov.get("backend_branch"),
    "datasets_used": _sl_prov.get("datasets_used"),
    "statistical_test": _sl_prov.get("statistical_test"),
    "effect_size_metric": _sl_prov.get("effect_size_metric"),
    "audit_artifact": _sl_prov.get("audit_artifact"),
    "generated_at": bundle.get("generated_at"),
    "bundle_sha256_api_echo": val.get("bundle_sha256") or (val.get("audit", {}) or {}).get("bundle_sha256"),
    "tumor_board_bundle_request_id": prov.get("request_id"),
    "evidence_matrix_axes": em_axes,
    "n_recommended_drugs": len(reco),
    "elo_seed": 20260619,
    "elo_modifiers": {k: {"weight": v[0], "label": v[1]} for k, v in MODIFIERS.items()},
    "contribution_boundary": {
        "manuscript_receipts": [
            "6 anchor p-values", "Cohen's d values", "n_mut / n_wt counts",
            "GDSC2 / DepMap 24Q2 datasets", "PARP1 falsification result",
            "Mann-Whitney U one-sided test choice",
        ],
        "arbiter_contribution": [
            "TumorBoardBundle contract validation",
            "Honesty gate: URLs not in seed_urls stripped from co-scientist reflect",
            "bundle_sha256 API-echoed audit-of-record",
            "Deterministic Elo audit sort (modifiers labeled, weights visible)",
        ],
        "not_invoked_on_this_case": [
            "MONAI RetinaNet (radiology)", "MedSigLIP-448 (pathology)",
            "TxGemma-9B chat (therapy reasoning)",
            "ClinicalBERT report parser", "CBIS-DDSM logreg (mammography)",
        ],
    },
})
print(f"[done] {len(transcript)} calls saved to {HERE}")
