#!/usr/bin/env python3
"""Maintain docs/PROGRESS_LEDGER.json — the honest LIVE / NOT_WIRED progress record.

This script is the ONLY sanctioned way to modify the ledger. It regenerates
the JSON in full from an in-code source of truth (this file), so there is no
schema drift and no partial-edit divergence between the docs copy and the
`/mnt/results/` mirror.

Rules baked into the design (from PLAN.md §2.3, §2.5, and honesty_notes):

  * LIVE = real code produces real output on real inputs, backed by an
    evidence path (file that exists, request_id from audit-*.jsonl, or a
    collectible test file).
  * NOT_WIRED = anything else (stub, placeholder, silent fallback, scaffolded
    but not connected). Non-empty `not_wired_reason` REQUIRED.
  * Hand-drafted math with caveat in a model card counts as LIVE.

Usage:
  python scripts/update_progress_ledger.py            # regenerate both files
  python scripts/update_progress_ledger.py --dry-run  # print diff, no writes

The script exits 0 on success and non-zero if any LIVE claim lacks a working
evidence path (missing files/tests) — that's the acceptance-gate mentioned
in PLAN.md §5 rule 3.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "docs" / "PROGRESS_LEDGER.json"
MIRROR_PATH = Path("/mnt/results/progress_ledger.json")


# --------------------------------------------------------------------------- #
# Data-driven ledger content
#
# When a subsystem, endpoint, or sprint changes state, edit the tables below
# and re-run this script. The JSON is derived deterministically; do NOT hand-
# edit docs/PROGRESS_LEDGER.json.
# --------------------------------------------------------------------------- #


LEDGER_VERSION = "1.0.0"


PROJECT_META = {
    "name": "oncology-arbiter",
    "one_line_goal": (
        "Open-architecture breast oncology reasoning platform spanning "
        "screening → biopsy → therapy, with calibrated per-stage arbiters, "
        "cited evidence, and honest performance disclosure."
    ),
    "regulatory_posture": (
        "Research Use Only. Investigational / IRB path. Not FDA-cleared. "
        "Not CE-marked. Not intended for clinical use."
    ),
    "architecture_layers": [
        "L1 data",
        "L2 evidence",
        "L3 arbiter",
        "L4a screening",
        "L4b biopsy",
        "L4c therapy",
        "L5 orchestrator",
    ],
}


USER_STORIES = [
    {
        "id": "US-01",
        "as_a": "IRB-validated radiology researcher",
        "i_want": "post a CBIS-DDSM DICOM to /v1/screening/analyze and receive a screening arbiter output with a calibrated risk bucket",
        "so_that": "I can prospectively validate a screening triage model against reader ground truth",
        "endpoints": ["/v1/screening/analyze"],
        "status": "LIVE",
        "not_wired_reason": None,
        "evidence": [
            "src/oncology_arbiter/api/app.py",
            "tests/unit/test_screening_medsiglip_wiring.py",
            "/mnt/results/screening_response_medsiglip_smoke_final.json",
        ],
    },
    {
        "id": "US-02",
        "as_a": "researcher",
        "i_want": "post a biopsy report to /v1/biopsy/analyze and receive a biopsy arbiter output",
        "so_that": "I can classify biopsy findings and choose downstream molecular tests",
        "endpoints": ["/v1/biopsy/analyze"],
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "Endpoint returns HTTP 200 with placeholder findings. The L4b "
            "biopsy model (MedSigLIP-448 for histopathology or a dedicated "
            "biopsy classifier) is not connected. Only the L3 biopsy arbiter "
            "template runs (on empty features), which is a separate LIVE "
            "subsystem — see L3-biopsy-arbiter."
        ),
        "evidence": [
            "src/oncology_arbiter/api/app.py",
        ],
    },
    {
        "id": "US-03",
        "as_a": "researcher",
        "i_want": "post biopsy output + patient context to /v1/therapy/reason and receive a therapy arbiter recommendation with cited literature",
        "so_that": "I can evaluate a research-only literature-grounded therapy proposal",
        "endpoints": ["/v1/therapy/reason"],
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "Endpoint returns HTTP 200 with placeholder findings. The L4c "
            "therapy reasoner (TxGemma-9B) is not connected. The L2 evidence "
            "retrieval layer (PubMed / arXiv / Europe PMC) is also not "
            "wired into this endpoint. Only the L3 therapy arbiter template "
            "runs — see L3-therapy-arbiter."
        ),
        "evidence": [
            "src/oncology_arbiter/api/app.py",
        ],
    },
    {
        "id": "US-04",
        "as_a": "researcher",
        "i_want": "post one DICOM + one biopsy report to /v1/case/full and receive one integrated envelope covering screening, biopsy, therapy",
        "so_that": "I can review a single case end-to-end without three separate calls",
        "endpoints": ["/v1/case/full"],
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "Only the screening leg is live. Biopsy and therapy legs "
            "return placeholders because US-02 and US-03 are NOT_WIRED."
        ),
        "evidence": ["src/oncology_arbiter/api/app.py"],
    },
    {
        "id": "US-05",
        "as_a": "IRB reviewer",
        "i_want": "every API response to carry disclaimer, caveat, provenance, honesty_gate, evidence, warnings",
        "so_that": "no consumer can accidentally treat a research output as clinical",
        "endpoints": ["all"],
        "status": "LIVE",
        "not_wired_reason": None,
        "evidence": [
            "src/oncology_arbiter/api/schemas.py::ApiEnvelope",
            "tests/unit/test_honesty.py",
        ],
    },
]


# Subsystems track LIVE vs NOT_WIRED at the code-path level, one row per
# component that either produces or is supposed to produce an output.
SUBSYSTEMS = [
    # --- L1: data ---
    {
        "id": "L1-dicom-ingest",
        "layer": "L1",
        "role": "Read a CBIS-DDSM DICOM (or arbitrary mammography DICOM), extract pixel array + laterality/view metadata.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "pydicom",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/data/cbis_ddsm.py",
            "tests/data/test_cbis_ddsm_ingest.py",
            "tests/fixtures/cbis_ddsm/",
        ],
        "evidence": [
            "tests/data/test_cbis_ddsm_ingest.py",
            "tests/data/test_api_real_dicom.py",
        ],
    },
    {
        "id": "L1-preprocessing",
        "layer": "L1",
        "role": "Orientation detection, breast mask, pectoral removal (MLO), float32 [0,1] normalization on real mammograms.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "in-house (numpy/scikit-image)",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/mammography/reader.py",
            "src/oncology_arbiter/mammography/segmentation.py",
            "src/oncology_arbiter/mammography/laterality.py",
            "src/oncology_arbiter/mammography/pipeline.py",
        ],
        "evidence": [
            "tests/unit/test_pectoral_removal_synthetic.py",
        ],
    },
    # --- L2: evidence ---
    {
        "id": "L2-evidence-fetchers",
        "layer": "L2",
        "role": "PubMed / arXiv / Europe PMC / web fetch under an SSRF guard.",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "The individual fetcher modules exist and their live integration "
            "tests can be run against the network, but no endpoint currently "
            "populates ApiEnvelope.evidence[] from them. Every LIVE response "
            "returns evidence=[]."
        ),
        "current_backend": "urllib + custom parsers",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/tools/pubmed_search.py",
            "src/oncology_arbiter/tools/arxiv_search.py",
            "src/oncology_arbiter/tools/europe_pmc_search.py",
            "src/oncology_arbiter/tools/web_fetch.py",
        ],
        "evidence": [
            "tests/integration/test_pubmed_live.py",
            "tests/integration/test_arxiv_live.py",
            "tests/integration/test_europe_pmc_live.py",
            "tests/integration/test_web_fetch_live.py",
        ],
    },
    # --- L3: arbiter ---
    {
        "id": "L3-screening-arbiter",
        "layer": "L3",
        "role": "L2-regularised logistic arbiter mapping BI-RADS-like features → p_positive → risk_bucket for screening triage.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "hand-drafted template coefficients (n_training=0)",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/arbiter/logistic.py",
            "src/oncology_arbiter/arbiter/models/screening_arbiter_template_v0.json",
        ],
        "evidence": [
            "tests/unit/test_arbiter_l2_logistic.py",
            "tests/unit/test_api_arbiter_wiring.py",
        ],
    },
    {
        "id": "L3-biopsy-arbiter",
        "layer": "L3",
        "role": "Logistic arbiter for biopsy stage classification.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "hand-drafted template coefficients (n_training=0)",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/arbiter/models/biopsy_arbiter_template_v0.json",
        ],
        "evidence": [
            "tests/unit/test_api_arbiter_wiring.py::test_biopsy_endpoint_returns_arbiter_score",
        ],
    },
    {
        "id": "L3-therapy-arbiter",
        "layer": "L3",
        "role": "Logistic arbiter for therapy recommendation.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "hand-drafted template coefficients (n_training=0)",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/arbiter/models/therapy_arbiter_template_v0.json",
        ],
        "evidence": [
            "tests/unit/test_api_arbiter_wiring.py::test_therapy_endpoint_returns_arbiter_score",
        ],
    },
    # --- L4a: screening model ---
    {
        "id": "L4a-screening-medsiglip",
        "layer": "L4a",
        "role": "Zero-shot vision-language backbone that scores a mammogram against `malignant mass` vs `no mass` labels. HAI-DEF gated.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "google/medsiglip-448",
        "hai_def_gate_state": "allowed",
        "wired_files": [
            "src/oncology_arbiter/models/medsiglip.py",
            "src/oncology_arbiter/models/hai_def.py",
            "src/oncology_arbiter/api/app.py",
        ],
        "evidence": [
            "tests/unit/test_medsiglip_wiring.py",
            "tests/unit/test_screening_medsiglip_wiring.py",
            "/mnt/results/screening_response_medsiglip_smoke_final.json",
        ],
        "live_smoke": {
            "input_file": "tests/fixtures/cbis_ddsm/Calc-Test_P_00038_LEFT_CC.dcm",
            "overall_score": 7.976839697221294e-06,
            "malignant_prob": 7.976839697221294e-06,
            "without_prob": 1.3905498235544655e-05,
            "checked_at": "2026-07-03T17:04:00+00:00",
            "note": "Bit-exact match to pre-hibernation reference (verified via rg on execution_trace/transcript.jsonl).",
        },
    },
    {
        "id": "L4a-screening-siglip-proxy",
        "layer": "L4a",
        "role": "Apache-2.0 ungated general-domain SigLIP used ONLY as a development proxy when HAI-DEF is denied. NEVER labeled as MedSigLIP.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "google/siglip-base-patch16-224",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/models/siglip_baseline.py",
            "src/oncology_arbiter/api/app.py",
        ],
        "evidence": [
            "tests/models/test_siglip_baseline.py",
            "tests/unit/test_screening_medsiglip_wiring.py::test_medsiglip_disabled_proxy_still_works",
            "/mnt/results/screening_response_medsiglip_gated_with_proxy_fallback.json",
        ],
    },
    {
        "id": "L4a-monai-efficientdet-detector",
        "layer": "L4a",
        "role": "MONAI EfficientDet lesion detector — README architecture item.",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "Only referenced in README architecture. No detector code path "
            "exists in the repo; no MONAI import, no EfficientDet weights, "
            "no detection call in the endpoint. Placeholder for a future sprint."
        ),
        "current_backend": "none",
        "hai_def_gate_state": None,
        "wired_files": [],
        "evidence": [],
    },
    # --- L4b: biopsy model ---
    {
        "id": "L4b-biopsy-model",
        "layer": "L4b",
        "role": "Biopsy histopathology classifier (MedSigLIP-448 embed + linear probe, or dedicated model).",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "/v1/biopsy/analyze returns placeholder output. No inference "
            "path from biopsy input → model → prediction. Endpoint only "
            "exercises the L3 biopsy arbiter template on empty features."
        ),
        "current_backend": "none",
        "hai_def_gate_state": None,
        "wired_files": [],
        "evidence": [],
    },
    # --- L4c: therapy model ---
    {
        "id": "L4c-therapy-model",
        "layer": "L4c",
        "role": "TxGemma-9B therapy reasoner over biopsy output + patient context, with cited literature.",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "/v1/therapy/reason returns placeholder. TxGemma is not "
            "downloaded, not loaded, not called. No L2 evidence "
            "retrieval feeds the endpoint."
        ),
        "current_backend": "none",
        "hai_def_gate_state": "unknown",
        "wired_files": [],
        "evidence": [],
    },
    # --- L5: orchestrator ---
    {
        "id": "L5-co-scientist-orchestrator",
        "layer": "L5",
        "role": "Multi-agent orchestrator (open-source re-implementation of Gottweis et al., Nature 2026).",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "Agent scaffolding exists under src/oncology_arbiter/agents/ "
            "(supervisor.py etc.) but no orchestration loop calls them "
            "from any endpoint. No agent-to-agent turn taking is running."
        ),
        "current_backend": "none",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/agents/supervisor.py",
        ],
        "evidence": [],
    },
    # --- HAI-DEF gate ---
    {
        "id": "hai-def-gate",
        "layer": "cross-cutting",
        "role": "Preflight HTTP HEAD probe against /resolve/main/config.json on HAI-DEF-gated HuggingFace repos. Returns ALLOWED / FORBIDDEN / UNAUTHENTICATED / UNKNOWN.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "requests + huggingface.co",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/models/hai_def.py",
        ],
        "evidence": [
            "tests/models/test_hai_def.py",
        ],
        "notes": (
            "Regression-guarded (2026-07-03): probes /resolve/main/config.json "
            "with allow_redirects=False; treats 30x as ALLOWED (token accepted "
            "and redirected to CDN). Old /api/models/{repo} path returned 200 "
            "for gated repos even without a token — silent proxy fallback bug, "
            "now fixed."
        ),
    },
    # --- API-level ---
    {
        "id": "endpoint-health",
        "layer": "API",
        "role": "Liveness probe. Returns {status: ok, endpoints: [...]}.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "in-process",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": [
            "tests/unit/test_api_model_cards_and_artifacts.py",
        ],
    },
    {
        "id": "endpoint-screening-analyze",
        "layer": "API",
        "role": "POST /v1/screening/analyze — DICOM in, screening arbiter + MedSigLIP score out.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "MedSigLIP-448 (default) or SigLIP proxy (opt-in fallback)",
        "hai_def_gate_state": "allowed",
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": [
            "tests/unit/test_screening_medsiglip_wiring.py",
            "tests/data/test_api_real_dicom.py",
        ],
    },
    {
        "id": "endpoint-biopsy-analyze",
        "layer": "API",
        "role": "POST /v1/biopsy/analyze — biopsy report in, biopsy arbiter out (placeholder body).",
        "status": "NOT_WIRED",
        "not_wired_reason": "Returns HTTP 200 with placeholder findings; L4b model NOT_WIRED (see subsystem L4b-biopsy-model).",
        "current_backend": "none",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": [],
    },
    {
        "id": "endpoint-therapy-reason",
        "layer": "API",
        "role": "POST /v1/therapy/reason — biopsy output + patient context in, therapy arbiter out (placeholder body).",
        "status": "NOT_WIRED",
        "not_wired_reason": "Returns HTTP 200 with placeholder findings; L4c model NOT_WIRED (see subsystem L4c-therapy-model).",
        "current_backend": "none",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": [],
    },
    {
        "id": "endpoint-case-full",
        "layer": "API",
        "role": "POST /v1/case/full — one DICOM + biopsy report → integrated screening/biopsy/therapy envelope.",
        "status": "NOT_WIRED",
        "not_wired_reason": "Screening leg is LIVE; biopsy and therapy legs return placeholders because their subsystems are NOT_WIRED.",
        "current_backend": "MedSigLIP for screening only",
        "hai_def_gate_state": "allowed",
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": [],
    },
    {
        "id": "endpoint-model-cards",
        "layer": "API",
        "role": "GET /v1/model-cards — enumerate docs/model_cards/*.md as JSON.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "file read of docs/model_cards/",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": ["tests/unit/test_api_model_cards_and_artifacts.py"],
    },
    {
        "id": "endpoint-artifacts",
        "layer": "API",
        "role": "GET /v1/artifacts/{category}/{filename} — path-traversal-safe file streaming.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "in-process file streaming",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/app.py"],
        "evidence": ["tests/unit/test_api_model_cards_and_artifacts.py"],
    },
    {
        "id": "audit-log",
        "layer": "cross-cutting",
        "role": "One JSONL event per request in artifacts/audit/audit-*.jsonl.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "in-house JSONL writer",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/audit.py"],
        "evidence": ["tests/unit/test_irb_artifacts.py"],
    },
    {
        "id": "response-warnings-field",
        "layer": "API",
        "role": "ApiEnvelope.warnings[] — non-fatal honesty warnings on every response.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "src/oncology_arbiter/api/schemas.py",
        "hai_def_gate_state": None,
        "wired_files": ["src/oncology_arbiter/api/schemas.py", "src/oncology_arbiter/api/app.py"],
        "evidence": ["tests/unit/test_screening_medsiglip_wiring.py"],
    },
    {
        "id": "response-gate-report-in-provenance",
        "layer": "API",
        "role": "Expose the GateReport (repo, level, status, reason) in ScreeningResponse.provenance so callers see the gate outcome inline.",
        "status": "NOT_WIRED",
        "not_wired_reason": (
            "MedSigLIP fills gate_report internally on MedSigLipResult and "
            "the app logs it, but the API schema Provenance object does not "
            "yet expose it as a field. The gate outcome is carried as a "
            "`medsiglip_gated:...` warning string but not as structured data."
        ),
        "current_backend": "in-house",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/models/medsiglip.py",
            "src/oncology_arbiter/api/app.py",
        ],
        "evidence": [],
    },
    {
        "id": "docker-image",
        "layer": "infra",
        "role": "Container image for the FastAPI app.",
        "status": "NOT_WIRED",
        "not_wired_reason": "Dockerfile not yet written. Wave 3 step 2.3 lands a multi-stage Dockerfile + .dockerignore; first real docker build happens in CI on push (no docker daemon in this sandbox).",
        "current_backend": None,
        "hai_def_gate_state": None,
        "wired_files": [],
        "evidence": [],
    },
]


# Sprints are cited to real commit SHAs; git log --format='%H|%aI|%s' produced these.
SPRINTS = [
    {
        "id": "sprint-01",
        "name": "Phase 1 baseline",
        "closed_at": "2026-07-02T00:22:27+00:00",
        "commits": ["bf72e51"],
        "goal": "Ship an honest Phase 1 skeleton: real DICOM preprocessing, placeholder API endpoints, Co-Scientist tools scaffold.",
        "delivered": [
            "src/oncology_arbiter/data/preprocess.py — orientation, mask, pectoral removal",
            "FastAPI app with /health and /v1/screening/analyze placeholder returning HTTP 200",
            "Co-Scientist agent scaffold under src/oncology_arbiter/agents/",
        ],
        "not_delivered": [
            "Real screening model: intentionally deferred to sprint-07/08",
            "L2 evidence retrieval endpoints",
        ],
        "artifacts": ["Dockerfile", "README.md"],
        "notes": "Established RUO disclaimer contract on every response envelope.",
    },
    {
        "id": "sprint-02",
        "name": "IRB-readiness artifacts",
        "closed_at": "2026-07-02T00:36:42+00:00",
        "commits": ["18489dc"],
        "goal": "Land the IRB-readiness paper trail so an academic research partner can drop this in.",
        "delivered": [
            "IRB protocol template",
            "Prediction ledger schema",
            "Consent template",
            "32 structural tests locking down the artifacts",
        ],
        "not_delivered": [],
        "artifacts": ["docs/irb/", "tests/unit/test_irb_artifacts.py"],
        "notes": "",
    },
    {
        "id": "sprint-03",
        "name": "Model cards + errata",
        "closed_at": "2026-07-02T00:47:18+00:00",
        "commits": ["7bf77ef"],
        "goal": "Publish model cards for every backbone we might load, in advance of any actual load.",
        "delivered": [
            "docs/model_cards/medsiglip_448.md",
            "docs/model_cards/medgemma_1p5_4b_it.md",
            "docs/model_cards/medgemma_27b.md",
            "docs/model_cards/siglip_base_patch16_224.md",
            "31 model-card structural tests",
        ],
        "not_delivered": [],
        "artifacts": ["docs/model_cards/", "tests/unit/test_model_cards.py"],
        "notes": "Cards carry the mammography-out-of-distribution warning that later goes into MedSigLIP warnings[].",
    },
    {
        "id": "sprint-04",
        "name": "CBIS ingest + HAI-DEF gating scaffolding",
        "closed_at": "2026-07-02T01:03:41+00:00",
        "commits": ["3ad3fa7"],
        "goal": "Real DICOM ingest on real CBIS-DDSM cases + HAI-DEF gate helpers.",
        "delivered": [
            "src/oncology_arbiter/data/cbis_ddsm.py — DICOM reader",
            "src/oncology_arbiter/models/hai_def.py — first cut (probe endpoint TBD)",
            "ModelState enum: PLACEHOLDER, LOADED, GATED, PROXY_SIGLIP, UNAVAILABLE",
            "68 tests",
        ],
        "not_delivered": [
            "MedSigLIP client — deferred to sprint-08",
            "HAI-DEF probe endpoint choice was wrong (/api/models/) — bug uncovered sprint-08",
        ],
        "artifacts": ["tests/data/test_cbis_ddsm_ingest.py", "tests/models/test_hai_def.py"],
        "notes": "The probe URL choice made in this sprint carried a silent-fallback bug; fixed in sprint-08.",
    },
    {
        "id": "sprint-05",
        "name": "SigLIP baseline + real DICOM smoke",
        "closed_at": "2026-07-02T01:04:01+00:00",
        "commits": ["c77370a"],
        "goal": "Wire the ungated general-domain SigLIP as a development proxy so we can run end-to-end on real DICOM before HAI-DEF is granted.",
        "delivered": [
            "src/oncology_arbiter/models/siglip_baseline.py — SigLIP proxy client with mammography honesty warning",
            "Real zero-shot smoke on Calc-Test_P_00038_LEFT_CC.dcm",
            "21 tests including a real-network CBIS-DDSM smoke",
        ],
        "not_delivered": [
            "MedSigLIP client — HAI-DEF access not granted yet",
        ],
        "artifacts": ["tests/models/test_siglip_baseline.py"],
        "notes": "Proxy label enforced at the type level (ModelState.PROXY_SIGLIP); tests ensure it is never re-labelled as LOADED_MEDSIGLIP.",
    },
    {
        "id": "sprint-06",
        "name": "L2 arbiter templates + wave3 conductor",
        "closed_at": "2026-07-02T05:15:49+00:00",
        "commits": ["3735a06"],
        "goal": "Ship the three L2 logistic arbiters (screening/biopsy/therapy) and wire them into every endpoint envelope.",
        "delivered": [
            "src/oncology_arbiter/arbiter/logistic.py",
            "arbiter/models/screening_arbiter_template_v0.json",
            "arbiter/models/biopsy_arbiter_template_v0.json",
            "arbiter/models/therapy_arbiter_template_v0.json",
            "Extended /v1/biopsy/analyze and /v1/therapy/reason to return arbiter_score blocks",
            "Extended schemas.py with ArbiterScore and ModelCardSummary",
        ],
        "not_delivered": [],
        "artifacts": ["src/oncology_arbiter/arbiter/", "tests/unit/test_arbiter_l2_logistic.py"],
        "notes": "Hand-drafted coefficients with n_training=0; TEMPLATE caveat on every arbiter output.",
    },
    {
        "id": "sprint-07",
        "name": "Worker fan-out + wave3 recovery",
        "closed_at": "2026-07-03T16:02:59+00:00",
        "commits": [
            "54ebae4",  # worker-1 IRB
            "f941dee",  # worker-2 model cards
            "d780631",  # worker-3 CBIS+HAI-DEF
            "c25948a",  # worker-4 SigLIP smoke
            "10403a0",  # merge origin/main (wave3-conductor)
        ],
        "goal": "Rejoin all four worker branches with the wave3-conductor tip after a hibernation-caused branch fork.",
        "delivered": [
            "Merged worker-1..4 into main sequentially",
            "Merged origin/main (wave3-conductor) via --allow-unrelated-histories",
            "Resolved AA conflicts on api/app.py and api/schemas.py by taking origin (superset)",
            "341 tests passing after merge",
        ],
        "not_delivered": [
            "MedSigLIP client — carried to sprint-08",
            "HAI-DEF probe bug — discovered but not yet fixed",
        ],
        "artifacts": [],
        "notes": "This sprint is a merge-only sprint; no new code. Verified via `git log --graph --all`.",
    },
    {
        "id": "sprint-08",
        "name": "HAI-DEF fix + MedSigLIP end-to-end",
        "closed_at": "2026-07-03T17:05:46+00:00",
        "commits": [
            "c41405d",  # hai_def fix
            "38f4d48",  # medsiglip client
            "075154b",  # endpoint wiring
            "b9083e7",  # doc comment on singleton mutation
        ],
        "goal": "Fix the silent proxy fallback bug in the HAI-DEF gate probe, land the real MedSigLIP-448 client, and wire it into /v1/screening/analyze.",
        "delivered": [
            "hai_def.py: probe /resolve/main/config.json (not /api/models/), treat 30x as ALLOWED, allow_redirects=False",
            "models/medsiglip.py: real MedSigLIP-448 client with HAI-DEF preflight, no silent fallback",
            "tests/unit/test_medsiglip_wiring.py: 13 client tests",
            "tests/unit/test_screening_medsiglip_wiring.py: 9 endpoint tests covering allowed / gated / proxy-fallback precedence",
            "api/schemas.py: ModelState.LOADED_MEDSIGLIP; ApiEnvelope.warnings[] added",
            "api/app.py: precedence MedSigLIP → (opt-in) proxy → placeholder with warnings surfaced",
            "Live smoke on Calc-Test_P_00038_LEFT_CC.dcm: overall_score=7.976839697221294e-06 (bit-exact to pre-hibernation)",
            "Live gated smoke: model_state=gated, no silent fallback",
            "Live proxy-fallback smoke: model_state=proxy_siglip with BOTH warnings",
            "Full regression 370 passed",
        ],
        "not_delivered": [
            "gate_report structured field on ScreeningResponse.provenance (subsystem response-gate-report-in-provenance still NOT_WIRED)",
        ],
        "artifacts": [
            "/mnt/results/screening_response_medsiglip_smoke_final.json",
            "/mnt/results/screening_response_medsiglip_gated_final.json",
            "/mnt/results/screening_response_medsiglip_gated_with_proxy_fallback.json",
        ],
        "notes": "HAI-DEF regression-guarded via test_check_probes_resolve_endpoint_not_api_metadata.",
    },
]


DEFERRED_DECISIONS = [
    {
        "id": "ADR-0001",
        "title": "MedSigLIP score does NOT yet feed the screening arbiter's feature vector",
        "why_deferred": (
            "The L3 screening arbiter has BI-RADS-shaped features "
            "(birads_BI_RADS_1..5); MedSigLIP produces a two-label zero-shot "
            "probability. Discretising the MedSigLIP score into pseudo-BI-RADS "
            "before we have a reader-annotated calibration set would inject "
            "opinion, not signal. We keep overall_score and arbiter_score as "
            "independent honest signals for now."
        ),
        "revisit_when": "Reader-annotated CBIS-DDSM subset arrives; then calibrate MedSigLIP → BI-RADS mapping.",
    },
    {
        "id": "ADR-0002",
        "title": "Preprocess is injected into MedSigLip / SiglipBaseline via singleton mutation",
        "why_deferred": (
            "Phase 2 helper functions mutate `_preprocess_fn` on the model "
            "singleton for the duration of a call. Safe under FastAPI's "
            "default single-threaded async event loop, unsafe under a "
            "threadpool. Documented in code."
        ),
        "revisit_when": "Phase 3 (multi-request throughput). Fix: per-request MedSigLip instance or `preprocess_result` argument on `.run()`.",
    },
    {
        "id": "ADR-0003",
        "title": "GateReport is not exposed on ScreeningResponse.provenance",
        "why_deferred": (
            "GateReport is populated internally on MedSigLipResult and logged "
            "in the audit stream, but only the string warning surfaces on "
            "the response. Adding a structured field is a schema-level "
            "change with downstream client impact; queued behind the "
            "provenance v2 sprint."
        ),
        "revisit_when": "Provenance v2 sprint (client-facing schema update).",
    },
]


EXTERNAL_GATES = [
    {
        "gate": "hai_def",
        "repos": [
            {
                "repo_id": "google/medsiglip-448",
                "state": "allowed",
                "last_checked_at": "2026-07-03T17:04:00+00:00",
                "evidence": "Live smoke ran end-to-end and produced overall_score=7.976839697221294e-06 with model_state=loaded_medsiglip. See /mnt/results/screening_response_medsiglip_smoke_final.json.",
            },
            {
                "repo_id": "google/medgemma-1.5-4b-it",
                "state": "allowed",
                "last_checked_at": "2026-07-03T16:30:00+00:00",
                "evidence": "check_hai_def_access returned AccessLevel.ALLOWED / status=200 with valid HF_TOKEN in sprint-08 verification. Transcript search key: `medgemma-1.5-4b-it`.",
            },
            {
                "repo_id": "google/medgemma-27b-text-it",
                "state": "allowed",
                "last_checked_at": "2026-07-03T16:30:00+00:00",
                "evidence": "check_hai_def_access returned AccessLevel.ALLOWED / status=200 with valid HF_TOKEN in sprint-08 verification.",
            },
        ],
    },
]


HONESTY_NOTES = [
    "Ledger only marks LIVE when real code produces real output from real inputs, with an evidence path that exists.",
    "Hand-drafted math with caveat in a model card counts as LIVE (per user decision 2026-07-02) — the illustrative caveat lives in the model card and in `warnings`, not in the ledger tier.",
    "Stubs, placeholders, silent-failure fallbacks, and scaffolded-but-unconnected features are NOT_WIRED with an explicit `not_wired_reason`. They may still ship.",
    "Live-smoke numbers cited in this ledger (e.g. overall_score=7.976839697221294e-06) are bit-verified against the transcript.jsonl record before landing.",
]


# --------------------------------------------------------------------------- #
# Ledger assembly + evidence validation
# --------------------------------------------------------------------------- #


def _current_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_evidence(entries: list[dict]) -> list[str]:
    """Return a list of validation errors for LIVE entries with missing evidence.

    An evidence path may be:
      * a filesystem path that exists (absolute or relative to REPO_ROOT)
      * a test node id (contains `::`) → validate the file portion exists
    """
    errors: list[str] = []
    for e in entries:
        if e.get("status") != "LIVE":
            continue
        ev = e.get("evidence") or []
        if not ev:
            errors.append(f"{e['id']}: LIVE but evidence[] is empty")
            continue
        for path in ev:
            # Strip `::testfunc` suffix for pytest node ids
            file_part = path.split("::", 1)[0]
            candidate = Path(file_part)
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if not candidate.exists():
                errors.append(f"{e['id']}: evidence path does not exist: {path}")
    return errors


def _validate_not_wired(entries: list[dict]) -> list[str]:
    errors: list[str] = []
    for e in entries:
        if e.get("status") != "NOT_WIRED":
            continue
        if not (e.get("not_wired_reason") or "").strip():
            errors.append(f"{e['id']}: NOT_WIRED but not_wired_reason is empty")
    return errors


def build_ledger() -> dict:
    return {
        "$schema_version": LEDGER_VERSION,
        "generated_at": _now_iso(),
        "git_sha": _current_git_sha(),
        "project": PROJECT_META,
        "user_stories": USER_STORIES,
        "subsystems": SUBSYSTEMS,
        "sprints": SPRINTS,
        "deferred_decisions": DEFERRED_DECISIONS,
        "external_gates": EXTERNAL_GATES,
        "honesty_notes": HONESTY_NOTES,
    }


def _serialise(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def _strip_volatile_fields(text: str) -> str:
    """Drop always-changing fields (generated_at, git_sha) for change detection.

    We want `--dry-run` on an unchanged repo to print `no changes` even though
    the timestamp bumps every second and the git sha could roll forward.
    """
    if not text:
        return ""
    try:
        doc = json.loads(text)
    except Exception:
        return text
    doc.pop("generated_at", None)
    doc.pop("git_sha", None)
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def _diff_summary(old: str, new: str) -> str:
    old_stable = _strip_volatile_fields(old)
    new_stable = _strip_volatile_fields(new)
    if old_stable == new_stable:
        return "no changes"
    old_lines = old_stable.splitlines()
    new_lines = new_stable.splitlines()
    return f"changed: {len(old_lines)} → {len(new_lines)} lines"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate docs/PROGRESS_LEDGER.json + /mnt/results/progress_ledger.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + print diff summary without writing.",
    )
    args = parser.parse_args(argv)

    ledger = build_ledger()

    # Validate before writing
    all_entries = USER_STORIES + SUBSYSTEMS
    errors = _validate_evidence(all_entries) + _validate_not_wired(all_entries)
    if errors:
        print("VALIDATION ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            f"\n{len(errors)} error(s). Ledger NOT written.",
            file=sys.stderr,
        )
        return 2

    new_text = _serialise(ledger)

    old_text = LEDGER_PATH.read_text() if LEDGER_PATH.exists() else ""
    summary = _diff_summary(old_text, new_text)

    if args.dry_run:
        print(f"dry-run: {summary}")
        print(f"    ledger: {LEDGER_PATH}")
        print(f"    mirror: {MIRROR_PATH}")
        return 0

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(new_text)
    try:
        MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        MIRROR_PATH.write_text(new_text)
    except OSError as e:
        print(f"warning: could not write mirror at {MIRROR_PATH}: {e}", file=sys.stderr)

    live_count = sum(
        1
        for e in all_entries
        if e.get("status") == "LIVE"
    )
    not_wired_count = sum(
        1
        for e in all_entries
        if e.get("status") == "NOT_WIRED"
    )
    print(f"wrote {LEDGER_PATH}")
    print(f"wrote {MIRROR_PATH}")
    print(f"summary: {summary}")
    print(
        f"  {live_count} LIVE, {not_wired_count} NOT_WIRED, "
        f"{len(SPRINTS)} sprints, {len(EXTERNAL_GATES[0]['repos'])} HAI-DEF repos"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
