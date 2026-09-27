"""Consolidate real, live-executed evidence for the 7 Modal endpoints named in
Section 0.C of the Alpha audit follow-up (medsiglip-448, clinicalbert,
case-storage, phikon-pathology, gemma-fallback, luna16-infer, medgemma-27b)
into one machine-checkable JSON receipt + a human-readable markdown table.

This script does NOT re-run the live network calls itself (that would make
the receipt's own generation nondeterministic and slow). It consolidates
facts already captured, this session, from:
  - Two real pytest -v -s runs against the actual Modal endpoints:
      artifacts/audit/live_endpoint_evidence/run_20260927T233108Z_v2.log
      (34 selected / 27 passed / 7 failed / 1 deselected [slow medgemma chat])
  - One background slow-marked run of the medgemma-27b chat completion test
    (artifacts/audit/live_endpoint_evidence/medgemma_chat_slow_*.log)
  - Independently-computed sha256sum of every local fixture file used
  - One direct, independent manifest() call against case-storage for the
    durable LUNA16 test case (892531e5e4dde1e8) to recover its full sha256
  - One direct, independent raw HTTP probe of the luna16-detect endpoint
    with a nonexistent case_id (confirms HTTP 200 + structured error body,
    not a 5xx, for that specific edge case)
  - One direct, independent GET of the live Render product /health endpoint
    (https://oncology-arbiter.onrender.com/health) to empirically verify
    which of these 7 capabilities the free-tier product surface actually
    reaches via Modal vs. serves as a local placeholder/proxy -- kept
    strictly separate from the direct-to-Modal results per explicit
    instruction not to conflate a Render free-tier gap with a Modal defect.

Every field below is either a literal value copied from one of those real,
timestamped artifacts, or computed directly (sha256sum). Nothing here is
invented, estimated, or backfilled from memory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "docs" / "proofs" / "seven_endpoint_live_io_test_evidence.json"
OUT_MD = REPO_ROOT / "docs" / "proofs" / "seven_endpoint_live_io_test_evidence.md"

GENERATED_UTC = datetime.now(timezone.utc).isoformat()

PRIMARY_LOG = "artifacts/audit/live_endpoint_evidence/run_20260927T233108Z_v2.log"
ENV_FIX_LOG = "artifacts/audit/live_endpoint_evidence/run_20260927T232614Z.log"  # first attempt, ModuleNotFoundError before `pip install -e .`; kept as provenance of the environment-fix step, not endpoint evidence

RENDER_HEALTH_HTTP = 200
RENDER_HEALTH_ELAPSED_S = 22.2
RENDER_MODELS_LOADED = {
    "monai_screening": "placeholder",
    "medsiglip_biopsy": "placeholder",
    "biopsy_report_parser": "proxy_regex_v0",
    "txgemma_therapy": "proxy_rules_lite",
    "co_scientist": "proxy_co_scientist",
    "l3_arbiter": "template",
    "nsclc_pipeline": "proxy_lung_heuristic",
}

ENDPOINTS = [
    {
        "endpoint": "medsiglip-448",
        "test_files": ["tests/integration/test_medsiglip_modal_client.py"],
        "url_base": "https://crispro--medsiglip",
        "urls_exercised": [
            "https://crispro--medsiglip-info.modal.run (via preflight())",
            "https://crispro--medsiglip-embed.modal.run",
            "https://crispro--medsiglip-embed-batch.modal.run",
        ],
        "fixtures": [
            {"path": "tests/fixtures/cbis_ddsm/Calc-Test_P_00038_LEFT_CC.dcm",
             "sha256": "229363355ffa1a5cb8a0129155389471175ab38a7dc99e5726c9264fc0489c01"},
            {"path": "tests/fixtures/cbis_ddsm/Mass-Test_P_00016_LEFT_CC.dcm",
             "sha256": "e4253ed10249b79249d1e417f35cfb8196b224db529015f439c501b2a4805bab"},
        ],
        "http_status_observed": [200, 200, 200, 200, 200],
        "n_pass": 5, "n_fail": 0, "n_total": 5,
        "key_response_fields": (
            "preflight reason contains 'dim=1152'; embed_dicom returns 1152-d "
            "vector with L2 norm in (1,100); embed_dicoms batch matches "
            "singleton within 1e-3; run() returns model_repo='google/"
            "medsiglip-448', input_resolution=448, model_state=LOADED_MEDSIGLIP, "
            "mammography off-label warning present, gate_report.access_level="
            "ALLOWED."
        ),
        "verdict": "PASS (5/5)",
        "separately_documented_honest_defect": (
            "Finding H (gate_violation_bf5e54f8.json, not a pytest assertion in "
            "this file): live app_version='medsiglip-modal-v0.3.0' vs the "
            "deploy script's intended 'v0.4.0-alpha' -- a real deploy-hygiene "
            "drift, root-caused via full git history + a live curl to confirm "
            "the vision-embedding weights/preprocessing have been IDENTICAL "
            "since the very first v0.3.0 deploy (no confound introduced into "
            "any AUROC measurement that used this endpoint), but the intended "
            "v0.4.0-alpha code has never actually been `modal deploy`-ed."
        ),
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "clinicalbert",
        "test_files": ["tests/integration/test_clinicalbert_live.py"],
        "url_base": "https://crispro--clinicalbert",
        "urls_exercised": [
            "https://crispro--clinicalbert-healthz.modal.run",
            "https://crispro--clinicalbert-info.modal.run",
            "https://crispro--clinicalbert-parse.modal.run",
        ],
        "fixtures": [
            {"path": "tests/fixtures/clinicalbert_reports/real_reports_sample.json",
             "sha256": "0eca05f013a4b70e85d7771e95c572d99a819051bba29907dc4bf1e32023d2ea",
             "note": "8 real, de-identified TCGA breast_crc + nsclc pathology reports"},
        ],
        "http_status_observed": [200, 200, 200, 200, 200, 200, 200, 200],
        "n_pass": 4, "n_fail": 4, "n_total": 8,
        "key_response_fields": (
            "live app_version='clinicalbert-modal-v0.5.1' (client/repo expect "
            "'clinicalbert-modal-v0.5.2-sliding-window'); provenance="
            "'REAL-v0.5.1-snorkel-openrouter-llm' (genuinely real, not the "
            "banned SYNTHETIC-v0.3.x fallback); live /info self-reported "
            "test_micro_f1=0.08089260808926081 (BIO-span, 296-report held-out "
            "set); this run's raw-endpoint, bypass-strict-contract, n=8 "
            "entity-TYPE-presence measurement: tp=24 fp=4 fn=21 "
            "precision=0.8571 recall=0.5333 micro_F1=0.6575 (different "
            "methodology/sample than the /info figure -- reported for "
            "direction only, not as a reproduction)."
        ),
        "verdict": "4 PASS / 4 HONEST_FAIL (4/8)",
        "honest_fail_detail": (
            "HONEST_FAIL x4: (1) app_version/sliding-window contract check -- "
            "live is missing window_tokens/overlap_tokens/window_aggregation/"
            "model_sha256/metrics_sha256 entirely (not just a version-string "
            "mismatch); (2-4) ClinicalBertModalClient.parse() on 3 real "
            "reports (idx 0,2,6) all raise ClinicalBertModalError for the "
            "identical root cause (validate_clinicalbert_contract() correctly "
            "refuses the v0.5.1 response shape). Kept as genuine failing "
            "tests, not mocked around."
        ),
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "case-storage",
        "test_files": ["tests/integration/test_case_storage_live.py"],
        "url_base": "https://crispro--case-storage",
        "urls_exercised": [
            "https://crispro--case-storage-upload.modal.run",
            "https://crispro--case-storage-manifest.modal.run (via CaseStorageClient.manifest())",
            "https://crispro--case-storage-get-file.modal.run (via CaseStorageClient.get_file())",
        ],
        "fixtures": [
            {"path": "tests/fixtures/cbis_ddsm/Calc-Test_P_00038_LEFT_CC.dcm",
             "sha256": "229363355ffa1a5cb8a0129155389471175ab38a7dc99e5726c9264fc0489c01"},
            {"path": "N/A (durable prior-session case, not a repo fixture file)",
             "sha256": "892531e5e4dde1e8b5b92240191d8015fff03f4e07519b4419bc447e8d357ed0",
             "note": "case_id 892531e5e4dde1e8 == sha256_full[:16], confirmed by an independent manifest() call this window (not merely re-asserting a stored value)"},
        ],
        "http_status_observed": [200, 200, 200, 200, 200, 200, 200],
        "n_pass": 6, "n_fail": 0, "n_total": 6,
        "key_response_fields": (
            "server-returned case_id/sha256_full for a real upload EXACTLY "
            "match independently-computed hashlib.sha256(raw); reupload of "
            "identical bytes returns the identical case_id (idempotent); "
            "manifest()/get_file() round-trip is byte-for-byte identical "
            "(verified by sha256, not just length); the 892531e5e4dde1e8 case "
            "uploaded in a DIFFERENT prior session still resolves (cross-"
            "session Modal Volume durability); empty-payload upload returns "
            "HTTP 200 + structured {'error':...} body, correctly raised "
            "client-side as SpecialistServiceError('upstream_error', ...) "
            "rather than a silent success. app_version="
            "'case-storage-modal-v0.5.0-alpha' consistent across every call."
        ),
        "verdict": "PASS (6/6) -- the one fully clean endpoint of the 7",
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "phikon-pathology",
        "test_files": ["tests/integration/test_phikon_pathology_live.py"],
        "url_base": None,
        "urls_exercised": ["https://crispro--phikon-embed.modal.run"],
        "fixtures": [
            {"path": "tests/fixtures/pathology/nct_crc_he100k_tum.png",
             "sha256": "25da480fb5449dd10c0a107e821f4eb2f065b88caf5bca1de41112c4580ba700"},
            {"path": "tests/fixtures/pathology/nct_crc_he100k_norm.png",
             "sha256": "2dd51c2d7820c8fda2c1c31f1772f2b4440f857f2ed24136bad599e6c51c6d7e"},
            {"path": "tests/fixtures/pathology/nct_crc_he100k_str.png",
             "sha256": "6260faa2dc3158067d0b9e57ad1f7f9adcae44ef6b9d7ea4e91e623886549c9b"},
            {"path": "tests/fixtures/pathology/nct_crc_he100k_lym.png",
             "sha256": "b2ce7ae570393fcca64038ca6e1af5b9accd4daff151da62081569397b8b658c"},
            {"path": "tests/fixtures/pathology/nct_crc_he100k_back.png",
             "sha256": "371d8a01f75979c8fca38959e850947b21bf408b80c055035dcfb32dc6fa5118"},
        ],
        "http_status_observed": [200, 200, 200, 200, 200, 200, 200],
        "n_pass": 4, "n_fail": 0, "n_total": 4,
        "key_response_fields": (
            "embedding_dim=768 on a real tumor-epithelium tile; identical "
            "input produces byte-identical vector_sha256 across 2 separate "
            "network calls (determinism); pairwise Euclidean distances across "
            "5 genuinely distinct real tissue classes (TUM/NORM/STR/LYM/BACK) "
            "are all >1e-6 (not degenerate/constant) and the largest is "
            ">0.01 (>=0.01 threshold used to rule out near-collapse); the "
            "audit receipt's input_reference embeds the exact sha256 of the "
            "bytes sent."
        ),
        "verdict": "PASS (4/4) -- the second fully clean endpoint of the 7",
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "gemma-fallback",
        "test_files": ["tests/integration/test_gemma_fallback_live.py"],
        "url_base": "https://crispro--gemma-fallback",
        "urls_exercised": [
            "https://crispro--gemma-fallback-healthz.modal.run",
            "https://crispro--gemma-fallback-chat.modal.run",
        ],
        "fixtures": [
            {"path": "N/A (text prompt, no file fixture)",
             "sha256": None,
             "note": "prompt 1: 'Reply with exactly the two words: hello world'; prompt 2: 'What model are you?'"},
        ],
        "http_status_observed": [200, 200, 200],
        "n_pass": 2, "n_fail": 1, "n_total": 3,
        "key_response_fields": (
            "healthz body: status='ok', app='gemma-fallback', "
            "model='Qwen/Qwen2.5-7B-Instruct'; chat-infra completion for a "
            "20-token reply returned in well under 30s with coherent "
            "'hello world' content; model-identity chat reported "
            "model='qwen2.5-7b-instruct' (lowercased in the response)."
        ),
        "verdict": "2 PASS / 1 HONEST_FAIL (2/3)",
        "honest_fail_detail": (
            "HONEST_FAIL: an app deployed and labeled 'gemma-fallback' "
            "serves Qwen/Qwen2.5-7B-Instruct, not any Gemma-family model. "
            "Confirmed live, real network call, no mock. Deliberately kept "
            "unmocked/unsilenced per the test file's own docstring; this is "
            "a genuine model-identity/labeling defect, and the serving infra "
            "itself is otherwise healthy (test_chat_completion_infra_is_"
            "healthy passes)."
        ),
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "luna16-infer",
        "test_files": [
            "tests/integration/test_luna16_infer_live.py",
            "(separately, population-scale: scripts/run_luna16_froc_subset0.py -- see below)",
        ],
        "url_base": None,
        "urls_exercised": [
            "https://crispro--luna16-detect.modal.run",
            "https://crispro--luna16-healthz.modal.run",
            "https://crispro--luna16-info.modal.run",
        ],
        "fixtures": [
            {"path": "N/A (durable case in case-storage, not a repo fixture file)",
             "sha256": "892531e5e4dde1e8b5b92240191d8015fff03f4e07519b4419bc447e8d357ed0",
             "note": "case_id 892531e5e4dde1e8, real 121-slice chest CT DICOM series"},
        ],
        "http_status_observed": [200, 200, 0, 200, 200, 200],
        "http_status_note": (
            "The malformed-case-id test makes ZERO network calls -- "
            "Luna16Client's own regex guard rejects it client-side before "
            "any request is sent (http_status=0/not-applicable by design, "
            "not a transport failure). The nonexistent-but-valid-format "
            "case_id path was independently re-verified this window with a "
            "raw, client-bypassing POST: returns HTTP 200 with a structured "
            "{'error': \"case_id read failed: FileNotFoundError...\"} body, "
            "which SpecialistServiceError then correctly raises as "
            "'upstream_error' -- i.e. a real HTTP 200, not a 5xx, for that "
            "specific edge case."
        ),
        "n_pass": 5, "n_fail": 1, "n_total": 6,
        "key_response_fields": (
            "healthz: app='luna16-infer', bundle_version='0.6.9'; info: "
            "model_state='loaded_luna16_retinanet', "
            "hu_range=[-1024.0,300.0], target_spacing_mm=[1.25,0.703125,"
            "0.703125]; real detect on the 121-slice case: "
            "n_detections_total=4, n_returned=4, top_score=0.7112138867378235, "
            "inference_seconds=2.9023, total_seconds=8.2557, device='cuda:0', "
            "input_shape=[121,512,512]; spacing probe (independently measured "
            "native DICOM header values vs the endpoint's own declared "
            "target): native dz=2.5mm, dy=dx=0.7617mm vs declared target "
            "dz=1.25mm, dy=dx=0.703125mm -- native z-spacing is exactly "
            "2.00x the declared target, in-plane 1.083x off."
        ),
        "verdict": "5 PASS / 1 HONEST_FAIL (5/6) on this 121-slice case",
        "honest_fail_detail": (
            "HONEST_FAIL: detect() has no Spacingd/Orientationd resample step "
            "for the case_id request path -- the endpoint's own /info "
            "declares a target spacing it does not actually resample "
            "incoming volumes to. Reproduced live with fresh numbers this "
            "run (native dz=2.5mm vs target 1.25mm, ratio 2.00x -- matches "
            "the historical finding exactly)."
        ),
        "important_scope_note": (
            "This file's 'real detect' test uses ONE modest 121-slice case "
            "and PASSES (no crash, plausible 4 detections). This is a "
            "DIFFERENT, much larger-scale finding from Finding G "
            "(gate_violation_bf5e54f8.json): a population-scale run of all "
            "89 series in LUNA16 subset0 via scripts/run_luna16_froc_subset0.py "
            "found a 38.2% overall server-side HTTP-500 failure rate (34/89 "
            "series), rising to 90.0% for series with slice count D>300 "
            "(the resample defect above is a plausible contributing "
            "mechanism, not yet proven as the sole cause of the D>300 "
            "crashes). The two findings are complementary, not "
            "contradictory: a small, normal-sized case succeeds; a "
            "population sweep including large-D series fails at scale. Both "
            "must be reported; neither supersedes the other."
        ),
        "provenance_log": PRIMARY_LOG,
    },
    {
        "endpoint": "medgemma-27b",
        "test_files": ["tests/integration/test_medgemma_27b_live.py"],
        "url_base": "https://crispro--medgemma-27b",
        "urls_exercised": [
            "https://crispro--medgemma-27b-healthz.modal.run",
            "https://crispro--medgemma-27b-chat.modal.run",
        ],
        "fixtures": [
            {"path": "N/A (text prompt, no file fixture)",
             "sha256": None,
             "note": "prompt: 'In one short sentence, what is non-small cell lung cancer?', max_tokens=64, temperature=0.0"},
        ],
        "http_status_observed": [200, 200, "PENDING_BACKGROUND_RUN"],
        "n_pass": "1 (+1 pending)", "n_fail": 1, "n_total": 3,
        "key_response_fields": (
            "healthz app_version (live, observed) != EXPECTED_APP_VERSION "
            "(both repo source and production GemmaClient agree with each "
            "other on 'medgemma-27b-modal-v0.5.0-production-ready'; live "
            "reports 'medgemma-27b-modal-v0.4.0-alpha'). Chat-completion "
            "real call result: see live_endpoint_evidence/medgemma_chat_slow_*.log "
            "(background run, historical reference only: 602.997s -> HTTP "
            "500, re-measured not re-asserted)."
        ),
        "verdict": "1 PASS / 1 HONEST_FAIL fast; slow chat test PENDING",
        "honest_fail_detail": (
            "HONEST_FAIL: deploy drift, reproduced live this run -- the "
            "audited/reviewed repo code (APP_VERSION) and the production "
            "client's own expected-version check already agree with each "
            "other on v0.5.0-production-ready; the live deployment has "
            "simply never been updated to match and still serves the older "
            "v0.4.0-alpha. This is a deploy-pipeline gap, not a code-"
            "correctness gap."
        ),
        "provenance_log": PRIMARY_LOG,
        "slow_chat_test_status": "PENDING_BACKGROUND_RUN -- see medgemma_chat_slow_*.log for the outcome once the background job completes.",
    },
]

RENDER_VS_MODAL_NOTE = {
    "purpose": (
        "Explicit, separate accounting of the free-tier Render product "
        "surface (the actual public API a real caller reaches) vs. the "
        "direct-to-Modal endpoints exercised by the 7 live pytest files "
        "above. These must not be conflated: a 'Modal FAIL' means the "
        "underlying model backend itself misbehaves; a 'Render FAIL' here "
        "means the deployed product does not even attempt to reach that "
        "backend, by an explicit, disclosed, resource-budget design choice "
        "(512 MB / 0.1 CPU free-tier dyno)."
    ),
    "render_url": "https://oncology-arbiter.onrender.com",
    "live_probe_http_status": RENDER_HEALTH_HTTP,
    "live_probe_elapsed_seconds": RENDER_HEALTH_ELAPSED_S,
    "live_probe_models_loaded": RENDER_MODELS_LOADED,
    "interpretation": {
        "medsiglip-448": "Render reports medsiglip_biopsy='placeholder' -- NOT wired to Modal at all on the live product surface (render.yaml: 'would OOM the 512 MB dyno... exercised locally... via generated capability proofs'). This is a RENDER_FREE_TIER_FAIL (never attempts the call), distinct from the direct-Modal PASS result above.",
        "luna16-infer": "Render reports monai_screening='placeholder' for the analogous reason -- same RENDER_FREE_TIER_FAIL class, distinct from the direct-Modal 5/6 PASS result above.",
        "clinicalbert": (
            "render.yaml declares CLINICALBERT_BACKEND=modal and "
            "CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert (note: "
            "the -test- variant, NOT the https://crispro--clinicalbert "
            "production endpoint exercised directly above). However, the "
            "LIVE /health probe's models_loaded.biopsy_report_parser reports "
            "'proxy_regex_v0', not a ClinicalBERT/Modal label. This is an "
            "UNRESOLVED DISCREPANCY, disclosed honestly rather than papered "
            "over: either 'biopsy_report_parser' names a genuinely different, "
            "regex-based capability from the ClinicalBERT NLP entity parser "
            "tested above (plausible, given a separate report_parser_regex "
            "lineage exists elsewhere in this repo's docs), or the deployed "
            "Render revision has not picked up the CLINICALBERT_BACKEND=modal "
            "config. Not adjudicated further this session -- flagged for "
            "follow-up, not silently resolved either way."
        ),
        "case-storage, phikon-pathology, gemma-fallback, medgemma-27b": (
            "None of these four appear in render.yaml's envVars or the live "
            "/health models_loaded dict at all -- the free-tier product does "
            "not expose case upload/pathology-embedding/LLM-fallback "
            "capabilities as user-facing endpoints in the first place. "
            "N/A rather than FAIL: there is no product surface to test."
        ),
    },
}


def main() -> None:
    receipt = {
        "receipt_type": "seven_endpoint_live_io_test_evidence",
        "generated_utc": GENERATED_UTC,
        "repo": "fjkiani/scientist-monai-moat",
        "branch": "audit/manski-gate-and-test-migration",
        "base_commit": "bf5e54f8d8801286ba4b0ad7ca03ce0c7fc75b10",
        "scope": (
            "Section 0.C of the Alpha audit follow-up: real I/O test "
            "evidence for all 7 named Modal endpoints, executed live this "
            "session (not from memory, not mocked)."
        ),
        "environment_fix_disclosed": (
            "The first live-run attempt this window failed 31/34 tests "
            "with ModuleNotFoundError: No module named 'oncology_arbiter' -- "
            "the package was not installed (editable) in this sandbox's "
            "pytest venv (/workspace/.venv). Fixed via "
            "`uv pip install -e .` (also pulled in pydicom as a transitive "
            "dependency, resolving a second, separate ModuleNotFoundError "
            "for the luna16 resample test). This was a test-harness/"
            "environment gap, not a genuine endpoint defect -- disclosed "
            "here rather than silently discarding the first failed run."
        ),
        "totals": {
            "n_endpoints": 7,
            "n_tests_collected": 35,
            "n_tests_deselected_slow": 1,
            "n_tests_selected": 34,
            "n_pass": 27,
            "n_fail_honest": 7,
            "n_pending_slow_background": 1,
        },
        "endpoints": ENDPOINTS,
        "render_vs_modal_free_tier_accounting": RENDER_VS_MODAL_NOTE,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")

    # Markdown evidence table
    lines = [
        "# Seven-endpoint live I/O test evidence",
        "",
        f"Generated {GENERATED_UTC}. Base commit `bf5e54f8d8801286ba4b0ad7ca03ce0c7fc75b10`, "
        "branch `audit/manski-gate-and-test-migration`. All results below are from a real, "
        "live pytest execution against the actual Modal endpoints this session "
        f"(primary log: `{PRIMARY_LOG}`), not from memory or mocks.",
        "",
        "| Endpoint | URL(s) | Fixture path | Input SHA256 | HTTP | Key response fields | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in ENDPOINTS:
        urls = "<br>".join(e["urls_exercised"])
        fixtures = e["fixtures"]
        fpaths = "<br>".join(f["path"] for f in fixtures)
        fshas = "<br>".join((f["sha256"] or "N/A")[:16] + "..." if f["sha256"] else "N/A" for f in fixtures)
        http_vals = e.get("http_status_observed", [])
        http_str = ",".join(str(h) for h in http_vals) if isinstance(http_vals, list) else str(http_vals)
        key_fields = e["key_response_fields"].replace("\n", " ")
        result = e["verdict"]
        lines.append(
            f"| {e['endpoint']} | {urls} | {fpaths} | {fshas} | {http_str} | {key_fields} | {result} |"
        )
    lines += [
        "",
        "## Render free-tier product surface vs. direct-Modal endpoints (kept separate, not conflated)",
        "",
        RENDER_VS_MODAL_NOTE["purpose"],
        "",
        f"Live probe: `GET {RENDER_VS_MODAL_NOTE['render_url']}/health` -> "
        f"HTTP {RENDER_VS_MODAL_NOTE['live_probe_http_status']} in "
        f"{RENDER_VS_MODAL_NOTE['live_probe_elapsed_seconds']}s. "
        f"`models_loaded` = `{json.dumps(RENDER_VS_MODAL_NOTE['live_probe_models_loaded'])}`",
        "",
    ]
    for k, v in RENDER_VS_MODAL_NOTE["interpretation"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")
    print(f"Wrote {OUT_MD} ({OUT_MD.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
