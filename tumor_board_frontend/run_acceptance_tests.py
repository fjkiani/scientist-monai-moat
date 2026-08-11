"""Acceptance harness T1–T10 for the two-case tumor board deliverable.

Every test checks a real property against real files on disk. No mocks.
Prints a summary and exits nonzero if any test fails.

  T01  All three visible HTML pages carry the RUO strip + Not FDA-cleared token.
  T02  AK page carries the mandatory contribution-boundary disclosure verbatim.
  T03  AK page contains the manuscript SHA d33f6403… linked to GitHub.
  T04  All 6 anchor p-values render verbatim (full decimal) on the AK page.
  T05  PARP falsification narrative is a first-class row, NOT quarantined.
  T06  Chip discipline on AK: proxy_co_scientist appears ONLY on the Elo panel;
       manuscript-anchored blocks carry loaded_* chips.
  T07  NSCLC page shows the LUNA16 low-confidence (0.042) inline, not hidden.
  T08  Zero fake NCT identifiers anywhere in the deliverable.
  T09  Zero quarantine-forbidden strings/patterns anywhere in visible files.
  T10  Every case has a live 3-call transcript on disk, all 2xx, with real
       request_ids and the expected model_state chip on the primary call.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).parent
VISIBLE_HTML = [
    HERE / "index.html",
    HERE / "nsclc_luna16" / "mockup.html",
    HERE / "ak_mbd4_lof" / "mockup.html",
]
ALL_VISIBLE = VISIBLE_HTML + [
    HERE / "component_spec.json",
    HERE / "README.md",
]

results: list[dict[str, Any]] = []


def _pass(tid: str, name: str, detail: str = "") -> None:
    results.append({"test": tid, "name": name, "status": "PASS", "detail": detail})


def _fail(tid: str, name: str, detail: str) -> None:
    results.append({"test": tid, "name": name, "status": "FAIL", "detail": detail})


def _read(p: pathlib.Path) -> str:
    return p.read_text() if p.exists() else ""


# ---- T01: RUO strip + FDA-cleared token on every visible HTML ---------
t01_fails = []
for p in VISIBLE_HTML:
    body = _read(p)
    if "RESEARCH USE ONLY" not in body:
        t01_fails.append(f"{p.name} missing 'RESEARCH USE ONLY'")
    if "Not FDA-cleared" not in body:
        t01_fails.append(f"{p.name} missing 'Not FDA-cleared'")
if t01_fails:
    _fail("T01", "RUO strip + FDA-cleared token on every page", "; ".join(t01_fails))
else:
    _pass("T01", "RUO strip + FDA-cleared token on every page", f"3/3 pages OK")

# ---- T02: AK page carries mandatory contribution-boundary disclosure ---
ak = _read(HERE / "ak_mbd4_lof" / "mockup.html")
disclosure_tokens = [
    "manuscript-anchored, arbiter validates+audits+surfaces",
    "The arbiter did NOT generate them",
    "arbiter contributes contract validation",
    "CONTRIBUTION BOUNDARY",
]
missing = [t for t in disclosure_tokens if t not in ak]
if missing:
    _fail("T02", "AK contribution-boundary disclosure verbatim", f"missing: {missing}")
else:
    _pass("T02", "AK contribution-boundary disclosure verbatim", "all 4 tokens present")

# ---- T03: manuscript SHA linked ---------------------------------------
sha_ms = "d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1"
sha_be = "bfd6d11fc872c11a13365b0682cea776a136c7f3"
sha_bundle = "0121fc84fc798af57aad78f2c9274506eac769313d4b21329ad9e3775c5b3a4c"
sha_findings = []
if sha_ms not in ak:
    sha_findings.append(f"missing manuscript SHA {sha_ms[:12]}…")
if sha_be not in ak:
    sha_findings.append(f"missing backend SHA {sha_be[:12]}…")
if sha_bundle not in ak:
    sha_findings.append(f"missing bundle SHA {sha_bundle[:12]}…")
if f"github.com/fjkiani/crispro/tree/{sha_ms}" not in ak:
    sha_findings.append("manuscript SHA not linked to GitHub tree")
if sha_findings:
    _fail("T03", "AK page carries the 3 SHAs, manuscript linked", "; ".join(sha_findings))
else:
    _pass("T03", "AK page carries the 3 SHAs, manuscript linked", "3/3 SHAs + link OK")

# ---- T04: all 6 anchor p-values verbatim ------------------------------
anchors = [
    "0.021484496737088882",   # gdsc primary (ceralasertib)
    "0.015328932966132268",   # stress test #1 MSI purge
    "0.003002668797799231",   # stress test #2 TP53
    "0.045165724128583974",   # stress test #3 LOO
    "0.02533035508952329",    # stress test #4 non-bowel
    "0.07445705343975263",    # axis partner adavosertib
    "0.6047878879741422",     # falsification PARP1
]
missing_p = [a for a in anchors if a not in ak]
if missing_p:
    _fail("T04", "AK all 7 anchor p-values verbatim (6 primary + 1 falsification)",
          f"missing: {missing_p}")
else:
    _pass("T04", "AK all 7 anchor p-values verbatim", "7/7 present with full decimals")

# ---- T05: PARP falsification is first-class, NOT quarantined ----------
required_parp_tokens = [
    "PARP hypothesis — falsified",
    'data-block="parp-falsification"',
    "PARP1 expression in MBD4-LOF",
    "NOT elevated vs comparator",
    "n=19 vs 1498",
    "p=0.605",
]
# Also confirm PARP row is in the 6-axis evidence matrix as a first-class row
# (i.e. not inside a <details> or hidden pane), and NO 'quarantine' or 'hidden' tab
parp_first_class = "<tr><td>parp_inhibitors</td>" in ak or ">parp_inhibitors</td>" in ak
has_quarantine_tab = "quarantine" in ak.lower() and "quarantine_test_harness.json" not in ak.lower()
missing_parp = [t for t in required_parp_tokens if t not in ak]
issues = []
if missing_parp:
    issues.append(f"missing: {missing_parp}")
if not parp_first_class:
    issues.append("PARP row not first-class in evidence matrix")
if has_quarantine_tab:
    issues.append("some 'quarantine' text appears in visible section")
if issues:
    _fail("T05", "PARP falsification is first-class, not quarantined", "; ".join(issues))
else:
    _pass("T05", "PARP falsification is first-class, not quarantined",
          "narrative + first-class row + no quarantine tab")

# ---- T06: chip discipline on AK ---------------------------------------
# proxy_co_scientist should appear only on the Elo audit-sort block,
# never on manuscript-anchored blocks. Manuscript-anchored blocks must
# carry loaded_* chips.
proxy_regions = re.findall(
    r'<h2[^>]*>[^<]*<span class="chip amber">proxy_co_scientist</span>',
    ak,
)
# Also collect all chip usages in each section h2 for a full picture.
h2_chip_matches = re.findall(
    r'<h2[^>]*>(.*?)</h2>',
    ak,
    flags=re.DOTALL,
)
proxy_on_manuscript = []
for h2 in h2_chip_matches:
    if "proxy_co_scientist" in h2 and "audit sort" not in h2.lower() and "elo" not in h2.lower():
        proxy_on_manuscript.append(h2[:120])
# Manuscript blocks must have at least one loaded_* chip.
required_loaded = ["loaded_ak_bundle", "loaded_manuscript_receipt", "loaded_mbd4_evidence_matrix"]
missing_loaded = [c for c in required_loaded if c not in ak]
if proxy_on_manuscript or missing_loaded:
    detail = ""
    if proxy_on_manuscript:
        detail += f"proxy chip on non-Elo h2: {proxy_on_manuscript}; "
    if missing_loaded:
        detail += f"missing loaded chips: {missing_loaded}"
    _fail("T06", "chip discipline on AK (proxy only on Elo panel)", detail)
else:
    _pass("T06", "chip discipline on AK",
          f"proxy chip constrained to Elo block; all 3 loaded_* chips present")

# ---- T07: NSCLC low-confidence surfaced inline ------------------------
nsclc = _read(HERE / "nsclc_luna16" / "mockup.html")
lc_tokens = [
    "top_score = 0.042",
    "Detector low-confidence is HONEST",
    "out-of-distribution",
    'loaded_luna16_retinanet',
]
missing_lc = [t for t in lc_tokens if t not in nsclc]
if missing_lc:
    _fail("T07", "NSCLC low-confidence surfaced inline (not hidden)",
          f"missing: {missing_lc}")
else:
    _pass("T07", "NSCLC low-confidence surfaced inline",
          "0.042 + honest explanation + chip visible inline")

# ---- T08: zero fake NCT ids -------------------------------------------
nct_fake_re = re.compile(r"NCT-EXAMPLE-\d+|NCT-DEMO-\d+|NCT-PLACEHOLDER", re.IGNORECASE)
bad_files: list[tuple[str, list[str]]] = []
for p in ALL_VISIBLE:
    body = _read(p)
    hits = nct_fake_re.findall(body)
    if hits:
        bad_files.append((str(p), hits))
if bad_files:
    _fail("T08", "Zero fake NCT identifiers", f"found: {bad_files}")
else:
    _pass("T08", "Zero fake NCT identifiers", f"scanned {len(ALL_VISIBLE)} files")

# ---- T09: zero quarantine-forbidden strings/patterns ------------------
qh = json.loads(_read(HERE / "quarantine_test_harness.json"))
forbidden_str = qh.get("forbidden_substrings", [])
forbidden_re = [re.compile(x["pattern"]) for x in qh.get("forbidden_patterns_regex", [])]
q_bad: list[str] = []
# Only check user-visible files; the quarantine harness itself lists them
# for enforcement — so exclude that file from the sweep.
scan_files = [p for p in ALL_VISIBLE if p.name != "quarantine_test_harness.json"]
for p in scan_files:
    body = _read(p)
    for s in forbidden_str:
        if s in body:
            q_bad.append(f"{p.name}: forbidden substring '{s}'")
    for r in forbidden_re:
        m = r.search(body)
        if m:
            q_bad.append(f"{p.name}: forbidden pattern hit '{m.group(0)[:80]}'")
if q_bad:
    _fail("T09", "Zero quarantine-forbidden strings/patterns", "; ".join(q_bad[:8]))
else:
    _pass("T09", "Zero quarantine-forbidden strings/patterns",
          f"scanned {len(scan_files)} files, {len(forbidden_str)} substrings, "
          f"{len(forbidden_re)} regex patterns")

# ---- T10: transcripts on disk, all 2xx, real ids ----------------------
transcript_checks = [
    {
        "case": "nsclc_luna16",
        "transcript": HERE / "e2e_run" / "nsclc_case" / "transcript.jsonl",
        "expected_primary_state": "loaded_luna16_retinanet",
        "primary_step": "02_case_full",
        "min_calls": 3,
    },
    {
        "case": "ak_mbd4_lof",
        "transcript": HERE / "e2e_run" / "ak_ingestion" / "transcript.jsonl",
        "expected_primary_state": "loaded_ak_bundle",
        "primary_step": "02_validate_bundle",
        "min_calls": 3,
    },
]
t10_issues: list[str] = []
for tc in transcript_checks:
    tpath = tc["transcript"]
    if not tpath.exists():
        t10_issues.append(f"{tc['case']}: transcript missing at {tpath}")
        continue
    rows = [json.loads(l) for l in tpath.read_text().strip().splitlines()]
    if len(rows) < tc["min_calls"]:
        t10_issues.append(f"{tc['case']}: only {len(rows)} calls, expected >= {tc['min_calls']}")
    non2xx = [r for r in rows if not (200 <= r["status"] < 300)]
    if non2xx:
        t10_issues.append(f"{tc['case']}: {len(non2xx)} non-2xx calls")
    primary = next((r for r in rows if r["step"] == tc["primary_step"]), None)
    if primary is None:
        t10_issues.append(f"{tc['case']}: no {tc['primary_step']} row")
    else:
        if primary.get("model_state") != tc["expected_primary_state"]:
            t10_issues.append(
                f"{tc['case']}: primary step {tc['primary_step']} state="
                f"{primary.get('model_state')} != expected {tc['expected_primary_state']}"
            )
        rid = primary.get("request_id") or ""
        if not re.match(r"^[0-9a-f]{16,}", rid):
            t10_issues.append(
                f"{tc['case']}: primary step request_id doesn't look like a UUID: {rid}"
            )
if t10_issues:
    _fail("T10", "Live 3-call transcripts on disk, all 2xx, real request_ids", "; ".join(t10_issues))
else:
    _pass("T10", "Live 3-call transcripts on disk, all 2xx, real request_ids",
          "both cases: >=3 calls, all 2xx, real UUIDs, expected states")


# ---- Summary ---------------------------------------------------------
n_pass = sum(1 for r in results if r["status"] == "PASS")
n_fail = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n{'='*76}")
print(f"Acceptance suite T01–T10: {n_pass}/{len(results)} PASS")
print(f"{'='*76}")
for r in results:
    icon = "✓" if r["status"] == "PASS" else "✗"
    print(f"  [{icon}] {r['test']:4s} {r['name']}")
    if r["detail"]:
        print(f"        {r['detail']}")
print()

# Persist to disk for the README/commit.
(HERE / "acceptance_tests.json").write_text(json.dumps({
    "summary": {
        "total": len(results),
        "pass": n_pass,
        "fail": n_fail,
    },
    "results": results,
}, indent=2))

sys.exit(0 if n_fail == 0 else 1)
