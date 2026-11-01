#!/usr/bin/env python3
"""Regenerate docs/proofs/seven_endpoint_live_io_test_evidence.md purely from
the (already-patched) JSON on disk, using the exact same table-column layout
as the original builder (scripts/build_seven_endpoint_evidence_receipt.py).

This exists because a prior in-place text patch to the .md mis-indexed the
markdown table columns (fixture-sha column vs http-status column). Rebuilding
straight from the structured JSON -- which was correctly patched via named
dict keys, not positional text splitting -- is the honest fix, not further
string surgery.
"""
import json
from pathlib import Path

JSON_PATH = Path("docs/proofs/seven_endpoint_live_io_test_evidence.json")
MD_PATH = Path("docs/proofs/seven_endpoint_live_io_test_evidence.md")
PRIMARY_LOG = "artifacts/audit/live_endpoint_evidence/run_20260927T233108Z_v2.log"


def main() -> None:
    data = json.loads(JSON_PATH.read_text())

    lines = [
        "# Seven-endpoint live I/O test evidence",
        "",
        f"Generated {data['generated_utc']}; patched {data.get('patched_utc', 'n/a')}. "
        "Base commit `bf5e54f8d8801286ba4b0ad7ca03ce0c7fc75b10`, branch "
        "`audit/manski-gate-and-test-migration`. All results below are from a "
        "real, live pytest execution against the actual Modal endpoints this "
        f"session (primary log: `{PRIMARY_LOG}`), not from memory or mocks.",
        "",
        f"**Totals**: {data['totals']['n_pass_total']} pass / "
        f"{data['totals']['n_fail_honest_total']} fail_honest out of "
        f"{data['totals']['n_tests_total_executed']} tests executed "
        f"({data['totals']['n_tests_collected']} collected) -- "
        f"{data['totals']['n_pass_main_run']}/{data['totals']['n_fail_honest_main_run']} "
        "in the main (non-slow) run, plus "
        f"{data['totals']['n_slow_tests_executed_separately']} slow test run "
        f"separately in background ({data['totals']['n_slow_pass']} pass / "
        f"{data['totals']['n_slow_fail_honest']} fail_honest).",
        "",
        "| Endpoint | URL(s) | Fixture path | Input SHA256 | HTTP | Key response fields | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in data["endpoints"]:
        urls = "<br>".join(e["urls_exercised"])
        fixtures = e["fixtures"]
        fpaths = "<br>".join(f["path"] for f in fixtures)
        fshas = "<br>".join(
            (f["sha256"][:16] + "...") if f.get("sha256") else "N/A" for f in fixtures
        )
        http_vals = e.get("http_status_observed", [])
        http_str = ",".join(str(h) for h in http_vals) if isinstance(http_vals, list) else str(http_vals)
        key_fields = e["key_response_fields"].replace("\n", " ")
        result = e["verdict"]
        lines.append(
            f"| {e['endpoint']} | {urls} | {fpaths} | {fshas} | {http_str} | {key_fields} | {result} |"
        )

    note = data["render_vs_modal_free_tier_accounting"]
    lines += [
        "",
        "## Render free-tier product surface vs. direct-Modal endpoints (kept separate, not conflated)",
        "",
        note["purpose"],
        "",
        f"Live probe: `GET {note['render_url']}/health` -> "
        f"HTTP {note['live_probe_http_status']} in "
        f"{note['live_probe_elapsed_seconds']}s. "
        f"`models_loaded` = `{json.dumps(note['live_probe_models_loaded'])}`",
        "",
    ]
    for k, v in note["interpretation"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Medgemma slow-chat resolution + mechanistic hypothesis, sourced from JSON.
    medgemma = next(x for x in data["endpoints"] if x["endpoint"] == "medgemma-27b")
    lines += [
        "---",
        "",
        "## Medgemma-27b slow chat-completion test -- resolved (was PENDING_BACKGROUND_RUN)",
        "",
        medgemma["slow_chat_test_status"],
        "",
        medgemma.get("honest_fail_detail_slow_chat", ""),
        "",
        "**Mechanistic hypothesis for the healthz-PASS vs chat-TIMEOUT divergence "
        "(NOT proven, hypothesis only):**",
        "",
        medgemma.get("mechanistic_hypothesis_healthz_vs_chat_divergence", ""),
        "",
    ]

    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"Regenerated {MD_PATH} ({MD_PATH.stat().st_size} bytes) from patched JSON.")


if __name__ == "__main__":
    main()
