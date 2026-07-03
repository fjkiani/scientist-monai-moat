"""Contract tests for the progress ledger + its maintainer script.

The ledger is not user-facing code, but it IS a discipline artifact and it
governs a promise we make to the user ("LIVE really means LIVE"). These
tests enforce that promise mechanically.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_progress_ledger.py"
LEDGER_PATH = REPO_ROOT / "docs" / "PROGRESS_LEDGER.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "update_progress_ledger", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["update_progress_ledger"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert LEDGER_PATH.exists(), (
        f"missing {LEDGER_PATH} — run scripts/update_progress_ledger.py"
    )
    return json.loads(LEDGER_PATH.read_text())


# --------------------------------------------------------------------------- #
# 1. Ledger file exists, is well-formed, and covers every required section.
# --------------------------------------------------------------------------- #


def test_ledger_top_level_keys_present(ledger: dict) -> None:
    for key in (
        "$schema_version",
        "generated_at",
        "git_sha",
        "project",
        "user_stories",
        "subsystems",
        "sprints",
        "deferred_decisions",
        "external_gates",
        "honesty_notes",
    ):
        assert key in ledger, f"ledger missing top-level key: {key}"


def test_ledger_has_at_least_one_of_every_category(ledger: dict) -> None:
    assert ledger["user_stories"], "user_stories is empty"
    assert ledger["subsystems"], "subsystems is empty"
    assert ledger["sprints"], "sprints is empty"
    assert ledger["external_gates"], "external_gates is empty"


# --------------------------------------------------------------------------- #
# 2. Every LIVE entry has a working evidence path. Every NOT_WIRED entry has
#    a non-empty not_wired_reason. (Same rules the script enforces at write.)
# --------------------------------------------------------------------------- #


def _iter_status_entries(ledger: dict):
    yield from ledger["user_stories"]
    yield from ledger["subsystems"]


def test_every_live_entry_has_working_evidence(ledger: dict) -> None:
    for e in _iter_status_entries(ledger):
        if e.get("status") != "LIVE":
            continue
        ev = e.get("evidence") or []
        assert ev, f"{e['id']} is LIVE but evidence[] is empty"
        for path in ev:
            file_part = path.split("::", 1)[0]
            # accept both repo-relative and absolute
            if file_part.startswith("/"):
                candidate = Path(file_part)
            else:
                candidate = REPO_ROOT / file_part
            assert candidate.exists(), (
                f"{e['id']} evidence path missing on disk: {path}"
            )


def test_every_wired_files_path_exists_on_disk(ledger: dict) -> None:
    """Subsystem wired_files must point at real files on disk (not aspirational paths).

    Regression guard against the honesty bug where L2-evidence-fetchers cited
    ``src/oncology_arbiter/evidence/*`` but the code actually lives in ``tools/``.
    Applies to BOTH LIVE and NOT_WIRED subsystems: if a wired_files entry doesn't
    exist, the ledger is lying about where the code lives.
    """
    for e in ledger["subsystems"]:
        wf = e.get("wired_files") or []
        for path in wf:
            file_part = path.split("::", 1)[0]
            if file_part.startswith("/"):
                candidate = Path(file_part)
            else:
                candidate = REPO_ROOT / file_part
            assert candidate.exists(), (
                f"{e['id']} wired_files path missing on disk: {path}"
            )


def test_every_not_wired_entry_has_reason(ledger: dict) -> None:
    for e in _iter_status_entries(ledger):
        if e.get("status") != "NOT_WIRED":
            continue
        reason = (e.get("not_wired_reason") or "").strip()
        assert reason, f"{e['id']} is NOT_WIRED but not_wired_reason is empty"


def test_status_is_one_of_the_two_tiers(ledger: dict) -> None:
    for e in _iter_status_entries(ledger):
        assert e["status"] in ("LIVE", "NOT_WIRED"), (
            f"{e['id']} has invalid status: {e['status']!r}"
        )


# --------------------------------------------------------------------------- #
# 3. External gate rows cite real evidence.
# --------------------------------------------------------------------------- #


def test_external_hai_def_gates_have_evidence(ledger: dict) -> None:
    gates = {g["gate"]: g for g in ledger["external_gates"]}
    assert "hai_def" in gates
    for repo in gates["hai_def"]["repos"]:
        assert repo["state"] in (
            "allowed",
            "forbidden",
            "unauthenticated",
            "unknown",
        ), f"invalid gate state for {repo['repo_id']}: {repo['state']!r}"
        assert repo.get("evidence"), f"no evidence for gate row {repo['repo_id']}"


# --------------------------------------------------------------------------- #
# 4. The regression-guarded LIVE claim for MedSigLIP smoke matches what the
#    live envelope on disk actually says.
# --------------------------------------------------------------------------- #


def test_medsiglip_live_smoke_number_matches_envelope(ledger: dict) -> None:
    subsystems = {s["id"]: s for s in ledger["subsystems"]}
    ms = subsystems.get("L4a-screening-medsiglip")
    assert ms is not None
    smoke = ms.get("live_smoke")
    assert smoke, "L4a-screening-medsiglip LIVE entry missing live_smoke block"

    envelope_path = Path("/mnt/results/screening_response_medsiglip_smoke_final.json")
    if not envelope_path.exists():
        pytest.skip(f"envelope not present at {envelope_path}")

    body = json.loads(envelope_path.read_text())
    assert body["provenance"]["model_state"] == "loaded_medsiglip"
    assert body["overall_score"] == smoke["overall_score"], (
        f"ledger says overall_score={smoke['overall_score']} "
        f"but envelope has {body['overall_score']}"
    )


# --------------------------------------------------------------------------- #
# 5. `--dry-run` on an unchanged repo prints "no changes" — PLAN.md §5 rule 6.
# --------------------------------------------------------------------------- #


def test_dry_run_reports_no_changes_when_repo_is_current() -> None:
    # Regenerate to ensure the on-disk copy is current, then dry-run.
    subprocess.check_call(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
    )
    out = subprocess.check_output(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=str(REPO_ROOT),
        text=True,
    )
    assert "no changes" in out, f"expected 'no changes' in dry-run output; got:\n{out}"


# --------------------------------------------------------------------------- #
# 6. Sprint entries cite real commit SHAs (short-sha or full-sha).
# --------------------------------------------------------------------------- #


def test_sprint_commits_exist_in_git_history(ledger: dict) -> None:
    for sprint in ledger["sprints"]:
        for sha in sprint.get("commits", []):
            try:
                subprocess.check_call(
                    ["git", "cat-file", "-e", sha],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                pytest.fail(f"sprint {sprint['id']} cites unknown commit: {sha}")


# --------------------------------------------------------------------------- #
# 7. LIVE + NOT_WIRED counts match what the script prints on success.
# --------------------------------------------------------------------------- #


def test_summary_counts_are_stable(ledger: dict) -> None:
    live = sum(
        1
        for e in _iter_status_entries(ledger)
        if e["status"] == "LIVE"
    )
    not_wired = sum(
        1
        for e in _iter_status_entries(ledger)
        if e["status"] == "NOT_WIRED"
    )
    # Just assert non-trivial: we want at least a handful of each.
    assert live >= 10, f"only {live} LIVE entries — did you break the ledger?"
    assert not_wired >= 5, f"only {not_wired} NOT_WIRED entries"
