"""Gate-input contract for release-gating tests (order A-13).

A test that gates a released number must FAIL when its input is missing, never
skip. Skipping makes a green run indistinguishable from absent data, which is how
the v0.5.0 real-text F1 floor sat unevaluated while asserting a threshold it would
have failed by 3.8x.

Genuinely optional fixtures (large DICOMs, network-fetched demo data) may still
skip, but they must say so explicitly via ``optional_fixture`` so the two cases
are distinguishable in the source.

Local development can downgrade gate failures to skips by exporting
``ONCOLOGY_ARBITER_ALLOW_MISSING_GATE_INPUTS=1``. CI must never set it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ALLOW_MISSING_ENV = "ONCOLOGY_ARBITER_ALLOW_MISSING_GATE_INPUTS"


def _allow_missing() -> bool:
    return os.environ.get(ALLOW_MISSING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def require_gate_input(path: str | Path, why: str) -> Path:
    """Return ``path``, or fail the test if it does not exist.

    ``why`` must name the released quantity this input gates, so the failure
    message tells the reader what is unverified rather than what is missing.
    """
    p = Path(path)
    if p.exists():
        return p
    msg = (
        f"RELEASE GATE INPUT MISSING: {p}\n"
        f"This input gates: {why}\n"
        f"The assertion below has NOT been evaluated, so this gate cannot vouch for "
        f"that quantity. Regenerate the input or explicitly retire the gate.\n"
        f"To downgrade to a skip for local work only, export {ALLOW_MISSING_ENV}=1."
    )
    if _allow_missing():
        pytest.skip(msg)
    pytest.fail(msg, pytrace=False)
    raise AssertionError("unreachable")


def optional_fixture(path: str | Path, what: str) -> Path:
    """Skip when a genuinely optional fixture is absent. Not for release gates."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"optional fixture absent ({what}): {p}")
    return p
