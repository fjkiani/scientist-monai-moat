"""Session-wide pytest configuration.

Test-suite auth policy
======================
Almost every unit test in this repo assumes the API is reachable without an
``X-API-Key`` header. In production (Render deploy) the operator sets
``ONCOLOGY_ARBITER_AUTH_MODE=on`` explicitly and provides a bootstrap-key hash
so the tenants table is seeded on startup. Locally and in CI we don't want
every keyless request to fail 401 — that would obscure the real behavior each
endpoint test is trying to prove.

We take the "explicit off for tests, explicit on for prod" position: set the
env var to ``off`` at the top of the pytest session unless a specific test
opts back in. Tests that DO exercise the auth path (see
``tests/unit/test_saas_hardening.py`` and ``tests/unit/test_auth_bootstrap.py``)
use ``monkeypatch.setenv(...)`` to override this default within their own
scope.

This mirrors the existing operator ergonomics (fresh clone → tests pass) while
leaving the enforcement code itself unchanged in production.
"""
from __future__ import annotations

import os


def pytest_configure(config):
    """Force AUTH_MODE=off at session start unless already explicitly set.

    We only default when unset — a test run that specifically wants to hit the
    enforced path (``AUTH_MODE=on`` from the shell) is left alone.
    """
    os.environ.setdefault("ONCOLOGY_ARBITER_AUTH_MODE", "off")

    # v0.2.2: the /v1/demo/case pre-warm tries to fetch a ~14 MB DICOM from
    # HuggingFace at every `create_app()`. Skipping it in tests keeps the
    # suite fast and network-free by default. Tests that WANT to exercise
    # the pre-warm path can un-set this via monkeypatch.
    os.environ.setdefault("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "1")
