"""Tests for oncology_arbiter.models.hai_def

Covers:
  * Repo-id constants match Google's HAI-DEF card verbatim
  * HTTP status classification (401 → UNAUTHENTICATED, 403 → FORBIDDEN, 200 → ALLOWED)
  * Token discovery precedence (env vars > cache file > None)
  * enforce_gate raises GatedAccessError for non-allowed reports
  * resolve_backend_for_task returns the correct repo per task
  * Ungated proxy always reports ALLOWED (public repo)

No live network required — a stub session captures HTTP calls.
"""
from __future__ import annotations

import types

import pytest

from oncology_arbiter.models.hai_def import (
    HAI_DEF_GATED_REPOS,
    UNGATED_PROXY_REPO,
    AccessLevel,
    GateReport,
    GatedAccessError,
    check_hai_def_access,
    enforce_gate,
    resolve_backend_for_task,
    _classify_http_status,
    _discover_hf_token,
)


# --------------------------------------------------------------------------- #
# Repo-id anchors (verbatim from Google model card pages fetched 2026-07-01)


def test_hai_def_gated_repos_exact_list() -> None:
    assert HAI_DEF_GATED_REPOS == (
        "google/medsiglip-448",
        "google/medgemma-1.5-4b-it",
        "google/medgemma-27b-it",
    )


def test_ungated_proxy_repo_is_general_domain_siglip() -> None:
    assert UNGATED_PROXY_REPO == "google/siglip-base-patch16-224"


def test_ungated_proxy_is_not_in_gated_list() -> None:
    assert UNGATED_PROXY_REPO not in HAI_DEF_GATED_REPOS


# --------------------------------------------------------------------------- #
# HTTP classification


@pytest.mark.parametrize(
    "status,expected_level",
    [
        (200, AccessLevel.ALLOWED),
        (401, AccessLevel.UNAUTHENTICATED),
        (403, AccessLevel.FORBIDDEN),
        (404, AccessLevel.UNKNOWN),
        (500, AccessLevel.UNKNOWN),
        (None, AccessLevel.UNKNOWN),
    ],
)
def test_classify_http_status(status: int | None, expected_level: AccessLevel) -> None:
    level, reason = _classify_http_status(status)
    assert level is expected_level
    assert isinstance(reason, str) and reason


def test_401_reason_mentions_token() -> None:
    _, reason = _classify_http_status(401)
    assert "token" in reason.lower()


def test_403_reason_mentions_terms() -> None:
    _, reason = _classify_http_status(403)
    r = reason.lower()
    assert "terms" in r or "hai-def" in r
    assert "access" in r or "accept" in r


# --------------------------------------------------------------------------- #
# Token discovery (env, cache, none)


def test_discover_token_prefers_hf_token_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_alpha_env")
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("os.path.expanduser", lambda p: "/nonexistent/hf_token")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert _discover_hf_token() == "hf_alpha_env"


def test_discover_token_falls_back_to_huggingface_hub_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_beta_env")
    monkeypatch.setattr("os.path.expanduser", lambda p: "/nonexistent/hf_token")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert _discover_hf_token() == "hf_beta_env"


def test_discover_token_none_when_all_missing(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert _discover_hf_token() is None


def test_discover_token_reads_cache_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    cache = tmp_path / "token"
    cache.write_text("hf_cached_1234\n")
    monkeypatch.setattr("os.path.expanduser", lambda p: str(cache))
    monkeypatch.setattr("os.path.isfile", lambda p: p == str(cache))
    assert _discover_hf_token() == "hf_cached_1234"


# --------------------------------------------------------------------------- #
# check_hai_def_access with stub session


class _StubResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _StubSession:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls: list[tuple[str, dict, float, bool]] = []

    def head(self, url, headers=None, timeout=None, allow_redirects=False):
        self.calls.append((url, dict(headers or {}), timeout, allow_redirects))
        return _StubResp(self.status_code)


def test_check_returns_allowed_on_200(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(200)
    rep = check_hai_def_access("google/medsiglip-448", session=sess)
    assert rep.access_level is AccessLevel.ALLOWED
    assert rep.status_code == 200
    assert rep.has_token is True
    assert rep.repo_id == "google/medsiglip-448"


def test_check_returns_unauthenticated_on_401_without_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(401)
    rep = check_hai_def_access("google/medsiglip-448", session=sess)
    assert rep.access_level is AccessLevel.UNAUTHENTICATED
    assert rep.status_code == 401
    assert rep.has_token is False


def test_check_returns_forbidden_on_403_with_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(403)
    rep = check_hai_def_access("google/medgemma-27b-it", session=sess)
    assert rep.access_level is AccessLevel.FORBIDDEN
    assert rep.status_code == 403
    assert rep.has_token is True


def test_check_hits_correct_hf_api_url(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(200)
    check_hai_def_access("google/medgemma-1.5-4b-it", session=sess)
    assert sess.calls, "session was not invoked"
    url, headers, timeout, follow = sess.calls[0]
    assert url == "https://huggingface.co/api/models/google/medgemma-1.5-4b-it"
    assert headers.get("Authorization") == "Bearer hf_dummy"
    assert isinstance(timeout, (int, float)) and timeout > 0
    assert follow is True


def test_check_omits_auth_header_when_no_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(401)
    check_hai_def_access("google/medsiglip-448", session=sess)
    _, headers, _, _ = sess.calls[0]
    assert "Authorization" not in headers


def test_check_returns_unknown_on_network_exception(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    class _Blowup:
        def head(self, *a, **kw):
            raise ConnectionError("dns fail")

    rep = check_hai_def_access("google/medsiglip-448", session=_Blowup())
    assert rep.access_level is AccessLevel.UNKNOWN
    assert rep.status_code is None
    assert "network" in rep.reason.lower()


def test_check_rejects_empty_repo_id() -> None:
    with pytest.raises(ValueError):
        check_hai_def_access("")
    with pytest.raises(ValueError):
        check_hai_def_access(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# enforce_gate


def test_enforce_gate_noop_on_allowed() -> None:
    rep = GateReport(
        repo_id="google/medsiglip-448",
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="ok",
        has_token=True,
    )
    enforce_gate(rep)  # must not raise


@pytest.mark.parametrize(
    "level,status",
    [
        (AccessLevel.UNAUTHENTICATED, 401),
        (AccessLevel.FORBIDDEN, 403),
        (AccessLevel.UNKNOWN, 500),
    ],
)
def test_enforce_gate_raises_for_denied(level: AccessLevel, status: int) -> None:
    rep = GateReport(
        repo_id="google/medsiglip-448",
        access_level=level,
        status_code=status,
        reason="denied",
        has_token=True,
    )
    with pytest.raises(GatedAccessError) as exc:
        enforce_gate(rep)
    assert exc.value.repo_id == "google/medsiglip-448"
    assert exc.value.access_level is level
    assert exc.value.status_code == status


# --------------------------------------------------------------------------- #
# resolve_backend_for_task


@pytest.mark.parametrize(
    "task,expected_repo",
    [
        ("screening", "google/medsiglip-448"),
        ("medsiglip", "google/medsiglip-448"),
        ("medgemma_small", "google/medgemma-1.5-4b-it"),
        ("medgemma_large", "google/medgemma-27b-it"),
        ("proxy", "google/siglip-base-patch16-224"),
    ],
)
def test_resolve_backend_for_task(monkeypatch, task: str, expected_repo: str) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(200)
    repo, rep = resolve_backend_for_task(task, session=sess)
    assert repo == expected_repo
    assert rep.repo_id == expected_repo
    assert rep.access_level is AccessLevel.ALLOWED


def test_resolve_backend_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        resolve_backend_for_task("nonsense-task")


def test_resolve_backend_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    sess = _StubSession(200)
    repo, _ = resolve_backend_for_task("  SCREENING  ", session=sess)
    assert repo == "google/medsiglip-448"


# --------------------------------------------------------------------------- #
# ModelState wire enum extension


def test_model_state_enum_has_gated_variant() -> None:
    """API surfaces MUST be able to say ModelState.GATED when HAI-DEF denies."""
    from oncology_arbiter.api.schemas import ModelState
    assert hasattr(ModelState, "GATED")
    assert ModelState.GATED.value == "gated"


def test_model_state_enum_has_proxy_siglip_variant() -> None:
    """Ungated proxy fallback MUST get a distinct wire value so downstream
    consumers know NOT to treat it as MedSigLIP output."""
    from oncology_arbiter.api.schemas import ModelState
    assert hasattr(ModelState, "PROXY_SIGLIP")
    assert ModelState.PROXY_SIGLIP.value == "proxy_siglip"


def test_model_state_placeholder_and_loaded_survive() -> None:
    """Existing states MUST not have been renamed or dropped."""
    from oncology_arbiter.api.schemas import ModelState
    assert ModelState.PLACEHOLDER.value == "placeholder"
    assert ModelState.LOADED.value == "loaded"
    assert ModelState.UNAVAILABLE.value == "unavailable"
    assert ModelState.CACHED.value == "cached"
    assert ModelState.LOADING.value == "loading"
