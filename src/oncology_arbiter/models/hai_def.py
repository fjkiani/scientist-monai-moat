"""HAI-DEF (Health AI Developer Foundations) access-gating utilities.

Google's medical models (MedSigLIP, MedGemma 1.5 4B, MedGemma 1 27B) are
distributed under the Health AI Developer Foundations (HAI-DEF) Terms of Use
and are **gated** on Hugging Face — pulling weights requires:

  1. an authenticated HuggingFace token (env var `HF_TOKEN` or
     `HUGGINGFACE_HUB_TOKEN`, or a token cached under `~/.cache/huggingface/`),
  2. explicit user acceptance of the HAI-DEF terms on the model repo page.

Unauthenticated or unauthorized requests to `huggingface.co` for those
weights return **HTTP 401** (no token / bad token) or **HTTP 403**
(token OK but user has not accepted the license). Both must be handled
without crashing the arbiter; instead we surface a machine-readable
`GatedAccessError` and the API layer maps that to `ModelState.GATED`
so the response envelope tells the caller exactly why no inference ran.

This module is deliberately I/O-light: it only checks whether a request
*would* be allowed without actually downloading the multi-GB weights.

References for verification:
  - HAI-DEF landing:  https://developers.google.com/health-ai-developer-foundations
  - MedSigLIP card:   https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card
  - MedGemma card:    https://developers.google.com/health-ai-developer-foundations/medgemma/model-card

RUO / mammography honesty note: MedSigLIP's training data does NOT include
mammography (see docs/model_cards/medsiglip_448.md). Even when this gate
returns `AccessLevel.ALLOWED`, downstream code MUST NOT report any
mammography-specific AUROC as a MedSigLIP number; the only Google-published
breast-related AUROC is on **histopathology** (Invasive Breast Cancer,
n=5000, 3 classes, zero-shot 0.933 / linear-probe 0.930 / HAI-DEF LP 0.943).
"""
from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# Model identifiers we support


# These are the exact HuggingFace repo IDs (verbatim from Google's model cards
# fetched 2026-07-01). Do not typo-fix these — they must match `hf_hub_download`
# call arguments exactly or the request will 404 instead of 401/403.
HAI_DEF_GATED_REPOS: tuple[str, ...] = (
    "google/medsiglip-448",
    "google/medgemma-1.5-4b-it",
    "google/medgemma-27b-it",
)


# Ungated proxy model used when HAI-DEF weights are unavailable. This is a
# GENERAL-DOMAIN vision-language model trained on WebLI (web image-text) —
# it has no medical curation and MUST NOT be reported as a substitute for
# MedSigLIP output. See docs/model_cards/siglip_base_patch16_224.md.
UNGATED_PROXY_REPO: str = "google/siglip-base-patch16-224"


class AccessLevel(str, enum.Enum):
    """Outcome of a preflight HAI-DEF access check.

    UNAUTHENTICATED — no HuggingFace token found in env or cache
    FORBIDDEN       — token present but user has not accepted HAI-DEF terms
                      for this specific repo (HTTP 403 semantic)
    ALLOWED         — preflight succeeded; weight download would proceed
    UNKNOWN         — network / registry error; not classifiable
    """

    UNAUTHENTICATED = "unauthenticated"   # HTTP 401 semantic
    FORBIDDEN = "forbidden"               # HTTP 403 semantic
    ALLOWED = "allowed"                   # ready to pull weights
    UNKNOWN = "unknown"                   # ambiguous network / server error


class GatedAccessError(RuntimeError):
    """Raised when preflight determined a gated repo cannot be pulled.

    Carries the underlying `AccessLevel`, HTTP status code (if any), the
    repo_id that was checked, and a human-readable reason. API handlers
    should map this to `ModelState.GATED` and preserve the reason string in
    the response envelope so the caller knows whether to retry with a token
    vs redirect the user to accept HAI-DEF terms.
    """

    def __init__(
        self,
        repo_id: str,
        access_level: AccessLevel,
        status_code: int | None,
        reason: str,
    ) -> None:
        self.repo_id = repo_id
        self.access_level = access_level
        self.status_code = status_code
        self.reason = reason
        super().__init__(
            f"HAI-DEF access denied for {repo_id!r}: "
            f"{access_level.value} (HTTP {status_code}) — {reason}"
        )


@dataclass(frozen=True)
class GateReport:
    """Machine-readable output of `check_hai_def_access`."""

    repo_id: str
    access_level: AccessLevel
    status_code: int | None
    reason: str
    has_token: bool

    @property
    def allowed(self) -> bool:
        return self.access_level is AccessLevel.ALLOWED


# --------------------------------------------------------------------------- #
# Token discovery


def _discover_hf_token() -> str | None:
    """Return the effective HuggingFace token (env or cache) or None.

    Order (matches huggingface_hub's own resolution):
      1. `HF_TOKEN` env var
      2. `HUGGINGFACE_HUB_TOKEN` env var
      3. token file at `~/.cache/huggingface/token`
    Returns None if no token can be discovered. Never raises.
    """
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    # Filesystem token cache (huggingface-cli login writes here).
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    try:
        if os.path.isfile(token_path):
            with open(token_path) as fh:
                val = fh.read().strip()
            if val:
                return val
    except OSError:
        # Any filesystem error → treat as no token, do not crash.
        pass
    return None


def _classify_http_status(status_code: int | None) -> tuple[AccessLevel, str]:
    """Map an HTTP status from the HF hub to (AccessLevel, reason).

    * 200 → ALLOWED
    * 401 → UNAUTHENTICATED (no token or invalid token)
    * 403 → FORBIDDEN (token OK but HAI-DEF terms not accepted)
    * 404 → UNKNOWN (repo not found — treat as configuration bug)
    * anything else → UNKNOWN
    """
    if status_code == 200:
        return AccessLevel.ALLOWED, "preflight succeeded"
    if status_code == 401:
        return (
            AccessLevel.UNAUTHENTICATED,
            "no valid HuggingFace token; set HF_TOKEN and re-run",
        )
    if status_code == 403:
        return (
            AccessLevel.FORBIDDEN,
            "HAI-DEF terms have not been accepted for this repo; visit the "
            "HuggingFace model page and click 'Access repository'",
        )
    if status_code == 404:
        return AccessLevel.UNKNOWN, f"repo not found (HTTP 404) — check repo_id"
    return AccessLevel.UNKNOWN, f"unexpected status HTTP {status_code}"


# --------------------------------------------------------------------------- #
# Public API


def check_hai_def_access(
    repo_id: str,
    *,
    session: Any = None,
    timeout_s: float = 10.0,
) -> GateReport:
    """Preflight-check whether `repo_id` can be pulled from HuggingFace.

    Uses HEAD `https://huggingface.co/api/models/<repo_id>` so we do NOT
    trigger a weight download (which would be several GB for MedSigLIP or
    tens of GB for MedGemma 27B).

    Parameters
    ----------
    repo_id:
        HuggingFace repo id. Must be one of `HAI_DEF_GATED_REPOS` for the
        gate semantics to be meaningful; other repos will still be probed
        but treated as UNKNOWN.
    session:
        Optional `requests.Session`-like object with a `head(url, ...)`
        method. Injected in tests to avoid real network I/O. If None,
        `requests.head` is used directly.
    timeout_s:
        Connection + read timeout in seconds.

    Returns
    -------
    GateReport
        Never raises for expected outcomes (401/403/404/timeout). Only
        raises for programmer errors (empty repo_id).

    Notes
    -----
    HuggingFace's public API historically returns 401 for unauthenticated
    requests to gated repos and 403 for authenticated-but-unlicensed
    requests. See https://huggingface.co/docs/hub/api and the /api/models
    surface — behaviour is model-registry-specific and this classifier is
    deliberately conservative (unknown status → AccessLevel.UNKNOWN).
    """
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")

    token = _discover_hf_token()
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://huggingface.co/api/models/{repo_id}"

    # Lazy-import requests so the honesty of this module (no accidental
    # network calls at import time) is preserved and so tests using a
    # stub session don't pull the real dependency.
    if session is None:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - dev env has requests
            return GateReport(
                repo_id=repo_id,
                access_level=AccessLevel.UNKNOWN,
                status_code=None,
                reason=f"requests not installed: {exc}",
                has_token=bool(token),
            )
        session = requests

    try:
        resp = session.head(url, headers=headers, timeout=timeout_s, allow_redirects=True)
    except Exception as exc:  # network flake, DNS, timeout etc.
        return GateReport(
            repo_id=repo_id,
            access_level=AccessLevel.UNKNOWN,
            status_code=None,
            reason=f"network error probing HAI-DEF: {type(exc).__name__}: {exc}",
            has_token=bool(token),
        )

    status = getattr(resp, "status_code", None)
    access_level, reason = _classify_http_status(status)

    # Extra semantics: if we got 200 but had NO token, that means the repo
    # is actually ungated (public). Callers who expected HAI-DEF behaviour
    # can spot the mismatch via `has_token=False`.
    return GateReport(
        repo_id=repo_id,
        access_level=access_level,
        status_code=status,
        reason=reason,
        has_token=bool(token),
    )


def enforce_gate(report: GateReport) -> None:
    """Raise GatedAccessError if `report` is not ALLOWED.

    Convenience wrapper for API handlers that want to short-circuit on
    denied access. If the report is ALLOWED this is a no-op.
    """
    if report.allowed:
        return
    raise GatedAccessError(
        repo_id=report.repo_id,
        access_level=report.access_level,
        status_code=report.status_code,
        reason=report.reason,
    )


def resolve_backend_for_task(
    task: str,
    *,
    session: Any = None,
) -> tuple[str, GateReport]:
    """Pick the HuggingFace repo to serve `task` and return its gate report.

    Task → primary repo map (verbatim from model cards, DO NOT ALIAS):
      * "screening"       → google/medsiglip-448
      * "medsiglip"       → google/medsiglip-448
      * "medgemma_small"  → google/medgemma-1.5-4b-it
      * "medgemma_large"  → google/medgemma-27b-it
      * "proxy"           → google/siglip-base-patch16-224  (always ungated)

    If the primary repo is denied, the caller is responsible for deciding
    whether to fall back to the ungated proxy (and, if so, must mark the
    response as `ModelState.PROXY_SIGLIP` NOT `ModelState.LOADED`).

    Returns
    -------
    (repo_id, gate_report)
    """
    task = task.strip().lower()
    if task in {"screening", "medsiglip"}:
        repo_id = "google/medsiglip-448"
    elif task == "medgemma_small":
        repo_id = "google/medgemma-1.5-4b-it"
    elif task == "medgemma_large":
        repo_id = "google/medgemma-27b-it"
    elif task == "proxy":
        repo_id = UNGATED_PROXY_REPO
    else:
        raise ValueError(
            f"unknown task {task!r}; expected one of "
            "'screening', 'medsiglip', 'medgemma_small', "
            "'medgemma_large', 'proxy'"
        )
    return repo_id, check_hai_def_access(repo_id, session=session)


__all__ = [
    "HAI_DEF_GATED_REPOS",
    "UNGATED_PROXY_REPO",
    "AccessLevel",
    "GateReport",
    "GatedAccessError",
    "check_hai_def_access",
    "enforce_gate",
    "resolve_backend_for_task",
]
