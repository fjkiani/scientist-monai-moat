"""Shape tests for render.yaml — the Render.com deployment blueprint.

These tests enforce the free-tier orchestration posture:
- The 512 MB process never loads MONAI, MedSigLIP, or TxGemma weights.
- Identity-locked specialists are addressed remotely.
- Proxy/rules-lite substitutes are not silently enabled.
- Every env var must be a static value; no secrets (HuggingFace tokens,
  API keys) get committed.
- The healthCheckPath MUST match the Dockerfile HEALTHCHECK target, or
  Render's proxy will 502 while the container is up.

If a maintainer later wants to enable one of the heavy backends, they
should either upgrade to a paid tier AND update these tests, or set
the flag as an unset envVar (Render dashboard override) that this file
NEVER commits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RENDER_YAML = Path(__file__).parent.parent.parent / "render.yaml"


def test_render_yaml_exists_at_repo_root() -> None:
    assert RENDER_YAML.exists(), f"render.yaml missing at {RENDER_YAML}"


@pytest.fixture(scope="module")
def render_yaml_text() -> str:
    return RENDER_YAML.read_text(encoding="utf-8")


def test_render_yaml_declares_docker_runtime(render_yaml_text: str) -> None:
    assert re.search(r"runtime:\s*docker", render_yaml_text), \
        "render.yaml must set runtime: docker so BuildKit uses our Dockerfile"


def test_render_yaml_targets_free_plan(render_yaml_text: str) -> None:
    # Guard against accidentally committing a paid-tier plan.
    assert re.search(r"plan:\s*free", render_yaml_text), \
        "render.yaml must set plan: free — anything else would silently bill"


def test_render_yaml_health_check_matches_dockerfile(render_yaml_text: str) -> None:
    # Both the Render proxy and the Dockerfile HEALTHCHECK must probe /health,
    # or Render will 502 while the container is actually alive.
    assert re.search(r"healthCheckPath:\s*/health", render_yaml_text)
    dockerfile = (RENDER_YAML.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "/health" in dockerfile, \
        "Dockerfile HEALTHCHECK must also probe /health"


def test_render_yaml_docker_context_is_repo_root(render_yaml_text: str) -> None:
    # We rely on the ./ context to include src/, pyproject.toml, and
    # src/oncology_arbiter/api/static/dist/.
    assert re.search(r"dockerContext:\s*\.", render_yaml_text)
    assert re.search(r"dockerfilePath:\s*\./Dockerfile", render_yaml_text)


def test_render_yaml_enables_frontend_mount(render_yaml_text: str) -> None:
    """SPA bundle is shipped in the Docker image — mount it or the
    /ui/ endpoint returns 404 on the deployed URL."""
    assert "ONCOLOGY_ARBITER_SERVE_FRONTEND" in render_yaml_text


def test_render_yaml_configures_remote_identity_locked_specialists(
    render_yaml_text: str,
) -> None:
    for key in (
        "MODAL_MEDSIGLIP_URL",
        "MEDGEMMA_MODAL_URL",
        "TXGEMMA_MODAL_URL",
    ):
        assert re.search(rf"^\s*-\s*key:\s*{key}\s*$", render_yaml_text, re.MULTILINE)


def test_render_yaml_does_not_enable_proxy_or_legacy_parser_paths(
    render_yaml_text: str,
) -> None:
    for key in (
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY",
        "ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST",
        "CLINICALBERT_BACKEND",
        "CLINICALBERT_MODAL_URL",
    ):
        assert not re.search(
            rf"^\s*-\s*key:\s*{key}\s*$",
            render_yaml_text,
            re.MULTILINE,
        )


def test_render_yaml_does_not_enable_heavy_backends(render_yaml_text: str) -> None:
    """The free tier's 512 MB dyno cannot hold MONAI (~500 MB),
    MedSigLIP (~2 GB with torch), or TxGemma. Enabling any of these
    flags in the committed blueprint would guarantee an OOM on first
    real request.

    A maintainer who wants to enable these later must either:
      - Upgrade to a paid plan first AND update this test, or
      - Set the flag as an untracked dashboard env override (never in this file).
    """
    forbidden = {
        "ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR",
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
    }
    for flag in forbidden:
        # Only flag it if the flag is actually declared as an active env var.
        # A comment mentioning the flag (explaining WHY it's off) is fine.
        # The distinction: an active env var appears as `- key: <FLAG>` in yaml.
        pattern = rf"^\s*-\s*key:\s*{flag}\s*$"
        assert not re.search(pattern, render_yaml_text, re.MULTILINE), (
            f"render.yaml enables {flag} — the 512 MB free tier would OOM. "
            "Upgrade the plan first and update this test."
        )


def test_render_yaml_does_not_bake_secrets(render_yaml_text: str) -> None:
    """No HuggingFace token, OpenAI key, or AWS key in a committed file.
    Every secret must live in the Render dashboard as a per-environment
    override, never in git.
    """
    secret_patterns = [
        r"HUGGINGFACE_TOKEN\s*[:=]\s*[\"']?hf_",
        r"HF_TOKEN\s*[:=]\s*[\"']?hf_",
        r"OPENAI_API_KEY\s*[:=]\s*[\"']?sk-",
        r"AWS_SECRET_ACCESS_KEY\s*[:=]",
        r"AWS_ACCESS_KEY_ID\s*[:=]\s*[\"']?AKIA",
        r"RENDER_API_KEY\s*[:=]\s*[\"']?rnd_",
    ]
    for pat in secret_patterns:
        assert not re.search(pat, render_yaml_text), \
            f"render.yaml appears to contain a secret matching /{pat}/"


def test_render_yaml_branch_is_main(render_yaml_text: str) -> None:
    """Deploy tracks main — the branch where all four merges landed.
    A stale branch name would silently deploy prior code."""
    assert re.search(r"branch:\s*main", render_yaml_text)
