"""Shape tests for the Dockerfile and .dockerignore.

These are DESIGN CONTRACT tests, not build tests — this sandbox has no
docker daemon. They verify:

  * The Dockerfile exists at the repo root and uses a multi-stage build
    with a slim base image, a non-root USER, and a HEALTHCHECK.
  * The .dockerignore exists and excludes the paths we never want in the
    build context (venvs, node_modules, tests, secrets).
  * The health endpoint referenced by HEALTHCHECK actually exists in the
    FastAPI app (regression guard against renaming /health).
  * The frontend static bundle path referenced by the runtime is present
    in the source tree so a fresh `docker build .` will succeed.

To actually build the image (outside this sandbox):
    docker build -t oncology-arbiter:dev .
    docker build --build-arg ONCOLOGY_ARBITER_INCLUDE_ML=1 -t oncology-arbiter:ml .
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


# --------------------------------------------------------------------------- #
# File existence

def test_dockerfile_exists_at_repo_root():
    assert DOCKERFILE.is_file(), f"Dockerfile not found at {DOCKERFILE}"


def test_dockerignore_exists_at_repo_root():
    assert DOCKERIGNORE.is_file(), f".dockerignore not found at {DOCKERIGNORE}"


# --------------------------------------------------------------------------- #
# Dockerfile contract

def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_is_multi_stage():
    text = _dockerfile_text()
    from_lines = [ln for ln in text.splitlines() if ln.strip().startswith("FROM ")]
    assert len(from_lines) >= 2, (
        "Dockerfile must be multi-stage (at least 2 FROM directives) so build "
        f"tooling is not in the runtime image. Got {len(from_lines)}."
    )
    # At least one FROM must be `AS runtime` (or similar terminal stage name).
    assert any("AS runtime" in ln for ln in from_lines), (
        "Dockerfile must have a `FROM ... AS runtime` terminal stage"
    )


def test_dockerfile_uses_slim_base():
    """We ship on python:3.11-slim — the -slim variant is ~50 MB vs ~1 GB for
    the default python:3.11 image. Regression guard so nobody switches back.
    Handles either a literal FROM python:X.Y-slim or the ARG-based form
    `ARG PYTHON_VERSION=X.Y-slim` + `FROM python:${PYTHON_VERSION}`.
    """
    text = _dockerfile_text()
    literal = re.search(r"FROM\s+python:[\d.]+-slim", text)
    arg_form = re.search(
        r"ARG\s+PYTHON_VERSION\s*=\s*[\d.]+-slim.*?FROM\s+python:\$\{PYTHON_VERSION\}",
        text,
        re.DOTALL,
    )
    assert literal or arg_form, (
        "Dockerfile must use a slim Python base image (python:X.Y-slim, "
        "either directly or via ARG PYTHON_VERSION)"
    )


def test_dockerfile_has_healthcheck_against_health_endpoint():
    text = _dockerfile_text()
    assert "HEALTHCHECK" in text, "Dockerfile is missing HEALTHCHECK"
    # The HEALTHCHECK command must reference /health, not /healthz or /.
    hc_match = re.search(
        r"HEALTHCHECK[^\n]*(?:\n\s*[^\n]+)*", text
    )
    assert hc_match, "HEALTHCHECK block not found"
    hc_text = hc_match.group(0)
    assert "/health" in hc_text, (
        f"HEALTHCHECK must probe /health, got:\n{hc_text}"
    )


def test_dockerfile_uses_non_root_user():
    text = _dockerfile_text()
    # Look for a USER directive with a non-root argument (name or non-zero UID).
    user_lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip().startswith("USER ")
    ]
    assert user_lines, "Dockerfile is missing USER directive (running as root)"
    for ln in user_lines:
        arg = ln.removeprefix("USER").strip()
        assert arg not in ("root", "0"), (
            f"Dockerfile must not run as root — found `{ln}`"
        )


def test_dockerfile_creates_dedicated_user():
    """Best practice: create a system user for the app, not reuse a stock one."""
    text = _dockerfile_text()
    assert re.search(r"useradd|adduser", text), (
        "Dockerfile should create a dedicated non-root user via useradd/adduser"
    )


def test_dockerfile_exposes_port_and_uvicorn_matches():
    text = _dockerfile_text()
    expose = re.search(r"EXPOSE\s+(\d+)", text)
    assert expose, "Dockerfile must EXPOSE a port"
    port = expose.group(1)
    # CMD (or ENTRYPOINT) must bind uvicorn to the same port.
    cmd_block = re.search(r"CMD\s*\[[^\]]+\]", text)
    assert cmd_block, "Dockerfile must have a CMD directive"
    assert port in cmd_block.group(0), (
        f"EXPOSE {port} but CMD does not reference it: {cmd_block.group(0)}"
    )
    assert "uvicorn" in cmd_block.group(0).lower(), (
        "CMD should launch uvicorn"
    )


def test_dockerfile_uses_factory_flag_for_uvicorn():
    """create_app() is a factory — uvicorn must be invoked with --factory
    or the app fails to start (regression guard, learned the hard way)."""
    text = _dockerfile_text()
    cmd_block = re.search(r"CMD\s*\[[^\]]+\]", text)
    assert cmd_block, "CMD directive missing"
    assert "--factory" in cmd_block.group(0), (
        "uvicorn CMD must include --factory since app.py exports create_app()"
    )


def test_dockerfile_does_not_bake_secrets():
    """Defense in depth: no HUGGINGFACE_TOKEN, no AWS keys, no anything that
    smells like a secret hardcoded via ENV."""
    text = _dockerfile_text()
    banned = [
        "HUGGINGFACE_TOKEN=",
        "HF_TOKEN=",
        "AWS_SECRET",
        "AWS_ACCESS_KEY",
        "OPENAI_API_KEY=",
        "SECRET_KEY=",
    ]
    for needle in banned:
        assert needle not in text, (
            f"Dockerfile appears to bake a secret ({needle!r}) — never commit "
            "credentials into an image layer"
        )


# --------------------------------------------------------------------------- #
# .dockerignore contract

def _dockerignore_text() -> str:
    return DOCKERIGNORE.read_text(encoding="utf-8")


def test_dockerignore_excludes_venv():
    text = _dockerignore_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert any(ln.startswith(".venv") for ln in lines), (
        ".dockerignore must exclude .venv/"
    )


def test_dockerignore_excludes_node_modules():
    text = _dockerignore_text()
    assert "node_modules" in text, (
        ".dockerignore must exclude node_modules (frontend build tooling)"
    )


def test_dockerignore_excludes_git_and_tests():
    text = _dockerignore_text()
    assert ".git/" in text or ".git" in text
    assert "tests/" in text or "tests\n" in text


def test_dockerignore_excludes_secret_patterns():
    text = _dockerignore_text()
    for needle in (".env", "*.pem", "*.key", "secrets/"):
        assert needle in text, (
            f".dockerignore should exclude {needle!r} as defense-in-depth"
        )


def test_dockerignore_does_not_exclude_src():
    """Regression guard: `src/` MUST be shipped or `pip install .` breaks."""
    text = _dockerignore_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        assert ln not in ("src", "src/", "src/**"), (
            ".dockerignore MUST NOT exclude src/ — pip install needs it"
        )


def test_dockerignore_does_not_exclude_pyproject():
    text = _dockerignore_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        assert ln != "pyproject.toml", (
            ".dockerignore MUST NOT exclude pyproject.toml"
        )


# --------------------------------------------------------------------------- #
# Runtime dependency guards

def test_frontend_static_bundle_present_in_source_tree():
    """If the runtime env has ONCOLOGY_ARBITER_SERVE_FRONTEND=1 (which the
    Dockerfile sets), the built SPA dist/ must be committed so the image
    can serve it — Docker does not run `npm build`."""
    bundle = REPO_ROOT / "src/oncology_arbiter/api/static/dist/index.html"
    if not bundle.is_file():
        pytest.skip(
            f"frontend bundle not present at {bundle} — this is a Dockerfile "
            "shape guard; skip on branches that predate the frontend merge"
        )
    assert bundle.is_file()


def test_healthcheck_endpoint_exists_in_app():
    """Ensure the /health endpoint referenced by HEALTHCHECK actually exists."""
    from oncology_arbiter.api.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/health" in paths, (
        f"/health must exist in the FastAPI app for the Dockerfile HEALTHCHECK "
        f"to be meaningful. Existing paths: {sorted(paths)[:10]}..."
    )


# --------------------------------------------------------------------------- #
# Path-resolution contract (added after live-deploy gap: /v1/model-cards
# returned [] and /ui/ was 404 because the arbiter package was imported
# from /opt/oa/lib/python3.11/site-packages/ where parents[3] jumps
# outside the repo tree, so docs/ and static/dist/ were unreachable).

def test_runtime_ships_source_tree_to_app_src():
    """
    Runtime stage must COPY src/ to /app/src. Combined with
    PYTHONPATH=/app/src:..., this makes
    Path(__file__).resolve().parents[3] resolve to /app inside the container,
    so _PROJECT_ROOT / 'docs' / 'model_cards' finds real files.
    """
    text = _dockerfile_text()
    pattern = re.compile(
        r"^COPY\s+(?:--chown=[^\s]+\s+)?src\s+/app/src\b",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "Dockerfile must COPY src/ to /app/src (needed so parents[3] from "
        "api/app.py resolves to /app inside the container)"
    )


def test_runtime_ships_docs_to_app_docs():
    """
    Runtime stage must COPY docs/ to /app/docs so /v1/model-cards and
    /v1/artifacts/docs/* have data to return.
    """
    text = _dockerfile_text()
    pattern = re.compile(
        r"^COPY\s+(?:--chown=[^\s]+\s+)?docs\s+/app/docs\b",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "Dockerfile must COPY docs/ to /app/docs — /v1/model-cards is empty "
        "otherwise because _PROJECT_ROOT / 'docs' / 'model_cards' misses."
    )


def test_pythonpath_prefers_app_src():
    """
    PYTHONPATH must place /app/src FIRST. Otherwise `import oncology_arbiter`
    resolves from the pip-installed copy under /opt/oa/lib/... and
    parents[3] jumps outside the repo tree — breaking /v1/model-cards + /ui.
    """
    text = _dockerfile_text()
    matches = re.findall(r"""PYTHONPATH\s*=\s*"?([^"\n]+)"?""", text)
    assert matches, "Dockerfile must set PYTHONPATH"
    for value in matches:
        parts = [p.strip() for p in value.split(":") if p.strip()]
        assert parts, f"PYTHONPATH is empty: {value!r}"
        assert parts[0] == "/app/src", (
            f"PYTHONPATH must start with /app/src (got {parts[0]!r}). "
            f"Full value: {value!r}"
        )


def test_audit_dir_points_to_writable_path():
    """
    audit.py default AUDIT_DIR is `$CWD/artifacts/audit`. Inside the
    container CWD is /app/ (root-owned) and the process runs as non-root
    arbiter (uid 10001), so mkdir fails and /v1/case/full 500s the moment
    it tries to log_event(). The Dockerfile must set
    ONCOLOGY_ARBITER_AUDIT_DIR to a path the non-root user CAN write —
    /tmp is the safe default on ephemeral tiers. Learned live at
    dep-d944aasv5n6c73bsdaqg: PermissionError: [Errno 13] '/app/artifacts'.
    """
    text = _dockerfile_text()
    pattern = re.compile(
        r"^\s*ONCOLOGY_ARBITER_AUDIT_DIR\s*=\s*(?P<path>/[^\s\\]+)",
        re.MULTILINE,
    )
    m = pattern.search(text)
    assert m, (
        "Dockerfile must set ONCOLOGY_ARBITER_AUDIT_DIR to a writable "
        "location, otherwise /v1/case/full and other endpoints that "
        "call audit.log_event() will 500 as non-root uid 10001."
    )
    path = m.group("path")
    # /app is root-owned, /opt/oa is baked read-only — reject those.
    assert not path.startswith("/app"), (
        f"AUDIT_DIR {path!r} is under /app/ which is root-owned; "
        "non-root arbiter user cannot mkdir there. Use /tmp or a "
        "persistent-volume mount point."
    )
    assert not path.startswith("/opt/oa"), (
        f"AUDIT_DIR {path!r} is under /opt/oa which ships the immutable "
        "dep set; do not write user data there."
    )


def test_saas_dependencies_in_pyproject():
    """v0.2 SaaS hardening MUST bring in slowapi + prometheus-fastapi-instrumentator."""
    pyproject = REPO_ROOT / "pyproject.toml"
    content = pyproject.read_text()
    assert "slowapi" in content, (
        "pyproject.toml must declare slowapi so the rate-limit middleware "
        "is available in the runtime image."
    )
    assert "prometheus-fastapi-instrumentator" in content, (
        "pyproject.toml must declare prometheus-fastapi-instrumentator so "
        "the /metrics endpoint is available in the runtime image."
    )


def test_saas_modules_ship_in_source_tree():
    """The auth/ and observability/ packages must exist alongside api/."""
    assert (REPO_ROOT / "src/oncology_arbiter/auth/__init__.py").is_file()
    assert (REPO_ROOT / "src/oncology_arbiter/auth/api_key.py").is_file()
    assert (REPO_ROOT / "src/oncology_arbiter/auth/middleware.py").is_file()
    assert (REPO_ROOT / "src/oncology_arbiter/observability/__init__.py").is_file()
    assert (REPO_ROOT / "src/oncology_arbiter/observability/request_id.py").is_file()
    assert (REPO_ROOT / "src/oncology_arbiter/observability/logging_config.py").is_file()


def test_auth_db_path_writable():
    """Dockerfile ENV must point AUTH_DB_PATH at a writable location."""
    content = (REPO_ROOT / "Dockerfile").read_text()
    m = re.search(r"^\s*ONCOLOGY_ARBITER_AUTH_DB_PATH\s*=\s*(?P<path>/[^\s\\]+)", content, re.MULTILINE)
    assert m, "Dockerfile must declare ONCOLOGY_ARBITER_AUTH_DB_PATH so tenants.sqlite has a writable home"
    path = m.group("path")
    assert not path.startswith("/app"), f"AUTH_DB_PATH {path!r} lands in root-owned /app; use /tmp or a volume"
    assert not path.startswith("/opt/oa"), f"AUTH_DB_PATH {path!r} lands in the immutable dep dir /opt/oa"


def test_auth_mode_default_is_off():
    """Free-tier image must ship auth OFF so the /health probes work without a key.

    Sites that want SaaS auth flip this to `on` via the platform's env-var
    UI; that must NEVER be baked into the image."""
    content = (REPO_ROOT / "Dockerfile").read_text()
    m = re.search(r"^\s*ONCOLOGY_ARBITER_AUTH_MODE\s*=\s*(?P<mode>[a-z]+)", content, re.MULTILINE)
    assert m, "Dockerfile must set ONCOLOGY_ARBITER_AUTH_MODE explicitly"
    assert m.group("mode").lower() == "off", (
        f"Image must ship with AUTH_MODE=off; got {m.group('mode')!r}. "
        "Enable per-deployment via platform env var, not the Dockerfile."
    )
