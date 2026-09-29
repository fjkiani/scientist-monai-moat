#!/usr/bin/env python3
"""Deterministic oncology delivery gauntlet with actionable JSON/Markdown output."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROOF_PREFIXES = (
    "artifacts/",
    "docs/",
    "tests/fixtures/",
    ".cursor/",
)
DELIVERY_PREFIXES = (
    "models/",
    "src/",
    "deploy/",
    "scripts/",
)
MODEL_SUFFIXES = (".joblib", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin")
LOCK_DEFAULT = ".github/oncology-gauntlet/skill-lock.json"
VALIDATOR_DEFAULT = (
    ".cursor/skills/oncology-moat-platform/validate_tonight_delivery.py"
)
BOOTSTRAP_REQUIRED = {
    ".github/workflows/oncology-gauntlet.yml",
    ".github/workflows/oncology-guardian.yml",
    ".github/oncology-gauntlet/skill-lock.json",
    ".github/oncology-gauntlet/Dockerfile",
    ".github/oncology-gauntlet/requirements.lock",
    "scripts/oncology_gauntlet.py",
    "scripts/oncology_runtime_verifier.py",
    "scripts/oncology_trusted_test_runner.py",
}
BOOTSTRAP_PREFIXES = (
    ".cursor/skills/",
    ".github/CODEOWNERS",
    ".github/oncology-gauntlet/",
    ".github/workflows/oncology-gauntlet.yml",
    ".github/workflows/oncology-guardian.yml",
    "scripts/oncology_",
    "tests/unit/test_oncology_gauntlet.py",
)
SECRET_PATTERNS = {
    "GitHub classic PAT": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained PAT": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "GitHub application token": re.compile(r"gh[osrup]_[A-Za-z0-9]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "OpenAI API key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "Assigned secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b"
        r"[ \t]*[:=][ \t]*[\"']?[A-Za-z0-9_./+=-]{16,}"
    ),
}


@dataclass
class Check:
    id: str
    status: str
    summary: str
    evidence: list[str]
    remediation: list[str]


def run(
    repo: Path, command: list[str], *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=repo, capture_output=True, text=True
    )
    if check and result.returncode:
        raise RuntimeError(
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_comparison_range(repo: Path, base: str, head: str) -> Check:
    errors: list[str] = []
    for label, ref in (("base", base), ("head", head)):
        result = run(repo, ["git", "cat-file", "-e", f"{ref}^{{commit}}"])
        if result.returncode:
            errors.append(f"{label} commit is unavailable: {ref}")
    if not errors:
        ancestor = run(repo, ["git", "merge-base", "--is-ancestor", base, head])
        if ancestor.returncode:
            errors.append("base is not an ancestor of head")
        if base == head:
            errors.append("base and head are identical")
    return Check(
        "comparison-range",
        "FAIL" if errors else "PASS",
        "Comparison range is invalid." if errors else "Comparison range is complete and ordered.",
        errors or [f"{base}..{head}"],
        ["Fetch full history and compare the protected base to the exact candidate head."] if errors else [],
    )


def git_changed_files(repo: Path, base: str, head: str) -> list[str]:
    result = run(repo, ["git", "diff", "--name-only", "--diff-filter=ACMRT", base, head], check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def git_diff(repo: Path, base: str, head: str) -> str:
    result = run(repo, ["git", "diff", "--no-ext-diff", "--no-textconv", "-U0", base, head], check=True)
    return result.stdout


def check_skill_lock(repo: Path, lock_path: str) -> Check:
    path = (repo / lock_path).resolve()
    if not path.is_file() or path.is_symlink():
        return Check(
            "skill-lock",
            "FAIL",
            "Skill lock file is missing.",
            [lock_path],
            ["Restore the owner-approved skill lock and rerun."],
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return Check(
            "skill-lock",
            "FAIL",
            "Skill lock file is invalid.",
            [type(exc).__name__],
            ["Restore the owner-approved skill lock and rerun."],
        )
    errors: list[str] = []
    evidence: list[str] = []
    files = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        errors.append("lock schema/files are invalid")
        files = {}
    for rel, expected in sorted(files.items()):
        candidate = repo / rel
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            errors.append(f"missing {rel}")
            continue
        if candidate.is_symlink() or resolved.is_symlink() or (
            repo != resolved and repo not in resolved.parents
        ):
            errors.append(f"unsafe path {rel}")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            errors.append(f"invalid expected hash {rel}")
            continue
        actual = file_sha256(resolved)
        evidence.append(f"{rel}: {actual}")
        if actual != expected:
            errors.append(f"hash mismatch {rel}")
    if errors:
        return Check(
            "skill-lock",
            "FAIL",
            "Owner-approved gauntlet/skills were changed.",
            errors,
            [
                "Do not weaken enforcement files.",
                "Ask Alpha to review and regenerate the lock for legitimate changes.",
            ],
        )
    return Check(
        "skill-lock",
        "PASS",
        "All locked enforcement files match.",
        evidence,
        [],
    )


def check_candidate_lock(repo: Path, trusted_lock_path: str, candidate_lock_path: str) -> Check:
    trusted = (repo / trusted_lock_path).resolve()
    candidate_raw = repo / candidate_lock_path
    candidate = candidate_raw.resolve()
    if not trusted.is_file() or not candidate.is_file():
        return Check(
            "lock-identity",
            "FAIL",
            "Trusted or candidate lock is missing.",
            [trusted_lock_path, candidate_lock_path],
            ["Restore the protected-base lock byte-for-byte."],
        )
    if candidate_raw.is_symlink():
        return Check(
            "lock-identity",
            "FAIL",
            "Candidate lock must not be a symlink.",
            [candidate_lock_path],
            ["Restore the protected-base lock byte-for-byte."],
        )
    if trusted.read_bytes() != candidate.read_bytes():
        return Check(
            "lock-identity",
            "FAIL",
            "Candidate lock differs from the protected-base lock.",
            [f"candidate sha256: {file_sha256(candidate)}", f"trusted sha256: {file_sha256(trusted)}"],
            ["Do not regenerate or weaken the enforcement lock in a capability PR."],
        )
    return Check(
        "lock-identity",
        "PASS",
        "Candidate lock matches protected-base lock byte-for-byte.",
        [f"sha256: {file_sha256(candidate)}"],
        [],
    )


def check_python_shadowing(repo: Path) -> Check:
    forbidden = []
    for path in repo.rglob("*.py"):
        try:
            relative = path.relative_to(repo)
        except ValueError:
            continue
        if ".git" in relative.parts:
            continue
        if path.name in {"sitecustomize.py", "usercustomize.py", "pytest.py"}:
            forbidden.append(str(relative))
    if forbidden:
        return Check(
            "python-shadowing",
            "FAIL",
            "Candidate contains Python startup/test-runner shadow modules.",
            sorted(forbidden),
            ["Remove modules that can execute before protected-base tests load."],
        )
    return Check(
        "python-shadowing",
        "PASS",
        "No Python startup/test-runner shadow modules are present.",
        [],
        [],
    )


def check_bootstrap(repo: Path, base: str, changed: list[str], requested: bool) -> Check | None:
    if not requested:
        return None
    guardian_at_base = run(
        repo,
        ["git", "cat-file", "-e", f"{base}:.github/workflows/oncology-guardian.yml"],
    )
    errors: list[str] = []
    if guardian_at_base.returncode == 0:
        errors.append("protected base already contains the immutable guardian")
    missing = sorted(BOOTSTRAP_REQUIRED - set(changed))
    if missing:
        errors.append(f"bootstrap is missing required files: {missing}")
    def allowed(path: str) -> bool:
        return any(
            path.startswith(prefix)
            if prefix.endswith(("/", "_"))
            else path == prefix
            for prefix in BOOTSTRAP_PREFIXES
        )

    disallowed = sorted(path for path in changed if not allowed(path))
    if disallowed:
        errors.append(f"bootstrap contains product/proof files: {disallowed}")
    return Check(
        "one-time-bootstrap",
        "FAIL" if errors else "PASS",
        "One-time enforcement bootstrap is invalid."
        if errors
        else "Protected base has no guardian; this diff contains only the complete enforcement plane.",
        errors or sorted(changed),
        [
            "Remove product/proof changes and install the complete enforcement plane in one commit."
        ]
        if errors
        else [],
    )


def check_diff_shape(changed: list[str]) -> Check:
    delivery = [path for path in changed if path.startswith(DELIVERY_PREFIXES)]
    proof = [path for path in changed if path.startswith(PROOF_PREFIXES)]
    models = [
        path for path in changed
        if (
            path.startswith("models/")
            or (path.startswith("src/") and "/models/" in path)
        )
        and (
            path.endswith(MODEL_SUFFIXES)
            or (path.endswith(".json") and "template" not in Path(path).name.lower())
        )
    ]
    implementation = [
        path for path in changed
        if path.startswith(("src/", "deploy/", "scripts/"))
        and path.endswith(".py")
        and not path.startswith("scripts/oncology_")
    ]
    tests = [path for path in changed if path.startswith("tests/") and path.endswith(".py")]
    manifest = [path for path in changed if path == "artifacts/tonight_delivery.json"]
    missing_surfaces = [
        label
        for label, files in (
            ("trained model artifact", models),
            ("product/training implementation", implementation),
            ("executable regression test", tests),
            ("artifacts/tonight_delivery.json", manifest),
        )
        if not files
    ]
    if proof and missing_surfaces:
        return Check(
            "diff-shape",
            "FAIL",
            "Proof/receipt changes are not accompanied by the complete delivery surface.",
            changed,
            [
                f"Add {surface}." for surface in missing_surfaces
            ],
        )
    if missing_surfaces:
        return Check(
            "diff-shape",
            "FAIL",
            "Push does not contain a complete model delivery.",
            changed or ["No changed files resolved from the exact event range."],
            [f"Add {surface}." for surface in missing_surfaces],
        )
    return Check(
        "diff-shape",
        "PASS",
        "Diff contains model, implementation, tests, and delivery contract.",
        models + implementation + tests + manifest,
        [],
    )


def check_delivery_contract(repo: Path, validator_path: str) -> Check:
    validator = repo / validator_path
    if not validator.is_file():
        return Check(
            "delivery-contract",
            "FAIL",
            "Delivery validator is missing.",
            [validator_path],
            ["Restore the locked validator."],
        )
    result = run(
        repo, [sys.executable, str(validator), "--repo", str(repo)]
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    if result.returncode:
        remediation = [
            line[2:] for line in lines if line.startswith("- ")
        ]
        if not remediation:
            remediation = [
                "Create every trained, wired, tested capability artifact required by the validator."
            ]
        return Check(
            "delivery-contract",
            "FAIL",
            "Ten-capability delivery contract is incomplete.",
            lines[:50],
            remediation[:50],
        )
    return Check(
        "delivery-contract",
        "PASS",
        "All ten capabilities are trained, wired, and tested.",
        lines,
        [],
    )


def added_lines(diff_text: str) -> str:
    return "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def check_secrets(diff_text: str) -> Check:
    matches: list[str] = []
    additions = added_lines(diff_text)
    for label, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(additions):
            token = match.group(0)
            safe_prefix = html.escape(token[:4])
            matches.append(
                f"{label}: {safe_prefix}… (fully redacted)"
            )
    if matches:
        return Check(
            "secret-scan",
            "FAIL",
            "Credential-shaped values were added.",
            matches,
            ["Remove and rotate every exposed credential before pushing again."],
        )
    return Check(
        "secret-scan",
        "PASS",
        "No targeted credential patterns were added.",
        [],
        [],
    )


def report_markdown(payload: dict) -> str:
    def safe(value: object) -> str:
        text = str(value).replace("@", "@\u200b")
        text = "".join(char if char.isprintable() else "�" for char in text)
        return html.escape(text, quote=False).replace("`", "&#96;")

    lines = [
        "<!-- oncology-gauntlet-report -->",
        "# Oncology Delivery Gauntlet",
        "",
        f"**Verdict: {payload['verdict']}**",
        f"`{payload['base'][:12]}…{payload['head'][:12]}`",
        "",
    ]
    for check in payload["checks"]:
        icon = "✅" if check["status"] == "PASS" else "❌"
        lines.extend(
            [
                f"## {icon} {safe(check['id'])}: {safe(check['status'])}",
                safe(check["summary"]),
                "",
            ]
        )
        if check["evidence"]:
            lines.append("Evidence:")
            lines.extend(f"- `{safe(item)}`" for item in check["evidence"][:50])
            lines.append("")
        if check["remediation"]:
            lines.append("Required correction:")
            lines.extend(
                f"{index}. {safe(item)}"
                for index, item in enumerate(check["remediation"][:50], 1)
            )
            lines.append("")
    lines.extend(
        [
            "## Acceptance law",
            "The LLM review is advisory. It cannot override a deterministic FAIL.",
            "Resubmit only after every failing row is corrected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--lock", default=LOCK_DEFAULT)
    parser.add_argument("--candidate-lock", default=LOCK_DEFAULT)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--validator", default=VALIDATOR_DEFAULT)
    parser.add_argument(
        "--report-json", default="artifacts/gauntlet/report.json"
    )
    parser.add_argument(
        "--report-md", default="artifacts/gauntlet/report.md"
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    range_check = check_comparison_range(repo, args.base, args.head)
    if range_check.status == "PASS":
        try:
            changed = git_changed_files(repo, args.base, args.head)
            diff_text = git_diff(repo, args.base, args.head)
        except RuntimeError as exc:
            changed = []
            diff_text = ""
            range_check = Check(
                "comparison-range",
                "FAIL",
                "The exact candidate diff could not be resolved.",
                [str(exc)[:2000]],
                ["Fetch full history and rerun against the exact base/head SHAs."],
            )
    else:
        changed = []
        diff_text = ""
    bootstrap_check = check_bootstrap(repo, args.base, changed, args.bootstrap)
    if bootstrap_check is not None and bootstrap_check.status == "PASS":
        diff_check = bootstrap_check
        delivery_check = Check(
            "delivery-contract",
            "PASS",
            "One-time enforcement bootstrap does not claim oncology model delivery.",
            ["The next candidate PR must satisfy all ten capability contracts."],
            [],
        )
    else:
        diff_check = bootstrap_check or check_diff_shape(changed)
        delivery_check = check_delivery_contract(repo, args.validator)
    checks = [
        range_check,
        check_candidate_lock(repo, args.lock, args.candidate_lock),
        check_skill_lock(repo, args.lock),
        check_python_shadowing(repo),
        diff_check,
        delivery_check,
        check_secrets(diff_text),
    ]
    verdict = (
        "PASS" if all(item.status == "PASS" for item in checks) else "REJECT"
    )
    payload = {
        "schema_version": 1,
        "verdict": verdict,
        "base": args.base,
        "head": args.head,
        "changed_files": changed,
        "checks": [asdict(item) for item in checks],
    }
    json_path = repo / args.report_json
    md_path = repo / args.report_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(report_markdown(payload) + "\n")
    print(md_path.read_text())
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
