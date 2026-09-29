from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "oncology_gauntlet", ROOT / "scripts/oncology_gauntlet.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_tonight_delivery",
    ROOT / ".cursor/skills/oncology-moat-platform/validate_tonight_delivery.py",
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def test_proof_only_diff_is_rejected() -> None:
    result = MODULE.check_diff_shape(
        [
            "docs/proofs/another_receipt.json",
            "artifacts/audit/another_log.json",
        ]
    )
    assert result.status == "FAIL"
    assert "complete delivery surface" in result.summary


def test_model_and_wiring_diff_is_accepted() -> None:
    result = MODULE.check_diff_shape(
        [
            "models/cbis_ddsm_logreg_v2.joblib",
            "src/oncology_arbiter/models/cbis_ddsm_probe.py",
            "tests/unit/test_cbis_ddsm_detection.py",
            "artifacts/tonight_delivery.json",
        ]
    )
    assert result.status == "PASS"


def test_unrelated_json_decoy_is_rejected() -> None:
    result = MODULE.check_diff_shape(
        [
            "artifacts/tonight_delivery.json",
            "src/receipt_theater.json",
            "tests/unit/test_receipt_theater.py",
        ]
    )
    assert result.status == "FAIL"
    assert any("trained model artifact" in item for item in result.remediation)


def test_secret_scan_rejects_and_redacts_token() -> None:
    token = "ghp_" + ("A" * 36)
    result = MODULE.check_secrets(f"+GH_TOKEN={token}")
    assert result.status == "FAIL"
    assert token not in "\n".join(result.evidence)
    assert "ghp_" in result.evidence[0]


def test_secret_scan_covers_openai_aws_and_private_keys() -> None:
    diff = "\n".join(
        [
            "+OPENAI_API_KEY=sk-proj-" + ("A" * 30),
            "+AWS_ACCESS_KEY_ID=AKIA" + ("B" * 16),
            "+" + "-----BEGIN " + "PRIVATE KEY-----",
        ]
    )
    result = MODULE.check_secrets(diff)
    assert result.status == "FAIL"
    evidence = "\n".join(result.evidence)
    assert "OpenAI" in evidence
    assert "AWS" in evidence
    assert "Private key" in evidence
    assert "A" * 20 not in evidence


def test_removed_secret_does_not_block_rotation() -> None:
    token = "ghp_" + ("A" * 36)
    result = MODULE.check_secrets(f"-GH_TOKEN={token}\n+GH_TOKEN_FROM_ENV=true")
    assert result.status == "PASS"


def test_skill_lock_rejects_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("owner policy")
    (repo / "locked.txt").symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    lock = repo / "lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "files": {"locked.txt": digest}}))
    result = MODULE.check_skill_lock(repo, "lock.json")
    assert result.status == "FAIL"
    assert "unsafe path locked.txt" in result.evidence


def test_candidate_lock_must_match_trusted_lock(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    trusted = tmp_path / "trusted-lock.json"
    trusted.write_text('{"schema_version":1,"files":{"a":"b"}}\n')
    candidate = repo / "lock.json"
    candidate.write_text('{"schema_version":1,"files":{}}\n')
    result = MODULE.check_candidate_lock(repo, "../trusted-lock.json", "lock.json")
    assert result.status == "FAIL"
    assert "differs" in result.summary


def test_python_startup_and_pytest_shadowing_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "sitecustomize.py").write_text("raise SystemExit(0)\n")
    (repo / "src" / "pytest.py").write_text("def main(*args): return 0\n")
    result = MODULE.check_python_shadowing(repo)
    assert result.status == "FAIL"
    assert sorted(result.evidence) == ["src/pytest.py", "src/sitecustomize.py"]


def test_one_time_bootstrap_requires_absent_base_guardian_and_exact_plane(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "--allow-empty", "-q", "-m", "base",
        ],
        cwd=repo,
        check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    changed = sorted(MODULE.BOOTSTRAP_REQUIRED | {".cursor/skills/INDEX.md"})
    result = MODULE.check_bootstrap(repo, base, changed, True)
    assert result is not None
    assert result.status == "PASS"
    rejected = MODULE.check_bootstrap(repo, base, changed + ["src/decoy.py"], True)
    assert rejected is not None
    assert rejected.status == "FAIL"


def test_validator_rejects_reused_fake_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    fake = repo / "fake.json"
    fake.write_text(json.dumps({"receipt_theater": True}))
    digest = hashlib.sha256(fake.read_bytes()).hexdigest()
    names = [
        "stage-screening", "stage-biopsy", "stage-therapy", "biopsy-probe",
        "cbis-medsiglip", "clinicalbert", "luna16", "phikon", "medgemma",
        "mammo-retinanet",
    ]
    entries = [
        {
            "name": name,
            "status": "trained_wired_tested",
            "artifact_path": "fake.json",
            "artifact_sha256": digest,
            "n_training": 999,
            "n_training_synthetic": False,
            "train_manifest_path": "fake.json",
            "train_manifest_sha256": digest,
            "evaluation_path": "fake.json",
            "evaluation_sha256": digest,
            "wiring_paths": ["fake.json"],
            "test_paths": ["fake.json"],
        }
        for name in names
    ]
    manifest = repo / "artifacts" / "tonight_delivery.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema_version": 2, "capabilities": entries}))
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    validator = ROOT / ".cursor/skills/oncology-moat-platform/validate_tonight_delivery.py"
    result = subprocess.run(
        [sys.executable, str(validator), "--repo", str(repo)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "TONIGHT DELIVERY GATE: FAIL" in result.stderr
    assert "reused" in result.stderr or "suffix is not allowed" in result.stderr
    assert "dataset manifest" in result.stderr


def test_runtime_verifier_fails_cleanly_without_manifest(tmp_path: Path) -> None:
    verifier = ROOT / "scripts/oncology_runtime_verifier.py"
    result = subprocess.run(
        [sys.executable, str(verifier), "--repo", str(tmp_path), "--run-tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "RUNTIME_VERIFIER_FAIL" in result.stderr
    assert "Traceback" not in result.stderr


def test_neural_capabilities_accept_only_safetensors() -> None:
    for name in ("clinicalbert", "luna16", "mammo-retinanet"):
        assert VALIDATOR.CAPABILITY_SPECS[name]["suffixes"] == {".safetensors"}


def test_malformed_safetensors_metadata_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "fake.safetensors"
    header = json.dumps(
        {
            "first": {
                "dtype": "NOT_A_DTYPE",
                "shape": ["not-an-int"],
                "data_offsets": [0, 50_000],
            },
            "second": {
                "dtype": "F32",
                "shape": [12_500],
                "data_offsets": [0, 50_000],
            },
        }
    ).encode()
    artifact.write_bytes(len(header).to_bytes(8, "little") + header + (b"\0" * 100_001))
    errors: list[str] = []
    VALIDATOR.validate_binary_container("luna16", artifact, errors)
    assert errors
    assert "invalid safetensors" in errors[0]


def test_structurally_valid_safetensors_is_accepted(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    header = json.dumps(
        {
            "first": {
                "dtype": "F32",
                "shape": [12_500],
                "data_offsets": [0, 50_000],
            },
            "second": {
                "dtype": "F32",
                "shape": [12_500],
                "data_offsets": [50_000, 100_000],
            },
        }
    ).encode()
    artifact.write_bytes(len(header).to_bytes(8, "little") + header + (b"\0" * 100_000))
    errors: list[str] = []
    VALIDATOR.validate_binary_container("luna16", artifact, errors)
    assert errors == []


def test_biopsy_floor_cannot_be_satisfied_with_substitute_metric() -> None:
    assert VALIDATOR.CAPABILITY_SPECS["biopsy-probe"]["metrics"] == {"auroc"}


def test_luna_delta_froc_uses_candidate_minus_baseline() -> None:
    rows = [
        {"baseline_sensitivity": 0.70, "candidate_sensitivity": 0.71},
        {"baseline_sensitivity": 0.68, "candidate_sensitivity": 0.69},
    ]
    value = VALIDATOR.recompute_metric("delta_froc_at_2", rows)
    assert abs(value - 0.01) < 1e-12
    assert value != 0.70


def test_markdown_has_stable_bot_marker() -> None:
    payload = {
        "verdict": "REJECT",
        "base": "a" * 40,
        "head": "b" * 40,
        "checks": [
            {
                "id": "delivery-contract",
                "status": "FAIL",
                "summary": "Missing trained artifact.",
                "evidence": ["biopsy-probe"],
                "remediation": ["Train, wire, test, and push it."],
            }
        ],
    }
    rendered = MODULE.report_markdown(payload)
    assert rendered.startswith("<!-- oncology-gauntlet-report -->")
    assert "Train, wire, test, and push it." in rendered


def test_checked_in_skill_lock_matches() -> None:
    result = MODULE.check_skill_lock(
        ROOT, ".github/oncology-gauntlet/skill-lock.json"
    )
    assert result.status == "PASS", result.evidence
