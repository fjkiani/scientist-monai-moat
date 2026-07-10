"""Unit tests for `oncology_arbiter.nsclc.luna16_finetune`.

These tests do NOT require the 66 GB LUNA16 corpus. They exercise:
    * FinetuneConfig defaults + overrides.
    * `run_finetune(dry_run=True)` command construction.
    * `write_refine_metrics` JSON schema.
    * `unpack_zenodo_subsets` on synthetic zip fixtures.

Real-data integration tests live in `tests/integration/luna16_train.py` and
are gated on `OA_LUNA16_DATASET_DIR` env var pointing to a resampled corpus.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from oncology_arbiter.nsclc.luna16_finetune import (
    FinetuneConfig,
    TARGET_SPACING_MM,
    run_finetune,
    unpack_zenodo_subsets,
    write_refine_metrics,
)


class TestFinetuneConfig:
    def test_defaults(self, tmp_path):
        cfg = FinetuneConfig(
            bundle_root=tmp_path / "bundle",
            dataset_dir=tmp_path / "dataset",
            datasplit_json=tmp_path / "fold0.json",
            output_dir=tmp_path / "out",
        )
        assert cfg.epochs == 20
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 4
        assert cfg.val_interval == 5
        assert cfg.initial_weights is None
        assert cfg.extra_overrides == {}

    def test_target_spacing_constant(self):
        assert TARGET_SPACING_MM == (0.703125, 0.703125, 1.25)


class TestRunFinetuneDryRun:
    def test_dry_run_returns_cmd(self, tmp_path):
        # Create a fake bundle root with a model.pt so the pre-check passes
        bundle = tmp_path / "bundle"
        (bundle / "models").mkdir(parents=True)
        (bundle / "models" / "model.pt").write_bytes(b"stub")
        (bundle / "configs").mkdir()
        (bundle / "configs" / "train.json").write_text("{}")

        cfg = FinetuneConfig(
            bundle_root=bundle,
            dataset_dir=tmp_path / "dataset",
            datasplit_json=tmp_path / "fold0.json",
            output_dir=tmp_path / "out",
            epochs=5,
            learning_rate=5e-4,
            batch_size=2,
        )

        result = run_finetune(cfg, dry_run=True)
        assert result["returncode"] == 0
        cmd = result["cmd"]
        # sys.executable should be in the cmd
        assert cmd[0] == sys.executable
        assert "monai.bundle" in cmd
        assert "training" in cmd
        # Overrides present
        assert "5" in cmd  # epochs
        assert "0.0005" in cmd  # learning_rate
        assert "2" in cmd  # batch_size
        # Output dir was created
        assert (tmp_path / "out").exists()

    def test_missing_weights_raises(self, tmp_path):
        # No models/model.pt exists -> should raise on run_finetune
        bundle = tmp_path / "bundle"
        (bundle / "configs").mkdir(parents=True)
        (bundle / "configs" / "train.json").write_text("{}")

        cfg = FinetuneConfig(
            bundle_root=bundle,
            dataset_dir=tmp_path / "dataset",
            datasplit_json=tmp_path / "fold0.json",
            output_dir=tmp_path / "out",
        )
        with pytest.raises(FileNotFoundError, match="initial_weights"):
            run_finetune(cfg, dry_run=True)

    def test_extra_overrides_included(self, tmp_path):
        bundle = tmp_path / "bundle"
        (bundle / "models").mkdir(parents=True)
        (bundle / "models" / "model.pt").write_bytes(b"stub")
        (bundle / "configs").mkdir()
        (bundle / "configs" / "train.json").write_text("{}")

        cfg = FinetuneConfig(
            bundle_root=bundle,
            dataset_dir=tmp_path / "dataset",
            datasplit_json=tmp_path / "fold0.json",
            output_dir=tmp_path / "out",
            extra_overrides={"seed": 42, "num_workers": 2},
        )
        result = run_finetune(cfg, dry_run=True)
        cmd = result["cmd"]
        assert "--seed" in cmd
        assert "42" in cmd
        assert "--num_workers" in cmd
        assert "2" in cmd


class TestUnpackZenodoSubsets:
    def _make_fake_subset_zip(self, out_path: Path, uids: list[str]) -> None:
        """Create a fake subset zip containing .mhd + .raw pairs for each uid."""
        with zipfile.ZipFile(out_path, "w") as zf:
            for uid in uids:
                zf.writestr(f"{uid}.mhd", f"stub mhd for {uid}\n")
                zf.writestr(f"{uid}.raw", b"\x00" * 1024)  # 1KB raw payload

    def test_unpacks_single_subset(self, tmp_path):
        zip_dir = tmp_path / "zips"
        zip_dir.mkdir()
        uids = [
            "1.3.6.1.4.1.14519.5.2.1.6279.6001.001",
            "1.3.6.1.4.1.14519.5.2.1.6279.6001.002",
        ]
        self._make_fake_subset_zip(zip_dir / "subset0.zip", uids)

        out = tmp_path / "unpacked"
        uid_to_mhd = unpack_zenodo_subsets(zip_dir, out, subsets=[0])
        assert len(uid_to_mhd) == 2
        for uid in uids:
            assert uid in uid_to_mhd
            assert uid_to_mhd[uid].exists()
            assert uid_to_mhd[uid].name == f"{uid}.mhd"

    def test_missing_subset_raises(self, tmp_path):
        zip_dir = tmp_path / "zips"
        zip_dir.mkdir()
        out = tmp_path / "unpacked"
        with pytest.raises(FileNotFoundError):
            unpack_zenodo_subsets(zip_dir, out, subsets=[0])

    def test_auto_detects_all_subsets_when_none(self, tmp_path):
        zip_dir = tmp_path / "zips"
        zip_dir.mkdir()
        self._make_fake_subset_zip(zip_dir / "subset0.zip", ["uid-a"])
        self._make_fake_subset_zip(zip_dir / "subset1.zip", ["uid-b"])
        out = tmp_path / "unpacked"
        uid_to_mhd = unpack_zenodo_subsets(zip_dir, out)
        assert set(uid_to_mhd.keys()) == {"uid-a", "uid-b"}


class TestWriteRefineMetrics:
    def test_delta_computation(self, tmp_path):
        baseline = {"froc_at_2fps": 0.75, "map_iou0.1": 0.80}
        refined = {"froc_at_2fps": 0.81, "map_iou0.1": 0.84}

        out = tmp_path / "metrics.json"
        write_refine_metrics(baseline, refined, out, fold_index=9,
                             n_train_series=534, n_val_series=67)

        with open(out) as f:
            d = json.load(f)

        assert d["schema_version"] == "v0.4.0-alpha"
        assert d["fold_index"] == 9
        assert d["n_train_series"] == 534
        assert d["n_val_series"] == 67
        assert d["baseline"] == baseline
        assert d["refined"] == refined
        assert abs(d["delta"]["froc_at_2fps"] - 0.06) < 1e-9
        assert abs(d["delta"]["map_iou0.1"] - 0.04) < 1e-9
        assert d["plan_target"]["froc_at_2fps_delta_min"] == 0.05
        assert d["plan_target"]["met"] is True

    def test_delta_below_target(self, tmp_path):
        baseline = {"froc_at_2fps": 0.75, "map_iou0.1": 0.80}
        refined = {"froc_at_2fps": 0.78, "map_iou0.1": 0.81}   # only +3%

        out = tmp_path / "metrics.json"
        write_refine_metrics(baseline, refined, out, fold_index=9,
                             n_train_series=534, n_val_series=67)

        with open(out) as f:
            d = json.load(f)

        assert d["plan_target"]["met"] is False
        assert d["delta"]["froc_at_2fps"] < 0.05

    def test_missing_metric_defaults_to_zero(self, tmp_path):
        # If a metric is absent, delta should still compute (against 0.0)
        baseline = {"map_iou0.1": 0.80}   # no froc_at_2fps
        refined = {"froc_at_2fps": 0.78, "map_iou0.1": 0.82}

        out = tmp_path / "metrics.json"
        write_refine_metrics(baseline, refined, out, fold_index=9,
                             n_train_series=534, n_val_series=67)

        with open(out) as f:
            d = json.load(f)

        assert d["delta"]["froc_at_2fps"] == 0.78  # 0.78 - 0.0
        assert d["plan_target"]["met"] is True     # 0.78 >= 0.05
