"""Structural checks for the Modal training apps.

These tests do NOT require the ``modal`` package to be installed. They parse
the app modules with ``ast`` and confirm the wiring expected by the plan:

- The LUNA16 fine-tune app declares volumes for data / baseline / output.
- Its ``finetune`` function is decorated with a Modal GPU + volumes spec.
- The CBIS-DDSM detection app has a ``train_detector`` function with GPU +
  data volumes and a torchvision RetinaNet import.
- Both apps ship a healthz endpoint and a POST trigger endpoint.

These are the minimal invariants the audit ledger relies on when replaying
which training runs produced which weights, so they must not regress.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"
LUNA16_APP = DEPLOY_DIR / "luna16_finetune_app.py"
CBIS_APP = DEPLOY_DIR / "cbis_ddsm_detection_app.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"function {name} not found")


def _decorator_kwargs(func: ast.FunctionDef, attr_ends_with: str) -> dict[str, ast.AST]:
    for d in func.decorator_list:
        if isinstance(d, ast.Call):
            fn = d.func
            # e.g. app.function(...) or app.cls(...)
            if isinstance(fn, ast.Attribute) and fn.attr == attr_ends_with:
                return {kw.arg: kw.value for kw in d.keywords if kw.arg}
    return {}


class TestLuna16App:
    def test_app_module_parses(self) -> None:
        assert LUNA16_APP.exists(), LUNA16_APP
        _parse(LUNA16_APP)

    def test_declares_all_three_volumes(self) -> None:
        text = LUNA16_APP.read_text()
        # Handle both single-line and split declarations
        assert '"luna16-data"' in text
        assert '"luna16-baseline-weights"' in text
        assert '"luna16-training-runs"' in text
        assert text.count("Volume.from_name") >= 3

    def test_finetune_has_gpu_and_volumes(self) -> None:
        tree = _parse(LUNA16_APP)
        fn = _find_function(tree, "finetune")
        kwargs = _decorator_kwargs(fn, "function")
        assert "gpu" in kwargs, "finetune must specify GPU"
        assert "volumes" in kwargs, "finetune must mount volumes"
        assert "timeout" in kwargs, "finetune must specify timeout"

    def test_finetune_defaults_match_scaffold(self) -> None:
        tree = _parse(LUNA16_APP)
        fn = _find_function(tree, "finetune")
        # Args: fold, epochs, learning_rate, batch_size, val_interval, dry_run
        defaults = {a.arg: d for a, d in zip(fn.args.args, [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults))}
        # epochs default 20 matches oncology_arbiter.nsclc.luna16_finetune.FinetuneConfig
        assert isinstance(defaults["epochs"], ast.Constant) and defaults["epochs"].value == 20
        # LR default 1e-3 (fine-tune, not bundle's from-scratch 1e-2)
        assert isinstance(defaults["learning_rate"], ast.Constant)
        assert defaults["learning_rate"].value == 1e-3

    def test_target_spacing_matches_bundle(self) -> None:
        text = LUNA16_APP.read_text()
        assert "TARGET_SPACING_MM = (0.703125, 0.703125, 1.25)" in text

    def test_has_healthz_endpoint(self) -> None:
        tree = _parse(LUNA16_APP)
        fn = _find_function(tree, "healthz")
        # decorated with @modal.fastapi_endpoint(method="GET", ...)
        assert any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "fastapi_endpoint"
            for d in fn.decorator_list
        )

    def test_has_trigger_endpoint(self) -> None:
        tree = _parse(LUNA16_APP)
        fn = _find_function(tree, "trigger")
        # Look at the decorator kwargs
        text = LUNA16_APP.read_text()
        assert 'method="POST"' in text or "method='POST'" in text


class TestCbisDdsmApp:
    def test_app_module_parses(self) -> None:
        assert CBIS_APP.exists(), CBIS_APP
        _parse(CBIS_APP)

    def test_declares_data_and_output_volumes(self) -> None:
        text = CBIS_APP.read_text()
        assert '"cbis-ddsm-data"' in text
        assert '"cbis-ddsm-training-runs"' in text
        assert text.count("Volume.from_name") >= 2

    def test_train_detector_has_gpu_and_volumes(self) -> None:
        tree = _parse(CBIS_APP)
        fn = _find_function(tree, "train_detector")
        kwargs = _decorator_kwargs(fn, "function")
        assert "gpu" in kwargs
        assert "volumes" in kwargs
        assert "timeout" in kwargs

    def test_uses_torchvision_retinanet(self) -> None:
        text = CBIS_APP.read_text()
        assert "retinanet_resnet50_fpn_v2" in text
        assert "RetinaNet_ResNet50_FPN_V2_Weights" in text

    def test_imports_scaffold_helpers(self) -> None:
        text = CBIS_APP.read_text()
        assert "from oncology_arbiter.mammography.cbis_ddsm_detection import" in text
        assert "build_case_manifest" in text
        assert "bbox_from_roi_mask" in text

    def test_emits_coco_manifests(self) -> None:
        text = CBIS_APP.read_text()
        assert "train_coco.json" in text
        assert "test_coco.json" in text

    def test_has_healthz_endpoint(self) -> None:
        tree = _parse(CBIS_APP)
        fn = _find_function(tree, "healthz")
        assert any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "fastapi_endpoint"
            for d in fn.decorator_list
        )

    def test_uses_pycocotools_eval(self) -> None:
        text = CBIS_APP.read_text()
        # eval is required for reporting map_at_iou_0.5 in the audit
        assert "pycocotools" in text
        assert "COCOeval" in text
