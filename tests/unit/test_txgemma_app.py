"""AST-level tests for the TxGemma-9B Modal app.

Verifies the deployment shape without requiring the ``modal`` package:

- App declares GPU + HF secret + memory floor.
- Class has ``@modal.enter()`` for weight loading + ``/reason`` endpoint.
- Honesty warning constant is present and non-trivial.
- App version bumps to v0.4.0-alpha.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


TXGEMMA_APP = Path(__file__).resolve().parents[2] / "deploy" / "modal" / "txgemma_app.py"


def _parse():
    return ast.parse(TXGEMMA_APP.read_text(), filename=str(TXGEMMA_APP))


def _find(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    pytest.fail(f"{name} not found in txgemma_app.py")


def _class_decorator_kwargs(cls):
    """Return kwargs of the @app.cls(...) decorator on a ClassDef."""
    for d in cls.decorator_list:
        if isinstance(d, ast.Call):
            fn = d.func
            if isinstance(fn, ast.Attribute) and fn.attr == "cls":
                return {kw.arg: kw.value for kw in d.keywords if kw.arg}
    return {}


class TestModuleShape:
    def test_module_parses(self):
        assert TXGEMMA_APP.exists()
        _parse()

    def test_app_version_is_v0_4_0(self):
        text = TXGEMMA_APP.read_text()
        assert "APP_VERSION = " in text
        assert "v0.4.0-alpha" in text

    def test_txgemma_repo_used(self):
        text = TXGEMMA_APP.read_text()
        # We use the chat variant as the primary
        assert 'google/txgemma-9b-chat' in text

    def test_honesty_warning_present(self):
        text = TXGEMMA_APP.read_text()
        assert "TXGEMMA_HONESTY_WARNING" in text
        assert "RESEARCH USE ONLY" in text

    def test_hf_secret_wired(self):
        text = TXGEMMA_APP.read_text()
        assert 'from_name("txgemma-hf-token")' in text


class TestClassAndEndpoints:
    def test_txgemma_class_exists(self):
        tree = _parse()
        cls = _find(tree, "TxGemma")
        assert cls is not None

    def test_class_has_gpu_and_secret(self):
        tree = _parse()
        cls = _find(tree, "TxGemma")
        kwargs = _class_decorator_kwargs(cls)
        assert "gpu" in kwargs, "TxGemma class must specify a GPU"
        assert "secrets" in kwargs, "TxGemma class must attach the HF secret"
        assert "memory" in kwargs, "TxGemma class must declare a memory floor"

    def test_load_method_uses_enter_decorator(self):
        text = TXGEMMA_APP.read_text()
        # @modal.enter() then def load
        assert "@modal.enter()" in text

    def test_reason_endpoint_declared(self):
        text = TXGEMMA_APP.read_text()
        assert 'method="POST"' in text
        assert 'label="txgemma-reason"' in text

    def test_info_endpoint_declared(self):
        text = TXGEMMA_APP.read_text()
        assert 'label="txgemma-info"' in text

    def test_healthz_endpoint_declared(self):
        text = TXGEMMA_APP.read_text()
        assert 'label="txgemma-healthz"' in text


class TestReasonSemantics:
    def test_prompt_required(self):
        text = TXGEMMA_APP.read_text()
        # Enforces prompt is a non-empty string
        assert "prompt must be a non-empty string" in text

    def test_deterministic_default(self):
        text = TXGEMMA_APP.read_text()
        # temperature default 0.0 → deterministic generation
        assert 'temperature", 0.0' in text or "temperature=0.0" in text

    def test_honesty_attached_to_result(self):
        text = TXGEMMA_APP.read_text()
        # /reason response always includes the honesty warning
        assert '"honesty_warning": TXGEMMA_HONESTY_WARNING' in text
