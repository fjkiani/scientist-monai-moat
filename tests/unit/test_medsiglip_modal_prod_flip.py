"""Structural tests for the MedSigLIP Modal app 'prod flip' knob.

The prod flip lets the same ``medsiglip_app.py`` deploy under either
``MEDSIGLIP_MODAL_MODE=staging`` (default, ``min_containers=0``) or
``MEDSIGLIP_MODAL_MODE=prod`` (``min_containers=1``, longer scaledown).
Toggling this at deploy time is how v0.4.0 promotes the endpoint from
'cold-start OK' to 'warm during clinic hours'.

These tests only parse the app module — they don't require the ``modal``
package. That keeps CI green on a plain runner.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


DEPLOY_APP = Path(__file__).resolve().parents[2] / "deploy" / "modal" / "medsiglip_app.py"


def _reload_app(env: dict[str, str]):
    """Import the app module fresh under a controlled environment.

    We drop the ``modal`` package into ``sys.modules`` as a stub so the
    ``import modal`` at the top of the app doesn't require the real client.
    """
    stub = _make_modal_stub()
    with patch.dict(os.environ, env, clear=False), patch.dict(
        sys.modules, {"modal": stub}, clear=False
    ):
        # Load module from source into a private namespace under a stable
        # name so re-imports don't cross-contaminate.
        spec = importlib.util.spec_from_file_location(
            "medsiglip_app_under_test", str(DEPLOY_APP)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa: E501
    return mod


def _make_modal_stub():
    """Minimal shim so ``import modal`` works without the real package."""
    import types

    stub = types.ModuleType("modal")

    class _Image:
        @staticmethod
        def debian_slim(**kwargs):
            return _Image()

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

    stub.Image = _Image

    class _App:
        def __init__(self, name):
            self.name = name
            self.fn_specs = {}
            self.cls_specs = {}

        def function(self, **kwargs):
            def _decorator(fn):
                self.fn_specs[fn.__name__] = kwargs
                return fn
            return _decorator

        def cls(self, **kwargs):
            def _decorator(cls):
                self.cls_specs[cls.__name__] = kwargs
                return cls
            return _decorator

    stub.App = _App

    def _fastapi_endpoint(**kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    stub.fastapi_endpoint = _fastapi_endpoint

    class _Secret:
        @staticmethod
        def from_name(name):
            return {"secret": name}

    stub.Secret = _Secret

    def _enter():
        def _decorator(fn):
            return fn
        return _decorator

    stub.enter = _enter

    class _Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            return {"volume": name}

    stub.Volume = _Volume

    class _Mount:
        @staticmethod
        def from_local_dir(local, remote_path=None):
            return {"mount": local, "remote": remote_path}

    stub.Mount = _Mount

    return stub


class TestProdFlipKnob:
    def test_default_is_staging(self):
        # With no env var, should default to min_containers=0
        with patch.dict(os.environ, {"MEDSIGLIP_MODAL_MODE": ""}, clear=False):
            mod = _reload_app({"MEDSIGLIP_MODAL_MODE": ""})
        assert mod._MIN_CONTAINERS == 0
        assert mod._SCALEDOWN_S == 300
        assert mod._MODAL_MODE == "staging"  # empty string → default via .get

    def test_explicit_staging(self):
        mod = _reload_app({"MEDSIGLIP_MODAL_MODE": "staging"})
        assert mod._MIN_CONTAINERS == 0
        assert mod._SCALEDOWN_S == 300
        assert mod._MODAL_MODE == "staging"

    def test_prod_bumps_to_one_container(self):
        mod = _reload_app({"MEDSIGLIP_MODAL_MODE": "prod"})
        assert mod._MIN_CONTAINERS == 1
        assert mod._SCALEDOWN_S == 900
        assert mod._MODAL_MODE == "prod"

    def test_prod_is_case_insensitive(self):
        mod = _reload_app({"MEDSIGLIP_MODAL_MODE": "PROD"})
        assert mod._MIN_CONTAINERS == 1
        assert mod._MODAL_MODE == "prod"

    def test_unknown_mode_falls_back_to_staging(self):
        mod = _reload_app({"MEDSIGLIP_MODAL_MODE": "bogus"})
        assert mod._MIN_CONTAINERS == 0
        assert mod._SCALEDOWN_S == 300

    def test_app_version_bumped_to_v0_4_0(self):
        mod = _reload_app({})
        assert mod.APP_VERSION == "medsiglip-modal-v0.4.0-alpha"

    def test_cls_decorator_receives_computed_values(self):
        mod = _reload_app({"MEDSIGLIP_MODAL_MODE": "prod"})
        # ``@app.cls(...)`` recorded the min_containers/scaledown_window
        # kwargs during module load; confirm they matched the prod values.
        spec = mod.app.cls_specs.get("MedSigLipModal")
        assert spec is not None
        assert spec["min_containers"] == 1
        assert spec["scaledown_window"] == 900
