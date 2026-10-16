"""Late-bound hooks so unit tests can monkeypatch oncology_arbiter.api.app.*"""
from __future__ import annotations

from typing import Any


def get_medsiglip() -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._get_medsiglip()


def run_medsiglip_on_preprocessed(*args: Any, **kwargs: Any) -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._run_medsiglip_on_preprocessed(*args, **kwargs)


def run_cbis_ddsm_probe_on_bytes(*args: Any, **kwargs: Any) -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._run_cbis_ddsm_probe_on_bytes(*args, **kwargs)


def elo_reason_for_modifier(*args: Any, **kwargs: Any) -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._elo_reason_for_modifier(*args, **kwargs)


def run_clinicalbert_parse(*args: Any, **kwargs: Any) -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._run_clinicalbert_parse(*args, **kwargs)


def score_explicit_therapy_triage(*args: Any, **kwargs: Any) -> Any:
    from oncology_arbiter.api import app as app_module
    return app_module._score_explicit_therapy_triage(*args, **kwargs)
