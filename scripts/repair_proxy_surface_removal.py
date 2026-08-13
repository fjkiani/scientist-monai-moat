#!/usr/bin/env python3
"""Delete the reachable proxy inference surface and the stale contract text.

Why this script exists
----------------------
An audit of the retired-proxy work found that the *implementation* of the
ungated general-domain SigLIP proxy was still reachable from
``src/oncology_arbiter/api/app.py`` (``_get_siglip_proxy`` →
``_run_siglip_proxy_on_preprocessed``), and that the public API schema still
advertised six ``ModelState`` wire values whose only purpose was to label a
substitute for a required specialist stage.  It also found that
``tests/unit/test_screening_medsiglip_wiring.py`` still pinned the proxy
contract (``model_state == "proxy_siglip"``,
``ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY``, and a template ``arbiter_score`` that
the production route no longer emits) and was green only because 8 of its 9
tests skip on a missing CBIS-DDSM fixture -- a silent pin that would fail the
moment the fixture appeared.

Every edit below is a *removal of a substitution path* or a correction of text
that describes a substitution as if it were production.  Nothing here changes a
number, a coefficient, or an inference result.

Design rules
------------
* Idempotent: every edit is skipped when already applied.
* Fail loud: a missing anchor is a hard error, never a silent no-op.
* No behaviour is added -- only proxy paths removed and text corrected.

Run:  python scripts/repair_proxy_surface_removal.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "oncology_arbiter"
TESTS = ROOT / "tests"


# --------------------------------------------------------------------------- #
# Edit primitives
# --------------------------------------------------------------------------- #


@dataclass
class Replace:
    """One exact-text replacement with a recorded justification."""

    path: Path
    old: str
    new: str
    reason: str
    count: int = 1

    def apply(self, check: bool) -> str:
        text = self.path.read_text(encoding="utf-8")
        if self.old not in text:
            # Deletions (new == "") are idempotent by definition: once the
            # anchor is gone the edit is done. Substitutions are idempotent
            # when the replacement text is already present.
            if self.new == "" or self.new in text:
                return "already-applied"
            raise SystemExit(f"ANCHOR MISSING in {self.path}:\n{self.old[:200]!r}")
        found = text.count(self.old)
        if found != self.count:
            raise SystemExit(
                f"ANCHOR COUNT {found} != {self.count} in {self.path}: {self.old[:120]!r}"
            )
        if not check:
            self.path.write_text(text.replace(self.old, self.new), encoding="utf-8")
        return "applied"


@dataclass
class WriteFile:
    path: Path
    content: str
    reason: str

    def apply(self, check: bool) -> str:
        if self.path.exists() and self.path.read_text(encoding="utf-8") == self.content:
            return "already-applied"
        if not check:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.content, encoding="utf-8")
        return "applied"


@dataclass
class GitRemove:
    path: Path
    reason: str

    def apply(self, check: bool) -> str:
        if not self.path.exists():
            return "already-applied"
        if not check:
            subprocess.run(
                ["git", "rm", "-f", str(self.path.relative_to(ROOT))],
                cwd=ROOT, check=True, capture_output=True,
            )
        return "applied"


EDITS: list[object] = []


# --------------------------------------------------------------------------- #
# 1. Move the generic image helper out of the proxy module, then delete it.
#
#    ``models/medsiglip.py`` imported ``_to_pil_from_float01`` and
#    ``DEFAULT_ZERO_SHOT_LABELS`` from ``siglip_baseline``.  Neither is a model
#    reference -- one is a PIL float->uint8 RGB conversion, the other a prompt
#    pair -- so they move to a neutral module and the proxy file goes away
#    entirely.  With the file deleted there is no ``SiglipBaseline`` class left
#    to construct, which is the only durable way to guarantee the general-domain
#    proxy can never serve a screening request again.
# --------------------------------------------------------------------------- #

IMAGE_IO = '''"""Generic image I/O helpers shared by the vision clients.

These helpers carry no model identity.  They previously lived in
``models/siglip_baseline.py`` (the ungated general-domain SigLIP proxy).  That
module was deleted when the proxy screening path was retired, so the two
model-agnostic pieces it held moved here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Neutral English prompt pair for a zero-shot breast screening probe. These are
# the prompts used across the SigLIP-family literature (Chen et al., 2023,
# arXiv:2303.15343) and are NOT clinical decision text. Zero-shot agreement
# scores against these prompts are uncalibrated and off-label.
DEFAULT_ZERO_SHOT_LABELS: tuple[str, str] = (
    "a mammogram of a breast with a malignant mass",
    "a mammogram of a breast without a malignant mass",
)


def to_pil_from_float01(arr: np.ndarray) -> Any:
    """Convert a preprocessed mammogram (float32 in [0,1], HxW) to a PIL image.

    * Grayscale [0,1] -> uint8 -> 3-channel RGB (SigLIP-family encoders expect 3ch).
    * No resizing here -- the processor handles resize/center-crop.
    """
    from PIL import Image

    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale mammogram, got shape {arr.shape}")
    a = np.clip(arr, 0.0, 1.0)
    a8 = (a * 255.0 + 0.5).astype(np.uint8)
    rgb = np.stack([a8, a8, a8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


# Back-compat alias: the private name is what medsiglip.py already imported.
_to_pil_from_float01 = to_pil_from_float01

__all__ = ["DEFAULT_ZERO_SHOT_LABELS", "to_pil_from_float01", "_to_pil_from_float01"]
'''

EDITS.append(WriteFile(SRC / "models" / "image_io.py", IMAGE_IO,
                       "neutral home for the two model-agnostic helpers"))

EDITS.append(Replace(
    SRC / "models" / "medsiglip.py",
    old="""# NOTE: we import ONE helper from siglip_baseline (the PIL float-array
# converter is generic image I/O, not a model reference) and the shared
# label pair, so the honesty test can still assert that we do NOT import
# the ``SiglipBaseline`` *class* itself.
from oncology_arbiter.models.siglip_baseline import (
    DEFAULT_ZERO_SHOT_LABELS,
    _to_pil_from_float01,
)""",
    new="""# Generic image I/O + the neutral prompt pair. These live in image_io.py so
# this module has no import edge to any proxy backbone; the SigLIP proxy module
# was deleted when the proxy screening path was retired.
from oncology_arbiter.models.image_io import (
    DEFAULT_ZERO_SHOT_LABELS,
    _to_pil_from_float01,
)""",
    reason="drop the import edge to the deleted proxy module",
))

EDITS.append(Replace(
    SRC / "models" / "medsiglip.py",
    old=":mod:`oncology_arbiter.models.siglip_baseline`. It differs from the proxy",
    new=":mod:`oncology_arbiter.models.siglip_baseline` (now deleted). It differed from that proxy",
    reason="docstring referenced a module that no longer exists",
))

EDITS.append(GitRemove(SRC / "models" / "siglip_baseline.py",
                       "ungated general-domain SigLIP proxy: deleted, not disabled"))


# --------------------------------------------------------------------------- #
# 2. app.py -- remove the reachable proxy plumbing and the dead NSCLC import.
# --------------------------------------------------------------------------- #

APP = SRC / "api" / "app.py"

EDITS.append(Replace(
    APP,
    old="    NsclcCTInput,\n",
    new="",
    reason="dead import; the client-supplied CT series path input is retired",
))

EDITS.append(Replace(
    APP,
    old="_SIGLIP_PROXY_SINGLETON: Any = None\n",
    new="",
    reason="no proxy singleton can exist once the proxy class is gone",
))

EDITS.append(Replace(
    APP,
    old='''def _get_siglip_proxy() -> Any:
    """Lazy-construct the SigLIP proxy client."""
    global _SIGLIP_PROXY_SINGLETON
    if _SIGLIP_PROXY_SINGLETON is None:
        from oncology_arbiter.models.siglip_baseline import SiglipBaseline
        _SIGLIP_PROXY_SINGLETON = SiglipBaseline()
    return _SIGLIP_PROXY_SINGLETON


''',
    new="",
    reason="removes the only constructor of the general-domain proxy",
))

EDITS.append(Replace(
    APP,
    old='''def _run_siglip_proxy_on_preprocessed(preprocess_result: Any) -> Any:
    """Run the SigLIP proxy on an already-preprocessed mammogram.

    Same Phase 2 caveat as :func:`_run_medsiglip_on_preprocessed` — the
    proxy singleton's ``_preprocess_fn`` is mutated for the call. Safe
    under single-thread async; not safe under a threadpool. Phase 3 fix.
    """
    proxy = _get_siglip_proxy()

    class _AlreadyPreprocessed:
        image = preprocess_result.image

    def _inject(_path: str) -> Any:
        return _AlreadyPreprocessed()

    proxy._preprocess_fn = _inject
    return proxy.run("(preprocessed)")


''',
    new="",
    reason="removes the reachable proxy inference call path",
))

# The screening receipt reported per-prompt SigLIP scores without stating that
# they are independent sigmoid agreement scores. The live-fire receipt showed
# probs = [9.475e-06, 8.807e-06] summing to 1.828e-05: emphatically not a
# distribution. Make the non-normalisation machine-checkable in the payload.
EDITS.append(Replace(
    APP,
    old='''                "score_semantics": "independent_uncalibrated_sigmoid_zero_shot",
            }''',
    new='''                "score_semantics": "independent_uncalibrated_sigmoid_zero_shot",
                "probs": [float(p) for p in result.probs],
                "probs_sum": float(sum(float(p) for p in result.probs)),
                "probs_are_normalised_distribution": False,
            }''',
    reason="SigLIP pairwise-sigmoid scores must never read as a distribution",
))

EDITS.append(Replace(
    APP,
    old="""            rules_sha256=None,
            rules_model_id=None,
            branch_id=None,
""",
    new="",
    reason="NCCN-lite rules fields removed from the public therapy contract",
))


# --------------------------------------------------------------------------- #
# 3. schemas.py -- retire the six substitute wire values and the stale text.
# --------------------------------------------------------------------------- #

SCHEMAS = SRC / "api" / "schemas.py"

for member, wire in [
    ('    PROXY_SIGLIP = "proxy_siglip"     # ungated general-domain SigLIP fallback (NOT MedSigLIP output)\n', "proxy_siglip"),
    ('    PROXY_MONAI_HEURISTIC = "proxy_monai_heuristic"  # L4a MONAI mask-gradient heuristic when weights unavailable\n', "proxy_monai_heuristic"),
    ('    PROXY_LUNG_HEURISTIC = "proxy_lung_heuristic"  # NSCLC HU-threshold + CC blobs (LIDC-IDRI) — not a trained detector\n', "proxy_lung_heuristic"),
    ('    PROXY_RULES_LITE = "proxy_rules_lite"  # L4c NCCN-lite rules fallback when TxGemma gated\n', "proxy_rules_lite"),
    ('    PROXY_REGEX_V0 = "proxy_regex_v0" # v0.2.1 pathology-report regex parser (stateless code, always available)\n', "proxy_regex_v0"),
    ('    FUSED_REGEX_CLINICALBERT = "fused_regex_clinicalbert"  # v0.3.0 regex ∧ ClinicalBERT fusion\n', "fused_regex_clinicalbert"),
]:
    EDITS.append(Replace(SCHEMAS, old=member, new="",
                         reason=f"{wire}: labels a substitute for a required specialist stage"))

EDITS.append(Replace(
    SCHEMAS,
    old='''    CONFIGURED_UNVERIFIED = "configured_unverified"  # endpoint configured; successful inference not yet proven
''',
    new='''    CONFIGURED_UNVERIFIED = "configured_unverified"  # endpoint configured; successful inference not yet proven
    # NOTE: the six substitute states (proxy_siglip, proxy_monai_heuristic,
    # proxy_lung_heuristic, proxy_rules_lite, proxy_regex_v0,
    # fused_regex_clinicalbert) were removed from this enum. See
    # RETIRED_MODEL_STATE_VALUES below.
''',
    reason="point readers at the retired-value record",
))

EDITS.append(Replace(
    SCHEMAS,
    old='''# --------------------------------------------------------------------------- #
# Shared


class ModelState(str, Enum):''',
    new='''# --------------------------------------------------------------------------- #
# Shared

# Wire values that appear in audit envelopes written before the proxy surface
# was retired. They are deliberately NOT members of :class:`ModelState`, so no
# route can emit one; the frozen set exists only so an auditor reading an
# archived receipt can still resolve the string. Each of these labelled a
# substitute standing in for a required specialist stage.
RETIRED_MODEL_STATE_VALUES: frozenset[str] = frozenset({
    "proxy_siglip",              # google/siglip-base-patch16-224, general-domain
    "proxy_monai_heuristic",     # mask-gradient heuristic, not a detector
    "proxy_lung_heuristic",      # HU threshold + connected components
    "proxy_rules_lite",          # NCCN-lite static rules
    "proxy_regex_v0",            # regex pathology parser
    "fused_regex_clinicalbert",  # regex AND ClinicalBERT fusion
})


class ModelState(str, Enum):''',
    reason="record the retired wire values without making them emittable",
))

# --- NSCLC: the client-controlled CT path input is gone. ------------------- #
EDITS.append(Replace(
    SCHEMAS,
    old='''# `NsclcCTInput.series_dir` points at a LIDC-IDRI CT series directory on the
# server (e.g. `/workspace/lidc_cohort/lidc_idri/LIDC-IDRI-0001/<StudyUID>/CT_<SeriesUID>`).
# Real pipeline execution is gated behind the `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1`
# env var so untrusted deployments never trust a client-controlled filesystem
# path. When gated off, the branch falls back to shape-only placeholder.


class NsclcCTInput(BaseModel):
    """Point at a CT series on disk for the real lung heuristic + NCCN rules.

    Only honored when `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` is set on the server;
    otherwise ignored to avoid client-controlled path traversal in shared
    deployments.
    """
    model_config = ConfigDict(extra="forbid")
    series_dir: str = Field(..., description="Absolute path to a CT_<SeriesUID> directory")
    patient_id: str | None = Field(default=None, description="LIDC patient id if known")
    top_n: int = Field(default=10, ge=1, le=100, description="Max candidate blobs to summarize")


''',
    new='''# NSCLC CT input by client-supplied server path (`NsclcCTInput.series_dir`,
# gated behind ONCOLOGY_ARBITER_ALLOW_SERIES_DIR) is RETIRED. Production NSCLC
# consumes a verified case-storage `case_id` only, so the detector runs on an
# uploaded, hash-receipted series rather than a path the caller chose.


''',
    reason="retire the client-controlled filesystem path input",
))

EDITS.append(Replace(
    SCHEMAS,
    old='''    # Provenance
    series_dir: str | None = None
    n_slices: int | None = None
    read_seconds: float | None = None
    heuristic_seconds: float | None = None
''',
    new='''    # Provenance. n_slices comes from the verified case manifest, not from a
    # local read; series_dir/read_seconds/heuristic_seconds belonged to the
    # retired local-CT + HU-heuristic path and were never set on this route.
    n_slices: int | None = None
''',
    reason="drop provenance fields that only the retired local path could fill",
))

# --- parser provenance text: no regex parser is wired. --------------------- #
EDITS.append(Replace(
    SCHEMAS,
    old="""    ``parse_state`` (v0.2.1) surfaces per-field provenance to the UI so the
    clinician can see whether each value came from a confident regex match,
    an ambiguous mention that needs review, or was never mentioned in the
    report at all. When the UI submits an override, the state becomes
    ``user_supplied``.""",
    new="""    ``parse_state`` surfaces per-field provenance to the UI so the clinician
    can see whether each value came from a confident ClinicalBERT span, an
    ambiguous mention that needs review, or was never mentioned in the report
    at all. When the UI submits an override, the state becomes
    ``user_supplied``. No regex parser is wired on any route.""",
    reason="the only wired parser is ClinicalBERT v0.5.2",
))

EDITS.append(Replace(
    SCHEMAS,
    old='''        description="Per-field provenance from the report parser (proxy_regex_v0). "
                    "Absent when biopsy analysis did not run a text parser.",''',
    new='''        description="Per-field provenance from the ClinicalBERT v0.5.2 "
                    "sliding-window pathology parser. Absent when biopsy "
                    "analysis did not run a text parser.",''',
    reason="stop advertising the retired regex parser id",
))

EDITS.append(Replace(
    SCHEMAS,
    old='''    """One field parsed by the v0.3.0 fused (regex + ClinicalBERT) parser.

    Only used for extended fields the v0.2.1 regex parser could not produce:
    ki67_pct, tumor_size_mm, T/N/M stage, margin, LVI. The four core fields''',
    new='''    """One extended field parsed by the ClinicalBERT v0.5.2 pathology parser.

    Carries the extended entities beyond the receptor core:
    ki67_pct, tumor_size_mm, T/N/M stage, margin, LVI. The four core fields''',
    reason="no regex/fusion parser exists to be fused with",
))

EDITS.append(Replace(
    SCHEMAS,
    old='''    parser_id: str = Field(
        ...,
        description="e.g. proxy_regex_v0, clinicalbert_v1, clinicalbert_v1+regex_v0",
    )
    fusion_mode: Literal["regex", "bert", "fused", "clinicalbert"] = "clinicalbert"''',
    new='''    parser_id: str = Field(
        ...,
        description="Always clinicalbert_v0.5.2_sliding_window; no regex or "
                    "fused parser is wired on any route.",
    )
    fusion_mode: Literal["clinicalbert"] = "clinicalbert"''',
    reason="close the type so a fused/regex value cannot be serialised",
))

# --- Co-Scientist naming: the wired endpoint is a deterministic ranker. ---- #
EDITS.append(Replace(
    SCHEMAS,
    old='''class CoScientistRunRequest(BaseModel):
    """Input for POST /v1/co_scientist/run.

    Callers pass whatever stage envelopes they already have (screening,
    biopsy, therapy), plus the union of URLs their tool-loop actually
    fetched. Anything the model returns citing an unseen URL will be
    dropped by REFLECT.
    """''',
    new='''class CoScientistRunRequest(BaseModel):
    """Input for POST /v1/offline_ranker/run.

    The wired endpoint is an OFFLINE DETERMINISTIC RANKER: a fixed Elo-style
    ordering over caller-supplied stage envelopes. It performs no LLM call and
    its output must never be labelled as model-generated reasoning. Callers
    pass whatever stage envelopes they already have (screening, biopsy,
    therapy) plus the union of URLs their own tool loop fetched; any citation
    to an unseen URL is dropped.
    """''',
    reason="request docstring named a live agent that is not wired",
))


# --------------------------------------------------------------------------- #
# 4. hai_def.py docstring pointed at a removed enum member.
# --------------------------------------------------------------------------- #

EDITS.append(Replace(
    SRC / "models" / "hai_def.py",
    old="response as `ModelState.PROXY_SIGLIP` NOT `ModelState.LOADED`).",
    new="response as `ModelState.GATED` NOT `ModelState.LOADED`; the former "
        "`proxy_siglip` substitute state has been retired).",
    reason="the referenced substitute state no longer exists",
))


# --------------------------------------------------------------------------- #
# 5. Tests: retire the proxy-era screening wiring file, fix the enum pin.
# --------------------------------------------------------------------------- #

EDITS.append(GitRemove(
    TESTS / "unit" / "test_screening_medsiglip_wiring.py",
    "pinned proxy_siglip + ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY + a template "
    "arbiter_score the production route no longer emits; 8/9 tests were "
    "skip-masked by a missing CBIS-DDSM fixture",
))

EDITS.append(Replace(
    TESTS / "models" / "test_hai_def.py",
    old='''def test_model_state_enum_has_proxy_siglip_variant() -> None:
    """Ungated proxy fallback MUST get a distinct wire value so downstream
    consumers know NOT to treat it as MedSigLIP output."""
    from oncology_arbiter.api.schemas import ModelState
    assert hasattr(ModelState, "PROXY_SIGLIP")
    assert ModelState.PROXY_SIGLIP.value == "proxy_siglip"''',
    new='''def test_model_state_enum_cannot_express_the_retired_siglip_proxy() -> None:
    """The ungated general-domain proxy is retired, so no route may emit it.

    Inverts the original assertion. The wire value stays recorded in
    RETIRED_MODEL_STATE_VALUES so archived receipts remain readable, but it is
    not constructible as a ModelState any more.
    """
    from oncology_arbiter.api.schemas import RETIRED_MODEL_STATE_VALUES, ModelState
    assert not hasattr(ModelState, "PROXY_SIGLIP")
    assert "proxy_siglip" not in {m.value for m in ModelState}
    assert "proxy_siglip" in RETIRED_MODEL_STATE_VALUES''',
    reason="invert a pin that required the proxy state to exist",
))


NEW_TESTS = '''

# --------------------------------------------------------------------------- #
# Retired-substitute contract. Folded in from the deleted proxy-era file
# tests/unit/test_screening_medsiglip_wiring.py, inverted: instead of pinning
# that a proxy state exists, these pin that no substitute can be emitted.
# --------------------------------------------------------------------------- #


def test_no_retired_substitute_state_is_constructible() -> None:
    from oncology_arbiter.api.schemas import RETIRED_MODEL_STATE_VALUES

    live = {m.value for m in ModelState}
    assert live.isdisjoint(RETIRED_MODEL_STATE_VALUES), live & RETIRED_MODEL_STATE_VALUES
    assert ModelState.LOADED_MEDSIGLIP.value == "loaded_medsiglip"
    for other in (ModelState.PLACEHOLDER, ModelState.GATED, ModelState.UNAVAILABLE):
        assert ModelState.LOADED_MEDSIGLIP.value != other.value


def test_app_module_imports_no_substitute_backend() -> None:
    """app.py must not import any retired stand-in for a required specialist."""
    import ast
    import importlib
    from pathlib import Path

    app_src = Path(importlib.import_module("oncology_arbiter.api.app").__file__)
    tree = ast.parse(app_src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    banned = {
        "oncology_arbiter.models.siglip_baseline",   # general-domain SigLIP proxy
        "oncology_arbiter.nlp.report_parser_v2",     # regex / regex+BERT fusion parser
        "oncology_arbiter.models.therapy_rules_lite",  # NCCN-lite static rules
        "oncology_arbiter.models.nccn_nsclc_rules",    # NCCN-lite static rules (NSCLC)
        "oncology_arbiter.models.txgemma_client",      # gated TxGemma therapy path
        "oncology_arbiter.lung.pipeline",              # HU-threshold nodule heuristic
        "oncology_arbiter.lung",
    }
    assert not (imported & banned), sorted(imported & banned)
    text = app_src.read_text(encoding="utf-8")
    for token in ("_get_siglip_proxy", "_run_siglip_proxy_on_preprocessed",
                  "SiglipBaseline", "NsclcCTInput"):
        assert token not in text, token


def test_zero_shot_probs_are_reported_as_independent_sigmoid_scores(monkeypatch) -> None:
    """SigLIP scores are pairwise-sigmoid, so they need not sum to 1.

    The live MedSigLIP receipt returned probs=[9.475e-06, 8.807e-06]
    (sum 1.83e-05). Presenting that as a normalised distribution -- or as a
    calibrated malignancy probability -- would be a false claim, so the receipt
    must carry the sum and the explicit non-normalisation flag.
    """
    class ModalClient:
        def run(self, path):
            return SimpleNamespace(
                labels=["malignant", "without malignancy"],
                probs=[9.475328624830581e-06, 8.806667210592423e-06],
                model_repo="google/medsiglip-448",
                app_version="medsiglip-modal-v0.4.0-alpha",
                input_resolution=448,
                embedding_dim=1152,
                embedding_sha256="c" * 64,
                prompts=["a mammogram showing malignancy", "a mammogram without malignancy"],
                inference_seconds=0.25,
                warnings=[],
                gate_report=None,
            )

    client = _client(monkeypatch, ModalClient)
    body = client.post("/v1/screening/analyze", json=_payload()).json()
    block = body["medsiglip"]
    assert block["score_semantics"] == "independent_uncalibrated_sigmoid_zero_shot"
    assert block["probs_are_normalised_distribution"] is False
    assert block["probs_sum"] == pytest.approx(1.8281995835423004e-05, rel=1e-9)
    assert block["probs_sum"] < 0.5  # emphatically not a distribution
    assert body["overall_score"] == pytest.approx(9.475328624830581e-06, rel=1e-12)
    assert body["arbiter_score"] is None
    assert any("uncalibrated" in w.lower() for w in body["warnings"]), body["warnings"]
'''

SCREEN_CONTRACT = TESTS / "unit" / "test_screening_production_contract.py"


class AppendTests:
    reason = "fold forward the one surviving assertion + pin non-normalisation"

    def apply(self, check: bool) -> str:
        text = SCREEN_CONTRACT.read_text(encoding="utf-8")
        if "test_no_retired_substitute_state_is_constructible" in text:
            return "already-applied"
        if "import pytest" not in text:
            text = text.replace(
                "import numpy as np\n", "import numpy as np\nimport pytest\n", 1
            )
        text = text.rstrip("\n") + "\n" + NEW_TESTS
        if not check:
            SCREEN_CONTRACT.write_text(text, encoding="utf-8")
        return "applied"


EDITS.append(AppendTests())


# --------------------------------------------------------------------------- #
# 6. Model card: keep it as the audit record, but say it is retired.
# --------------------------------------------------------------------------- #

CARD = ROOT / "docs" / "model_cards" / "siglip_base_patch16_224.md"
CARD_BANNER = """> **RETIRED (this pass).** The proxy client
> `src/oncology_arbiter/models/siglip_baseline.py` has been **deleted** and the
> `proxy_siglip` wire value removed from `ModelState`, so no route can serve a
> general-domain SigLIP score. This card is retained only as the audit record of
> what the proxy was and why its output was never MedSigLIP output.

"""


class CardBanner:
    reason = "the card described a live proxy; mark it retired without deleting evidence"

    def apply(self, check: bool) -> str:
        text = CARD.read_text(encoding="utf-8")
        if "**RETIRED (this pass).**" in text:
            return "already-applied"
        lines = text.split("\n")
        insert_at = 1 if lines and lines[0].startswith("#") else 0
        new = "\n".join(lines[:insert_at] + ["", CARD_BANNER.rstrip("\n")] + lines[insert_at:])
        if not check:
            CARD.write_text(new, encoding="utf-8")
        return "applied"


EDITS.append(CardBanner())


# --------------------------------------------------------------------------- #
# 7. Ledger generator: two current_backend fields still advertised retired
#    backends, and the screening endpoint role claimed an arbiter it no longer
#    emits (the route hard-codes arbiter_score=None).
# --------------------------------------------------------------------------- #

LEDGER_GEN = ROOT / "scripts" / "update_progress_ledger.py"

EDITS.append(Replace(
    LEDGER_GEN,
    old='''        "current_backend": "hand-drafted template coefficients (n_training=0)",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/arbiter/models/biopsy_arbiter_template_v0.json",
        ],
        "superseded_by": "breast_dss_v3",''',
    new='''        # RETIRED entries carry no backend: the biopsy route emits
        # arbiter_score=None, so naming a backend here would imply the template
        # still serves traffic (the L4a siglip proxy entry already uses None).
        "current_backend": None,
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/arbiter/models/biopsy_arbiter_template_v0.json",
        ],
        "superseded_by": "breast_dss_v3",''',
    reason="RETIRED status must not advertise a current backend",
))

EDITS.append(Replace(
    LEDGER_GEN,
    old='''        "role": "POST /v1/screening/analyze — DICOM in, screening arbiter + MedSigLIP score out.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "MedSigLIP-448 (default) or SigLIP proxy (opt-in fallback)",''',
    new='''        "role": (
            "POST /v1/screening/analyze — de-identified DICOM in, MedSigLIP-448 "
            "zero-shot agreement scores out. arbiter_score is always None on "
            "this route. Scores are independent uncalibrated sigmoid values "
            "(they do not sum to 1) and are not diagnostic evidence."
        ),
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "MedSigLIP-448 via Modal (strict; no proxy fallback exists)",''',
    reason="route emits no arbiter and has no proxy fallback any more",
))

EDITS.append(Replace(
    LEDGER_GEN,
    old='''            "src/oncology_arbiter/models/siglip_baseline.py — SigLIP proxy client with mammography honesty warning",''',
    new='''            "src/oncology_arbiter/models/siglip_baseline.py — DELETED this pass (proxy client removed, not disabled)",''',
    reason="file no longer exists",
))


# --------------------------------------------------------------------------- #
# 8. Ledger evidence pointers to the deleted proxy-era test file.
#
#    tests/unit/test_progress_ledger.py::test_every_live_entry_has_working_evidence
#    requires every LIVE entry's evidence path to exist on disk, which is
#    exactly the guard that caught these. The historical sprint record is NOT
#    rewritten -- past claims stay as written and get a retirement annotation
#    appended instead.
# --------------------------------------------------------------------------- #

EDITS.append(Replace(
    LEDGER_GEN,
    old='            "tests/unit/test_screening_medsiglip_wiring.py",\n',
    new='            "tests/unit/test_screening_production_contract.py",\n',
    reason="evidence pointer: retired file -> the strict production contract",
    count=3,
))

EDITS.append(Replace(
    LEDGER_GEN,
    old='        "evidence": ["tests/unit/test_screening_medsiglip_wiring.py"],',
    new='        "evidence": ["tests/unit/test_screening_production_contract.py"],',
    reason="evidence pointer: retired file -> the strict production contract",
))

EDITS.append(Replace(
    LEDGER_GEN,
    old='''            "Live proxy-fallback smoke: model_state=proxy_siglip with BOTH warnings",''',
    new='''            "Live proxy-fallback smoke: model_state=proxy_siglip with BOTH warnings",
            "RETIRED LATER: the proxy fallback described in this sprint no longer "
            "exists. models/siglip_baseline.py was deleted, ModelState.PROXY_SIGLIP "
            "was removed, and tests/unit/test_screening_medsiglip_wiring.py was "
            "retired. The lines above are kept as the historical record of what "
            "this sprint shipped, not as a description of the current surface.",''',
    reason="annotate history instead of rewriting it",
))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    applied = skipped = 0
    for edit in EDITS:
        status = edit.apply(args.check)
        label = getattr(edit, "path", type(edit).__name__)
        print(f"[{status:>15}] {label} :: {getattr(edit, 'reason', '')[:90]}")
        applied += status == "applied"
        skipped += status == "already-applied"

    # Guard: after the edits, no src module may reference the deleted proxy or
    # the retired substitute states.
    if not args.check:
        offenders: list[str] = []
        pattern = re.compile(
            r"ModelState\.(PROXY_SIGLIP|PROXY_MONAI_HEURISTIC|PROXY_LUNG_HEURISTIC"
            r"|PROXY_RULES_LITE|PROXY_REGEX_V0|FUSED_REGEX_CLINICALBERT)"
        )
        for py in SRC.rglob("*.py"):
            body = py.read_text(encoding="utf-8")
            if pattern.search(body) or "siglip_baseline import" in body:
                offenders.append(str(py.relative_to(ROOT)))
        if offenders:
            raise SystemExit(f"residual substitute references: {offenders}")

    manifest = ROOT / "artifacts" / "audit" / "proxy_surface_removal.json"
    record = {
        "script": "scripts/repair_proxy_surface_removal.py",
        "applied": applied,
        "already_applied": skipped,
        "deleted_source_modules": ["src/oncology_arbiter/models/siglip_baseline.py"],
        "deleted_test_files": ["tests/unit/test_screening_medsiglip_wiring.py"],
        "deleted_test_reason": (
            "pinned model_state==proxy_siglip, ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY, "
            "and a template arbiter_score the production route no longer emits; "
            "8 of 9 tests were skip-masked by the missing CBIS-DDSM fixture"
        ),
        "retired_model_state_values": sorted(
            ["proxy_siglip", "proxy_monai_heuristic", "proxy_lung_heuristic",
             "proxy_rules_lite", "proxy_regex_v0", "fused_regex_clinicalbert"]
        ),
        "replacement_coverage": [
            "tests/unit/test_screening_production_contract.py::test_no_retired_substitute_state_is_constructible",
            "tests/unit/test_screening_production_contract.py::test_app_module_imports_no_substitute_backend",
            "tests/unit/test_screening_production_contract.py::test_zero_shot_probs_are_reported_as_independent_sigmoid_scores",
            "tests/models/test_hai_def.py::test_model_state_enum_cannot_express_the_retired_siglip_proxy",
        ],
        "residual_unreachable_modules": [
            "src/oncology_arbiter/nlp/report_parser_v2.py",
            "src/oncology_arbiter/models/therapy_rules_lite.py",
            "src/oncology_arbiter/models/nccn_nsclc_rules.py",
            "src/oncology_arbiter/lung/pipeline.py",
        ],
        "residual_note": (
            "These modules are NOT imported by api/app.py (verified by AST test), so "
            "no route can reach them. They are retained as tested research code; the "
            "import ban test is what prevents re-wiring."
        ),
    }
    if not args.check:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest}")
    print(f"applied={applied} already_applied={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
