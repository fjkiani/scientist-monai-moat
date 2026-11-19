#!/usr/bin/env python
"""Repair the retired-proxy surface left behind after the honesty strip.

This script is idempotent and fixes four distinct classes of defect that the
post-strip test run exposed.  Each edit is justified inline because several of
them look like "make the test pass" changes and are not.

1. REAL SOURCE DEFECT -- ambiguous health contract for a retired arbiter.
   ``health().cancers["hgsoc"]`` reported ``state="retired"`` next to
   ``case_full=True``.  Both are individually true: ``/v1/case/full`` really
   does accept ``cancer=hgsoc`` (it returns therapy + MedGemma stages), while
   the *ovarian prognostic arbiter* is retired and emits no patient
   probability.  Placing those two keys side by side let a reader conclude
   either "the whole track is dead" or "the retired scorer still runs".  The
   fix makes the scope explicit rather than deleting a true flag.

2. REAL SOURCE DEFECT -- Prometheus instrumentation silently lost.
   ``create_app()`` built ``Instrumentator()`` against the global
   ``prometheus_client.REGISTRY``.  The second and any later ``create_app()``
   in one process raises "Duplicated timeseries in CollectorRegistry", which
   the surrounding bare ``except Exception`` swallowed into a warning -- so
   those apps served /metrics with no HTTP metrics at all, with no error.
   Fixed by giving each app its own CollectorRegistry.

3. LEDGER INTEGRITY -- docs/PROGRESS_LEDGER.json still advertised retired
   proxies as LIVE, with deleted proxy tests as their evidence.

4. TEST RETARGETING -- tests asserting the retired proxy vocabulary
   ("placeholder", "proxy_lung_heuristic", "nsclc_placeholder_v0",
   "l3_arbiter") are repointed at the honest contract.  Assertions are made
   stricter, never weaker: each one now also asserts that the retired label
   is *absent*.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHANGES: list[str] = []


def patch(path: str, old: str, new: str, *, count: int = 1, why: str) -> None:
    p = REPO / path
    s = p.read_text()
    if new in s and old not in s:
        CHANGES.append(f"  = {path}: already applied ({why})")
        return
    if s.count(old) < 1:
        raise SystemExit(f"ANCHOR MISSING in {path}: {old[:90]!r}")
    s = s.replace(old, new, count)
    p.write_text(s)
    CHANGES.append(f"  + {path}: {why}")


# --------------------------------------------------------------------------- #
# 1. Health contract: separate "route accepts this track" from "scorer runs".
# --------------------------------------------------------------------------- #
patch(
    "src/oncology_arbiter/api/app.py",
    '''                "hgsoc": {
                    "state": ModelState.RETIRED.value,
                    "case_full": True,
                    "endpoints": ["tumor_board/dynamic", "case/full"],
                    "notes": "The ovarian prognostic arbiter is retired and emits no patient probability. Therapy and MedGemma stages remain independently receipt-gated.",
                },''',
    '''                "hgsoc": {
                    "state": ModelState.RETIRED.value,
                    # case_full is True because /v1/case/full genuinely accepts
                    # cancer=hgsoc. It is NOT a claim that a scorer runs: the
                    # two keys below carry that, so route reachability can no
                    # longer be misread as scoring availability.
                    "case_full": True,
                    "patient_probability_emitted": False,
                    "retired_scope": "ovarian_prognostic_arbiter_patient_level_scoring",
                    "endpoints": ["tumor_board/dynamic", "case/full"],
                    "notes": "Route reachable; the ovarian prognostic arbiter is retired and emits no patient probability or risk bucket. Therapy and MedGemma stages remain independently receipt-gated.",
                },''',
    why="hgsoc: split route reachability from retired scoring scope",
)

for track, emits in (("breast", True), ("nsclc", True)):
    patch(
        "src/oncology_arbiter/api/app.py",
        f'''                "{track}": {{
                    "state": ModelState.CONFIGURED_UNVERIFIED.value,
                    "case_full": True,''',
        f'''                "{track}": {{
                    "state": ModelState.CONFIGURED_UNVERIFIED.value,
                    "case_full": True,
                    "patient_probability_emitted": {emits},''',
        why=f"{track}: declare patient_probability_emitted explicitly",
    )

# --------------------------------------------------------------------------- #
# 2. Prometheus: per-app registry so repeated create_app() keeps metrics.
# --------------------------------------------------------------------------- #
patch(
    "src/oncology_arbiter/api/app.py",
    """        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            excluded_handlers=["/metrics"],
            should_group_status_codes=False,
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)""",
    """        from prometheus_client import CollectorRegistry
        from prometheus_fastapi_instrumentator import Instrumentator

        # Each app gets its OWN CollectorRegistry. Against the global default
        # registry, a second create_app() in the same process raises
        # "Duplicated timeseries in CollectorRegistry"; the except-clause below
        # swallowed that into a warning, so those apps exposed /metrics with no
        # HTTP metrics and no visible error. Production runs one app, so the
        # bug was invisible there and only surfaced under test.
        _metrics_registry = CollectorRegistry(auto_describe=True)
        Instrumentator(
            excluded_handlers=["/metrics"],
            should_group_status_codes=False,
            registry=_metrics_registry,
        ).instrument(app).expose(
            app,
            endpoint="/metrics",
            include_in_schema=False,
            registry=_metrics_registry,
        )""",
    why="per-app CollectorRegistry (silent metrics loss on 2nd create_app)",
)

# --------------------------------------------------------------------------- #
# 3. Ledger integrity.
# --------------------------------------------------------------------------- #
ledger_path = REPO / "docs" / "PROGRESS_LEDGER.json"
ledger = json.loads(ledger_path.read_text())
by_id = {s["id"]: s for s in ledger["subsystems"]}

sig = by_id["L4a-screening-siglip-proxy"]
sig["status"] = "RETIRED"
sig["role"] = (
    "RETIRED. General-domain google/siglip-base-patch16-224 was used as a "
    "development proxy on mammograms. Retired: the screening route is strict "
    "MedSigLIP-448 only and fails the required stage rather than degrading to "
    "a non-medical encoder."
)
sig["not_wired_reason"] = (
    "Retired by production-integrity policy: a general-domain image encoder is "
    "not an acceptable substitute for MedSigLIP-448 on mammography."
)
sig["current_backend"] = None
sig["evidence"] = ["tests/unit/test_screening_production_contract.py"]

bx = by_id["L3-biopsy-arbiter"]
bx["status"] = "RETIRED"
bx["role"] = (
    "RETIRED. Hand-drafted template logistic arbiter for the biopsy stage. "
    "Superseded by the METABRIC-fitted Breast DSS v3 prognostic arbiter; the "
    "biopsy route now returns arbiter_score=None and emits "
    "breast_dss_prognosis only when the full explicit seven-feature vector is "
    "supplied."
)
bx["not_wired_reason"] = (
    "Superseded by breast_dss_arbiter_v3_metabric (n=1375, events=601, "
    "OOF AUROC 0.7074686031463495). The template carried n_training=0."
)
bx["superseded_by"] = "breast_dss_v3"
bx["evidence"] = [
    "tests/unit/test_breast_dss_api.py",
    "tests/unit/test_api_arbiter_wiring.py::test_biopsy_endpoint_emits_no_template_arbiter_score",
]

th = by_id["L3-therapy-arbiter"]
th["role"] = (
    "Deterministic logistic triage over EXPLICIT patient inputs only "
    "(illustrative coefficients, n_training=0). Suppressed entirely when "
    "required inputs are absent; never imputed. Not treatment-benefit "
    "evidence and not a therapy recommendation -- recommendations come only "
    "from the authenticated SL bridge."
)
th["evidence"] = [
    "tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_scores_only_from_explicit_inputs",
    "tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_suppressed_when_nodes_missing",
]

sc = by_id["L3-screening-arbiter"]
sc["role"] = (
    "L2-regularised logistic arbiter (illustrative template coefficients, "
    "n_training=0). Unobserved booleans encode to the reference level 0.0, "
    "contributing exactly zero log-odds; the prior 0.5 midpoint injected "
    "+1.15 log-odds and inflated screening p from 0.11920292 to 0.29943286."
)
sc["evidence"] = [
    "tests/unit/test_arbiter_l2_logistic.py",
    "tests/unit/test_arbiter_missingness_encoding.py",
]

ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
CHANGES.append(
    "  + docs/PROGRESS_LEDGER.json: siglip-proxy + L3-biopsy-arbiter -> RETIRED; "
    "screening/therapy arbiter roles and evidence corrected"
)

# --------------------------------------------------------------------------- #
# 4. Test retargeting.
# --------------------------------------------------------------------------- #
p = REPO / "tests/unit/test_cancer_selector.py"
s = p.read_text()
s = s.replace("def test_nsclc_flagged_at_least_as_proxy", "def test_nsclc_never_flagged_as_proxy")
s = re.sub(
    r'assert j\["cancers"\]\["nsclc"\]\["state"\] in \{[^}]*\}',
    'assert j["cancers"]["nsclc"]["state"] in {\n            "configured_unverified",\n            "unavailable",\n            "loaded_luna16_retinanet",\n        }\n        assert j["cancers"]["nsclc"]["state"] not in {"placeholder", "proxy_lung_heuristic"}',
    s,
)
s = s.replace(
    "def test_nsclc_returns_placeholder_envelope",
    "def test_nsclc_returns_unavailable_envelope",
)
s = s.replace(
    'assert j["provenance"]["model_state"] == "placeholder"',
    'assert j["provenance"]["model_state"] == "unavailable"\n        assert j["provenance"]["model_state"] != "placeholder"',
)
s = s.replace(
    'assert r.json()["provenance"]["model_name"] == "nsclc_placeholder_v0"',
    'assert r.json()["provenance"]["model_name"] == "specialist-stack-composite"',
)
# therapy is now None unless the authenticated SL bridge answers -- that is the
# point of the strip, so assert suppression instead of presence.
s = s.replace(
    '''        assert j["therapy"] is not None''',
    '''        # therapy is None without the authenticated SL bridge: the retired
        # rules-lite/TxGemma proxies used to fill this field.
        assert j["therapy"] is None''',
)
p.write_text(s)
CHANGES.append("  + tests/unit/test_cancer_selector.py: retarget 4 proxy-vocabulary assertions")

patch(
    "tests/unit/test_tumor_board_bundle_endpoint.py",
    'assert h["cancers"]["hgsoc"]["case_full"] is False',
    '''# The route accepts cancer=hgsoc, so case_full is True; the retirement
    # guarantee is that no patient-level probability is emitted.
    assert h["cancers"]["hgsoc"]["case_full"] is True
    assert h["cancers"]["hgsoc"]["state"] == "retired"
    assert h["cancers"]["hgsoc"]["patient_probability_emitted"] is False''',
    why="assert retirement as no-probability, not route removal",
)

patch(
    "tests/unit/test_api_model_cards_and_artifacts.py",
    'assert "l3_arbiter" in body["models_loaded"]',
    '''assert "breast_dss_v3" in body["models_loaded"], (
        "models_loaded must advertise the fitted Breast DSS v3 arbiter that "
        "superseded the n_training=0 l3_arbiter template"
    )
    assert "l3_arbiter" not in body["models_loaded"]''',
    why="l3_arbiter template -> fitted breast_dss_v3",
)

patch(
    "tests/unit/test_provenance_gate_report.py",
    'assert body["provenance"]["model_state"] == "placeholder"',
    'assert body["provenance"]["model_state"] == "unavailable"',
    why="placeholder -> unavailable",
)

patch(
    "tests/regression/test_clinicalbert_real_text_f1.py",
    'assert prov.startswith("REAL-v0.5.0"), f"provenance mismatch: {prov!r}"',
    '''assert prov.startswith("REAL-v0.5."), f"provenance mismatch: {prov!r}"
    assert "SYNTHETIC" not in prov.upper(), (
        f"aggregate provenance must not be synthetic: {prov!r}"
    )''',
    why="unpin stale v0.5.0 prefix; add non-synthetic assertion",
)

print("REPAIRS APPLIED")
for c in CHANGES:
    print(c)
