#!/usr/bin/env python
"""Live-fire proof that the Manski gate has no public bypass.

Runs the real ASGI app (not a mock) and records, for every route that can
emit an L2 point estimate, what happens when the declared feature panel is
complete, partially unobserved, missing a one-hot level, and missing a
continuous covariate with no declared support.

Three claims are under test and each one is falsifiable here:

1.  A public route never returns 200 with an unidentified point estimate.
    Falsified by any 200 whose `manski.identified` is False.
2.  No header, query parameter or body field relaxes the gate. Falsified by
    any bypass attempt that turns a 422 into a 200.
3.  Blind inference is reachable only with the privileged scope. Falsified by
    a 200 from the research route without it — including with
    ONCOLOGY_ARBITER_AUTH_MODE=off, which is checked explicitly because that
    switch is exactly how a privileged route quietly opens in production.

Writes a JSON receipt; exits non-zero if any claim is falsified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Env must be set BEFORE oncology_arbiter.auth is imported: `DEFAULT_DB` is
# bound to os.environ at module import time, so setting the path afterwards
# leaves `verify_api_key` reading a different sqlite file than the one the
# keys were minted into — which surfaces as a 401 for a perfectly valid key
# and would be very easy to misread as "the scope check is broken".
_TMP = Path(tempfile.mkdtemp(prefix="manski-verify-"))
os.environ.setdefault("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", "1")
os.environ["ONCOLOGY_ARBITER_AUTH_DB_PATH"] = str(_TMP / "tenants.sqlite")
os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"] = str(_TMP / "audit")
os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = "off"
os.environ["ONCOLOGY_ARBITER_RATE_LIMIT"] = "100000/minute"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


# Complete therapy panel. `grade` is an int Literal[1,2,3] on the wire model,
# not a string: passing "2" produces a 200 with arbiter_score=None and an
# `invalid_triage_features` receipt, which looks like a broken gate but is
# actually schema rejection. Distinguish the two by checking for `error`.
BASE_PANEL: dict[str, Any] = {
    "histology": "invasive_ductal",
    "grade": 2,
    "er_positive": True,
    "pr_positive": True,
    "her2_positive": False,
    "ki67_pct": 20.0,
    "tumor_size_mm": 22.0,
    "lymph_nodes_pos": 0,
    "brca_pathogenic": False,
    "age_years": 58.0,
}

# Feature-space panel handed straight to the arbiter (research route), using
# the artefact's own feature names.
#
# UNIT TRAP, verified: the keys are named `*_norm` but the artefact expects the
# RAW value and applies the divisor itself — `feature_encodings["ki67_norm"]`
# is the string "ki67_pct / 100.0" and `_encode_continuous` divides. Passing an
# already-normalised 0.20 is silently accepted and attenuates the contribution
# 100-fold: p drops 0.155251 -> 0.100763, a -35.10% relative error, with zero
# errors and zero warnings. `_encode_continuous` performs no range validation,
# so nothing distinguishes a percentage from a fraction. The public wire path
# is correct (app.py maps ki67_pct -> ki67_norm raw); the exposure is the
# research route's free-form feature dict.
BASE_FEATURES: dict[str, Any] = {
    "histology": "invasive_ductal",
    "grade": "2",
    "er_status_positive": True,
    "pr_status_positive": True,
    "her2_status_positive": False,
    "ki67_norm": 20.0,          # raw ki67_pct, NOT 0.20
    "tumor_size_norm": 22.0,    # raw size_mm, NOT 0.44
    "node_status_positive": False,
    "brca_status_known_pathogenic": False,
    "age_at_diagnosis_norm": 58.0,  # raw age_years, NOT 0.58
}

# The pre-normalised panel above, kept so the receipt can quantify the trap.
PRE_NORMALISED_TRAP: dict[str, Any] = dict(
    BASE_FEATURES, ki67_norm=0.20, tumor_size_norm=0.44,
    age_at_diagnosis_norm=0.58,
)

CASES: list[tuple[str, list[str], str]] = [
    # Expectations are the measured widths against MANSKI_MAX_WIDTH = 0.25,
    # not guesses. Dropping `histology` costs only 0.147815 of width, so a 200
    # there is the gate working, not the gate leaking.
    ("complete", [], "identified"),                                  # w 0.000000
    ("drop_pr", ["pr_positive"], "identified"),                      # w 0.095615
    ("drop_onehot_histology", ["histology"], "identified"),          # w 0.147815
    ("drop_onehot_grade", ["grade"], "ManskiBoundsExceeded"),        # w 0.339560
    ("drop_er_her2", ["er_positive", "her2_positive"],
     "ManskiBoundsExceeded"),                                        # w 0.468617
    ("drop_all_bools",
     ["er_positive", "pr_positive", "her2_positive", "lymph_nodes_pos",
      "brca_pathogenic"],
     "ManskiBoundsExceeded"),                                        # w 0.826868
    ("drop_continuous_ki67", ["ki67_pct"], "IDENTIFIED_SET_UNBOUNDED"),
    ("drop_continuous_size", ["tumor_size_mm"], "IDENTIFIED_SET_UNBOUNDED"),
    ("drop_continuous_age", ["age_years"], "IDENTIFIED_SET_UNBOUNDED"),
]

FEATURE_CASES: list[tuple[str, list[str]]] = [
    ("complete", []),
    ("drop_er_her2", ["er_status_positive", "her2_status_positive"]),
    ("drop_onehot_grade", ["grade"]),
    ("drop_continuous_ki67", ["ki67_norm"]),
]

# Expected widths on the raw panel, measured directly from the artefact.
EXPECTED_WIDTH: dict[str, float] = {
    "complete": 0.000000,
    "drop_pr": 0.095615,
    "drop_onehot_histology": 0.147815,
    "drop_onehot_grade": 0.339560,
    "drop_er_her2": 0.468617,
    "drop_all_bools": 0.826868,
    "drop_continuous_ki67": 1.000000,
    "drop_continuous_size": 1.000000,
    "drop_continuous_age": 1.000000,
}

BYPASS_HEADER_NAMES = [
    "X-Allow-Blind-Inference", "x-allow-blind-inference",
    "X-ALLOW-BLIND-INFERENCE", "X-Manski-Override", "X-Debug-Bypass",
    "X-Research-Mode", "Allow-Blind-Inference", "X-Blind-Inference",
    "X-Manski-Max-Width", "X-Override-Gate",
]
BYPASS_HEADER_VALUES = [
    "true", "TRUE", "True", " yes ", "1", "on", "false", "0", "",
    "maybe", "TrUe", "yes,true", "1 ", "\ttrue",
]


def _panel(drop: list[str]) -> dict[str, Any]:
    return {k: v for k, v in BASE_PANEL.items() if k not in drop}


def _features(drop: list[str]) -> dict[str, Any]:
    return {k: v for k, v in BASE_FEATURES.items() if k not in drop}


def _body(resp) -> dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_unparseable": resp.text[:400]}


def _gate_error(body: dict[str, Any]) -> str | None:
    """Return the gate error code, or None if this 422 was schema validation.

    FastAPI answers request-model violations with 422 too. Only a gate 422
    carries an `error` key at the top level; a schema 422 carries `detail`.
    Conflating them is how a broken gate passes its own test.
    """
    err = body.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return err.get("code") or err.get("error")
    return None


def run(out_path: Path) -> int:
    from fastapi.testclient import TestClient
    from oncology_arbiter.api.app import create_app
    from oncology_arbiter.auth import APIKeyDB
    from oncology_arbiter.arbiter.manski import (
        MANSKI_ERROR_CODE, MANSKI_MAX_WIDTH, RESEARCH_BLIND_INFERENCE_SCOPE,
        UNBOUNDED_ERROR_CODE,
    )

    # Rate limit is raised to 100000/minute at module scope: the production
    # default is 30/minute and the bypass sweep fires ~150 requests, so
    # without raising it the sweep measures the limiter and every 429
    # masquerades as a passed bypass test.
    db = APIKeyDB(Path(os.environ["ONCOLOGY_ARBITER_AUTH_DB_PATH"]))
    scoped_key, scoped_rec = db.issue(
        "research-tenant", scopes=[RESEARCH_BLIND_INFERENCE_SCOPE]
    )
    plain_key, plain_rec = db.issue("plain-tenant")

    client = TestClient(create_app())
    receipt: dict[str, Any] = {
        "receipt_type": "manski_gate_no_public_bypass",
        "generated_at": time.time(),
        "max_width": MANSKI_MAX_WIDTH,
        "transport": "in-process ASGI TestClient (localhost) — NOT production evidence",
        "public_routes": {},
        "bypass_attempts": [],
        "research_route": {},
        "violations": [],
    }
    violations: list[str] = []

    # ---------------------------------------------------------------- #
    # 1. Public routes, four levels of unobservedness each.
    for route, build in (
        ("/v1/therapy/reason",
         lambda p: {"patient_context": {"cancer_type": "breast"},
                    "therapy_features": p}),
        # DynamicTumorBoardRequest: the key is `cancer`, not `cancer_type`,
        # and extra fields are forbidden.
        ("/v1/tumor_board/dynamic",
         lambda p: {"cancer": "breast", "therapy_features": p}),
        # FullCaseRequest — there is no CaseFullRequest.
        ("/v1/case/full",
         lambda p: {"therapy_features": p}),
    ):
        per_route: dict[str, Any] = {}
        for name, drop, expect in CASES:
            r = client.post(route, json=build(_panel(drop)))
            body = _body(r)
            code = _gate_error(body)
            entry = {
                "status": r.status_code,
                "gate_error": code,
                "expected": expect,
                "dropped": drop,
            }
            if r.status_code == 422 and code is None:
                entry["schema_rejection_not_gate"] = True
                violations.append(
                    f"{route}[{name}]: 422 came from schema validation, not the "
                    f"gate — the probe payload is wrong, not the gate"
                )
            if code:
                entry["bounds"] = body.get("bounds")
                entry["width"] = body.get("width")
                entry["missing_features"] = body.get("missing_features")
                entry["unobserved_unbounded"] = body.get("unobserved_unbounded")
                entry["public_bypass_available"] = body.get("public_bypass_available")
                if body.get("public_bypass_available") is not False:
                    violations.append(
                        f"{route}[{name}]: public_bypass_available is not False"
                    )
            if r.status_code == 200:
                score = _find_arbiter_score(body)
                entry["arbiter_score_present"] = score is not None
                if score is not None:
                    m = score.get("manski") or {}
                    entry["p_positive"] = score.get("p_positive")
                    entry["width"] = m.get("width")
                    entry["identified"] = m.get("identified")
                    if m.get("identified") is False:
                        violations.append(
                            f"{route}[{name}]: 200 released an UNIDENTIFIED point "
                            f"estimate (width={m.get('width')})"
                        )
                    if (m.get("width") or 0.0) > MANSKI_MAX_WIDTH + 1e-9:
                        violations.append(
                            f"{route}[{name}]: 200 with width {m.get('width')} > "
                            f"{MANSKI_MAX_WIDTH}"
                        )
            exp_w = EXPECTED_WIDTH.get(name)
            if exp_w is not None and entry.get("width") is not None:
                entry["expected_width"] = exp_w
                if abs(entry["width"] - exp_w) > 1e-5:
                    violations.append(
                        f"{route}[{name}]: width {entry['width']} != measured "
                        f"reference {exp_w}"
                    )
            if r.status_code == 429:
                violations.append(
                    f"{route}[{name}]: 429 rate limited — sweep is measuring the "
                    f"limiter, not the gate"
                )
            if expect == "identified" and r.status_code != 200:
                violations.append(
                    f"{route}[{name}]: expected 200 (width below max), got "
                    f"{r.status_code} code={code}"
                )
            if expect in (MANSKI_ERROR_CODE, UNBOUNDED_ERROR_CODE):
                if code != expect:
                    violations.append(
                        f"{route}[{name}]: expected gate {expect}, got "
                        f"status={r.status_code} code={code}"
                    )
            per_route[name] = entry
        receipt["public_routes"][route] = per_route

    # ---------------------------------------------------------------- #
    # 2. Bypass sweep against a panel that is definitely gated.
    gated = {"patient_context": {"cancer_type": "breast"},
             "therapy_features": _panel(["er_positive", "her2_positive"])}
    for hname in BYPASS_HEADER_NAMES:
        for hval in BYPASS_HEADER_VALUES:
            r = client.post("/v1/therapy/reason", json=gated,
                            headers={hname: hval})
            ok = r.status_code == 422 and _gate_error(_body(r)) == MANSKI_ERROR_CODE
            receipt["bypass_attempts"].append(
                {"kind": "header", "name": hname, "value": hval,
                 "status": r.status_code, "still_gated": ok}
            )
            if r.status_code == 429:
                violations.append(
                    f"INCONCLUSIVE header {hname}={hval!r}: 429 rate limited"
                )
            elif not ok:
                violations.append(f"BYPASS via header {hname}={hval!r}")

    for qs in (
        "?allow_blind_inference=true",
        "?allow_blind_inference=1&manski_max_width=1.0",
        "?manski_max_width=0.99",
        "?bypass=true&debug=1",
    ):
        r = client.post(f"/v1/therapy/reason{qs}", json=gated)
        ok = r.status_code == 422 and _gate_error(_body(r)) == MANSKI_ERROR_CODE
        receipt["bypass_attempts"].append(
            {"kind": "query", "value": qs, "status": r.status_code,
             "still_gated": ok})
        if not ok:
            violations.append(f"BYPASS via query {qs}")

    for extra in (
        {"allow_blind_inference": True},
        {"manski_max_width": 1.0},
        {"bypass_manski": True},
    ):
        r = client.post("/v1/therapy/reason", json={**gated, **extra})
        body = _body(r)
        code = _gate_error(body)
        # Extra-field rejection (422 schema) is an acceptable outcome; a 200
        # is not.
        ok = r.status_code in (400, 422) and not (
            r.status_code == 200
        )
        receipt["bypass_attempts"].append(
            {"kind": "body_field", "value": list(extra), "status": r.status_code,
             "gate_error": code, "still_gated": ok})
        if r.status_code == 200:
            violations.append(f"BYPASS via body field {list(extra)}")

    # ---------------------------------------------------------------- #
    # 3. Research route: locked by scope, not by AUTH_MODE.
    rr = "/v1/research/arbiter/identified_set"
    probe = {"arbiter": "therapy",
             "features": _features(["er_status_positive", "her2_status_positive"]),
             "acknowledge_not_for_clinical_use": True}

    # AUTH_MODE=off is live right now. The anonymous principal must still fail.
    r = client.post(rr, json=probe)
    receipt["research_route"]["no_key_auth_mode_off"] = {
        "status": r.status_code, "body": _body(r)}
    if r.status_code != 403:
        violations.append(
            f"research route returned {r.status_code} to the anonymous "
            f"principal with AUTH_MODE=off; expected 403"
        )

    # Flip auth on. Everything below exercises real key lookup; with
    # AUTH_MODE=off `require_api_key` short-circuits to the anonymous
    # principal and never reads the header at all, so a scoped key would be
    # invisible. That is correct fail-closed behaviour and is exactly what the
    # check above proves; it also means the happy path must be tested with
    # auth genuinely enabled.
    os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = "on"
    receipt["research_route"]["auth_mode_for_scoped_tests"] = "on"

    r = client.post(rr, json=probe)
    receipt["research_route"]["no_key_auth_mode_on"] = {"status": r.status_code}
    if r.status_code != 401:
        violations.append(
            f"research route returned {r.status_code} with no API key and "
            f"auth on; expected 401"
        )

    r = client.post(rr, json=probe, headers={"X-API-Key": "oa_live_deadbeef"})
    receipt["research_route"]["bogus_key"] = {"status": r.status_code}
    if r.status_code not in (401, 403):
        violations.append(
            f"research route returned {r.status_code} to a bogus key"
        )

    r = client.post(rr, json=probe, headers={"X-API-Key": plain_key})
    receipt["research_route"]["valid_key_no_scope"] = {
        "status": r.status_code,
        "error": _body(r).get("detail", {}).get("error")
        if isinstance(_body(r).get("detail"), dict) else None,
    }
    if r.status_code != 403:
        violations.append(
            f"research route returned {r.status_code} to a key without the "
            f"scope; expected 403"
        )

    r = client.post(rr, json={**probe, "acknowledge_not_for_clinical_use": False},
                    headers={"X-API-Key": scoped_key})
    receipt["research_route"]["scoped_key_no_acknowledgement"] = {
        "status": r.status_code}
    if r.status_code != 422:
        violations.append(
            "research route accepted a request without the "
            "not-for-clinical-use acknowledgement"
        )

    released: dict[str, Any] = {}
    for name, drop in FEATURE_CASES:
        r = client.post(
            rr,
            json={"arbiter": "therapy", "features": _features(drop),
                  "acknowledge_not_for_clinical_use": True},
            headers={"X-API-Key": scoped_key},
        )
        body = _body(r)
        entry = {"status": r.status_code, "dropped": drop}
        if r.status_code == 200:
            entry.update({
                "point_estimate": body["point_estimate"],
                "bounds": [body["bounds"]["lower"], body["bounds"]["upper"]],
                "width": body["bounds"]["width"],
                "identified": body["bounds"]["identified"],
                "would_have_been_rejected": body["would_have_been_rejected"],
                "public_route_error_code": body["public_route_error_code"],
                "missing_features": body["missing_features"],
                "unobserved_unbounded": body["unobserved_unbounded"],
                "n_warnings": len(body["warnings"]),
                "warnings": body["warnings"],
                "receipt_sha_field": body["audit_receipt"]["request_features_sha256"],
                "receipt_tenant": body["audit_receipt"]["tenant_id"],
                "not_for_clinical_use": body["not_for_clinical_use"],
            })
            if not body["bounds"]["identified"] and not body["warnings"]:
                violations.append(
                    f"research[{name}]: unidentified release carried no warnings"
                )
            if body["bounds"]["identified"] != (not body["would_have_been_rejected"]):
                violations.append(
                    f"research[{name}]: identified/would_have_been_rejected disagree"
                )
        else:
            violations.append(
                f"research[{name}]: scoped key got {r.status_code}, expected 200"
            )
        released[name] = entry
    receipt["research_route"]["scoped_releases"] = released
    receipt["research_route"]["tenant_ids"] = {
        "scoped": scoped_rec.tenant_id, "plain": plain_rec.tenant_id}

    # Audit ledger must actually contain the receipts we just generated.
    audit_root = Path(os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"])
    logged = []
    for p in sorted(audit_root.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("endpoint") == rr:
                logged.append(ev["extra"]["blind_inference_receipt"])
    receipt["research_route"]["audit_ledger_entries"] = len(logged)
    receipt["research_route"]["audit_ledger_sample"] = logged[:1]
    if len(logged) < len(FEATURE_CASES):
        violations.append(
            f"audit ledger holds {len(logged)} blind-inference receipts, "
            f"expected >= {len(FEATURE_CASES)}"
        )

    # ---------------------------------------------------------------- #
    # 4. Record the continuous-encoding unit trap as a first-class finding.
    from oncology_arbiter.arbiter import load_arbiter as _la
    _arb = _la("therapy")
    _raw = _arb.score(BASE_FEATURES)
    _pre = _arb.score(PRE_NORMALISED_TRAP)
    receipt["findings"] = {
        "continuous_encoding_unit_trap": {
            "severity": "high",
            "statement": (
                "Continuous feature keys are named `*_norm` but the artefact "
                "declares the divisor and applies it, so the caller must pass "
                "the RAW value. Passing an already-normalised value is "
                "silently accepted."
            ),
            "declared_encodings": {
                k: v for k, v in _arb.feature_encodings.items()
                if isinstance(v, str)
            },
            "raw_panel_p": _raw.p_positive,
            "pre_normalised_panel_p": _pre.p_positive,
            "absolute_error": _pre.p_positive - _raw.p_positive,
            "relative_error_pct": 100.0 * (_pre.p_positive - _raw.p_positive)
            / _raw.p_positive,
            "errors_raised": 0,
            "warnings_raised": 0,
            "range_validation_present": False,
            "public_route_affected": False,
            "public_route_reason": (
                "app.py maps the wire field ki67_pct straight onto ki67_norm "
                "without pre-dividing, so the public path is correct"
            ),
            "research_route_affected": True,
            "research_route_reason": (
                "the research route accepts a free-form feature dict, so a "
                "caller who reads the key name and pre-normalises gets a "
                "wrong-but-plausible probability"
            ),
            "root_cause_shared_with_unbounded_gate": (
                "the artefact declares a divisor but no admissible support; "
                "the same missing declaration is what forces "
                "IDENTIFIED_SET_UNBOUNDED and what makes the unit error "
                "undetectable"
            ),
        }
    }

    receipt["violations"] = violations
    receipt["verdict"] = "NO_PUBLIC_BYPASS" if not violations else "BYPASS_FOUND"
    receipt["n_bypass_attempts"] = len(receipt["bypass_attempts"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({
        "verdict": receipt["verdict"],
        "n_bypass_attempts": receipt["n_bypass_attempts"],
        "n_violations": len(violations),
        "violations": violations[:10],
        "out": str(out_path),
    }, indent=2))
    return 1 if violations else 0


def _find_arbiter_score(body: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the arbiter score block wherever a route happens to nest it."""
    for key in ("arbiter_score", "therapy_triage"):
        v = body.get(key)
        if isinstance(v, dict):
            return v
    for v in body.values():
        if isinstance(v, dict):
            found = _find_arbiter_score(v)
            if found is not None:
                return found
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "artifacts/audit/manski_gate_verification.json")
    a = ap.parse_args()
    raise SystemExit(run(a.out))
