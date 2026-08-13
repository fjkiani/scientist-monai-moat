"""Fail-closed acceptance gate for the clinical report parser.

Directive 3. The deployed ClinicalBERT NER app answers HTTP 200 and returns
well-formed entities, so *transport* success is not evidence of *clinical*
fitness. Two independent facts about the live deployment disqualify it:

1. It self-reports ``test_micro_f1 = 0.0808926081`` on its own held-out set.
   The acceptance floor for clinical routing is 0.70.
2. It self-reports ``clinicalbert-modal-v0.5.1`` while the in-repo client
   contract pins ``clinicalbert-modal-v0.5.2-sliding-window``. The 192/32
   sliding-window aggregation the contract depends on is not deployed, so the
   long-report truncation behaviour is unknown, not merely unverified.

Measured consequences on the audited report (see
``artifacts/audit/clinicalbert_field_errors.json``): the parser returned
``HER2_VALUE = "positive"`` for a report documenting ``HER2 IHC 1+ (NEGATIVE)``,
and ``KI67_PCT = 67`` for a report documenting ``Ki-67 ... 20%`` -- reading the
digits out of the analyte name. Both wrong values sit inside spans whose
``surface`` text contains the correct answer, so localisation worked and value
assignment did not.

This module is the single chokepoint. It does not "warn"; it refuses. There is
no regex fallback, no fusion with a rules parser, and no inferred substitution,
because a wrong receptor value is more dangerous than a null one: a null is
visibly missing, whereas ``HER2 = positive`` is silently actionable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from oncology_arbiter.nlp.clinicalbert_modal_client import EXPECTED_APP_VERSION

#: Minimum held-out micro-F1 for the parser to be eligible for clinical routing.
PARSER_MIN_MICRO_F1: float = 0.70

#: The only app version whose wire contract this repo implements.
PARSER_REQUIRED_APP_VERSION: str = EXPECTED_APP_VERSION

#: Held-out metrics observed from each deployment's own ``/info`` endpoint, keyed
#: by the ``app_version`` that deployment reports. Recorded from live probes in
#: ``artifacts/audit/modal_fleet_live_inference.json`` so the F1 floor can be
#: enforced without a second network round-trip per request. A version absent
#: from this table has unverifiable metrics and is therefore refused.
DEPLOYED_PARSER_METRICS: dict[str, dict[str, float]] = {
    "clinicalbert-modal-v0.5.1": {
        "test_micro_f1": 0.08089260808926081,
        "real_text_micro_f1_breast_crc": 0.08785529715762275,
        "real_text_micro_f1_nsclc": 0.06961325966850829,
    },
}

BELOW_FLOOR_CODE: str = "PARSER_BELOW_VALIDATION_FLOOR"
VERSION_MISMATCH_CODE: str = "PARSER_VERSION_MISMATCH"
UNVERIFIABLE_CODE: str = "PARSER_METRICS_UNVERIFIABLE"

#: Fields that may only ever be populated from an ACCEPTED parser. If the gate
#: refuses, every one of these must be absent or ``None`` in the response --
#: including anything derived downstream (DSS, subtype, therapy options).
PARSER_DERIVED_FIELDS: frozenset[str] = frozenset(
    {
        "report_parse",
        "her2_status",
        "her2_value",
        "ki67_pct",
        "ki67",
        "grade",
        "parsed_grade",
        "er_status",
        "er_value",
        "pr_status",
        "pr_value",
        "receptor_panel",
        "molecular_subtype",
        "subtype",
        "dss_prognosis",
        "therapy_options",
        "tumor_size_mm_parsed",
        "lvi",
        "t_stage",
        "n_stage",
        "m_stage",
        "margin",
    }
)


@dataclass(frozen=True)
class ParserAcceptance:
    """Outcome of evaluating a parser deployment against the gate."""

    accepted: bool
    error_code: str | None
    detail: str
    observed_app_version: str | None
    observed_micro_f1: float | None
    required_app_version: str = PARSER_REQUIRED_APP_VERSION
    required_micro_f1: float = PARSER_MIN_MICRO_F1
    blocked_fields: tuple[str, ...] = field(default=())
    fallback_permitted: bool = False

    def __post_init__(self) -> None:
        if self.accepted and self.error_code is not None:
            raise ValueError("an accepted parser cannot carry an error code")
        if not self.accepted and self.error_code is None:
            raise ValueError("a refused parser must carry an error code")
        if self.fallback_permitted:
            raise ValueError(
                "no fallback is permitted: substituting a regex or rules parser "
                "for a refused NER model reintroduces the retired "
                "regex_report_parser capability"
            )

    def payload(self) -> dict[str, Any]:
        """Machine-readable refusal body for a required-stage receipt."""
        return {
            "error": self.error_code,
            "detail": self.detail,
            "stage": "clinicalbert_pathology_parse",
            "observed_app_version": self.observed_app_version,
            "required_app_version": self.required_app_version,
            "observed_test_micro_f1": self.observed_micro_f1,
            "required_test_micro_f1": self.required_micro_f1,
            "report_parse": None,
            "parser_derived_fields_withheld": sorted(self.blocked_fields),
            "regex_fallback_used": False,
            "fusion_used": False,
            "inferred_value_substitution_used": False,
            "required_action": (
                "Deploy a parser that reports "
                f"app_version == {self.required_app_version!r} and held-out "
                f"micro-F1 >= {self.required_micro_f1} on a token-level "
                "annotated corpus with a patient-level holdout, then re-run "
                "acceptance."
            ),
        }


#: Env var an operator sets to declare which parser build is actually deployed.
#: It defaults to the version observed live during the audit, so the gate is
#: closed by default: an operator who deploys a conforming build must say so
#: explicitly, and the post-parse layer still checks the claim against the wire.
DEPLOYED_VERSION_ENV: str = "CLINICALBERT_DEPLOYED_APP_VERSION"
AUDITED_DEPLOYED_APP_VERSION: str = "clinicalbert-modal-v0.5.1"


def declared_deployment_info() -> dict[str, Any]:
    """Operator-declared deployment identity for the pre-flight check."""
    import os

    version = os.environ.get(DEPLOYED_VERSION_ENV, AUDITED_DEPLOYED_APP_VERSION)
    return {"app_version": version, **DEPLOYED_PARSER_METRICS.get(version, {})}


def _coerce_f1(info: Mapping[str, Any]) -> float | None:
    for key in ("test_micro_f1", "micro_f1", "held_out_micro_f1"):
        v = info.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    metrics = info.get("metrics")
    if isinstance(metrics, Mapping):
        return _coerce_f1(metrics)
    return None


def evaluate_parser_acceptance(info: Mapping[str, Any] | None) -> ParserAcceptance:
    """Decide whether a parser deployment may serve clinical routing.

    ``info`` is the deployment's own ``/info`` payload. We deliberately trust
    the deployment's self-reported metrics as an *upper* bound only: a model
    claiming 0.08 cannot be better than 0.08, so a claim below the floor is
    sufficient to refuse without re-running evaluation. A claim *above* the
    floor is NOT sufficient to accept on its own -- it must also match the
    pinned app version, which is what ties the number to a known corpus and
    aggregation scheme.
    """
    blocked = tuple(sorted(PARSER_DERIVED_FIELDS))
    if info is None:
        return ParserAcceptance(
            accepted=False,
            error_code=UNVERIFIABLE_CODE,
            detail=(
                "the parser deployment did not return an /info payload, so its "
                "held-out micro-F1 and app version cannot be verified; a parser "
                "whose validation state is unknown is refused"
            ),
            observed_app_version=None,
            observed_micro_f1=None,
            blocked_fields=blocked,
        )

    version = info.get("app_version") or info.get("version")
    version = str(version) if version is not None else None
    f1 = _coerce_f1(info)

    # Version is checked FIRST: a mismatched version means the reported F1
    # belongs to a different artifact, so the floor comparison is meaningless.
    if version != PARSER_REQUIRED_APP_VERSION:
        return ParserAcceptance(
            accepted=False,
            error_code=VERSION_MISMATCH_CODE,
            detail=(
                f"deployed app_version {version!r} does not match the pinned "
                f"contract {PARSER_REQUIRED_APP_VERSION!r}; the sliding-window "
                "aggregation this client depends on is not proven present, so "
                "long-report truncation behaviour is unknown"
            ),
            observed_app_version=version,
            observed_micro_f1=f1,
            blocked_fields=blocked,
        )

    if f1 is None:
        return ParserAcceptance(
            accepted=False,
            error_code=UNVERIFIABLE_CODE,
            detail=(
                "the parser deployment reported no held-out micro-F1; an "
                "unmeasured parser is refused rather than assumed adequate"
            ),
            observed_app_version=version,
            observed_micro_f1=None,
            blocked_fields=blocked,
        )

    if f1 < PARSER_MIN_MICRO_F1:
        return ParserAcceptance(
            accepted=False,
            error_code=BELOW_FLOOR_CODE,
            detail=(
                f"deployed parser self-reports held-out micro-F1 {f1!r}, below "
                f"the clinical-routing floor {PARSER_MIN_MICRO_F1}; parser-derived "
                "receptor, proliferation, grade, subtype, DSS and therapy fields "
                "are withheld"
            ),
            observed_app_version=version,
            observed_micro_f1=f1,
            blocked_fields=blocked,
        )

    return ParserAcceptance(
        accepted=True,
        error_code=None,
        detail=(
            f"parser {version} meets the pinned contract and reports held-out "
            f"micro-F1 {f1!r} >= {PARSER_MIN_MICRO_F1}"
        ),
        observed_app_version=version,
        observed_micro_f1=f1,
    )


def assert_no_parser_derived_values(payload: Mapping[str, Any]) -> list[str]:
    """Return the list of parser-derived keys that leaked a non-null value.

    Used by the end-to-end tests to prove treatment output did not consume a
    refused parser's values. Recurses so a value buried in ``dss_prognosis`` or
    ``therapy.options[i]`` is still caught.
    """
    leaks: list[str] = []

    def has_nonnull_leaf(node: Any) -> bool:
        """True if any *scalar* leaf under ``node`` carries a value.

        A container whose every leaf is ``None`` is an empty shell, not a leak:
        ``receptor_panel = {er_positive: None, her2_status: None, ...}`` is the
        correct fail-closed serialisation of "we parsed nothing", and flagging
        it would make the invariant unfalsifiable.
        """
        if isinstance(node, Mapping):
            return any(has_nonnull_leaf(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return any(has_nonnull_leaf(v) for v in node)
        return node not in (None, "", [], {})

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                if k in PARSER_DERIVED_FIELDS and has_nonnull_leaf(v):
                    leaks.append(p)
                walk(v, p)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload, "")
    return leaks
