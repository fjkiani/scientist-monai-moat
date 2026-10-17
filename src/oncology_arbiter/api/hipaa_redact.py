"""Field redaction for tumor-board bundles when HIPAA_MODE is on.

Replaces identifier fields and any later copy of those raw strings.
This is a dictionary-plus-regex scrub of known identifier fields, not a
clinical de-identification product.
"""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
PHI_FIELD_NAMES = frozenset({
    "patient_id",
    "name",
    "patient_name",
    "full_name",
    "patientname",
})


def _collect_secrets(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in PHI_FIELD_NAMES and isinstance(value, str):
                stripped = value.strip()
                if stripped and stripped != REDACTED:
                    found.add(stripped)
            _collect_secrets(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_secrets(item, found)


def _scrub_string(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        redacted = re.sub(re.escape(secret), REDACTED, redacted)
    return redacted


def _redact_node(node: Any, secrets: list[str]) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if str(key).lower() in PHI_FIELD_NAMES:
                out[key] = REDACTED if value not in (None, "") else value
            else:
                out[key] = _redact_node(value, secrets)
        return out
    if isinstance(node, list):
        return [_redact_node(item, secrets) for item in node]
    if isinstance(node, str):
        return _scrub_string(node, secrets)
    return node


def redact_phi(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with patient_id and name fields replaced.

    The raw identifier is also stripped from every other string in the
    payload so a note cannot echo the value the field redaction removed.
    """
    secrets: set[str] = set()
    _collect_secrets(payload, secrets)
    ordered = sorted(secrets, key=len, reverse=True)
    redacted = _redact_node(payload, ordered)
    if not isinstance(redacted, dict):
        raise TypeError("HIPAA redactor expected an object payload")
    return redacted
