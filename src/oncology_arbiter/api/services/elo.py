"""Elo rank modifier reason helper."""
from __future__ import annotations

def elo_reason_for_modifier(
    drug_id: str,
    delta: float,
    disease_context: dict,
) -> str:
    """Best-effort human-readable reason for the applied modifier delta.

    Kept deliberately terse — the SPA renders `rank_delta`, `applied_modifier`
    and `reason` side-by-side, so this text is a *hint*, not the source of
    truth.
    """
    if delta == 0.0:
        return "no modifier"

    d_low = drug_id.lower()
    ctx = disease_context or {}
    hrd = bool(ctx.get("hrd_positive")) or bool(ctx.get("brca_mutated"))
    pdl1 = ctx.get("pd_l1_cps")
    prior_lines = ctx.get("prior_lines")

    parp_names = ("olaparib", "niraparib", "rucaparib", "talazoparib")
    if any(n in d_low for n in parp_names) and hrd and delta > 0:
        return f"HRD+PARP boost ({delta:+.2f})"

    if "bevacizumab" in d_low and delta > 0:
        return f"Bevacizumab GOG-0218 posture ({delta:+.2f})"

    if any(n in d_low for n in ("pembrolizumab", "atezolizumab", "durvalumab", "nivolumab")):
        try:
            cps = float(pdl1) if pdl1 is not None else None
        except (TypeError, ValueError):
            cps = None
        if delta > 0 and cps is not None and cps >= 10:
            return f"PD-L1 CPS>=10 boost ({delta:+.2f})"
        if delta < 0 and cps is not None and cps < 10:
            return f"PD-L1 CPS<10 penalty ({delta:+.2f})"

    try:
        pl = int(prior_lines) if prior_lines is not None else None
    except (TypeError, ValueError):
        pl = None
    if delta < 0 and pl is not None and pl >= 3:
        return f"prior_lines={pl} penalty ({delta:+.2f})"

    if "ceralasertib" in d_low and delta != 0:
        return f"CAPRI ATR/PARP posture ({delta:+.2f})"

    if delta > 0:
        return f"manual boost ({delta:+.2f})"
    return f"manual penalty ({delta:+.2f})"



class EloModifierReasoner:
    """Class facade for Elo modifier reason text."""

    reason_for_modifier = staticmethod(elo_reason_for_modifier)
