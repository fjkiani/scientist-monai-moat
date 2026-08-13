#!/usr/bin/env python3
"""Quantify what the retired 0.5-midpoint missingness encoding injected.

Recomputes everything from the COMMITTED arbiter template artefacts -- no
hard-coded probabilities -- and writes one PNG.

Question: when every boolean feature of a template arbiter is unobserved, what
probability comes out under each encoding?

  * reference level 0.0  -> logit = intercept, so p = sigmoid(intercept),
                            which is exactly the intercept-only base rate.
  * prevalence 0.1192    -> the user's stated target. 0.1192 is a PROBABILITY
                            (sigmoid(-2.0) for the screening template), not a
                            feature value, so substituting it as the feature
                            value still injects 0.1192 * sum(coef) log-odds.
                            sigmoid(E[x]) != E[sigmoid(x)] (Jensen).
  * midpoint 0.5 (retired) -> injects 0.5 * sum(coef) log-odds out of nothing.

The right panel gives the assumption-free (Manski) bounds obtained by letting
each unobserved boolean take its worst/best admissible value, which is the only
honest summary when the value is genuinely unknown.

Usage: python scripts/plot_missingness_encoding_impact.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "oncology_arbiter" / "arbiter" / "models"
TEMPLATES = ("screening", "biopsy", "therapy")

# The user's "true base rate 0.1192" is sigmoid(-2.0) to full precision.
PREVALENCE = 1.0 / (1.0 + math.exp(2.0))
MIDPOINT = 0.5

OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"
BLACK = "#000000"
GREEN = "#009E73"


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def analyse(name: str) -> dict:
    """Recompute the encoding impact for one committed template artefact."""
    artefact = MODELS / f"{name}_arbiter_template_v0.json"
    payload = json.loads(artefact.read_text(encoding="utf-8"))
    coefs: dict[str, float] = payload["coefficients"]
    encodings: dict[str, dict] = payload["feature_encodings"]

    bool_feats: list[str] = []
    for feature, enc in encodings.items():
        if not isinstance(enc, dict) or "true" not in enc:
            continue  # one-hot categorical or numeric normaliser
        # Post-repair invariant: an unobserved boolean is never the 0.5 midpoint.
        unknown = enc.get("unknown", None)
        assert unknown in (None, 0.0), f"{name}/{feature}: unknown={unknown!r}"
        bool_feats.append(feature)

    intercept = float(payload["intercept"])
    bool_coefs = [float(coefs[f]) for f in bool_feats]
    total = sum(bool_coefs)

    return {
        "template": name,
        "n_bool": len(bool_feats),
        "sum_bool_coef": total,
        "intercept": intercept,
        "p_reference": sigmoid(intercept),
        "p_prevalence": sigmoid(intercept + PREVALENCE * total),
        "p_midpoint": sigmoid(intercept + MIDPOINT * total),
        "injected_logodds_at_midpoint": MIDPOINT * total,
        "injected_logodds_at_prevalence": PREVALENCE * total,
        # Manski bounds: each unobserved boolean is free in {0, 1}.
        "p_lower": sigmoid(intercept + sum(c for c in bool_coefs if c < 0)),
        "p_upper": sigmoid(intercept + sum(c for c in bool_coefs if c > 0)),
    }


def plot(rows: list[dict], out: Path) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.0, 5.6))
    names = [r["template"] for r in rows]
    x = list(range(len(rows)))
    width = 0.26

    series = [
        ("unobserved encoded at 0.0 (reference level)", "p_reference", OKABE_BLUE),
        (f"unobserved encoded at {PREVALENCE:.4f} (prevalence)", "p_prevalence", OKABE_ORANGE),
        ("unobserved encoded at 0.5 (retired midpoint)", "p_midpoint", BLACK),
    ]
    for i, (label, key, colour) in enumerate(series):
        offs = [xi + (i - 1) * width for xi in x]
        vals = [r[key] for r in rows]
        ax_l.bar(offs, vals, width, label=label, color=colour,
                 edgecolor="black", linewidth=0.6)
        for xo, v in zip(offs, vals):
            ax_l.text(xo, v + 0.006, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    for xi, r in zip(x, rows):
        ax_l.hlines(r["p_reference"], xi - 1.6 * width, xi + 1.6 * width,
                    colors=GREEN, linestyles="--", linewidth=1.6,
                    label="intercept-only base rate" if xi == 0 else None)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(names)
    ax_l.set_xlabel("Arbiter template")
    ax_l.set_ylabel("Predicted probability when the whole\nboolean panel is unobserved")
    ax_l.set_ylim(0.0, max(r["p_midpoint"] for r in rows) * 1.42)
    ax_l.legend(loc="upper left", fontsize=9, frameon=False)
    ax_l.spines[["top", "right"]].set_visible(False)

    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        lo, hi = r["p_lower"], r["p_upper"]
        ax_r.hlines(yi, lo, hi, color=OKABE_BLUE, linewidth=3.0,
                    label="assumption-free (Manski) bounds" if yi == y[0] else None)
        ax_r.vlines([lo, hi], yi - 0.16, yi + 0.16, color=OKABE_BLUE, linewidth=3.0)
        ax_r.plot([r["p_reference"]], [yi], "o", color=OKABE_BLUE, markersize=11,
                  zorder=3, label="point estimate (reference encoding)" if yi == y[0] else None)
        ax_r.plot([r["p_midpoint"]], [yi], "X", color=BLACK, markersize=12,
                  zorder=3, label="retired 0.5 midpoint estimate" if yi == y[0] else None)
        ax_r.text(lo, yi + 0.26, f"{lo:.4f}", ha="center", va="bottom", fontsize=9)
        ax_r.text(hi, yi + 0.26, f"{hi:.4f}", ha="center", va="bottom", fontsize=9)
        # Width label sits clear of the interval, right of the upper cap.
        ax_r.text(hi + 0.035, yi, f"width {hi - lo:.4f}", ha="left", va="center",
                  fontsize=9, color="#333333")
        # Point label below the row, offset from whichever marker is leftmost.
        ax_r.text(min(r["p_reference"], r["p_midpoint"]), yi - 0.30,
                  f"point {r['p_reference']:.4f}", ha="center", va="top",
                  fontsize=9, color=OKABE_BLUE)

    ax_r.set_yticks(y)
    ax_r.set_yticklabels(names)
    ax_r.set_ylim(-0.85, len(rows) - 0.35)
    ax_r.set_xlim(-0.04, 1.28)
    ax_r.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_r.set_xlabel("Predicted probability")
    ax_r.set_ylabel("Arbiter template")
    ax_r.legend(loc="lower center", bbox_to_anchor=(0.42, -0.02), ncol=3,
                fontsize=9, frameon=False)
    ax_r.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/results/audit_v2/figures/missingness_encoding_impact_v3.png")
    args = ap.parse_args()

    rows = [analyse(t) for t in TEMPLATES]
    header = (
        f"{'template':<10} {'n_bool':>6} {'sum_coef':>9} {'p_ref':>11} "
        f"{'p_prev':>11} {'p_mid':>11} {'inj@0.5':>10} {'lower':>11} {'upper':>11} {'width':>11}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['template']:<10} {r['n_bool']:>6d} {r['sum_bool_coef']:>9.4f} "
            f"{r['p_reference']:>11.8f} {r['p_prevalence']:>11.8f} {r['p_midpoint']:>11.8f} "
            f"{r['injected_logodds_at_midpoint']:>+10.4f} {r['p_lower']:>11.8f} "
            f"{r['p_upper']:>11.8f} {r['p_upper'] - r['p_lower']:>11.8f}"
        )
    scr = rows[0]
    print(f"\nPREVALENCE constant = {PREVALENCE!r}")
    print(
        "screening: encoding the FEATURE at the prevalence still injects "
        f"{scr['injected_logodds_at_prevalence']:.6f} log-odds and overshoots the "
        f"base rate by {100 * (scr['p_prevalence'] / scr['p_reference'] - 1):.2f}%; "
        "only the reference level 0.0 reproduces it exactly."
    )
    plot(rows, Path(args.out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
