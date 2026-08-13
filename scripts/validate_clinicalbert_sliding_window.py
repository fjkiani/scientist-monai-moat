"""Validate the deployed 192/32 ClinicalBERT inference on 296 real reports.

This is an in-corpus reproducibility measurement, not a held-out estimate:
the deployment rule (192-token horizon, 32-token overlap, mean-logit merge)
was fixed from checkpoint metadata and the user directive, but the same 296
reports were previously inspected during the audit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, "/mnt/shared-workspace/shared/arbiter_audit")
from cbert_rescore import decode_spans, load_test, score_perdoc

from transformers import AutoModelForTokenClassification, AutoTokenizer

from oncology_arbiter.nlp.clinicalbert_sliding_window import (
    AGGREGATION,
    OVERLAP_TOKENS,
    WINDOW_TOKENS,
    predict_words_sliding_window,
)

CKPT = Path("/mnt/shared-workspace/shared/clinicalbert_runs_v51/seed_123")
OUT = Path("/workspace/smm/sliding_window_validation.json")


def main() -> None:
    reports = load_test()
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForTokenClassification.from_pretrained(CKPT).eval()
    label_map = json.load(open(CKPT / "label_map.json"))
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    gold_docs, pred_docs, cancers = [], [], []
    n_windows = []
    t0 = time.time()
    for i, report in enumerate(reports, 1):
        pred = predict_words_sliding_window(
            tokenizer=tokenizer,
            model=model,
            words=report["tokens"],
            id2label=id2label,
            window_tokens=WINDOW_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
        )
        gold_docs.append(decode_spans(report["labels"]))
        pred_docs.append(decode_spans(pred.labels))
        cancers.append(report["_cancer"])
        n_windows.append(pred.n_windows)
        if i % 25 == 0 or i == len(reports):
            print(f"{i}/{len(reports)} reports", flush=True)

    overall = score_perdoc(gold_docs, pred_docs)
    by_cancer = {}
    for cancer in sorted(set(cancers)):
        idx = [i for i, c in enumerate(cancers) if c == cancer]
        by_cancer[cancer] = score_perdoc(
            [gold_docs[i] for i in idx], [pred_docs[i] for i in idx]
        )

    out = {
        "status": "IN_CORPUS_REPRODUCIBILITY_NOT_HELD_OUT_VALIDATION",
        "checkpoint": str(CKPT),
        "n_reports": len(reports),
        "total_words": sum(len(r["tokens"]) for r in reports),
        "window_tokens": WINDOW_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "aggregation": AGGREGATION,
        "n_windows_total": sum(n_windows),
        "n_windows_min": min(n_windows),
        "n_windows_max": max(n_windows),
        "overall": overall,
        "by_cancer": by_cancer,
        "seconds": time.time() - t0,
        "comparison_only": {
            "prior_audit_configuration_f1": 0.25724104074619536,
            "prior_configuration": (
                "256-token budget, approximately 50% word-window overlap, "
                "highest-confidence overlap merge; same 296 reports"
            ),
            "same_estimator": False,
            "warning": (
                "A numerical difference is expected because the mandated "
                "192/32/mean-logit deployment estimator is not the prior "
                "256/50%-overlap/max-confidence audit estimator."
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"overall": overall["micro"], "by_cancer": {
        k: v["micro"] for k, v in by_cancer.items()
    }, "n_windows_total": sum(n_windows), "seconds": out["seconds"]}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
