"""
Fine-tune Bio_ClinicalBERT for pathology-report entity extraction.

Trainer for the token-classification model that backs the
`clinicalbert_v1` report parser. Reads the synthetic JSONL splits produced by
`corpus_synth.py`, aligns whitespace/punctuation tokens to WordPiece
sub-tokens, fine-tunes for a small number of epochs on CPU (or GPU if
available), and evaluates entity-level precision/recall/F1 on the held-out
test split.

Design honesty
--------------
- Base: `emilyalsentzer/Bio_ClinicalBERT` (BioBERT + MIMIC-III clinical notes
  pretraining). This is a public model and the license permits derivative
  weights.
- Training data: 100% synthetic (see `corpus_synth.py`). Every prediction
  the model makes downstream is stamped with `parser_id="clinicalbert_v1"`
  and `provenance="SYNTHETIC-v0.3.0"` so the UI can flag the honesty
  contract.
- Sub-token label propagation: the first WordPiece of a whitespace token gets
  the whitespace token's label; continuation sub-tokens get label -100 so
  cross-entropy ignores them. This is the standard HuggingFace pattern.
- We report entity-level F1 (a predicted span is correct only if the entity
  type AND the surface span match a gold span), not just per-token
  accuracy, because per-token accuracy is inflated by O tags.

Artifacts saved (under `--out-dir`, default `models/report_parser_clinicalbert_v1/`)
- `config.json`, `tokenizer.json`, `vocab.txt`, `model.safetensors` — HF-format
  weights
- `label_map.json` — LABEL2ID / ID2LABEL used at inference time
- `metrics.json` — per-entity precision / recall / F1 + overall micro-F1
- `train_summary.json` — corpus size, epochs, batch, timing, base model repo
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from oncology_arbiter.nlp.corpus_synth import (
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
    generate_corpus,
    save_split,
)


class ReportTaggerDataset(Dataset):
    """Turn (tokens, labels) whitespace tuples into padded model inputs."""

    def __init__(self, reports, tokenizer, max_len: int = 256):
        self.reports = reports
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        r = self.reports[idx]
        enc = self.tokenizer(
            r.tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(LABEL2ID[r.labels[wid]])
                prev = wid
            else:
                # Continuation sub-token: ignore in loss.
                aligned.append(-100)
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


def _decode_spans(tokens, labels):
    """From (tokens, BIO labels) recover entity spans as
    (entity_type, start_tok, end_tok_exclusive)."""
    out = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            out.append((etype, i, j))
            i = j
        else:
            i += 1
    return out


def _entity_f1(gold, pred):
    """Micro + per-entity precision / recall / F1.

    gold / pred are lists of (etype, start, end) tuples.
    """
    tp_by = defaultdict(int)
    fp_by = defaultdict(int)
    fn_by = defaultdict(int)

    gset = set(gold)
    pset = set(pred)

    for e in gset & pset:
        tp_by[e[0]] += 1
    for e in pset - gset:
        fp_by[e[0]] += 1
    for e in gset - pset:
        fn_by[e[0]] += 1

    etypes = set(list(tp_by.keys()) + list(fp_by.keys()) + list(fn_by.keys()))
    per_entity = {}
    for et in etypes:
        tp, fp, fn = tp_by[et], fp_by[et], fn_by[et]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_entity[et] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    tp = sum(tp_by.values())
    fp = sum(fp_by.values())
    fn = sum(fn_by.values())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"micro": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1},
            "per_entity": per_entity}


def evaluate(model, tokenizer, reports, device, max_len: int = 256):
    """Return entity-level metrics."""
    model.eval()
    gold_all, pred_all = [], []
    with torch.no_grad():
        for r in reports:
            enc = tokenizer(
                r.tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits
            preds = logits.argmax(-1)[0].cpu().tolist()
            word_ids = enc.word_ids(batch_index=0)
            pred_labels_per_word: list[str | None] = [None] * len(r.tokens)
            prev = None
            for tok_idx, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    continue
                pred_labels_per_word[wid] = ID2LABEL[preds[tok_idx]]
                prev = wid
            # Any tokens truncated stay None; we treat truncated positions as O
            # so we don't leak gold labels into pred.
            pred_labels = [l if l is not None else "O" for l in pred_labels_per_word]
            gold_spans = _decode_spans(r.tokens, r.labels)
            pred_spans = _decode_spans(r.tokens, pred_labels)
            gold_all.extend(gold_spans)
            pred_all.extend(pred_spans)
    return _entity_f1(gold_all, pred_all)


def main():
    # Line-buffer stdout/stderr so long CPU epochs still surface progress
    # to the training log even when stdout isn't a TTY (background jobs).
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--out-dir", default="models/report_parser_clinicalbert_v1")
    p.add_argument("--corpus-size", type=int, default=2000)
    p.add_argument("--corpus-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--corpus-out-dir", default="data/report_parser_v0_3_0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_out_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] generating {args.corpus_size} synthetic reports (seed={args.corpus_seed})")
    corpus = generate_corpus(args.corpus_size, seed=args.corpus_seed)
    for split, reports in corpus.items():
        save_split(reports, corpus_dir / f"{split}.jsonl")
        print(f"[corpus]   {split}: {len(reports)} reports -> {corpus_dir / (split + '.jsonl')}")

    print(f"[model] loading {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[model] device={device}, num_labels={len(BIO_LABELS)}")

    train_ds = ReportTaggerDataset(corpus["train"], tokenizer, args.max_len)
    val_ds = ReportTaggerDataset(corpus["val"], tokenizer, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps
    )

    t0 = time.time()
    train_losses = []
    val_losses = []
    val_f1s = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            total_loss += float(loss.item())
            if step % 25 == 0:
                print(f"[train] epoch={epoch} step={step}/{len(train_dl)} loss={loss.item():.4f}")
        avg_loss = total_loss / max(1, len(train_dl))
        train_losses.append(avg_loss)

        # Val loss (fast)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                vl += float(out.loss.item())
        vl /= max(1, len(val_dl))
        val_losses.append(vl)

        # Val entity F1 (slower — one report at a time)
        val_metrics = evaluate(model, tokenizer, corpus["val"], device, args.max_len)
        val_f1s.append(val_metrics["micro"]["f1"])
        print(
            f"[epoch {epoch}] train_loss={avg_loss:.4f} val_loss={vl:.4f} "
            f"val_micro_f1={val_metrics['micro']['f1']:.4f}"
        )
        # v0.3.0: per-epoch checkpoint so a mid-run kill still leaves usable
        # weights. Overwrites on each epoch (we keep the LAST epoch, not the
        # best-val, because 3-epoch runs on synthetic data are stable).
        ckpt_dir = out_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[ckpt] saved epoch {epoch} weights to {ckpt_dir}")

    train_seconds = time.time() - t0

    # Final test evaluation
    print("[eval] running test-set entity-level evaluation")
    test_metrics = evaluate(model, tokenizer, corpus["test"], device, args.max_len)
    print("[eval] test micro F1:", test_metrics["micro"]["f1"])
    for et, m in sorted(test_metrics["per_entity"].items()):
        print(f"[eval]   {et:14s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    # Save weights and label map
    print(f"[save] writing model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_map.json").write_text(
        json.dumps({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, indent=2)
    )

    metrics = {
        "base_model": args.base_model,
        "corpus_size": args.corpus_size,
        "corpus_seed": args.corpus_seed,
        "provenance": "SYNTHETIC-v0.3.0",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": args.max_len,
        "device": device,
        "train_seconds": train_seconds,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1s": val_f1s,
        "test": test_metrics,
        "label_map": {"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "train_summary.json").write_text(json.dumps({
        "base_model": args.base_model,
        "corpus_size": args.corpus_size,
        "epochs": args.epochs,
        "train_seconds": train_seconds,
        "test_micro_f1": test_metrics["micro"]["f1"],
        "device": device,
    }, indent=2))

    print("[done]", json.dumps({
        "train_seconds": train_seconds,
        "test_micro_f1": test_metrics["micro"]["f1"],
        "out_dir": str(out_dir),
    }))


if __name__ == "__main__":
    main()
