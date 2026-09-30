#!/usr/bin/env python3
"""Train and evaluate a compact ClinicalBERT token-classification head.

The immutable Bio_ClinicalBERT encoder is frozen. Only the linear BIO
classification head is newly trained on real TCGA pathology reports. Report
splits are fixed before this script creates overlapping tokenizer windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BASE_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
TOKEN_RE = re.compile(r"\w+(?:[-\.]\w+)*|[^\s\w]|\S+", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def strict_target(entities: list[dict[str, Any]]) -> tuple[int, str]:
    structured_sha = canonical_sha(entities)
    return int(structured_sha[:15], 16), structured_sha


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def limited(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:n])
    return [rows[index] for index in selected]


def derive_label_map(splits: Iterable[list[dict[str, Any]]]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = {label for rows in splits for row in rows for label in row["labels"]}
    labels.discard("O")
    ordered = ["O", *sorted(labels)]
    return ordered, {label: index for index, label in enumerate(ordered)}, {index: label for index, label in enumerate(ordered)}


class WindowDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int,
        stride: int,
    ) -> None:
        self.windows: list[dict[str, Any]] = []
        for row in rows:
            encoding = tokenizer(
                row["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                padding=False,
            )
            for window_index, input_ids in enumerate(encoding["input_ids"]):
                word_ids = encoding.word_ids(batch_index=window_index)
                aligned: list[int] = []
                previous: int | None = None
                for word_id in word_ids:
                    if word_id is None or word_id == previous:
                        aligned.append(-100)
                    else:
                        aligned.append(label2id[row["labels"][word_id]])
                    previous = word_id
                self.windows.append({
                    "input_ids": input_ids,
                    "attention_mask": encoding["attention_mask"][window_index],
                    "labels": aligned,
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.windows[index]


def compute_class_weights(rows: list[dict[str, Any]], labels: list[str], label2id: dict[str, int]) -> torch.Tensor:
    counts = Counter(label for row in rows for label in row["labels"])
    frequencies = torch.tensor([max(1, counts[label]) for label in labels], dtype=torch.float32)
    inverse_sqrt = 1.0 / frequencies.sqrt()
    weights = inverse_sqrt / inverse_sqrt[label2id["O"]]
    return weights.clamp(min=1.0, max=25.0)


def decode_entities(text: str, source_tokens: list[str], predicted_labels: list[str]) -> list[dict[str, Any]]:
    token_matches = list(TOKEN_RE.finditer(text))
    reconstructed = [match.group(0) for match in token_matches]
    if reconstructed != source_tokens:
        raise ValueError("training-tokenizer reconstruction mismatch")
    entities: list[dict[str, Any]] = []
    index = 0
    while index < len(predicted_labels):
        label = predicted_labels[index]
        if label.startswith("B-") or label.startswith("I-"):
            entity_type = label[2:]
            end_index = index + 1
            while end_index < len(predicted_labels) and predicted_labels[end_index] == f"I-{entity_type}":
                end_index += 1
            start_char = token_matches[index].start()
            end_char = token_matches[end_index - 1].end()
            entities.append({
                "entity_type": entity_type,
                "char_start": start_char,
                "char_end": end_char,
                "surface": text[start_char:end_char],
            })
            index = end_index
        else:
            index += 1
    entities.sort(key=lambda item: (item["char_start"], item["char_end"], item["entity_type"], item["surface"]))
    return entities


def predict_report(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    id2label: dict[int, str],
    device: torch.device,
    max_length: int,
    stride: int,
    inference_batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    encoding = tokenizer(
        row["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        padding=True,
        return_tensors="pt",
    )
    word_ids_by_window = [encoding.word_ids(batch_index=index) for index in range(len(encoding["input_ids"]))]
    sums = np.zeros((len(row["tokens"]), len(id2label)), dtype=np.float64)
    counts = np.zeros(len(row["tokens"]), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoding["input_ids"]), inference_batch_size):
            stop = min(start + inference_batch_size, len(encoding["input_ids"]))
            batch = {
                "input_ids": encoding["input_ids"][start:stop].to(device),
                "attention_mask": encoding["attention_mask"][start:stop].to(device),
            }
            logits = model(**batch).logits.detach().float().cpu().numpy()
            for local_index, window_index in enumerate(range(start, stop)):
                previous: int | None = None
                for token_index, word_id in enumerate(word_ids_by_window[window_index]):
                    if word_id is None or word_id == previous:
                        continue
                    sums[word_id] += logits[local_index, token_index]
                    counts[word_id] += 1
                    previous = word_id
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)[:10].tolist()
        raise RuntimeError(f"sliding windows did not cover source words: {missing}")
    predicted_ids = np.argmax(sums / counts[:, None], axis=1)
    predicted_labels = [id2label[int(value)] for value in predicted_ids]
    return decode_entities(row["text"], row["tokens"], predicted_labels), len(word_ids_by_window)


def span_metrics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in prediction_rows:
        gold = {(item["entity_type"], int(item["char_start"]), int(item["char_end"])) for item in row["gold_entities"]}
        predicted = {(item["entity_type"], int(item["char_start"]), int(item["char_end"])) for item in row["predicted_entities"]}
        for item in gold & predicted:
            counts[item[0]]["tp"] += 1
        for item in predicted - gold:
            counts[item[0]]["fp"] += 1
        for item in gold - predicted:
            counts[item[0]]["fn"] += 1
    per_entity: dict[str, Any] = {}
    total = Counter()
    for entity_type in sorted(counts):
        c = counts[entity_type]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_entity[entity_type] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        total.update(c)
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "micro": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "per_entity": per_entity,
    }


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    id2label: dict[int, str],
    device: torch.device,
    max_length: int,
    stride: int,
    inference_batch_size: int,
    base_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        predicted_entities, window_count = predict_report(
            model, tokenizer, row, id2label, device, max_length, stride, inference_batch_size
        )
        predicted_target, predicted_structured_sha = strict_target(predicted_entities)
        predictions.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "report_id": row["report_id"],
            "payload_sha256": row["payload_sha256"],
            "target_sha256": row["target_sha256"],
            "y_true": row["target"],
            "y_pred": predicted_target,
            "gold_structured_target_sha256": row["structured_target_sha256"],
            "predicted_structured_target_sha256": predicted_structured_sha,
            "gold_entities": row["structured_target"],
            "predicted_entities": predicted_entities,
            "strict_exact_match": predicted_target == row["target"],
            "window_count": window_count,
            "tumor_source": row["tumor_source"],
            "report_length_chars": row["report_length_chars"],
            "report_length_tokens": row["report_length_tokens"],
            "negation_term_count": row["negation_term_count"],
            "uncertainty_term_count": row["uncertainty_term_count"],
            "base_model": BASE_MODEL,
            "base_revision": base_revision,
        })
        if index % 25 == 0:
            print(f"[evaluate] {index}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    metrics = span_metrics(predictions)
    metrics["strict_report_exact_match_rate"] = (
        sum(bool(row["strict_exact_match"]) for row in predictions) / len(predictions) if predictions else 0.0
    )
    metrics["n_reports"] = len(predictions)
    metrics["n_windows"] = sum(int(row["window_count"]) for row in predictions)
    metrics["seconds"] = elapsed
    return predictions, metrics


def save_head(model: Any, path: Path, metadata: dict[str, str]) -> None:
    tensors = {
        "classifier.weight": model.classifier.weight.detach().cpu().contiguous(),
        "classifier.bias": model.classifier.bias.detach().cpu().contiguous(),
    }
    save_file(tensors, str(path), metadata=metadata)


def load_head(model: Any, path: Path) -> None:
    state = load_file(str(path))
    model.classifier.weight.data.copy_(state["classifier.weight"].to(model.classifier.weight.device))
    model.classifier.bias.data.copy_(state["classifier.bias"].to(model.classifier.bias.device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--limit-external", type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    wall_start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_rows = limited(load_jsonl(args.data_dir / "clinicalbert_train_reports.jsonl"), args.limit_train, args.seed)
    validation_rows = limited(load_jsonl(args.data_dir / "clinicalbert_validation_reports.jsonl"), args.limit_validation, args.seed + 1)
    test_rows = limited(load_jsonl(args.data_dir / "clinicalbert_test_reports.jsonl"), args.limit_test, args.seed + 2)
    external_rows = limited(load_jsonl(args.data_dir / "clinicalbert_external_reports.jsonl"), args.limit_external, args.seed + 3)
    # The label vocabulary is learned from training data only; held-out labels
    # never influence model shape or optimization. The full frozen training
    # split contains every schema-v2 entity type.
    labels, label2id, id2label = derive_label_map((train_rows,))

    print(f"[model] {args.base_model}@{args.base_revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[model] device={device} labels={len(labels)} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    train_dataset = WindowDataset(train_rows, tokenizer, label2id, args.max_length, args.stride)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    class_weights = compute_class_weights(train_rows, labels, label2id).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = -1
    artifact_path = args.out_dir / "clinicalbert_head_v2.safetensors"
    history: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    train_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        model.base_model.eval()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_tensor = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_function(logits.reshape(-1, len(labels)), labels_tensor.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_optimizer_steps += 1
            epoch_loss += float(loss.detach().cpu())
            epoch_tokens += int((labels_tensor != -100).sum().detach().cpu())
            if step % 25 == 0 or step == len(loader):
                print(f"[train] epoch={epoch} step={step}/{len(loader)} loss={float(loss):.5f}", flush=True)
        validation_predictions, validation_metrics = evaluate_rows(
            model, tokenizer, validation_rows, id2label, device,
            args.max_length, args.stride, args.inference_batch_size, args.base_revision,
        )
        validation_f1 = float(validation_metrics["micro"]["f1"])
        epoch_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "mean_train_loss": epoch_loss / max(1, len(loader)),
            "validation_span_micro_f1": validation_f1,
            "validation_span_metrics": validation_metrics,
            "seconds": epoch_seconds,
            "labeled_tokens": epoch_tokens,
        }
        history.append(record)
        print(f"[epoch] {json.dumps(record, sort_keys=True)}", flush=True)
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            save_head(model, artifact_path, {
                "artifact_type": "clinicalbert-token-classification-head",
                "base_model": args.base_model,
                "base_revision": args.base_revision,
                "dataset_sha256": args.dataset_sha256,
                "split_sha256": args.split_sha256,
                "training_seed": str(args.seed),
                "best_epoch": str(best_epoch),
            })
    train_seconds = time.perf_counter() - train_started
    load_head(model, artifact_path)
    artifact_sha = sha_file(artifact_path)

    validation_predictions, validation_metrics = evaluate_rows(
        model, tokenizer, validation_rows, id2label, device,
        args.max_length, args.stride, args.inference_batch_size, args.base_revision,
    )
    test_predictions, test_metrics = evaluate_rows(
        model, tokenizer, test_rows, id2label, device,
        args.max_length, args.stride, args.inference_batch_size, args.base_revision,
    )
    external_predictions, external_metrics = evaluate_rows(
        model, tokenizer, external_rows, id2label, device,
        args.max_length, args.stride, args.inference_batch_size, args.base_revision,
    )
    for collection in (validation_predictions, test_predictions, external_predictions):
        for row in collection:
            row["artifact_sha256"] = artifact_sha

    write_jsonl(args.out_dir / "clinicalbert_validation_predictions_v2.jsonl", validation_predictions)
    write_jsonl(args.out_dir / "clinicalbert_test_predictions_v2.jsonl", test_predictions)
    write_jsonl(args.out_dir / "clinicalbert_external_predictions_v2.jsonl", external_predictions)

    completed_at = utc_now()
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    label_map = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    config = {
        "schema_version": 2,
        "artifact_type": "clinicalbert-token-classification-linear-head",
        "artifact_sha256": artifact_sha,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "hidden_size": int(model.config.hidden_size),
        "num_labels": len(labels),
        "max_length": args.max_length,
        "stride": args.stride,
        "label_map": label_map,
        "dataset_sha256": args.dataset_sha256,
        "split_sha256": args.split_sha256,
        "n_training_synthetic": False,
    }
    write_json(args.out_dir / "clinicalbert_head_v2_config.json", config)
    metrics = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "artifact_sha256": artifact_sha,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "dataset_sha256": args.dataset_sha256,
        "split_sha256": args.split_sha256,
        "best_epoch": best_epoch,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "external_validation": external_metrics,
        "label_map": label_map,
    }
    write_json(args.out_dir / "clinicalbert_metrics_v2.json", metrics)

    summary = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "started_at": started_at,
        "completed_at": completed_at,
        "command": " ".join(sys.argv),
        "source_commit": args.source_commit,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "dataset_sha256": args.dataset_sha256,
        "split_sha256": args.split_sha256,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_path.stat().st_size,
        "n_artifact_tensors": 2,
        "n_training": len(train_rows),
        "n_validation": len(validation_rows),
        "n_test": len(test_rows),
        "n_external_validation": len(external_rows),
        "n_training_synthetic": False,
        "n_training_windows": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "train_windows_per_second": (len(train_dataset) * args.epochs) / max(train_seconds, 1e-12),
        "total_seconds": time.perf_counter() - wall_start,
        "optimizer_steps": total_optimizer_steps,
        "device": str(device),
        "peak_cuda_memory_bytes": peak_memory,
        "test_span_micro_f1": test_metrics["micro"]["f1"],
        "test_strict_report_exact_match_rate": test_metrics["strict_report_exact_match_rate"],
        "external_span_micro_f1": external_metrics["micro"]["f1"],
    }
    write_json(args.out_dir / "clinicalbert_training_summary_v2.json", summary)
    print(f"[done] {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
