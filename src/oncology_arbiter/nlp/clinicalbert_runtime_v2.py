"""Identity-bound sliding-window runtime for the real-report ClinicalBERT v2 head."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_FILENAME = "clinicalbert_head_v2.safetensors"
CONFIG_FILENAME = "clinicalbert_head_v2_config.json"
METRICS_FILENAME = "clinicalbert_metrics_v2.json"
EXPECTED_ARTIFACT_SHA256 = "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50"
BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BASE_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
PROVENANCE = "REAL-TCGA-PATHOLOGY-v2"
TOKEN_RE = re.compile(r"\w+(?:[-\.]\w+)*|[^\s\w]|\S+", re.UNICODE)
DISCLAIMER = (
    "Research Use Only. Not FDA-cleared. Not CE-marked. "
    "Not intended for clinical use."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse exact spans to one highest-confidence value per entity type."""
    per_type: dict[str, dict[str, Any]] = {}
    for span in spans:
        entity_type = span["entity_type"]
        if entity_type not in per_type or span["confidence"] > per_type[entity_type]["confidence"]:
            per_type[entity_type] = span

    parsed: dict[str, Any] = {}
    for entity_type, span in per_type.items():
        surface = span["surface"]
        low = re.sub(r"\s*-\s*", "-", surface.lower())
        value: Any = surface
        if entity_type in ("KI67_PCT", "PD_L1_TPS"):
            match = re.search(r"(\d+)\s*%?", surface)
            value = int(match.group(1)) if match else None
        elif entity_type in ("TMB", "TUMOR_SIZE_MM"):
            match = re.search(r"(\d+(?:\.\d+)?)", surface)
            value = float(match.group(1)) if match else None
        elif entity_type == "GRADE":
            match = re.search(r"(\d)", surface)
            value = int(match.group(1)) if match else None
        elif entity_type in ("T_STAGE", "N_STAGE", "M_STAGE"):
            value = surface.upper().lstrip("PC")
        elif entity_type in ("KRAS", "EGFR", "BRAF"):
            value = "wild_type" if any(term in low for term in (
                "wild-type", "wild type", "wildtype", "not detected",
                "no mutation", "no pathogenic", "no activating",
            )) else "mutated"
        elif entity_type in ("ALK", "ROS1"):
            value = "negative" if any(term in low for term in (
                "no rearrangement", "not identified", "not detected", "negative",
                "no staining", "no fusion",
            )) else "fusion_positive"
        elif entity_type == "MET":
            value = "not_detected" if any(term in low for term in (
                "not detected", "no exon 14", "no mutation", "not amplified",
                "wild-type", "negative",
            )) else "mutated"
        elif entity_type == "HER2_AMP":
            value = "not_amplified" if any(term in low for term in (
                "not amplified", "not detected", "no gene amp", "normal copy", "negative",
            )) else "amplified"
        elif entity_type == "MSI":
            if any(term in low for term in ("mss", "stable")):
                value = "mss"
            elif any(term in low for term in ("msi-h", "high", "unstable")):
                value = "msi_high"
            else:
                value = "unknown"
        elif entity_type in ("ER_VALUE", "PR_VALUE", "HER2_VALUE"):
            if any(term in low for term in ("no nuclear", "no staining", "negative", "1+", " 0")):
                value = "negative"
            elif any(term in low for term in ("equivocal", "2+", "1-5%", "weakly", "borderline")):
                value = "equivocal"
            else:
                value = "positive"
        elif entity_type == "MARGIN":
            if "close" in low:
                value = "close"
            elif any(term in low for term in ("negative", "uninvolved")):
                value = "negative"
            else:
                value = "positive"
        elif entity_type == "LVI":
            value = "absent" if any(term in low for term in (
                "absent", "not identified", "not present",
            )) else "present"
        parsed[entity_type] = {
            "surface": surface,
            "value": value,
            "char_start": span["char_start"],
            "char_end": span["char_end"],
            "confidence": span["confidence"],
        }
    return parsed


class ClinicalBertV2Runtime:
    """Load the pinned Bio_ClinicalBERT encoder plus the exact compact v2 head."""

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        base_model_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        started = time.perf_counter()
        self.bundle_dir = Path(bundle_dir)
        artifact_path = self.bundle_dir / ARTIFACT_FILENAME
        config_path = self.bundle_dir / CONFIG_FILENAME
        metrics_path = self.bundle_dir / METRICS_FILENAME
        for path in (artifact_path, config_path, metrics_path):
            if not path.is_file():
                raise FileNotFoundError(f"ClinicalBERT v2 bundle is missing {path.name}")
        self.artifact_sha256 = sha256_file(artifact_path)
        if self.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise RuntimeError(
                f"ClinicalBERT head identity mismatch: {self.artifact_sha256} "
                f"!= {EXPECTED_ARTIFACT_SHA256}"
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if self.config.get("artifact_sha256") != self.artifact_sha256:
            raise RuntimeError("ClinicalBERT config does not bind the head SHA")
        if self.metrics.get("artifact_sha256") != self.artifact_sha256:
            raise RuntimeError("ClinicalBERT metrics do not bind the head SHA")
        if self.config.get("base_model") != BASE_MODEL or self.config.get("base_revision") != BASE_REVISION:
            raise RuntimeError("ClinicalBERT base model identity mismatch")

        self.id2label = {int(key): value for key, value in self.config["label_map"]["id2label"].items()}
        self.label2id = {key: int(value) for key, value in self.config["label_map"]["label2id"].items()}
        if self.id2label.get(0) != "O" or len(self.id2label) != int(self.config["num_labels"]):
            raise RuntimeError("ClinicalBERT label-map contract is invalid")
        self.max_length = int(self.config["max_length"])
        self.stride = int(self.config["stride"])
        self.non_o_margin_threshold = float(self.config["non_o_margin_threshold"])

        base_source = str(base_model_dir) if base_model_dir is not None else BASE_MODEL
        revision = None if base_model_dir is not None else BASE_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(base_source, revision=revision)
        self.model = AutoModelForTokenClassification.from_pretrained(
            base_source,
            revision=revision,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        head = load_file(str(artifact_path))
        if set(head) != {"classifier.weight", "classifier.bias"}:
            raise RuntimeError("ClinicalBERT head must contain exactly classifier.weight and classifier.bias")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in head.values()):
            raise RuntimeError("ClinicalBERT head contains non-finite tensors")
        self.model.classifier.weight.data.copy_(head["classifier.weight"])
        self.model.classifier.bias.data.copy_(head["classifier.bias"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - started

    def identity(self) -> dict[str, Any]:
        test_metric = self.metrics.get("test", {}).get("micro", {})
        return {
            "artifact_filename": ARTIFACT_FILENAME,
            "artifact_sha256": self.artifact_sha256,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "provenance": PROVENANCE,
            "n_training": 2187,
            "n_training_synthetic": False,
            "num_labels": len(self.id2label),
            "max_length": self.max_length,
            "stride": self.stride,
            "non_o_margin_threshold": self.non_o_margin_threshold,
            "test_span_micro_f1": test_metric.get("f1"),
            "device": str(self.device),
            "load_seconds": round(self.load_seconds, 3),
        }

    def parse(self, report_text: str, *, inference_batch_size: int = 16) -> dict[str, Any]:
        import torch

        started = time.perf_counter()
        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("report_text must be a non-empty string")
        if len(report_text) > 20_000:
            raise ValueError("report_text too long (max 20000 chars)")
        token_matches = list(TOKEN_RE.finditer(report_text))
        tokens = [match.group(0) for match in token_matches]
        report_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        if not tokens:
            return {
                "parsed": {}, "spans": [], "n_tokens": 0, "window_count": 0,
                "report_sha256": report_sha256,
                "latency_seconds": round(time.perf_counter() - started, 6),
                **self.identity(), "disclaimer": DISCLAIMER,
            }

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding=True,
            return_tensors="pt",
        )
        window_count = len(encoding["input_ids"])
        word_ids_by_window = [encoding.word_ids(batch_index=index) for index in range(window_count)]
        sums = np.zeros((len(tokens), len(self.id2label)), dtype=np.float64)
        counts = np.zeros(len(tokens), dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, window_count, inference_batch_size):
                stop = min(start + inference_batch_size, window_count)
                logits = self.model(
                    input_ids=encoding["input_ids"][start:stop].to(self.device),
                    attention_mask=encoding["attention_mask"][start:stop].to(self.device),
                ).logits.detach().float().cpu().numpy()
                for local_index, window_index in enumerate(range(start, stop)):
                    previous: int | None = None
                    for token_index, word_id in enumerate(word_ids_by_window[window_index]):
                        if word_id is None or word_id == previous:
                            continue
                        sums[word_id] += logits[local_index, token_index]
                        counts[word_id] += 1
                        previous = word_id
        if np.any(counts == 0):
            raise RuntimeError("sliding windows did not cover every report token")
        averaged = sums / counts[:, None]
        best_non_o = np.argmax(averaged[:, 1:], axis=1) + 1
        margins = averaged[np.arange(len(tokens)), best_non_o] - averaged[:, 0]
        predicted_ids = np.where(margins >= self.non_o_margin_threshold, best_non_o, 0)
        shifted = averaged - averaged.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = [self.id2label[int(index)] for index in predicted_ids]
        confidences = probabilities[np.arange(len(tokens)), predicted_ids]

        spans: list[dict[str, Any]] = []
        index = 0
        while index < len(labels):
            label = labels[index]
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label[2:]
                end_index = index + 1
                while end_index < len(labels) and labels[end_index] == f"I-{entity_type}":
                    end_index += 1
                char_start = token_matches[index].start()
                char_end = token_matches[end_index - 1].end()
                spans.append({
                    "entity_type": entity_type,
                    "surface": report_text[char_start:char_end],
                    "char_start": char_start,
                    "char_end": char_end,
                    "start_tok": index,
                    "end_tok": end_index,
                    "confidence": float(np.mean(confidences[index:end_index])),
                })
                index = end_index
            else:
                index += 1
        spans.sort(key=lambda item: (item["char_start"], item["char_end"], item["entity_type"]))
        latency = time.perf_counter() - started
        return {
            "parsed": _canonicalize(spans),
            "spans": spans,
            "n_tokens": len(tokens),
            "window_count": window_count,
            "report_sha256": report_sha256,
            "latency_seconds": round(latency, 6),
            "seconds": round(latency, 6),
            **self.identity(),
            "disclaimer": DISCLAIMER,
        }
