"""In-process ClinicalBERT parser client for local smoke tests.

This is a **drop-in for :class:`ClinicalBertModalClient`** that loads the
fine-tuned Bio_ClinicalBERT + token-classification head from a local
weight directory (default: ``/workspace/clinicalbert_best``) and runs
inference in the same Python process as the API.

Only used when ``CLINICALBERT_BACKEND=local``. Prod path is Modal; this
client exists to smoke-test the /v1/case/full → parsed_report wiring
without a live Modal deployment. Not shipped to Render (torch would blow
the 512 MB free-tier ceiling).

Public surface matches ``ClinicalBertModalClient`` verbatim so the API
layer can select backends via env without further branching:

    parse(report_text) -> {
      "provenance", "base_model", "training_seed", "test_micro_f1",
      "parsed", "spans", "n_tokens", "seconds", "app_version",
      "disclaimer",
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

APP_VERSION = "clinicalbert-local-v0.4.1-alpha"

_DEFAULT_WEIGHT_DIR = os.environ.get(
    "CLINICALBERT_LOCAL_WEIGHT_DIR", "/workspace/clinicalbert_best"
)

_DISCLAIMER = (
    "Research Use Only. Not FDA-cleared. Not CE-marked. Not intended "
    "for clinical use."
)


class ClinicalBertLocalError(RuntimeError):
    """Raised when local ClinicalBERT inference fails."""


# --------------------------- token / decode helpers --------------------------

_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?|[^\sA-Za-z0-9]")


def _tokenize(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in _TOKEN_SPLIT_RE.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def _decode_bio_spans(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            out.append(
                {
                    "entity_type": etype,
                    "surface": " ".join(tokens[i:j]),
                    "start_tok": i,
                    "end_tok": j,
                }
            )
            i = j
        else:
            i += 1
    return out


def _canonicalize(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_type: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        et = span["entity_type"]
        if et not in per_type or len(span["surface"]) > len(per_type[et]["surface"]):
            per_type[et] = span

    parsed: Dict[str, Any] = {}
    for et, span in per_type.items():
        surface = span["surface"].strip()
        parsed[et] = {
            "surface": surface,
            "start_tok": span["start_tok"],
            "end_tok": span["end_tok"],
        }

        # Normalise tokenizer-injected whitespace around hyphens so
        # "wild - type" (produced by our whitespace tokenizer) still
        # matches lexicon terms written as "wild-type". Otherwise the
        # per-entity value-decoder misclassifies wild-type genes as
        # "mutated". Cheap; done pre-lower to preserve the substring.
        low = re.sub(r"\s*-\s*", "-", surface.lower())
        if et in ("KI67_PCT", "PD_L1_TPS"):
            m = re.search(r"(\d+)\s*%?", surface)
            parsed[et]["value"] = int(m.group(1)) if m else None
        elif et in ("TMB", "TUMOR_SIZE_MM"):
            m = re.search(r"(\d+(?:\.\d+)?)", surface)
            parsed[et]["value"] = float(m.group(1)) if m else None
        elif et == "GRADE":
            m = re.search(r"(\d)", surface)
            parsed[et]["value"] = int(m.group(1)) if m else None
        elif et in ("T_STAGE", "N_STAGE", "M_STAGE"):
            parsed[et]["value"] = surface.upper().lstrip("PC")
        elif et in ("KRAS", "EGFR", "BRAF"):
            if any(k in low for k in (
                "wild-type", "wild type", "wildtype", "not detected",
                "no mutation", "no pathogenic", "no activating",
            )):
                parsed[et]["value"] = "wild_type"
            else:
                parsed[et]["value"] = "mutated"
        elif et in ("ALK", "ROS1"):
            if any(k in low for k in (
                "no rearrangement", "not identified", "not detected",
                "negative", "no staining", "no fusion",
            )):
                parsed[et]["value"] = "negative"
            else:
                parsed[et]["value"] = "fusion_positive"
        elif et == "MET":
            if any(k in low for k in (
                "not detected", "no exon 14", "no mutation",
                "not amplified", "wild-type", "negative",
            )):
                parsed[et]["value"] = "not_detected"
            else:
                parsed[et]["value"] = "mutated"
        elif et == "HER2_AMP":
            if any(k in low for k in (
                "not amplified", "not detected", "no gene amp",
                "normal copy", "negative",
            )):
                parsed[et]["value"] = "not_amplified"
            else:
                parsed[et]["value"] = "amplified"
        elif et == "MSI":
            if any(k in low for k in ("mss", "stable")):
                parsed[et]["value"] = "mss"
            elif any(k in low for k in ("msi-h", "high", "unstable")):
                parsed[et]["value"] = "msi_high"
            else:
                parsed[et]["value"] = "unknown"
        elif et in ("ER_VALUE", "PR_VALUE", "HER2_VALUE"):
            if any(k in low for k in (
                "no nuclear", "no staining", "negative",
                "1+", " 0", "no mutation",
            )):
                parsed[et]["value"] = "negative"
            elif any(k in low for k in (
                "equivocal", "2+", "1-5%", "weakly", "borderline",
            )):
                parsed[et]["value"] = "equivocal"
            else:
                parsed[et]["value"] = "positive"
        elif et == "MARGIN":
            if "close" in low:
                parsed[et]["value"] = "close"
            elif any(k in low for k in ("negative", "uninvolved")):
                parsed[et]["value"] = "negative"
            else:
                parsed[et]["value"] = "positive"
        elif et == "LVI":
            if any(k in low for k in ("absent", "not identified", "not present")):
                parsed[et]["value"] = "absent"
            else:
                parsed[et]["value"] = "present"
        else:
            parsed[et]["value"] = surface
    return parsed


# --------------------------- client -----------------------------------------

_MODEL_CACHE: Dict[str, Tuple[Any, Any, Dict[int, str], Dict[str, Any]]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model_once(weight_dir: str) -> Tuple[Any, Any, Dict[int, str], Dict[str, Any]]:
    """Load model + tokenizer once per process, cache thereafter."""
    with _MODEL_CACHE_LOCK:
        if weight_dir in _MODEL_CACHE:
            return _MODEL_CACHE[weight_dir]
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ClinicalBertLocalError(
                f"torch/transformers not installed: {exc}. Install with "
                "`pip install torch transformers` before setting "
                "CLINICALBERT_BACKEND=local."
            ) from exc

        model_path = Path(weight_dir)
        if not model_path.is_dir():
            raise ClinicalBertLocalError(
                f"CLINICALBERT_LOCAL_WEIGHT_DIR not found: {weight_dir}"
            )

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        model.eval()

        metrics_path = model_path / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            metrics = {}

        # id2label priority: metrics.json > model.config
        id2label_raw = (metrics.get("label_map") or {}).get("id2label") or {}
        if id2label_raw:
            id2label = {int(k): v for k, v in id2label_raw.items()}
        else:
            id2label = {int(k): v for k, v in model.config.id2label.items()}

        meta = {
            "provenance": metrics.get("provenance", "SYNTHETIC-v0.3.1"),
            "base_model": metrics.get("base_model", "emilyalsentzer/Bio_ClinicalBERT"),
            "training_seed": metrics.get("training_seed"),
            "test_micro_f1": (metrics.get("test") or {}).get("micro", {}).get("f1"),
            "load_seconds": round(time.time() - t0, 3),
        }
        logger.info(
            "clinicalbert_local: loaded %s in %.2fs (seed=%s, f1=%s)",
            weight_dir, meta["load_seconds"], meta["training_seed"], meta["test_micro_f1"],
        )
        _MODEL_CACHE[weight_dir] = (tokenizer, model, id2label, meta)
        return _MODEL_CACHE[weight_dir]


class ClinicalBertLocalClient:
    """In-process drop-in for :class:`ClinicalBertModalClient`.

    Same method surface: ``healthz()``, ``info()``, ``parse(report_text)``.
    """

    def __init__(
        self,
        *,
        weight_dir: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.weight_dir = weight_dir or _DEFAULT_WEIGHT_DIR
        self.max_length = int(max_length)

    # ---------- lifecycle ----------
    def _get(self):
        return _load_model_once(self.weight_dir)

    # ---------- endpoints ----------
    def healthz(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": "clinicalbert-local",
            "version": APP_VERSION,
            "disclaimer": _DISCLAIMER,
        }

    def info(self) -> Dict[str, Any]:
        _, _, _id2label, meta = self._get()
        return {
            "app": "clinicalbert-local",
            "app_version": APP_VERSION,
            "base_model": meta["base_model"],
            "training_seed": meta["training_seed"],
            "test_micro_f1": meta["test_micro_f1"],
            "provenance": meta["provenance"],
            "num_labels": len(_id2label),
            "device": "cpu",
            "load_seconds": meta["load_seconds"],
            "disclaimer": _DISCLAIMER,
        }

    def parse(self, report_text: str) -> Dict[str, Any]:
        if not isinstance(report_text, str) or not report_text.strip():
            raise ClinicalBertLocalError("report_text must be a non-empty string")
        if len(report_text) > 20_000:
            raise ClinicalBertLocalError("report_text too long (max 20000 chars)")

        import torch

        tokenizer, model, id2label, meta = self._get()
        t0 = time.time()

        common = {
            "provenance": meta["provenance"],
            "base_model": meta["base_model"],
            "training_seed": meta["training_seed"],
            "test_micro_f1": meta["test_micro_f1"],
            "app_version": APP_VERSION,
            "disclaimer": _DISCLAIMER,
        }

        toks = _tokenize(report_text)
        tokens = [t for t, _, _ in toks]

        if not tokens:
            return {
                "parsed": {},
                "spans": [],
                "n_tokens": 0,
                "seconds": round(time.time() - t0, 3),
                **common,
            }

        try:
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            raise ClinicalBertLocalError(f"tokenize: {type(e).__name__}: {e}") from e

        try:
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).logits
            preds = logits.argmax(-1)[0].cpu().tolist()
        except Exception as e:
            raise ClinicalBertLocalError(f"forward: {type(e).__name__}: {e}") from e

        word_ids = enc.word_ids(batch_index=0)
        pred_labels_per_word: List[str] = ["O"] * len(tokens)
        prev = None
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                continue
            if wid < len(pred_labels_per_word):
                pred_labels_per_word[wid] = id2label.get(int(preds[tok_idx]), "O")
            prev = wid

        spans = _decode_bio_spans(tokens, pred_labels_per_word)
        parsed = _canonicalize(spans)

        return {
            "parsed": parsed,
            "spans": spans,
            "n_tokens": len(tokens),
            "seconds": round(time.time() - t0, 3),
            **common,
        }
