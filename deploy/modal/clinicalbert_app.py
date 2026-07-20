"""Modal app: fine-tuned Bio_ClinicalBERT report parser for OncologyArbiter.

Endpoints
---------
- `GET  /clinicalbert-healthz` → liveness (no model touch)
- `GET  /clinicalbert-info`    → warm the model, return metadata
                                 (base_model, provenance, entity types,
                                 training seed, test micro-F1)
- `POST /clinicalbert-parse`   → JSON `{"report_text": "..."}` → parsed
                                 pathology fields extracted via BIO-tagged
                                 token classification

Design notes
------------
- Base: emilyalsentzer/Bio_ClinicalBERT + fine-tuned classifier head.
- Fine-tune weights are shipped into the Modal image via
  `modal.Image.add_local_dir()` at deploy time. The weights include
  `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.txt`,
  `label_map.json`, `metrics.json`.
- CPU-only. ~430 MB weights + tokenizer; loads in ~5 s. Inference on a
  full pathology report (~500-800 tokens) is <500 ms.
- No HF token required at inference time — the tuned weights carry
  everything the tokenizer needs.
- Provenance: SYNTHETIC-v0.3.1 (breast + NSCLC). This is stamped in
  every response so callers can honor the "training data was synthetic"
  contract in downstream UI.

Deploy
------
    modal deploy deploy/modal/clinicalbert_app.py

Env vars at deploy time (optional)
----------------------------------
- `CLINICALBERT_WEIGHT_DIR`: path to fine-tuned weights on the host doing
  the deploy. Default: `/workspace/clinicalbert_best/`.
- `CLINICALBERT_MODAL_MODE`: 'prod' → min_containers=1 (warm replica);
  default 'staging' → min_containers=0 (zero-cost).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import modal

APP_VERSION = "clinicalbert-modal-v0.4.1-alpha"

_MODAL_MODE = (os.environ.get("CLINICALBERT_MODAL_MODE") or "staging").lower()
_MIN_CONTAINERS = 1 if _MODAL_MODE == "prod" else 0
_SCALEDOWN_S = 900 if _MODAL_MODE == "prod" else 300

_WEIGHT_DIR = Path(os.environ.get("CLINICALBERT_WEIGHT_DIR", "/workspace/clinicalbert_best"))
_MOUNT_TARGET = "/model"

CLINICALBERT_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "safetensors==0.4.5",
        "huggingface_hub==0.24.7",
        "fastapi==0.115.0",
        "numpy==1.26.4",
    )
    .add_local_dir(str(_WEIGHT_DIR), _MOUNT_TARGET)
)

app = modal.App("clinicalbert")


HEALTH_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.115.0"
)


@app.function(image=HEALTH_IMAGE)
@modal.fastapi_endpoint(method="GET", label="clinicalbert-healthz")
def healthz() -> Dict[str, str]:
    return {
        "status": "ok",
        "app": "clinicalbert",
        "version": APP_VERSION,
        "disclaimer": (
            "Research Use Only. Not FDA-cleared. Not CE-marked. Not intended "
            "for clinical use."
        ),
    }


# -- Token-classification decoder helpers ----------------------------

_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?|[^\sA-Za-z0-9]")


def _tokenize(text: str) -> List[tuple]:
    out: List[tuple] = []
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
    """Collapse BIO spans into a per-entity-type parsed dict."""
    per_type: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        et = span["entity_type"]
        if et not in per_type or len(span["surface"]) > len(per_type[et]["surface"]):
            per_type[et] = span

    parsed: Dict[str, Any] = {}
    for et, span in per_type.items():
        surface = span["surface"].strip()
        parsed[et] = {"surface": surface, "start_tok": span["start_tok"], "end_tok": span["end_tok"]}

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
            if any(k in low for k in ("no rearrangement", "not identified",
                                       "not detected", "negative", "no staining",
                                       "no fusion")):
                parsed[et]["value"] = "negative"
            else:
                parsed[et]["value"] = "fusion_positive"
        elif et == "MET":
            if any(k in low for k in ("not detected", "no exon 14", "no mutation",
                                       "not amplified", "wild-type", "negative")):
                parsed[et]["value"] = "not_detected"
            else:
                parsed[et]["value"] = "mutated"
        elif et == "HER2_AMP":
            if any(k in low for k in ("not amplified", "not detected", "no gene amp",
                                       "normal copy", "negative")):
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
            if any(k in low for k in ("no nuclear", "no staining", "negative",
                                       "1+", " 0", "no mutation")):
                parsed[et]["value"] = "negative"
            elif any(k in low for k in ("equivocal", "2+", "1-5%", "weakly", "borderline")):
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


@app.cls(
    image=CLINICALBERT_IMAGE,
    scaledown_window=_SCALEDOWN_S,
    timeout=180,
    min_containers=_MIN_CONTAINERS,
)
class ClinicalBertModal:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        t0 = time.time()

        model_path = _MOUNT_TARGET
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        metrics_path = Path(model_path) / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            self.metrics = metrics
            self.id2label = {
                int(k): v for k, v in metrics.get("label_map", {}).get("id2label", {}).items()
            }
            self.provenance = metrics.get("provenance", "SYNTHETIC-v0.3.1")
            self.training_seed = metrics.get("training_seed")
            self.test_micro_f1 = metrics.get("test", {}).get("micro", {}).get("f1")
            self.base_model = metrics.get("base_model", "emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.metrics = None
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            self.provenance = "SYNTHETIC-v0.3.1"
            self.training_seed = None
            self.test_micro_f1 = None
            self.base_model = "emilyalsentzer/Bio_ClinicalBERT"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.load_seconds = round(time.time() - t0, 3)
        self.warmed_at = time.time()

    @modal.fastapi_endpoint(method="GET", label="clinicalbert-info")
    def info(self) -> Dict[str, Any]:
        return {
            "app": "clinicalbert",
            "app_version": APP_VERSION,
            "base_model": self.base_model,
            "training_seed": self.training_seed,
            "test_micro_f1": self.test_micro_f1,
            "provenance": self.provenance,
            "num_labels": len(self.id2label),
            "device": self.device,
            "load_seconds": self.load_seconds,
            "warmed_at": self.warmed_at,
            "disclaimer": (
                "Research Use Only. Not FDA-cleared. Not CE-marked. Not intended "
                "for clinical use."
            ),
        }

    @modal.fastapi_endpoint(method="POST", label="clinicalbert-parse")
    def parse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON: `{"report_text": "..."}`."""
        import torch

        t0 = time.time()
        report_text = payload.get("report_text")
        if not isinstance(report_text, str) or not report_text.strip():
            return {"error": "report_text must be a non-empty string"}
        if len(report_text) > 20_000:
            return {"error": "report_text too long (max 20000 chars)"}

        toks = _tokenize(report_text)
        tokens = [t for t, _, _ in toks]

        common = {
            "provenance": self.provenance,
            "base_model": self.base_model,
            "training_seed": self.training_seed,
            "test_micro_f1": self.test_micro_f1,
            "app_version": APP_VERSION,
            "disclaimer": (
                "Research Use Only. Not FDA-cleared. Not CE-marked. "
                "Not intended for clinical use."
            ),
        }

        if not tokens:
            return {"parsed": {}, "spans": [], "n_tokens": 0,
                    "seconds": round(time.time() - t0, 3), **common}

        try:
            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            return {"error": f"tokenize: {type(e).__name__}: {e}"}

        try:
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                ).logits
            preds = logits.argmax(-1)[0].cpu().tolist()
        except Exception as e:
            return {"error": f"forward: {type(e).__name__}: {e}"}

        word_ids = enc.word_ids(batch_index=0)
        pred_labels_per_word: List[str] = ["O"] * len(tokens)
        prev = None
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                continue
            if wid < len(pred_labels_per_word):
                pred_labels_per_word[wid] = self.id2label.get(int(preds[tok_idx]), "O")
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
