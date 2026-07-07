"""Bio_ClinicalBERT-backed pathology report entity extractor.

This module loads the fine-tuned weights produced by
`clinicalbert_train.py` and exposes a `parse_pathology_report_clinicalbert(text)`
function that returns the SAME `ParsedReport` shape as the existing
regex-based `report_parser.parse_pathology_report` — so it can be swapped
in behind an env flag without touching the biopsy endpoint.

Design honesty
--------------
1. The trained model is `parser_id="clinicalbert_v1"` and stamps that
   into the `ParsedReport.parser_id` field. The frontend MUST render this
   so the operator can distinguish regex vs BERT parses.
2. Every value extraction the model makes carries a per-field softmax
   probability (max prob across the token(s) that formed the value). Any
   field with prob < the field-specific threshold is downgraded to
   `ambiguous` — the model refuses to commit to values it doesn't
   score highly. Thresholds are stored in the trained model card.
3. When BOTH regex and BERT are enabled (see `parse_pathology_report_fused`),
   we use a strict conservative fusion:
     - both match AND agree → use that value, state=matched, source=fused
     - both match but disagree → state=ambiguous, source=disagreement
     - only one matches → use that value, state=matched, source=<origin>
     - neither matches → state=no_match

Extending the regex parser's four fields
----------------------------------------
The BERT model tags additional entities beyond ER/PR/HER2/grade:
KI67_PCT, T/N/M_STAGE, TUMOR_SIZE_MM, MARGIN, LVI. These live on
`ParsedReport.extended_fields` — a dict of `ExtendedParseField` that
mirrors the ParseField contract. The biopsy endpoint can consume these
directly for the receptor_panel.ki67_percent field and pass the rest
through to the response for the UI.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# The trained model directory.  Resolution order (first hit wins):
#   1. env var ``ONCOLOGY_ARBITER_CLINICALBERT_DIR`` — for tests / prod
#      overrides / air-gapped mounts.
#   2. ``/workspace/models/report_parser_clinicalbert_v1`` — worker-local
#      dev store, fastest to load and where training writes.
#   3. ``/mnt/results/models/report_parser_clinicalbert_v1`` — S3-backed
#      deliverable mirror, portable across machines in a session.
# A path counts as "present" if it contains ``label_map.json`` — the same
# marker file ``_load()`` checks below.
_DEFAULT_MODEL_SEARCH_PATH: list[Path] = [
    Path("/workspace/models/report_parser_clinicalbert_v1"),
    Path("/mnt/results/models/report_parser_clinicalbert_v1"),
]


def _resolve_default_model_dir() -> Path:
    env = os.environ.get("ONCOLOGY_ARBITER_CLINICALBERT_DIR")
    if env:
        return Path(env)
    for p in _DEFAULT_MODEL_SEARCH_PATH:
        if (p / "label_map.json").exists():
            return p
    # No local checkpoint found — return the first candidate; ``_load()``
    # will emit a clear FileNotFoundError with actionable guidance.
    return _DEFAULT_MODEL_SEARCH_PATH[0]


_DEFAULT_MODEL_DIR = _resolve_default_model_dir()

# Per-entity probability thresholds. Anything below → downgrade to ambiguous.
# These are conservative defaults; they can be tuned from the metrics.json
# by picking the recall≥0.85 point on the ROC of the max-token probability.
_ENTITY_THRESHOLDS: dict[str, float] = {
    "ER_VALUE": 0.7,
    "PR_VALUE": 0.7,
    "HER2_VALUE": 0.7,
    "KI67_PCT": 0.7,
    "GRADE": 0.7,
    "T_STAGE": 0.6,
    "N_STAGE": 0.6,
    "M_STAGE": 0.6,
    "TUMOR_SIZE_MM": 0.6,
    "MARGIN": 0.7,
    "LVI": 0.7,
}


@dataclass
class ClinicalBertField:
    """One extracted field, shape-compatible with regex ParseField."""

    value: Any
    match_state: str            # matched | ambiguous | no_match
    matched_text: str | None
    span: tuple[int, int] | None
    confidence: float           # 0..1 max prob across the value tokens
    source: str = "clinicalbert_v1"


@dataclass
class ClinicalBertParsedReport:
    """ParsedReport shape returned by clinicalbert parser."""

    er: ClinicalBertField
    pr: ClinicalBertField
    her2: ClinicalBertField
    grade: ClinicalBertField
    extended_fields: dict[str, ClinicalBertField]   # KI67_PCT, T_STAGE, ...
    parser_id: str = "clinicalbert_v1"
    provenance: str = "SYNTHETIC-v0.3.0"

    def as_dict(self) -> dict:
        def _fld(f: ClinicalBertField) -> dict:
            return {
                "value": f.value,
                "match_state": f.match_state,
                "matched_text": f.matched_text,
                "span": list(f.span) if f.span else None,
                "confidence": f.confidence,
                "source": f.source,
            }
        return {
            "er": _fld(self.er),
            "pr": _fld(self.pr),
            "her2": _fld(self.her2),
            "grade": _fld(self.grade),
            "extended_fields": {k: _fld(v) for k, v in self.extended_fields.items()},
            "parser_id": self.parser_id,
            "provenance": self.provenance,
        }


class ClinicalBertReportParser:
    """Singleton wrapper around the fine-tuned Bio_ClinicalBERT token
    classifier. Load once with `.get()`, call `.parse(text)` per report."""

    _instance: Optional["ClinicalBertReportParser"] = None

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if not (self.model_dir / "label_map.json").exists():
            raise FileNotFoundError(
                f"Model directory {self.model_dir} is missing label_map.json. "
                "Run `python -m oncology_arbiter.nlp.clinicalbert_train` first."
            )
        with (self.model_dir / "label_map.json").open() as f:
            lm = json.load(f)
        self._label2id = lm["label2id"]
        self._id2label = {int(k): v for k, v in lm["id2label"].items()}

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.to(self._device).eval()

    @classmethod
    def get(cls, model_dir: Path = _DEFAULT_MODEL_DIR) -> "ClinicalBertReportParser":
        if cls._instance is None or cls._instance.model_dir != model_dir:
            cls._instance = ClinicalBertReportParser(model_dir=model_dir)
        return cls._instance

    def parse(self, text: str, max_len: int = 512) -> ClinicalBertParsedReport:
        """Tokenize, run inference, decode BIO spans, and roll them up
        into per-field values matching the regex ParsedReport contract."""
        tokens = self._whitespace_tokenize(text)
        surface_tokens = [t for t, _, _ in tokens]

        enc = self._tokenizer(
            surface_tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_offsets_mapping=False,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_ids = probs.argmax(-1)

        # Roll sub-token predictions back to word-level. For each source
        # word we take the FIRST sub-token's argmax as the prediction and
        # its softmax prob as the confidence.
        word_ids = enc.word_ids(batch_index=0)
        word_labels: list[str | None] = [None] * len(surface_tokens)
        word_confs: list[float] = [0.0] * len(surface_tokens)
        prev_wid: int | None = None
        for i, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            if wid < len(surface_tokens):
                lab = self._id2label[int(pred_ids[i])]
                word_labels[wid] = lab
                word_confs[wid] = float(probs[i, int(pred_ids[i])])
            prev_wid = wid

        # Fill any word truncated out of the encoding with O @ 0.0 conf.
        for i, l in enumerate(word_labels):
            if l is None:
                word_labels[i] = "O"

        # Decode BIO spans and their per-span mean confidence + surface.
        spans = self._decode_spans_with_conf(surface_tokens, word_labels, word_confs, tokens)

        # Aggregate spans per entity type — take the highest-confidence one
        # if multiple, since last-match-wins made sense for regex but a
        # classifier can hedge across the report.
        best_by_entity: dict[str, dict] = {}
        for sp in spans:
            et = sp["entity_type"]
            if et not in best_by_entity or sp["confidence"] > best_by_entity[et]["confidence"]:
                best_by_entity[et] = sp

        return self._to_parsed_report(best_by_entity, text)

    # -- helpers ----------------------------------------------------------
    _TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?|[^\sA-Za-z0-9]")

    def _whitespace_tokenize(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in self._TOKEN_RE.finditer(text)]

    @staticmethod
    def _decode_spans_with_conf(
        tokens: list[str],
        labels: list[str],
        confs: list[float],
        offsets: list[tuple[str, int, int]],
    ) -> list[dict]:
        out = []
        i = 0
        while i < len(labels):
            lab = labels[i]
            if lab.startswith("B-"):
                et = lab[2:]
                j = i + 1
                while j < len(labels) and labels[j] == f"I-{et}":
                    j += 1
                span_toks = tokens[i:j]
                mean_conf = float(np.mean(confs[i:j]))
                out.append({
                    "entity_type": et,
                    "start_tok": i,
                    "end_tok": j,
                    "surface": " ".join(span_toks),
                    "confidence": mean_conf,
                    "char_start": offsets[i][1],
                    "char_end": offsets[j - 1][2],
                })
                i = j
            else:
                i += 1
        return out

    @staticmethod
    def _to_parsed_report(
        best_by_entity: dict[str, dict], text: str
    ) -> ClinicalBertParsedReport:
        def _mk(entity: str, canonicalizer) -> ClinicalBertField:
            sp = best_by_entity.get(entity)
            if sp is None:
                return ClinicalBertField(
                    value=None, match_state="no_match",
                    matched_text=None, span=None, confidence=0.0,
                )
            # Prefer the raw text slice as the canonicalizer input: our
            # whitespace tokenizer splits digit/symbol/letter boundaries
            # (e.g. "3+" → ["3", "+"], "T2" → ["T", "2"]) and re-joins
            # them with a space when it builds `sp["surface"]`. That
            # breaks substring / regex checks like "3+" in s or the T-stage
            # regex `[TNM]\d+`. Using the original character slice avoids
            # that entire class of bug.
            char_span = (sp["char_start"], sp["char_end"])
            surface = text[char_span[0]:char_span[1]]
            conf = sp["confidence"]
            threshold = _ENTITY_THRESHOLDS.get(entity, 0.7)
            value, state = canonicalizer(surface, conf, threshold)
            return ClinicalBertField(
                value=value, match_state=state,
                matched_text=text[char_span[0]:char_span[1]],
                span=char_span, confidence=conf,
            )

        # Canonicalizers: turn the surface text into the boolean / int /
        # literal value the schema expects.  The order of checks matters:
        # weakly-positive / focal / equivocal patterns must match BEFORE
        # the plain "positive" substring check, because "weakly positive"
        # contains the substring "positive".
        def _receptor(surface: str, conf: float, thr: float):
            # Normalize whitespace: the whitespace tokenizer splits "3+"
            # into ["3", "+"] and joins them back with a space, so a
            # HER2 surface arrives here as "3 +" (or "2 + ( equivocal )").
            # Collapse "N + " → "N+" so substring checks work.
            s = surface.lower().strip()
            s = re.sub(r"(\d)\s*\+", r"\1+", s)
            s = re.sub(r"\s+", " ", s)
            state = "ambiguous" if conf < thr else "matched"
            match_state = "matched" if state == "matched" else "ambiguous"

            def _out(val):
                return (val if state == "matched" else None, match_state)

            # --- equivocal / borderline patterns first ---
            equiv_markers = (
                "equivocal", "borderline",
                "weakly positive", "weak positivity", "focal weak",
                "1-5% of tumor cells", "1% weakly positive",
                "~2% of cells", "clinical significance uncertain",
                "reflex fish pending", "fish not yet resulted",
                "awaiting fish",
            )
            if "2+" in s and "3+" not in s:
                return _out("equivocal")
            if any(m in s for m in equiv_markers):
                return _out("equivocal")

            # --- positive patterns ---
            positive_markers = (
                "positivity", "strongly positive", "moderate to strong",
                "focally positive", "focal positive",
                "moderate", "strong",
            )
            if s == "3+" or "3+" in s:
                return _out("positive")
            if "positive" in s:
                return _out("positive")
            if any(m in s for m in positive_markers):
                return _out("positive")

            # --- negative patterns ---
            negative_markers = (
                "no nuclear staining", "no staining", "no expression",
                "no reactivity", "not expressed",
            )
            if s in {"0", "1+"}:
                return _out("negative")
            if "negative" in s:
                return _out("negative")
            if any(m in s for m in negative_markers):
                return _out("negative")

            # Model tagged something we didn't recognize — abstain.
            return (None, "ambiguous")

        def _er_bool(surface, conf, thr):
            val, state = _receptor(surface, conf, thr)
            if val == "positive": return (True, state)
            if val == "negative": return (False, state)
            return (None, state if state != "matched" else "ambiguous")

        def _pr_bool(surface, conf, thr):
            return _er_bool(surface, conf, thr)

        def _her2_lit(surface, conf, thr):
            val, state = _receptor(surface, conf, thr)
            return (val, state)

        def _int(surface, conf, thr):
            state = "ambiguous" if conf < thr else "matched"
            m = re.search(r"\d+", surface)
            if m:
                v = int(m.group(0))
                if entity_range := _RANGE.get(_current_entity):
                    lo, hi = entity_range
                    if not (lo <= v <= hi):
                        return (None, "ambiguous")
                return (v if state == "matched" else None, state)
            return (None, "ambiguous")

        def _float_mm(surface, conf, thr):
            state = "ambiguous" if conf < thr else "matched"
            m = re.search(r"\d+(?:\.\d+)?", surface)
            if m:
                return (float(m.group(0)) if state == "matched" else None, state)
            return (None, "ambiguous")

        def _stage(surface, conf, thr):
            state = "ambiguous" if conf < thr else "matched"
            m = re.search(r"[TNM](\d+[a-z]{0,3}|x)", surface, flags=re.I)
            if m:
                return (m.group(0).upper() if state == "matched" else None, state)
            return (None, "ambiguous")

        def _margin(surface, conf, thr):
            state = "ambiguous" if conf < thr else "matched"
            s = surface.lower()
            if "negative" in s or "uninvolved" in s: return ("negative" if state == "matched" else None, state)
            if "positive" in s or "involved" in s: return ("positive" if state == "matched" else None, state)
            if "close" in s: return ("close" if state == "matched" else None, state)
            return (None, "ambiguous")

        def _lvi(surface, conf, thr):
            state = "ambiguous" if conf < thr else "matched"
            s = surface.lower()
            if "absent" in s or "not identified" in s: return ("absent" if state == "matched" else None, state)
            if "present" in s or "identified" in s: return ("present" if state == "matched" else None, state)
            return (None, "ambiguous")

        _RANGE = {"GRADE": (1, 3), "KI67_PCT": (0, 100)}
        # `_current_entity` cheat: closure into _int since Python closures
        # capture lexical scope. We rebind inside the loop below.
        _current_entity = "GRADE"

        # ER/PR/HER2/GRADE map to the regex ParsedReport slots
        er = _mk("ER_VALUE", _er_bool)
        pr = _mk("PR_VALUE", _pr_bool)
        her2 = _mk("HER2_VALUE", _her2_lit)
        _current_entity = "GRADE"
        grade = _mk("GRADE", _int)

        ext: dict[str, ClinicalBertField] = {}
        _current_entity = "KI67_PCT"
        ext["ki67_pct"] = _mk("KI67_PCT", _int)
        ext["tumor_size_mm"] = _mk("TUMOR_SIZE_MM", _float_mm)
        ext["t_stage"] = _mk("T_STAGE", _stage)
        ext["n_stage"] = _mk("N_STAGE", _stage)
        ext["m_stage"] = _mk("M_STAGE", _stage)
        ext["margin"] = _mk("MARGIN", _margin)
        ext["lvi"] = _mk("LVI", _lvi)

        return ClinicalBertParsedReport(
            er=er, pr=pr, her2=her2, grade=grade, extended_fields=ext,
        )


def parse_pathology_report_clinicalbert(
    text: str, model_dir: Path = _DEFAULT_MODEL_DIR
) -> ClinicalBertParsedReport:
    """Convenience one-shot call. Loads the singleton on first invocation."""
    return ClinicalBertReportParser.get(model_dir=model_dir).parse(text)
