#!/usr/bin/env python3
"""Build a leakage-free real-report ClinicalBERT corpus and schema-v2 manifests.

The training/validation source is the public TCGA-Reports corpus. Labels are
span-anchored deterministic weak labels generated from the report text; no
synthetic report or synthetic patient is created. The final test cohort is the
pathologist-adjudicated TCGA-242 breast/colorectal subset (categories 1 and 2).
Categories 3-5 are frozen as an independent external-validation cohort.

All TCGA-242 patients and exact/normalized report payloads are reserved before
training-label generation and before the patient split. Sliding-window creation
is intentionally deferred to the trainer so a report can never cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from oncology_arbiter.nlp import corpus_real

TCGA_REPORTS_URI = "https://doi.org/10.17632/hyg5xkznpx.1"
TCGA242_URI = "https://doi.org/10.5281/zenodo.20263861"
LICENSE = "CC BY 4.0"
PATIENT_RE = re.compile(r"\b(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(?:no|not|negative|absent|without|free of|unremarkable)\b", re.IGNORECASE)
UNCERTAINTY_RE = re.compile(r"\b(?:possible|possibly|probable|suspicious|cannot exclude|equivocal|indeterminate)\b", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_json_sha(value: Any) -> str:
    return sha256_text(canonical_json(value))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def patient_id(value: str) -> str:
    match = PATIENT_RE.search(value)
    if match is None:
        raise ValueError(f"cannot derive TCGA patient ID from {value!r}")
    return match.group(1).upper()


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return sha256_text(normalized)


def canonical_entities(entities: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in entities:
        entity = asdict(raw) if not isinstance(raw, dict) else dict(raw)
        item = {
            "entity_type": str(entity["entity_type"]),
            "char_start": int(entity["char_start"]),
            "char_end": int(entity["char_end"]),
            "surface": str(entity.get("surface", entity.get("text_span", ""))),
        }
        out.append(item)
    out.sort(key=lambda item: (item["char_start"], item["char_end"], item["entity_type"], item["surface"]))
    return out


def strict_target(structured_target_sha256: str) -> int:
    # A deterministic 60-bit class code. Equality means the entire ordered
    # entity set matches; the sidecar evaluation separately reports proper
    # span-level micro/per-entity F1.
    return int(structured_target_sha256[:15], 16)


def extract_tcga242(path: Path) -> tuple[str, str]:
    raw = path.read_text(errors="replace")
    lines = raw.split("\n")
    if not lines or not lines[0].startswith("patient_filename:"):
        raise ValueError(f"missing patient_filename header: {path}")
    report_id = lines[0].split(":", 1)[1].strip()
    text = corpus_real._extract_text_body(raw)
    return report_id, text


def make_row(
    *,
    sample_id: str,
    report_id: str,
    patient: str,
    text: str,
    tumor_source: str,
    tokens: list[str],
    labels: list[str],
    entities: Iterable[Any],
    label_source: str,
    label_source_inputs: dict[str, Any],
    source_uri: str,
    source_record: str,
    source_payload_sha256: str,
    gold_annotation_sha256: str | None = None,
) -> dict[str, Any]:
    if len(tokens) != len(labels):
        raise ValueError(f"token/label mismatch for {sample_id}")
    structured_target = canonical_entities(entities)
    for entity in structured_target:
        start, end = entity["char_start"], entity["char_end"]
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"bad entity offsets for {sample_id}: {entity}")
        if text[start:end].casefold() != entity["surface"].casefold():
            raise ValueError(f"entity surface mismatch for {sample_id}: {entity}")
    structured_sha = canonical_json_sha(structured_target)
    target = strict_target(structured_sha)
    payload_sha = sha256_text(text)
    label_source_payload = {
        "label_source": label_source,
        "payload_sha256": payload_sha,
        "structured_target_sha256": structured_sha,
        **label_source_inputs,
    }
    row = {
        "schema_version": 2,
        "sample_id": sample_id,
        "report_id": report_id,
        "patient_id": patient,
        "patient_identity_basis": "TCGA barcode first 12 characters",
        "source_uri": source_uri,
        "source_record": source_record,
        "source_payload_sha256": source_payload_sha256,
        "tumor_source": tumor_source,
        "text": text,
        "payload_sha256": payload_sha,
        "normalized_payload_sha256": normalized_text_hash(text),
        "tokens": tokens,
        "labels": labels,
        "structured_target": structured_target,
        "structured_target_sha256": structured_sha,
        "target": target,
        "target_sha256": canonical_json_sha(target),
        "label_source": label_source,
        "label_source_sha256": canonical_json_sha(label_source_payload),
        "n_entities": len(structured_target),
        "entity_types": sorted({item["entity_type"] for item in structured_target}),
        "report_length_chars": len(text),
        "report_length_tokens": len(tokens),
        "negation_term_count": len(NEGATION_RE.findall(text)),
        "uncertainty_term_count": len(UNCERTAINTY_RE.findall(text)),
        "n_training_synthetic": False,
    }
    if gold_annotation_sha256 is not None:
        row["gold_annotation_sha256"] = gold_annotation_sha256
    return row


def build_gold_rows(
    reports_root: Path,
    annotations_root: Path,
    categories: tuple[int, ...],
    converter_code_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for category in categories:
        report_dir = reports_root / str(category)
        annotation_dir = annotations_root / str(category)
        for report_path in sorted(report_dir.glob("*.txt")):
            annotation_path = annotation_dir / f"{report_path.stem}.json"
            if not annotation_path.is_file():
                raise FileNotFoundError(annotation_path)
            report_id, text = extract_tcga242(report_path)
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            entities = corpus_real.gold_to_entities(text, annotation)
            if not entities:
                exclusions.append({
                    "category": category,
                    "source_record": report_path.name,
                    "report_id": report_id,
                    "patient_id": patient_id(report_id),
                    "reason": "no_schema_entity_could_be_surface_anchored",
                    "payload_sha256": sha256_text(text),
                    "annotation_sha256": sha256_file(annotation_path),
                })
                continue
            tokens, labels = corpus_real.bio_from_char_spans(text, entities)
            annotation_sha = sha256_file(annotation_path)
            row = make_row(
                sample_id=f"TCGA242-C{category}-{report_path.stem}",
                report_id=report_id,
                patient=patient_id(report_id),
                text=text,
                tumor_source=str(annotation.get("cancer_category") or f"tcga242_category_{category}"),
                tokens=tokens,
                labels=labels,
                entities=entities,
                label_source="TCGA-242 pathologist-adjudicated structured gold, deterministically surface-anchored",
                label_source_inputs={
                    "annotation_sha256": annotation_sha,
                    "converter_code_sha256": converter_code_sha256,
                    "tcga242_category": category,
                },
                source_uri=TCGA242_URI,
                source_record=report_path.name,
                source_payload_sha256=sha256_file(report_path),
                gold_annotation_sha256=annotation_sha,
            )
            row["tcga242_category"] = category
            rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    return rows, exclusions


def select_validation(
    rows: list[dict[str, Any]],
    fraction: float,
    seed_start: int,
    n_seed_candidates: int,
) -> tuple[set[str], dict[str, Any]]:
    if not (0.0 < fraction < 1.0):
        raise ValueError("validation fraction must be in (0,1)")
    n_validation = max(1, int(round(len(rows) * fraction)))
    features: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        features[f"study:{row['tumor_source']}"] .add(index)
        features[f"entity_count:{min(4, int(row['n_entities']))}"] .add(index)
        for entity_type in row["entity_types"]:
            features[f"entity:{entity_type}"] .add(index)
    eligible_features = {key: values for key, values in features.items() if len(values) >= 10}

    best: tuple[float, int, set[int], dict[str, Any]] | None = None
    population = list(range(len(rows)))
    for seed in range(seed_start, seed_start + n_seed_candidates):
        rng = random.Random(seed)
        shuffled = population.copy()
        rng.shuffle(shuffled)
        chosen = set(shuffled[:n_validation])
        components: dict[str, float] = {}
        penalty = 0.0
        for key, members in eligible_features.items():
            expected = len(members) * fraction
            observed = len(members & chosen)
            standardized = abs(observed - expected) / math.sqrt(expected + 1.0)
            components[key] = standardized
            if expected >= 2.0 and observed == 0:
                penalty += 10.0
        score = sum(value * value for value in components.values()) / max(1, len(components)) + penalty
        candidate = (score, seed, chosen, components)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    score, seed, indices, components = best
    selected_ids = {rows[index]["sample_id"] for index in indices}
    return selected_ids, {
        "strategy": "patient-level deterministic multi-feature balance search on training-source labels only",
        "validation_fraction": fraction,
        "seed_search_start": seed_start,
        "n_seed_candidates": n_seed_candidates,
        "selected_seed": seed,
        "balance_score": score,
        "n_features_scored": len(components),
        "largest_standardized_feature_deviations": dict(
            sorted(components.items(), key=lambda item: (-item[1], item[0]))[:20]
        ),
    }


def manifest_sample(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "sample_id", "patient_id", "patient_identity_basis", "report_id",
        "source_uri", "source_record", "source_payload_sha256", "tumor_source",
        "payload_sha256", "normalized_payload_sha256", "target", "target_sha256",
        "structured_target", "structured_target_sha256", "label_source",
        "label_source_sha256", "n_entities", "entity_types", "report_length_chars",
        "report_length_tokens", "negation_term_count", "uncertainty_term_count",
        "n_training_synthetic", "gold_annotation_sha256", "tcga242_category",
    )
    return {key: row[key] for key in keep if key in row}


def validate_disjoint(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = list(splits)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            a, b = splits[name_a], splits[name_b]
            overlaps = {}
            for field in ("sample_id", "patient_id", "payload_sha256", "normalized_payload_sha256"):
                common = sorted({row[field] for row in a} & {row[field] for row in b})
                overlaps[field] = {"n": len(common), "examples": common[:10]}
                if common:
                    raise AssertionError(f"{field} overlap {name_a}/{name_b}: {common[:3]}")
            result[f"{name_a}__{name_b}"] = overlaps
    return result


def build(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labeler_code_sha = sha256_file(Path(corpus_real.__file__).resolve())
    source_parquet_sha = sha256_file(args.tcga_reports_parquet)

    # Load and freeze every TCGA-242 report before touching the training source.
    all_gold_raw: list[dict[str, Any]] = []
    for category in range(1, 6):
        for report_path in sorted((args.tcga242_reports_dir / str(category)).glob("*.txt")):
            report_id, text = extract_tcga242(report_path)
            all_gold_raw.append({
                "patient_id": patient_id(report_id),
                "payload_sha256": sha256_text(text),
                "normalized_payload_sha256": normalized_text_hash(text),
            })
    heldout_patients = {row["patient_id"] for row in all_gold_raw}
    heldout_payloads = {row["payload_sha256"] for row in all_gold_raw}
    heldout_normalized = {row["normalized_payload_sha256"] for row in all_gold_raw}

    df = pd.read_parquet(args.tcga_reports_parquet)
    required_columns = {"patient_filename", "study", "text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"source parquet lacks columns: {sorted(required_columns - set(df.columns))}")

    candidate_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for source in sorted(df.to_dict(orient="records"), key=lambda row: str(row["patient_filename"])):
        report_id = str(source["patient_filename"])
        patient = patient_id(report_id)
        text = source["text"]
        if not isinstance(text, str) or len(text.strip()) < 50:
            exclusions.append({"report_id": report_id, "patient_id": patient, "reason": "invalid_or_short_text"})
            continue
        payload_sha = sha256_text(text)
        normalized_sha = normalized_text_hash(text)
        if patient in heldout_patients:
            exclusions.append({"report_id": report_id, "patient_id": patient, "reason": "reserved_tcga242_patient", "payload_sha256": payload_sha})
            continue
        if payload_sha in heldout_payloads or normalized_sha in heldout_normalized:
            exclusions.append({"report_id": report_id, "patient_id": patient, "reason": "reserved_tcga242_text_equivalent", "payload_sha256": payload_sha})
            continue
        entities = corpus_real.weak_label(text)
        if not entities:
            exclusions.append({"report_id": report_id, "patient_id": patient, "reason": "no_deterministic_span_label", "payload_sha256": payload_sha})
            continue
        tokens, labels = corpus_real.bio_from_char_spans(text, entities)
        row = make_row(
            sample_id=report_id,
            report_id=report_id,
            patient=patient,
            text=text,
            tumor_source=str(source["study"]),
            tokens=tokens,
            labels=labels,
            entities=entities,
            label_source="deterministic span-anchored weak supervision over real TCGA report text",
            label_source_inputs={
                "labeler_code_sha256": labeler_code_sha,
                "source_parquet_sha256": source_parquet_sha,
            },
            source_uri=TCGA_REPORTS_URI,
            source_record=report_id,
            source_payload_sha256=payload_sha,
        )
        candidate_records.append(row)

    # Remove exact or formatting-only duplicates before split. The earliest
    # source accession is retained, and exclusions retain the full audit trail.
    deduplicated: list[dict[str, Any]] = []
    seen_normalized: dict[str, str] = {}
    for row in sorted(candidate_records, key=lambda item: item["sample_id"]):
        key = row["normalized_payload_sha256"]
        if key in seen_normalized:
            exclusions.append({
                "report_id": row["report_id"],
                "patient_id": row["patient_id"],
                "reason": "duplicate_normalized_report_text",
                "kept_sample_id": seen_normalized[key],
                "payload_sha256": row["payload_sha256"],
            })
            continue
        seen_normalized[key] = row["sample_id"]
        deduplicated.append(row)

    validation_ids, split_search = select_validation(
        deduplicated,
        fraction=args.validation_fraction,
        seed_start=args.split_seed,
        n_seed_candidates=args.split_seed_candidates,
    )
    validation_rows = sorted((row for row in deduplicated if row["sample_id"] in validation_ids), key=lambda row: row["sample_id"])
    train_rows = sorted((row for row in deduplicated if row["sample_id"] not in validation_ids), key=lambda row: row["sample_id"])

    test_rows, test_exclusions = build_gold_rows(
        args.tcga242_reports_dir,
        args.tcga242_annotations_dir,
        categories=(1, 2),
        converter_code_sha256=labeler_code_sha,
    )
    external_rows, external_exclusions = build_gold_rows(
        args.tcga242_reports_dir,
        args.tcga242_annotations_dir,
        categories=(3, 4, 5),
        converter_code_sha256=labeler_code_sha,
    )
    exclusions.extend({**item, "cohort": "primary_test"} for item in test_exclusions)
    exclusions.extend({**item, "cohort": "external_validation"} for item in external_exclusions)

    splits = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "external_validation": external_rows,
    }
    disjointness = validate_disjoint(splits)

    all_rows = train_rows + validation_rows + test_rows + external_rows
    sample_ids = [row["sample_id"] for row in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
        raise AssertionError(f"duplicate dataset sample IDs: {duplicates[:10]}")
    target_codes: dict[int, str] = {}
    for row in all_rows:
        previous = target_codes.setdefault(row["target"], row["structured_target_sha256"])
        if previous != row["structured_target_sha256"]:
            raise AssertionError("strict target code collision")

    dataset = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "dataset": "TCGA pathology reports: real weak-supervision train plus TCGA-242 pathologist gold",
        "source_uri": f"{TCGA_REPORTS_URI}; {TCGA242_URI}",
        "license": LICENSE,
        "created_at": args.created_at,
        "patient_identity_basis": "TCGA barcode first 12 characters",
        "split_unit": "patient before any token/window expansion",
        "training_label_policy": "deterministic span-anchored weak supervision; no LLM labels and no synthetic reports",
        "test_label_policy": "pathologist-adjudicated TCGA-242 structured gold with deterministic surface anchoring",
        "labeler_code_sha256": labeler_code_sha,
        "source_parquet_sha256": source_parquet_sha,
        "strict_target_definition": "first 60 bits of canonical structured-entity-set SHA256; equality is report-level exact entity-set match",
        "samples": [manifest_sample(row) for row in all_rows],
        "external_validation_sample_ids": [row["sample_id"] for row in external_rows],
    }
    dataset_path = args.out_dir / "clinicalbert_dataset_v2.json"
    write_json(dataset_path, dataset)
    dataset_sha = sha256_file(dataset_path)

    split_manifest = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "dataset_manifest_path": "artifacts/clinicalbert/clinicalbert_dataset_v2.json",
        "dataset_manifest_sha256": dataset_sha,
        "seed": split_search["selected_seed"],
        "strategy": split_search["strategy"],
        "split_search": split_search,
        "patient_disjoint": True,
        "window_disjoint": True,
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "validation_sample_ids": [row["sample_id"] for row in validation_rows],
        "test_sample_ids": [row["sample_id"] for row in test_rows],
        "external_validation_sample_ids": [row["sample_id"] for row in external_rows],
        "train_patient_ids": sorted({row["patient_id"] for row in train_rows}),
        "validation_patient_ids": sorted({row["patient_id"] for row in validation_rows}),
        "test_patient_ids": sorted({row["patient_id"] for row in test_rows}),
        "external_validation_patient_ids": sorted({row["patient_id"] for row in external_rows}),
        "disjointness": disjointness,
    }
    split_path = args.out_dir / "clinicalbert_split_v2.json"
    write_json(split_path, split_manifest)

    split_files = {
        "train": args.out_dir / "clinicalbert_train_reports.jsonl",
        "validation": args.out_dir / "clinicalbert_validation_reports.jsonl",
        "test": args.out_dir / "clinicalbert_test_reports.jsonl",
        "external_validation": args.out_dir / "clinicalbert_external_reports.jsonl",
    }
    for name, path in split_files.items():
        write_jsonl(path, splits[name])
    exclusions_path = args.out_dir / "clinicalbert_data_exclusions_v2.json"
    write_json(exclusions_path, {
        "schema_version": 2,
        "capability": "clinicalbert",
        "created_at": args.created_at,
        "counts_by_reason": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
        "rows": sorted(exclusions, key=lambda item: (str(item.get("reason")), str(item.get("report_id", item.get("source_record", ""))))),
    })

    entity_coverage = {
        name: dict(sorted(Counter(entity for row in rows for entity in row["entity_types"]).items()))
        for name, rows in splits.items()
    }
    # Also record mention-level counts, not only report-level presence.
    entity_mentions = {
        name: dict(sorted(Counter(entity["entity_type"] for row in rows for entity in row["structured_target"]).items()))
        for name, rows in splits.items()
    }
    build_manifest = {
        "schema_version": 2,
        "capability": "clinicalbert",
        "created_at": args.created_at,
        "n_source_reports": int(len(df)),
        "n_reserved_tcga242_reports": len(all_gold_raw),
        "n_reserved_tcga242_patients": len(heldout_patients),
        "n_training": len(train_rows),
        "n_validation": len(validation_rows),
        "n_test": len(test_rows),
        "n_external_validation": len(external_rows),
        "n_training_synthetic": False,
        "patient_disjoint": True,
        "dataset_sha256": dataset_sha,
        "split_sha256": sha256_file(split_path),
        "labeler_code_sha256": labeler_code_sha,
        "source_parquet_sha256": source_parquet_sha,
        "entity_report_coverage": entity_coverage,
        "entity_mention_coverage": entity_mentions,
        "split_files": {
            name: {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in split_files.items()
        },
        "exclusions": {
            "filename": exclusions_path.name,
            "bytes": exclusions_path.stat().st_size,
            "sha256": sha256_file(exclusions_path),
            "counts_by_reason": dict(sorted(Counter(item["reason"] for item in exclusions).items())),
        },
        "source_licenses": {
            TCGA_REPORTS_URI: LICENSE,
            TCGA242_URI: LICENSE,
        },
    }
    build_path = args.out_dir / "clinicalbert_data_build_v2.json"
    write_json(build_path, build_manifest)

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "n_training": len(train_rows),
        "n_validation": len(validation_rows),
        "n_test": len(test_rows),
        "n_external_validation": len(external_rows),
        "selected_split_seed": split_search["selected_seed"],
        "dataset_sha256": dataset_sha,
        "split_sha256": sha256_file(split_path),
        "build_manifest_sha256": sha256_file(build_path),
        "exclusion_counts": build_manifest["exclusions"]["counts_by_reason"],
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga-reports-parquet", type=Path, required=True)
    parser.add_argument("--tcga242-reports-dir", type=Path, required=True)
    parser.add_argument("--tcga242-annotations-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--split-seed-candidates", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
