#!/usr/bin/env python3
"""Prepare real BACH microscopy images for the MedSigLIP biopsy probe.

The official patient-origin table is only partial. To prevent unprovable
patient overlap, every image with an unresolved patient identity is assigned
to one conservative identity-equivalence group and that group is forced into
the training split. Identified patients are assigned as indivisible groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

DATASET_REPO = "1aurent/BACH"
DATASET_REVISION = "55effe690290867396fc4403792fd27553376b34"
SOURCE_URI = f"https://huggingface.co/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}"
PATIENT_METADATA_URI = (
    "https://www.dropbox.com/sh/sc7yg21bcs3wr0z/"
    "AACiavY0BQPF6GYna9Fkjzola?e=1&dl=0"
)
LICENSE = "CC BY-NC-ND 4.0"
UNRESOLVED_GROUP = "BACH-UNRESOLVED-IDENTITY-EQUIVALENCE"
LABELS = {
    0: ("Benign", "benign_or_normal", 0),
    1: ("InSitu", "in_situ_carcinoma", 0),
    2: ("Invasive", "invasive_carcinoma", 1),
    3: ("Normal", "benign_or_normal", 0),
}
# Prespecified from labels + patient groups only, before embedding or fitting.
# Seed 20260928 deterministic grouped balance search; unresolved identities
# are one conservative group and can appear only in train.
TRAIN_PATIENTS = {
    "BACH-P12", "BACH-P18", "BACH-P21", "BACH-P22", "BACH-P24",
    "BACH-P25", "BACH-P26", "BACH-P31", "BACH-P32", "BACH-P38",
    UNRESOLVED_GROUP,
}
VALIDATION_PATIENTS = {
    "BACH-P11", "BACH-P14", "BACH-P15", "BACH-P20", "BACH-P30",
    "BACH-P33", "BACH-P34", "BACH-P36", "BACH-P37",
}
TEST_PATIENTS = {
    "BACH-P13", "BACH-P16", "BACH-P17", "BACH-P19", "BACH-P23",
    "BACH-P27", "BACH-P28", "BACH-P29", "BACH-P35", "BACH-P39",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_patient_map(path: Path) -> dict[str, str]:
    table = pd.read_excel(path)
    result: dict[str, str] = {}
    for start in (0, 4, 8, 12):
        for _, row in table.iloc[:, [start, start + 1, start + 2]].iterrows():
            filename = str(row.iloc[0])
            patient = row.iloc[2]
            patient_id = (
                f"BACH-P{int(patient):02d}" if pd.notna(patient) else UNRESOLVED_GROUP
            )
            if filename in result:
                raise ValueError(f"duplicate patient metadata filename: {filename}")
            result[filename] = patient_id
    if len(result) != 400:
        raise ValueError(f"expected 400 metadata rows, got {len(result)}")
    return result


def split_for_patient(patient_id: str) -> str:
    memberships = [
        patient_id in TRAIN_PATIENTS,
        patient_id in VALIDATION_PATIENTS,
        patient_id in TEST_PATIENTS,
    ]
    if sum(memberships) != 1:
        raise ValueError(f"patient {patient_id} has invalid split membership {memberships}")
    return ("train", "validation", "test")[memberships.index(True)]


def prepare(snapshot_dir: Path, metadata_xlsx: Path, out_dir: Path) -> dict[str, Any]:
    patient_map = load_patient_map(metadata_xlsx)
    parquet_paths = sorted((snapshot_dir / "data").glob("train-*.parquet"))
    if len(parquet_paths) != 15:
        raise ValueError(f"expected 15 train parquet shards, got {len(parquet_paths)}")
    images_dir = out_dir / "images_896x672_png"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for shard in parquet_paths:
        table = pq.read_table(shard, columns=["image", "label"])
        for image_record, label_value in zip(
            table["image"].to_pylist(), table["label"].to_pylist(), strict=True
        ):
            label = int(label_value)
            raw = image_record["bytes"]
            source_name = str(image_record["path"])
            if source_name in seen_names:
                raise ValueError(f"duplicate image name {source_name}")
            seen_names.add(source_name)
            if source_name not in patient_map:
                raise ValueError(f"image {source_name} missing patient metadata row")
            original_label, class3, binary_target = LABELS[label]
            prefix_expected = {0: "b", 1: "is", 2: "iv", 3: "n"}[label]
            if not source_name.startswith(prefix_expected):
                raise ValueError(f"label/path mismatch: {label=} {source_name=}")
            patient_id = patient_map[source_name]
            split = split_for_patient(patient_id)
            with Image.open(BytesIO(raw)) as image:
                image = image.convert("RGB").resize((896, 672), Image.Resampling.LANCZOS)
                output_name = f"{Path(source_name).stem}.png"
                output_path = images_dir / output_name
                image.save(output_path, format="PNG", optimize=True)
            payload_sha = sha256_file(output_path)
            samples.append(
                {
                    "sample_id": f"BACH-{Path(source_name).stem}",
                    "patient_id": patient_id,
                    "patient_identity_basis": (
                        "official_partial_patient_id"
                        if patient_id != UNRESOLVED_GROUP
                        else "conservative_equivalence_group_for_all_unresolved_ids"
                    ),
                    "split": split,
                    "source_name": source_name,
                    "source_label": original_label,
                    "multiclass_target": class3,
                    "target": binary_target,
                    "source_payload_sha256": sha256_bytes(raw),
                    "payload_name": output_name,
                    "payload_sha256": payload_sha,
                    "payload_bytes": output_path.stat().st_size,
                    "source_shard": shard.name,
                    "source_shard_sha256": sha256_file(shard),
                }
            )
    if len(samples) != 400 or len(seen_names) != 400 or set(seen_names) != set(patient_map):
        raise ValueError("BACH source/metadata coverage is not exactly 400 images")
    sample_ids = [row["sample_id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs are not unique")
    counts: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in samples if row["split"] == split]
        counts[split] = {
            "n": len(subset),
            "n_patients": len({row["patient_id"] for row in subset}),
            "class3": {
                name: sum(row["multiclass_target"] == name for row in subset)
                for name in ("benign_or_normal", "in_situ_carcinoma", "invasive_carcinoma")
            },
            "binary": {
                "negative": sum(row["target"] == 0 for row in subset),
                "positive": sum(row["target"] == 1 for row in subset),
            },
        }
    if [counts[s]["n"] for s in ("train", "validation", "test")] != [292, 54, 54]:
        raise ValueError(f"unexpected split counts: {counts}")
    patient_sets = {
        split: {row["patient_id"] for row in samples if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(
        patient_sets[a] & patient_sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("patient identities overlap across splits")
    payload = {
        "schema_version": 2,
        "dataset": "BACH breast histology microscopy (three-class biopsy probe)",
        "source_uri": SOURCE_URI,
        "source_revision": DATASET_REVISION,
        "patient_metadata_uri": PATIENT_METADATA_URI,
        "patient_metadata_sha256": sha256_file(metadata_xlsx),
        "license": LICENSE,
        "seed": 20260928,
        "split_method": "grouped label-balance search; unresolved identity equivalence forced to train",
        "positive_class_for_validator_auroc": "invasive_carcinoma",
        "classes": ["benign_or_normal", "in_situ_carcinoma", "invasive_carcinoma"],
        "patient_disjoint": True,
        "counts": counts,
        "samples": sorted(samples, key=lambda row: row["sample_id"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_table.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--patient-metadata-xlsx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.snapshot_dir, args.patient_metadata_xlsx, args.out_dir)
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
