"""Phikon NCT-CRC linear-probe delivery on Modal.

Downloads NCT-CRC-HE-100K + CRC-VAL-HE-7K, embeds with owkin/phikon (768-d),
fits StandardScaler+LogisticRegression, writes joblib + delivery manifests
to volume ``phikon-nct-crc``.

Run::
  MODAL_PROFILE=fjkiani modal run deploy/modal/phikon_probe_train_app.py
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import modal

APP_VERSION = "phikon-probe-train-v0.1.0"
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
ZENODO = "https://zenodo.org/records/1214456/files"
TRAIN_ZIP = "NCT-CRC-HE-100K.zip"
VAL_ZIP = "CRC-VAL-HE-7K.zip"

VOL = modal.Volume.from_name("phikon-nct-crc", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "safetensors==0.4.5",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "Pillow==10.4.0",
        "requests==2.32.3",
        "tqdm==4.66.5",
    )
)

app = modal.App("phikon-probe-train")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _download(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest} exists ({dest.stat().st_size} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)
    print(f"[dl] wrote {dest} ({dest.stat().st_size} bytes)", flush=True)


def _ensure_unzipped(zip_path: Path, out_dir: Path) -> Path:
    marker = out_dir / ".unzip_ok"
    if marker.is_file() and out_dir.is_dir():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text("ok")
    return out_dir


def _find_class_root(root: Path) -> Path:
    # zip may nest one extra directory
    if all((root / c).is_dir() for c in CLASSES):
        return root
    for child in root.iterdir():
        if child.is_dir() and all((child / c).is_dir() for c in CLASSES):
            return child
    raise FileNotFoundError(f"class dirs not found under {root}")


def _gather(root: Path, split: str) -> List[Dict[str, Any]]:
    rows = []
    for cls in CLASSES:
        for fp in sorted((root / cls).glob("*.tif")):
            rows.append(
                {
                    "path": str(fp),
                    "class": cls,
                    "label": CLASS_TO_IDX[cls],
                    "split": split,
                    "stem": fp.stem,
                }
            )
    return rows


@app.function(
    image=IMAGE,
    gpu="A10G",
    timeout=60 * 60 * 6,
    memory=65536,
    volumes={"/vol/phikon": VOL},
)
def train_probe(batch_size: int = 64) -> Dict[str, Any]:
    import numpy as np
    import torch
    from joblib import dump as joblib_dump
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoImageProcessor, AutoModel

    t0 = time.time()
    root = Path("/vol/phikon")
    raw = root / "raw"
    emb_dir = root / "embeddings"
    out_dir = root / "delivery"
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_zip = raw / TRAIN_ZIP
    val_zip = raw / VAL_ZIP
    _download(f"{ZENODO}/{TRAIN_ZIP}", train_zip)
    _download(f"{ZENODO}/{VAL_ZIP}", val_zip)
    train_root = _find_class_root(_ensure_unzipped(train_zip, raw / "NCT-CRC-HE-100K"))
    val_root = _find_class_root(_ensure_unzipped(val_zip, raw / "CRC-VAL-HE-7K"))

    manifest = _gather(train_root, "train") + _gather(val_root, "val")
    n = len(manifest)
    print(f"[manifest] n={n}", flush=True)
    (emb_dir / "manifest.json").write_text(json.dumps(manifest))

    emb_path = emb_dir / "embeddings.npy"
    done_path = emb_dir / "done_mask.npy"
    if (
        emb_path.is_file()
        and done_path.is_file()
        and done_path.stat().st_size > 0
        and int(np.load(done_path).sum()) == n
    ):
        embeddings = np.load(emb_path)
        done_mask = np.load(done_path).astype(bool)
        print(f"[embed] resume complete {done_mask.sum()}/{n}", flush=True)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon").to(device).eval()
        embeddings = np.zeros((n, 768), dtype=np.float32)
        done_mask = np.zeros((n,), dtype=bool)
        if emb_path.is_file() and done_path.is_file():
            old = np.load(emb_path)
            old_done = np.load(done_path).astype(bool)
            if old.shape == embeddings.shape:
                embeddings, done_mask = old, old_done
                print(f"[embed] resume from {done_mask.sum()}/{n}", flush=True)

        idxs = [i for i in range(n) if not done_mask[i]]
        print(f"[embed] remaining={len(idxs)} device={device}", flush=True)
        for start in range(0, len(idxs), batch_size):
            batch_i = idxs[start : start + batch_size]
            images = [Image.open(manifest[i]["path"]).convert("RGB") for i in batch_i]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                # CLS token
                feats = out.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            for j, i in enumerate(batch_i):
                embeddings[i] = feats[j]
                done_mask[i] = True
            if (start // batch_size) % 20 == 0:
                np.save(emb_path, embeddings)
                np.save(done_path, done_mask)
                VOL.commit()
                print(
                    f"[embed] {done_mask.sum()}/{n} "
                    f"({100*done_mask.sum()/n:.1f}%)",
                    flush=True,
                )
        np.save(emb_path, embeddings)
        np.save(done_path, done_mask)
        VOL.commit()

    assert int(done_mask.sum()) == n
    assert embeddings.shape == (n, 768)
    assert np.isfinite(embeddings).all()

    labels = np.asarray([m["label"] for m in manifest], dtype=np.int64)
    splits = np.asarray([m["split"] for m in manifest])
    train_mask = splits == "train"
    val_mask = splits == "val"
    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]

    # Hold out 10% of CRC-VAL as delivery "validation"; rest = test
    val_idx = np.where(val_mask)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(val_idx)
    n_val_hold = max(1, len(val_idx) // 10)
    delivery_val_idx = set(val_idx[:n_val_hold].tolist())
    delivery_test_idx = set(val_idx[n_val_hold:].tolist())

    print(
        f"[fit] train={train_mask.sum()} delivery_val={len(delivery_val_idx)} "
        f"delivery_test={len(delivery_test_idx)}",
        flush=True,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=42,
            class_weight="balanced",
        ),
    )
    t_fit = time.time()
    clf.fit(X_train, y_train)
    fit_s = time.time() - t_fit

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)
    macro = float(f1_score(y_val, y_pred, average="macro"))
    acc = float(accuracy_score(y_val, y_pred))
    bal = float(balanced_accuracy_score(y_val, y_pred))
    print(f"[eval] full CRC-VAL macro_f1={macro:.4f} acc={acc:.4f} bal={bal:.4f}", flush=True)

    joblib_path = out_dir / "phikon_probe_v1.joblib"
    joblib_dump(clf, joblib_path)
    joblib_sha = _sha256_file(joblib_path)
    emb_sha = _sha256_file(emb_path)

    # Build delivery sample table (all tiles)
    samples = []
    split_ids = {"train": [], "validation": [], "test": []}
    for i, m in enumerate(manifest):
        sample_id = f"phikon-{m['split']}-{m['class']}-{m['stem']}"
        patient_id = f"{m['split']}:{m['stem']}"  # unique → patient-disjoint
        target = int(m["label"])
        payload_sha = hashlib.sha256(Path(m["path"]).read_bytes()).hexdigest()
        target_sha = _canonical_json_sha(target)
        samples.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "payload_sha256": payload_sha,
                "target": target,
                "target_sha256": target_sha,
                "class_name": m["class"],
            }
        )
        if m["split"] == "train":
            split_ids["train"].append(sample_id)
        elif i in delivery_val_idx:
            split_ids["validation"].append(sample_id)
        else:
            split_ids["test"].append(sample_id)

    # Evaluation rows = delivery test only (recompute macro_f1)
    test_rows = []
    # Map sample_id -> index
    id_to_i = {
        f"phikon-{manifest[i]['split']}-{manifest[i]['class']}-{manifest[i]['stem']}": i
        for i in range(n)
    }
    y_pred_all = clf.predict(embeddings)
    for sid in split_ids["test"]:
        i = id_to_i[sid]
        s = next(x for x in samples if x["sample_id"] == sid)
        test_rows.append(
            {
                "sample_id": sid,
                "y_true": int(labels[i]),
                "y_pred": int(y_pred_all[i]),
                "target_sha256": s["target_sha256"],
            }
        )

    # recompute macro_f1 on test_rows
    labels_u = sorted({r["y_true"] for r in test_rows} | {r["y_pred"] for r in test_rows})
    f1s = []
    for lab in labels_u:
        tp = sum(r["y_true"] == lab and r["y_pred"] == lab for r in test_rows)
        fp = sum(r["y_true"] != lab and r["y_pred"] == lab for r in test_rows)
        fn = sum(r["y_true"] == lab and r["y_pred"] != lab for r in test_rows)
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else 2 * tp / den)
    test_macro = sum(f1s) / len(f1s)

    receipt = {
        "status": "ok",
        "app_version": APP_VERSION,
        "joblib_path": str(joblib_path),
        "joblib_sha256": joblib_sha,
        "embeddings_sha256": emb_sha,
        "n_total": n,
        "n_train": int(train_mask.sum()),
        "n_validation": len(split_ids["validation"]),
        "n_test": len(split_ids["test"]),
        "fit_seconds": fit_s,
        "full_crcval_macro_f1": macro,
        "full_crcval_accuracy": acc,
        "full_crcval_balanced_accuracy": bal,
        "delivery_test_macro_f1": test_macro,
        "elapsed_seconds": round(time.time() - t0, 3),
        "model": "owkin/phikon",
        "probe": "StandardScaler+LogisticRegression(multinomial,C=1.0,class_weight=balanced)",
    }
    (out_dir / "train_receipt.json").write_text(json.dumps(receipt, indent=2))
    (out_dir / "samples.json").write_text(json.dumps({"samples": samples}))
    (out_dir / "split_ids.json").write_text(json.dumps(split_ids))
    (out_dir / "test_rows.json").write_text(json.dumps({"rows": test_rows, "macro_f1": test_macro}))
    VOL.commit()
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


@app.local_entrypoint()
def main() -> None:
    r = train_probe.remote()
    print(json.dumps(r, indent=2))
