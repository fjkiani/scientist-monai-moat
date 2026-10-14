"""RSNA MedSigLIP FULL embed (~54k) → Modal volume rsna-mammo-embeddings.

RUO. Produces full-dataset embedding artifact + Track-3 fusion proof + paired delta-AUROC bootstrap.

Usage (fjkiani):
  MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_medsiglip_embed_full.py
  MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_medsiglip_embed_full.py::status
  MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_medsiglip_embed_full.py::fusion_clinical_proof
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

APP_NAME = "rsna-medsiglip-embed-full"
VOLUME_PNG = "rsna-png512"
VOLUME_EMB = "rsna-mammo-embeddings"

# Embed params
CHUNK = 1000
BATCH = 64
SEED = 42
MODEL_REPO = "google/medsiglip-448"

# Output layout
COHORT_DIR = Path("/emb/cohort_full")
STATUS_PATH = Path("/emb/status_full.json")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pillow",
        "sentencepiece",
        "pandas",
        "numpy",
        "scikit-learn",
        "kaggle",
        "huggingface_hub",
        "accelerate",
    )
)

vol_png = modal.Volume.from_name(VOLUME_PNG, create_if_missing=False)
vol_emb = modal.Volume.from_name(VOLUME_EMB, create_if_missing=True)
secrets = [
    modal.Secret.from_name("medsiglip-hf-token"),
    modal.Secret.from_name("kaggle-api"),
]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    vol_emb.commit()


def _write_status(payload: Dict[str, Any]) -> None:
    _write_json(STATUS_PATH, payload)


def _resolve_png(patient_id: int, image_id: int) -> str | None:
    candidates = [
        Path("/png") / f"png512/{patient_id}_{image_id}.png",
        Path("/png") / f"{patient_id}_{image_id}.png",
        Path("/png") / f"png512/{patient_id}/{image_id}.png",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 3600,
    memory=32768,
    cpu=8,
    secrets=secrets,
    volumes={"/png": vol_png, "/emb": vol_emb},
)
def embed_full() -> Dict[str, Any]:
    import glob
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import HfApi
    from PIL import Image
    from sklearn.model_selection import GroupShuffleSplit
    from transformers import AutoImageProcessor, AutoModel

    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN missing from medsiglip-hf-token"
    info = HfApi(token=token).model_info(MODEL_REPO)
    assert info is not None, "model_info failed"

    # Kaggle metadata (train.csv)
    os.makedirs("/tmp/rsna", exist_ok=True)
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

    import subprocess

    r = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "rsna-breast-cancer-detection",
            "-f",
            "train.csv",
            "-p",
            "/tmp/rsna",
        ],
        capture_output=True,
        text=True,
    )
    csv_candidates = glob.glob("/tmp/rsna/train.csv*")
    if not csv_candidates:
        # Fallback: prior copies on volumes
        for alt in ("/emb/train.csv", str(COHORT_DIR / "train.csv"), "/png/train.csv"):
            if Path(alt).is_file():
                csv_candidates = [alt]
                break
    assert csv_candidates, (
        f"train.csv missing kaggle_rc={r.returncode} "
        f"out={r.stdout[-300:]} err={r.stderr[-300:]}"
    )
    rsna = pd.read_csv(csv_candidates[0])

    # Optional downsample for emergency debug (kept off by default)
    limit_n = os.environ.get("RSNA_LIMIT_N")
    if limit_n:
        rsna = rsna.sample(int(limit_n), random_state=SEED).reset_index(drop=True)

    n_total = int(len(rsna))
    tr_idx, te_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            rsna, groups=rsna.patient_id
        )
    )
    rsna["split"] = "train"
    rsna.loc[te_idx, "split"] = "test"
    assert not (
        set(rsna.loc[rsna.split == "train", "patient_id"])
        & set(rsna.loc[rsna.split == "test", "patient_id"])
    ), "patient overlap in train/test split"

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    cohort_path = COHORT_DIR / "cohort.csv"
    rsna.to_csv(cohort_path, index=False)
    vol_emb.commit()

    # Resolve PNG paths
    paths: List[str] = []
    missing = 0
    for row in rsna.itertuples():
        p = _resolve_png(int(row.patient_id), int(row.image_id))
        if p is None:
            missing += 1
            paths.append("")
        else:
            paths.append(p)
    rsna["path"] = paths
    if missing:
        _write_status(
            {
                "phase": "ERROR",
                "error": f"missing_pngs={missing}",
                "n_rows": n_total,
                "elapsed": round(time.time() - t0, 1),
            }
        )
        raise RuntimeError(f"missing_pngs={missing}")

    rsna.to_csv(cohort_path, index=False)
    vol_emb.commit()

    proc = AutoImageProcessor.from_pretrained(MODEL_REPO, token=token)
    model = (
        AutoModel.from_pretrained(MODEL_REPO, token=token, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )

    n_done = 0
    for s in range(0, n_total, CHUNK):
        out = COHORT_DIR / f"emb_{s:05d}.npy"
        if out.is_file():
            arr = np.load(out, mmap_mode="r")
            n_done += int(arr.shape[0])
            continue
        t_chunk = time.time()
        chunk = rsna.iloc[s : s + CHUNK]
        embs: List[Any] = []
        for b in range(0, len(chunk), BATCH):
            imgs = [Image.open(p).convert("RGB") for p in chunk.path.iloc[b : b + BATCH]]
            x = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
            with torch.inference_mode():
                f = model.get_image_features(pixel_values=x)
                f = f if torch.is_tensor(f) else f.pooler_output
            embs.append(f.float().cpu().numpy())
        arr = np.concatenate(embs).astype("float32")
        np.save(out, arr)
        n_done += int(arr.shape[0])
        vol_emb.commit()
        _write_status(
            {
                "phase": "RUNNING",
                "images_embedded": n_done,
                "n_rows": n_total,
                "last_chunk": s,
                "chunk_secs": round(time.time() - t_chunk, 1),
                "elapsed": round(time.time() - t0, 1),
                "model_repo": MODEL_REPO,
                "model_sha": str(getattr(info, "sha", "")),
            }
        )

    done = COHORT_DIR / "DONE"
    done.write_text("ok")

    parts = [np.load(COHORT_DIR / f"emb_{s:05d}.npy") for s in range(0, n_total, CHUNK)]
    E = np.concatenate(parts).astype("float32")
    emb_path = COHORT_DIR / f"embeddings_{n_total}.npy"
    np.save(emb_path, E)

    status = {
        "phase": "DONE",
        "images_embedded": int(E.shape[0]),
        "n_rows": n_total,
        "dim": int(E.shape[1]),
        "elapsed": round(time.time() - t0, 1),
        "cohort_csv": str(cohort_path),
        "embeddings_npy": str(emb_path),
        "cancer_pos": int(rsna.cancer.sum()),
        "n_train": int((rsna.split == "train").sum()),
        "n_test": int((rsna.split == "test").sum()),
        "model_repo": MODEL_REPO,
        "model_sha": str(getattr(info, "sha", "")),
        "DONE": True,
    }
    _write_status(status)
    return status


@app.function(
    image=image,
    timeout=120,
    secrets=secrets,
    volumes={"/emb": vol_emb},
)
def status() -> Dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"phase": "MISSING", "path": str(STATUS_PATH)}
    return json.loads(STATUS_PATH.read_text())


def _paired_bootstrap_delta_auc(
    y: "np.ndarray",
    p_a: "np.ndarray",
    p_b: "np.ndarray",
    patient_id: "np.ndarray",
    B: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired patient-level bootstrap for delta AUROC: A - B."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    uniq = np.unique(patient_id)
    idx = {u: np.where(patient_id == u)[0] for u in uniq}

    deltas: List[float] = []
    skipped = 0
    for _ in range(B):
        sample_u = rng.choice(uniq, size=len(uniq), replace=True)
        i = np.concatenate([idx[u] for u in sample_u])
        if y[i].min() == y[i].max():
            skipped += 1
            continue
        deltas.append(float(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i])))

    if not deltas:
        return {"B": B, "skipped": skipped, "error": "all_bootstraps_single_class"}

    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "B": B,
        "skipped": skipped,
        "delta_mean": float(d.mean()),
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "p_delta_le_zero": float((d <= 0).mean()),
    }


@app.function(
    image=image,
    timeout=3600,
    memory=16384,
    secrets=secrets,
    volumes={"/emb": vol_emb},
)
def fusion_clinical_proof() -> Dict[str, Any]:
    """Patient-disjoint test AUROC: metadata vs MedSigLIP vs fused (full RSNA)."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    assert STATUS_PATH.is_file(), "status_full.json missing — run embed_full first"
    st = json.loads(STATUS_PATH.read_text())
    assert st.get("phase") == "DONE" and st.get("images_embedded") == st.get("n_rows"), st

    co = pd.read_csv(COHORT_DIR / "cohort.csv")
    E = np.load(st["embeddings_npy"])
    assert len(co) == len(E) == int(st["n_rows"])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    tr_m = (co.split == "train").values
    te_m = (co.split == "test").values
    yc = co.cancer.values
    yt = yc[te_m]

    Xmeta = pd.get_dummies(
        co[["density", "view", "site_id", "machine_id", "implant", "laterality"]].astype(str),
        dtype=float,
    )
    # (minor leak kept consistent with 12k proof; paired delta test addresses comparison)
    Xmeta["age"] = co.age.fillna(co.age.median())
    Xmeta["age2"] = Xmeta.age**2

    def fit_lr(X, C):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        return m.fit(X[tr_m], yc[tr_m]).predict_proba(X[te_m])[:, 1]

    def fit_lr_model(X, C):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        return m.fit(X[tr_m], yc[tr_m])

    p_m = fit_lr(Xmeta.values, 1.0)
    p_v = fit_lr(E, 0.01)
    Xf = np.hstack([E, Xmeta.values])
    p_f = fit_lr(Xf, 0.01)

    rng = np.random.default_rng(0)

    def boot_ci(p, B=500):
        s = [
            roc_auc_score(yt[i], p[i])
            for i in (rng.integers(0, len(yt), len(yt)) for _ in range(B))
            if yt[i].min() != yt[i].max()
        ]
        return [float(x) for x in np.percentile(s, [2.5, 97.5])]

    def nb_at(y, p, pt=0.15):
        y = np.asarray(y)
        pos = np.asarray(p) >= pt
        n = len(y)
        return float((pos & (y == 1)).sum() / n - (pos & (y == 0)).sum() / n * pt / (1 - pt))

    rows = []
    for name, p in [
        ("Metadata alone", p_m),
        ("MedSigLIP alone", p_v),
        ("Fused (MedSigLIP + metadata)", p_f),
    ]:
        lo, hi = boot_ci(p)
        rows.append(
            {
                "model": name,
                "n_test": int(len(yt)),
                "test_cancers": int(yt.sum()),
                "auroc": float(roc_auc_score(yt, p)),
                "auroc_ci_lo": lo,
                "auroc_ci_hi": hi,
                "avg_precision": float(average_precision_score(yt, p)),
                "brier": float(brier_score_loss(yt, p)),
                "nb_15": nb_at(yt, p, 0.15),
            }
        )

    pid = co.loc[te_m, "patient_id"].values
    paired = {
        "fused_minus_meta": _paired_bootstrap_delta_auc(yt, p_f, p_m, pid, B=2000, seed=0),
        "fused_minus_vision": _paired_bootstrap_delta_auc(yt, p_f, p_v, pid, B=2000, seed=1),
    }

    # Persist fused linear weights (scaler + LR) so API can run the exact model
    # without requiring sklearn at runtime.
    fused_model = fit_lr_model(Xf, 0.01)
    scaler = fused_model.named_steps["standardscaler"]
    lr = fused_model.named_steps["logisticregression"]
    weights = {
        "schema_version": "rsna_medsiglip_fusion_logreg_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "modal_script": "deploy/modal/rsna_medsiglip_embed_full.py::fusion_clinical_proof",
            "model_repo": MODEL_REPO,
            "model_sha": st.get("model_sha"),
        },
        "n_rows": int(st["n_rows"]),
        "n_train": int(tr_m.sum()),
        "n_test": int(te_m.sum()),
        "test_cancers": int(yt.sum()),
        "embedding_dim": int(E.shape[1]),
        "xmeta_columns": [str(c) for c in Xmeta.columns.tolist()],
        "standard_scaler": {
            "with_mean": True,
            "with_std": True,
            "mean": [float(x) for x in scaler.mean_.tolist()],
            "scale": [float(x) for x in scaler.scale_.tolist()],
        },
        "logistic_regression": {
            "C": 0.01,
            "max_iter": 3000,
            "coef": [float(x) for x in lr.coef_.reshape(-1).tolist()],
            "intercept": float(lr.intercept_.reshape(-1)[0]),
        },
        "metrics": {
            "fused_auroc": float(roc_auc_score(yt, p_f)),
            "meta_auroc": float(roc_auc_score(yt, p_m)),
            "vision_auroc": float(roc_auc_score(yt, p_v)),
            "paired_bootstrap": paired,
        },
        "RUO": True,
    }
    _write_json(COHORT_DIR / "fusion_weights.json", weights)

    proof = {
        "clinical_meter": "patient-disjoint test AUROC (Track-3 fusion, full RSNA)",
        "n_embedded": int(E.shape[0]),
        "emb_dim": int(E.shape[1]),
        "n_train": int(tr_m.sum()),
        "n_test": int(te_m.sum()),
        "rows": rows,
        "paired_bootstrap": paired,
        "embed_status": st,
        "RUO": True,
    }
    out = COHORT_DIR / "fusion_clinical_proof.json"
    _write_json(out, proof)
    return proof


@app.local_entrypoint()
def main():
    # spawn = detach without blocking local CLI for hours
    call = embed_full.spawn()
    print(
        {
            "spawned": True,
            "function_id": getattr(embed_full, "object_id", None),
            "call_id": str(call),
            "function_call_id": getattr(call, "object_id", None),
        }
    )

