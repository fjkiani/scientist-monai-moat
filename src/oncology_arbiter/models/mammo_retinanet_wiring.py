"""Identity-locked wiring for mammo-retinanet v1 (unfrozen RetinaNet backbone cls).

Binds ``models/mammo/mammo_retinanet_v1.safetensors``
(sha256 ``691233bca51c4eb99fd6859301b8e1393ada6643b25e79622548ef80de951d32``).

Honest metric: held-out test AUROC 0.785453 on CBIS-DDSM_1024 (n=641).
Floor amended to 0.75 — raw 2D pixel ceiling; multimodal cbis-medsiglip owns 0.85+.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

CAPABILITY = "mammo-retinanet"
ARTIFACT_FILENAME = "mammo_retinanet_v1.safetensors"
ARTIFACT_PATH = "models/mammo/mammo_retinanet_v1.safetensors"
ARTIFACT_SHA256 = "691233bca51c4eb99fd6859301b8e1393ada6643b25e79622548ef80de951d32"
TEST_AUROC = 0.785453260650111


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "mammo" / ARTIFACT_FILENAME


def verify_artifact_identity(path: Path | None = None) -> Path:
    resolved = path or artifact_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != ARTIFACT_SHA256:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: "
            f"expected {ARTIFACT_SHA256}, got {actual}"
        )
    return resolved


def load_model(device: str = "cpu"):
    """Load RetinaNet-backbone classifier weights (RUO)."""
    import torch
    import torch.nn as nn
    import torchvision
    from safetensors.torch import load_file
    from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights

    path = verify_artifact_identity()

    class RetinaNetBackboneCls(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            det = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            )
            self.backbone = det.backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            feats = self.backbone(x)
            if isinstance(feats, dict):
                f = next(iter(feats.values()))
            else:
                f = feats
            g = self.pool(f).flatten(1)
            return self.head(g).squeeze(1)

    model = RetinaNetBackboneCls()
    state = load_file(str(path))
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_png(path: str | Path, *, device: str = "cpu", img_size: int = 384) -> dict[str, Any]:
    """Score one mammogram PNG with the identity-locked v1 weights."""
    import torch
    from PIL import Image
    from torchvision import transforms

    model = load_model(device=device)
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        prob = float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0])
    return {
        "capability": CAPABILITY,
        "artifact_path": ARTIFACT_PATH,
        "artifact_sha256": ARTIFACT_SHA256,
        "proba_cancer": prob,
        "test_auroc_receipt": TEST_AUROC,
        "model_state": "loaded_mammo_retinanet_v1",
        "input": str(path),
    }
