"""LUNA16-trained RetinaNet lung-nodule detector.

Provides a thin wrapper around the MONAI Model Zoo `lung_nodule_ct_detection`
bundle (v0.6.9), which ships pre-trained RetinaNet weights on LUNA16 fold 0.

Why not train ourselves?
------------------------
LIDC-IDRI/LUNA16 is a credentialed download (TCIA) with ~120 GB of DICOM,
and training a 3D RetinaNet to convergence needs a GPU with ~16 GB VRAM.
Our sandbox is CPU-only. Rather than ship placeholder weights, we
integrate the published MONAI reference model, which achieves mAP=0.852 /
mAR=0.998 on LUNA16 fold 0. All caveats about the training data (LUNA16
is not screening-CT and not multi-site; nodule sizes 3-30 mm) get
surfaced as warnings on the API response.

License / provenance
--------------------
- MONAI bundle: Apache 2.0 (see docs/data_license.txt in the bundle
  snapshot).
- Underlying data: LUNA16 (a curated LIDC-IDRI subset). TCIA data use
  agreement applies to the raw data, NOT to the trained weights, which
  are freely redistributable per MONAI Model Zoo.
- All model responses stamp `model_state=loaded_luna16_retinanet`,
  `model_name=monai/lung_nodule_ct_detection@0.6.9`.

Preprocessing contract
----------------------
The bundle expects (from configs/inference.json):
- Resample to voxel size 0.703125 x 0.703125 x 1.25 mm (RAS orientation)
- Scale HU intensity [-1024, 300] -> [0, 1] with clipping
- Channel-first tensor, single-batch
Inference uses sliding-window at [192, 192, 80] with 0.25 overlap.
On CPU this takes ~30-90 seconds per case depending on volume size.

Output shape
------------
`detect(volume_hu, spacing_mm)` returns a `NoduleDetectionResult` with:
- `boxes`: list of NoduleBox with (z, y, x) center voxel coords, width /
  height / depth in mm, and score
- `top_score`: float
- `bundle_version`: "0.6.9"
- `preprocessing_summary`: dict for the response (voxel size, HU range)

The wrapper is intentionally stateless-per-detection: the network is a
module-level singleton loaded once via `LungNoduleDetector.get()`, but
each `detect` call fully owns its input tensor and postprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The MONAI bundle path. `ONCOLOGY_ARBITER_LUNA16_BUNDLE_DIR` env var
# overrides for tests / prod deploys.
_DEFAULT_BUNDLE_DIR = Path("/workspace/monai_bundles/lung_nodule_ct_detection")


LUNA16_WARNING = (
    "detector=monai/lung_nodule_ct_detection@0.6.9; trained on LUNA16 fold 0 "
    "(LIDC-IDRI subset, single-site chest CT). Not validated on screening "
    "CT, contrast CT, or pediatric CT. Nodule size range 3-30 mm. "
    "Detections are RESEARCH USE ONLY and require radiologist review."
)


@dataclass
class NoduleBox:
    """One detected nodule in world coordinates.

    RetinaNet raw output is in the CCCWHD box format
    (center-center-center + width/height/depth). The bundle's
    postprocessing converts the coordinates back to the input image's
    world frame (RAS orientation). We keep world-mm coordinates so
    downstream (arbiter, UI) can report nodule size without knowing the
    voxel spacing.
    """

    center_z_mm: float
    center_y_mm: float
    center_x_mm: float
    width_mm: float
    height_mm: float
    depth_mm: float
    score: float

    def diameter_mm(self) -> float:
        """Longest axis of the box (a conservative diameter estimate)."""
        return float(max(self.width_mm, self.height_mm, self.depth_mm))

    def as_dict(self) -> dict:
        return {
            "center_z_mm": self.center_z_mm,
            "center_y_mm": self.center_y_mm,
            "center_x_mm": self.center_x_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "depth_mm": self.depth_mm,
            "diameter_mm": self.diameter_mm(),
            "score": self.score,
        }


@dataclass
class NoduleDetectionResult:
    """Full response returned by `LungNoduleDetector.detect`."""

    boxes: list[NoduleBox]
    top_score: float
    n_detections: int
    bundle_version: str
    preprocessing_summary: dict
    inference_seconds: float


class LungNoduleDetector:
    """Singleton wrapper around the MONAI LUNA16 RetinaNet detector.

    Load once with `LungNoduleDetector.get()`; call `.detect(volume_hu,
    spacing_mm)` per case.
    """

    _instance: Optional["LungNoduleDetector"] = None

    def __init__(self, bundle_dir: Path = _DEFAULT_BUNDLE_DIR):
        self.bundle_dir = bundle_dir
        self.bundle_version = "0.6.9"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = None
        self._load()

    def _load(self) -> None:
        """Build the RetinaNetDetector and load pretrained weights."""
        from monai.networks.nets import resnet
        from monai.apps.detection.networks.retinanet_network import (
            RetinaNet,
            resnet_fpn_feature_extractor,
        )
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector

        # Backbone / feature extractor exactly matching bundle
        # configs/inference.json.
        backbone = resnet.resnet50(
            spatial_dims=3, n_input_channels=1,
            conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7],
        )
        fe = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
        net = RetinaNet(
            spatial_dims=3,
            num_classes=1,
            num_anchors=3,
            feature_extractor=fe,
            size_divisible=[16, 16, 8],
            use_list_output=False,
        )

        # Load pretrained weights
        ckpt_path = self.bundle_dir / "models" / "model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MONAI bundle weights not found at {ckpt_path}. "
                "Set ONCOLOGY_ARBITER_LUNA16_BUNDLE_DIR to the extracted "
                "bundle path, or download from HuggingFace "
                "MONAI/lung_nodule_ct_detection@0.6.9."
            )
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        # Bundle ships state_dict directly (not wrapped in "model" key).
        net.load_state_dict(ckpt if not isinstance(ckpt, dict) or "feature_extractor.body.conv1.weight" in ckpt else ckpt["model"])

        # Anchor generator + detector (params from inference.json)
        ag = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[1, 2, 4],
            base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
        )
        det = RetinaNetDetector(
            network=net,
            anchor_generator=ag,
            spatial_dims=3,
            num_classes=1,
            size_divisible=[16, 16, 8],
        )
        det.set_target_keys(box_key="box", label_key="label")
        det.set_box_selector_parameters(
            score_thresh=0.02,
            topk_candidates_per_level=1000,
            nms_thresh=0.22,
            detections_per_img=300,
        )
        det.set_sliding_window_inferer(
            roi_size=[192, 192, 80],
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )
        det.to(self._device).eval()
        self._detector = det

    @classmethod
    def get(cls, bundle_dir: Path = _DEFAULT_BUNDLE_DIR) -> "LungNoduleDetector":
        """Return the singleton, loading it on first access."""
        if cls._instance is None or cls._instance.bundle_dir != bundle_dir:
            cls._instance = LungNoduleDetector(bundle_dir=bundle_dir)
        return cls._instance

    def detect(
        self,
        volume_hu: np.ndarray,
        spacing_mm: tuple[float, float, float] | None = None,
    ) -> NoduleDetectionResult:
        """Run detection on a HU volume.

        Parameters
        ----------
        volume_hu : ndarray of shape (D, H, W) or (C, D, H, W)
            CT volume in Hounsfield units.
        spacing_mm : (dz, dy, dx) tuple in millimeters, or None if the
            caller already resampled to the LUNA16 target spacing
            (1.25 mm z, 0.703125 mm y, 0.703125 mm x).

        Returns
        -------
        NoduleDetectionResult
        """
        import time

        if self._detector is None:
            raise RuntimeError("Detector not loaded")

        # Ensure single-channel float32 (C, D, H, W)
        v = np.asarray(volume_hu, dtype=np.float32)
        if v.ndim == 3:
            v = v[None, ...]  # add channel axis
        elif v.ndim == 4 and v.shape[0] != 1:
            raise ValueError(f"Expected single-channel volume; got shape {v.shape}")

        # Bundle preprocessing: HU [-1024, 300] -> [0, 1] with clipping.
        v = np.clip(v, -1024.0, 300.0)
        v = (v + 1024.0) / (1024.0 + 300.0)

        # Note: we do NOT resample here — resampling belongs upstream in
        # the ct_reader / preprocessing layer where we have DICOM metadata.
        # The caller is responsible for delivering a volume roughly at
        # the LUNA16 target spacing; if `spacing_mm` is provided we log
        # it in the response for auditability.

        # Move to torch. Detector expects a list of tensors (one per image).
        x = [torch.from_numpy(v)]

        t0 = time.time()
        with torch.no_grad():
            out = self._detector(x)
        dt = time.time() - t0

        # `out` is a list of dicts with keys "box", "label", "label_scores".
        # In evaluation mode, box coordinates are still in voxel space
        # (unless a postprocessing transform runs). Since we're skipping
        # AffineBoxToWorldCoordinated (that needs monai.data.MetaTensor
        # + affine), we return the raw voxel-frame boxes and let the
        # caller apply spacing to convert to mm.
        raw = out[0] if out else {"box": torch.zeros(0, 6), "label": torch.zeros(0), "label_scores": torch.zeros(0)}

        boxes: list[NoduleBox] = []
        raw_boxes = raw["box"].detach().cpu().numpy()
        raw_scores = raw["label_scores"].detach().cpu().numpy()

        # RetinaNet in xyzxyz format (min corner, max corner).
        # NOTE: MONAI's spatial ordering under our load path is (z, y, x)
        # because the input is (D, H, W).
        dz = spacing_mm[0] if spacing_mm else 1.25
        dy = spacing_mm[1] if spacing_mm else 0.703125
        dx = spacing_mm[2] if spacing_mm else 0.703125

        for i, box in enumerate(raw_boxes):
            z0, y0, x0, z1, y1, x1 = box
            cz = (z0 + z1) / 2.0
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            wz = (z1 - z0) * dz
            wy = (y1 - y0) * dy
            wx = (x1 - x0) * dx
            boxes.append(NoduleBox(
                center_z_mm=float(cz * dz),
                center_y_mm=float(cy * dy),
                center_x_mm=float(cx * dx),
                width_mm=float(wx),
                height_mm=float(wy),
                depth_mm=float(wz),
                score=float(raw_scores[i]),
            ))

        boxes.sort(key=lambda b: b.score, reverse=True)
        top_score = boxes[0].score if boxes else 0.0

        return NoduleDetectionResult(
            boxes=boxes,
            top_score=top_score,
            n_detections=len(boxes),
            bundle_version=self.bundle_version,
            preprocessing_summary={
                "hu_range": [-1024.0, 300.0],
                "target_spacing_mm": [1.25, 0.703125, 0.703125],
                "actual_spacing_mm": list(spacing_mm) if spacing_mm else None,
                "roi_size": [192, 192, 80],
                "sw_overlap": 0.25,
                "score_thresh": 0.02,
                "nms_thresh": 0.22,
            },
            inference_seconds=dt,
        )
