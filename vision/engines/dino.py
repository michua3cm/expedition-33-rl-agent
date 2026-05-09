"""
DINOv2-based vision engine for few-shot, prototype-based detection.

Uses cosine similarity between DINOv2 ViT-S/14 embeddings to match a live ROI
crop against one or more reference template images.  Because DINOv2 encodes
local visual structure (not image-text semantics), it is robust to rotation,
scale, animation states, and visual variants of the same icon — without any
task-specific training.

Usage in calibration/config.py:
    "JUMP_CUE": {
        "files": ["jump_cue_ref_1.png", "jump_cue_ref_2.png", ...],
        "roi": (0.15, 0.15, 0.70, 0.50),
        "autolabel_engine": "DINO",
    }

Multiple reference files (``"files"`` key) let the engine cover the full range
of visual variants.  A single reference can also be supplied via ``"file"``.

Requires: torch (install via --group cuda for GPU support).
"""

from __future__ import annotations

import os

import numpy as np

from ..engine import Detection, VisionEngine, apply_roi
from ..registry import register

_DEFAULT_THRESHOLD = 0.75
_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@register("DINO")
class DINOv2VisionEngine(VisionEngine):
    """
    Few-shot detection via DINOv2 ViT-S/14 image similarity.

    For each target the engine:
      1. Encodes all reference template images at load() time.
      2. At detect() time: crops the target's ROI, encodes it, computes the
         maximum cosine similarity across all reference embeddings.
      3. Fires a Detection when that maximum exceeds the threshold.

    Unlike SIFT (single-instance matching), this approach is prototype-based:
    multiple references collectively define the visual category, so the engine
    generalises to unseen variants without retraining.
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._ref_embeddings: dict[str, list[np.ndarray]] = {}
        self._target_rois: dict[str, tuple] = {}
        self._model = None
        self._device = None
        self._torch = None

    @property
    def name(self) -> str:
        return "DINO"

    @property
    def needs_color(self) -> bool:
        return True

    def _init_model(self) -> None:
        if self._model is not None:
            return
        import torch
        self._torch = torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        self._model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
            .to(self._device)
            .eval()
        )

    def _encode(self, bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised float32 DINOv2 class-token embedding."""
        import cv2

        resized = cv2.resize(bgr, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor_np = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        tensor = self._torch.from_numpy(tensor_np).to(self._device)
        with self._torch.no_grad():
            features = self._model(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def _load_references(self, cfg: dict, assets_dir: str) -> list[np.ndarray]:
        """Load and encode all reference template images for one target."""
        import cv2

        file_names: list[str] = cfg.get("files") or (
            [cfg["file"]] if cfg.get("file") else []
        )
        embeddings: list[np.ndarray] = []
        for file_name in file_names:
            path = os.path.join(assets_dir, file_name)
            if not os.path.exists(path):
                print(f"[DINOEngine] Warning: '{path}' not found, skipping.")
                continue
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[DINOEngine] Warning: could not read '{path}', skipping.")
                continue
            embeddings.append(self._encode(bgr))
        return embeddings

    def load(self, targets: dict, assets_dir: str) -> None:
        self._init_model()
        self._ref_embeddings.clear()
        self._target_rois.clear()

        for label, cfg in targets.items():
            roi = cfg.get("roi")
            if roi is None:
                continue
            if not cfg.get("file") and not cfg.get("files"):
                continue

            embeddings = self._load_references(cfg, assets_dir)
            if not embeddings:
                continue

            self._ref_embeddings[label] = embeddings
            self._target_rois[label] = roi
            print(f"[DINOEngine] Loaded '{label}' with {len(embeddings)} reference(s).")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        for label, ref_embs in self._ref_embeddings.items():
            roi = self._target_rois[label]
            crop, off_x, off_y = apply_roi(frame, roi)
            if crop.size == 0:
                continue

            crop_emb = self._encode(crop)
            similarity = max(float(np.dot(ref, crop_emb)) for ref in ref_embs)

            if similarity >= self._threshold:
                detections.append(Detection(
                    label=label,
                    x=off_x,
                    y=off_y,
                    w=crop.shape[1],
                    h=crop.shape[0],
                    confidence=similarity,
                ))

        return detections
