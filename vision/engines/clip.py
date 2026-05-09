"""
CLIP-based vision engine for style-invariant auto-labeling.

Compares each target's ROI crop against a reference template embedding using
cosine similarity.  Because CLIP's image embedding space is robust to visual
style variations (italic, scaled, animated, recoloured text), this engine
catches training examples that pixel-exact template matching misses.

Intended use: set ``"autolabel_engine": "CLIP"`` on targets in
``calibration/config.py`` where style variation is expected.  Runtime
detection in the RL environment should continue to use YOLO.
"""

from __future__ import annotations

import os

import numpy as np

from ..engine import Detection, VisionEngine, apply_roi
from ..registry import register

_DEFAULT_THRESHOLD = 0.75


@register("CLIP")
class CLIPVisionEngine(VisionEngine):
    """
    Style-invariant detection via CLIP ViT-B/32 image-to-image similarity.

    For each target the engine:
      1. Encodes the template PNG once at load() time.
      2. At detect() time: crops the target's ROI from the live frame,
         encodes it, and computes cosine similarity against the reference.
      3. Fires a Detection when similarity >= threshold.

    The bounding box returned covers the full ROI region in full-frame
    pixel coordinates (same convention as PixelEngine / SIFTEngine).
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._ref_embeddings: dict[str, np.ndarray] = {}
        self._target_rois: dict[str, tuple] = {}
        self._model = None
        self._preprocess = None
        self._device = None
        self._torch = None

    @property
    def name(self) -> str:
        return "CLIP"

    @property
    def needs_color(self) -> bool:
        return True

    def _init_model(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch
        self._torch = torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess

    def _encode(self, bgr: np.ndarray) -> np.ndarray:
        """Return a L2-normalised float32 embedding for a BGR image."""
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def load(self, targets: dict, assets_dir: str) -> None:
        self._init_model()
        self._ref_embeddings.clear()
        self._target_rois.clear()

        for label, cfg in targets.items():
            roi = cfg.get("roi")
            file_name = cfg.get("file")
            if roi is None or file_name is None:
                continue

            template_path = os.path.join(assets_dir, file_name)
            if not os.path.exists(template_path):
                print(f"[CLIPEngine] Warning: template not found for '{label}', skipping.")
                continue

            import cv2
            template_bgr = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template_bgr is None:
                print(f"[CLIPEngine] Warning: could not read template for '{label}', skipping.")
                continue

            self._ref_embeddings[label] = self._encode(template_bgr)
            self._target_rois[label] = roi
            print(f"[CLIPEngine] Loaded '{label}'.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        for label, ref_emb in self._ref_embeddings.items():
            roi = self._target_rois[label]
            crop, off_x, off_y = apply_roi(frame, roi)
            if crop.size == 0:
                continue

            crop_emb = self._encode(crop)
            similarity = float(np.dot(ref_emb, crop_emb))

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
