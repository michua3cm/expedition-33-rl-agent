from __future__ import annotations

import os

import cv2
import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import register

_DEFAULT_THRESHOLD = 0.6


@register("PIXEL")
class PixelEngine(VisionEngine):
    """
    Template matching via TM_CCOEFF_NORMED.
    Fast and deterministic; sensitive to resolution and scale changes.
    """

    def __init__(self) -> None:
        self._templates: dict = {}

    @property
    def name(self) -> str:
        return "PIXEL"

    def load(self, targets: dict, assets_dir: str) -> None:
        self._templates.clear()
        for label, cfg in targets.items():
            path = os.path.join(assets_dir, cfg["file"])
            if not os.path.exists(path):
                print(f"[PixelEngine] Warning: '{cfg['file']}' not found, skipping '{label}'.")
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[PixelEngine] Error: failed to load '{cfg['file']}'.")
                continue
            threshold = cfg.get("threshold", _DEFAULT_THRESHOLD)
            self._templates[label] = {
                "image": img,
                "w": img.shape[1],
                "h": img.shape[0],
                "threshold": threshold,
            }
            print(f"[PixelEngine] Loaded '{label}' (threshold={threshold})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results: list[Detection] = []
        for label, data in self._templates.items():
            res = cv2.matchTemplate(frame, data["image"], cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= data["threshold"])
            for pt in zip(*loc[::-1]):
                results.append(Detection(
                    label=label,
                    x=int(pt[0]),
                    y=int(pt[1]),
                    w=data["w"],
                    h=data["h"],
                    confidence=float(res[pt[1], pt[0]]),
                ))
        return results
