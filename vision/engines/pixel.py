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

    Targets with ``color_mode: True`` in config are loaded and matched in BGR.
    All other targets use greyscale.  When at least one color template is
    loaded, needs_color returns True and callers must pass a BGR frame to
    detect(); the engine converts to grey internally for non-color templates.
    """

    def __init__(self) -> None:
        self._templates: dict = {}

    @property
    def name(self) -> str:
        return "PIXEL"

    @property
    def needs_color(self) -> bool:
        return any(t["color_mode"] for t in self._templates.values())

    def load(self, targets: dict, assets_dir: str) -> None:
        self._templates.clear()
        for label, cfg in targets.items():
            path = os.path.join(assets_dir, cfg["file"])
            if not os.path.exists(path):
                print(f"[PixelEngine] Warning: '{cfg['file']}' not found, skipping '{label}'.")
                continue
            color_mode = bool(cfg.get("color_mode", False))
            flags = cv2.IMREAD_COLOR if color_mode else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(path, flags)
            if img is None:
                print(f"[PixelEngine] Error: failed to load '{cfg['file']}'.")
                continue
            threshold = cfg.get("threshold", _DEFAULT_THRESHOLD)
            self._templates[label] = {
                "image": img,
                "w": img.shape[1],
                "h": img.shape[0],
                "threshold": threshold,
                "color_mode": color_mode,
            }
            mode_tag = "BGR" if color_mode else "grey"
            print(f"[PixelEngine] Loaded '{label}' ({mode_tag}, threshold={threshold})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Pre-compute grey once if the frame is BGR (needs_color path) so that
        # greyscale templates don't pay the conversion cost on every match.
        grey_frame: np.ndarray | None = None
        if self.needs_color:
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results: list[Detection] = []
        for label, data in self._templates.items():
            src = frame if data["color_mode"] else (grey_frame if grey_frame is not None else frame)
            res = cv2.matchTemplate(src, data["image"], cv2.TM_CCOEFF_NORMED)
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
