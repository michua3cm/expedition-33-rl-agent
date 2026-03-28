from __future__ import annotations

import os

import cv2
import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import register

_DEFAULT_THRESHOLD = 0.6

# Internal kind tags — used only inside this module.
_KIND_GREY    = "grey"     # greyscale template matching (default)
_KIND_COLOR   = "color"    # BGR template matching (color_mode: True)
_KIND_HSV_SAT = "hsv_sat"  # frame-wide saturation drop (no template file)


@register("PIXEL")
class PixelEngine(VisionEngine):
    """
    Fast, training-free detection engine.  Three detection modes in one pass:

    grey     — greyscale template matching via TM_CCOEFF_NORMED (default).
    color    — BGR template matching for targets whose key feature is colour
               (set ``color_mode: True`` in the target config).
    hsv_sat  — frame-wide HSV saturation check for screen-overlay effects
               that have no crisp icon (set ``hsv_sat_max`` in target config,
               omit ``file``).  Fires when mean S-channel < hsv_sat_max.

    Callers must pass a BGR frame when needs_color is True; the engine
    derives grey and HSV internally as needed.
    """

    def __init__(self) -> None:
        self._targets: dict = {}

    @property
    def name(self) -> str:
        return "PIXEL"

    @property
    def needs_color(self) -> bool:
        return any(t["kind"] in (_KIND_COLOR, _KIND_HSV_SAT) for t in self._targets.values())

    def load(self, targets: dict, assets_dir: str) -> None:
        self._targets.clear()
        for label, cfg in targets.items():
            file_name = cfg.get("file")

            # --- hsv_sat target (no template file) ---
            if file_name is None:
                hsv_sat_max = cfg.get("hsv_sat_max")
                if hsv_sat_max is None:
                    print(f"[PixelEngine] Warning: '{label}' has no file and no hsv_sat_max, skipping.")
                    continue
                self._targets[label] = {
                    "kind": _KIND_HSV_SAT,
                    "hsv_sat_max": float(hsv_sat_max),
                }
                print(f"[PixelEngine] Loaded '{label}' (hsv_sat, max_sat={hsv_sat_max})")
                continue

            # --- template-based targets ---
            path = os.path.join(assets_dir, file_name)
            if not os.path.exists(path):
                print(f"[PixelEngine] Warning: '{file_name}' not found, skipping '{label}'.")
                continue
            color_mode = bool(cfg.get("color_mode", False))
            flags = cv2.IMREAD_COLOR if color_mode else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(path, flags)
            if img is None:
                print(f"[PixelEngine] Error: failed to load '{file_name}'.")
                continue
            kind = _KIND_COLOR if color_mode else _KIND_GREY
            threshold = cfg.get("threshold", _DEFAULT_THRESHOLD)
            self._targets[label] = {
                "kind": kind,
                "image": img,
                "w": img.shape[1],
                "h": img.shape[0],
                "threshold": threshold,
            }
            print(f"[PixelEngine] Loaded '{label}' ({kind}, threshold={threshold})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Derive auxiliary frames lazily — only when a target needs them.
        grey_frame: np.ndarray | None = None
        hsv_frame:  np.ndarray | None = None

        def _grey() -> np.ndarray:
            nonlocal grey_frame
            if grey_frame is None:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.needs_color else frame
            return grey_frame

        def _hsv() -> np.ndarray:
            nonlocal hsv_frame
            if hsv_frame is None:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv_frame

        results: list[Detection] = []

        for label, data in self._targets.items():
            kind = data["kind"]

            # --- HSV saturation threshold ---
            if kind == _KIND_HSV_SAT:
                mean_sat = float(np.mean(_hsv()[:, :, 1]))
                sat_max = data["hsv_sat_max"]
                if mean_sat <= sat_max:
                    h, w = frame.shape[:2]
                    # Confidence: 1.0 when fully grey, 0.0 at the threshold boundary.
                    conf = 1.0 - (mean_sat / sat_max)
                    results.append(Detection(
                        label=label, x=0, y=0, w=w, h=h,
                        confidence=min(conf, 1.0),
                    ))
                continue

            # --- Template matching (grey or BGR) ---
            src = frame if kind == _KIND_COLOR else _grey()
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
