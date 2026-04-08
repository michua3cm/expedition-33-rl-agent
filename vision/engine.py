from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str
    x: int          # top-left x, relative to the captured ROI
    y: int          # top-left y, relative to the captured ROI
    w: int
    h: int
    confidence: float   # always 0.0 – 1.0, normalised per engine


@dataclass
class GameState:
    """Complete observation snapshot produced by the vision pipeline."""
    detections: list[Detection]
    timestamp: float
    engine_name: str
    frame: np.ndarray | None = field(default=None, repr=False)  # opt-in raw frame


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-Union for two axis-aligned bounding boxes."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.3) -> list[Detection]:
    """
    Greedy Non-Maximum Suppression.

    Detections of *different* labels never suppress each other.  Within each
    label group, boxes are sorted by confidence (descending) and any box that
    overlaps an already-kept box by more than ``iou_threshold`` is discarded.

    Args:
        detections:    Raw detection list, possibly with many overlapping boxes.
        iou_threshold: Overlap fraction above which a lower-confidence box is
                       suppressed.  0.3 works well for rigid template matches.

    Returns:
        Filtered list with at most one box per cluster per label.
    """
    if len(detections) <= 1:
        return detections

    groups: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        groups[d.label].append(d)

    kept: list[Detection] = []
    for group in groups.values():
        remaining = sorted(group, key=lambda d: d.confidence, reverse=True)
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [d for d in remaining if _iou(best, d) < iou_threshold]
    return kept


# HSV hue ranges (OpenCV scale: 0–180) shared across engines for color
# masking and validation.  Red wraps around 0, so it has two ranges.
# Gold/amber (JUMP_CUE icon, TURN_ALLY border): standard ~27–62°, OpenCV 13–31.
HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red":    [(0, 10), (170, 180)],
    "blue":   [(100, 130)],
    "green":  [(40, 80)],
    "yellow": [(20, 35)],
    "gold":   [(15, 35)],
    "purple": [(130, 160)],
}


def apply_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, int, int]:
    """Crop *frame* to a fractional ROI region.

    Args:
        frame: Input frame — greyscale (H×W) or BGR (H×W×3).
        roi:   ``(x_frac, y_frac, w_frac, h_frac)`` as fractions of the frame
               dimensions.  ``None`` skips cropping and returns the frame as-is.

    Returns:
        ``(cropped_frame, offset_x, offset_y)`` — the sub-frame and the pixel
        offset that must be added back to any Detection coordinates produced
        from the cropped frame so they are relative to the original full frame.
    """
    if roi is None:
        return frame, 0, 0
    h, w = frame.shape[:2]
    x = int(roi[0] * w)
    y = int(roi[1] * h)
    rw = max(1, int(roi[2] * w))
    rh = max(1, int(roi[3] * h))
    return frame[y:y + rh, x:x + rw], x, y


class VisionEngine(ABC):
    """
    Abstract base for all vision engines (PIXEL, SIFT, ORB, YOLO, …).

    Contract:
      - load()   prepares the engine once at startup
      - detect() takes a ROI frame (greyscale by default, BGR if needs_color)
      - Engines never touch the overlay, logger, or offset — callers handle that
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and the registry (e.g. 'PIXEL')."""

    @property
    def needs_color(self) -> bool:
        """
        Return True if this engine requires a BGR frame instead of greyscale.

        Callers (GameInstance, StateBuffer) check this after load() and pass
        the appropriate frame format to detect().  Default: False.
        """
        return False

    @abstractmethod
    def load(self, targets: dict, assets_dir: str) -> None:
        """
        Prepare the engine: load templates, model weights, etc.

        Args:
            targets:    Target config dict (same shape as TARGETS in config.py).
            assets_dir: Path to the directory that contains template/asset files.
        """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single ROI frame.

        Args:
            frame: Greyscale uint8 array by default.  BGR uint8 array when the
                   engine's needs_color property returns True.

        Returns:
            List of Detection objects; empty list if nothing found.
        """
