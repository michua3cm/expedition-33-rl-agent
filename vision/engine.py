from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

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
    frame: Optional[np.ndarray] = field(default=None, repr=False)  # opt-in raw frame


class VisionEngine(ABC):
    """
    Abstract base for all vision engines (PIXEL, SIFT, ORB, YOLO, …).

    Contract:
      - load()   prepares the engine once at startup
      - detect() takes a greyscale ROI frame, returns detections
      - Engines never touch the overlay, logger, or offset — callers handle that
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and the registry (e.g. 'PIXEL')."""

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
        Run detection on a single greyscale frame cropped to the active ROI.

        Args:
            frame: Greyscale numpy array (dtype uint8).

        Returns:
            List of Detection objects; empty list if nothing found.
        """
