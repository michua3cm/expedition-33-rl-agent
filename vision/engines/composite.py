from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import create, register


@register("COMPOSITE")
class CompositeEngine(VisionEngine):
    """
    Meta-engine that routes each target to its own sub-engine.

    Each target in the config may specify an ``engine`` field (e.g.
    ``"engine": "SIFT"``).  Targets without the field default to PIXEL.
    CompositeEngine creates one sub-engine per unique engine name, loads
    each with only its assigned targets, then merges results on every
    detect() call.

    This lets targets with different matching needs (template pixel
    matching, colour detection, scale-invariant keypoint matching) coexist
    in a single detection pass without any code changes in callers.

    Frame handling:
        needs_color returns True when any sub-engine needs a BGR frame.
        detect() downgrades BGR→GREY for sub-engines that don't need colour,
        so each sub-engine always receives the format it expects.
    """

    def __init__(self) -> None:
        self._sub_engines: list[VisionEngine] = []

    @property
    def name(self) -> str:
        return "COMPOSITE"

    @property
    def needs_color(self) -> bool:
        return any(eng.needs_color for eng in self._sub_engines)

    def load(self, targets: dict, assets_dir: str) -> None:
        groups: dict[str, dict] = defaultdict(dict)
        for label, cfg in targets.items():
            engine_name = cfg.get("engine", "PIXEL").upper()
            groups[engine_name][label] = cfg

        self._sub_engines = []
        for engine_name, group_targets in groups.items():
            eng = create(engine_name)
            eng.load(group_targets, assets_dir)
            self._sub_engines.append(eng)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        is_bgr = frame.ndim == 3
        grey_frame: np.ndarray | None = None

        def _grey() -> np.ndarray:
            nonlocal grey_frame
            if grey_frame is None:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if is_bgr else frame
            return grey_frame

        results: list[Detection] = []
        for eng in self._sub_engines:
            eng_frame = frame if eng.needs_color else _grey()
            results.extend(eng.detect(eng_frame))
        return results
