from __future__ import annotations

import cv2
import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import create, register

_DEFAULT_ENGINE = "PIXEL"


@register("COMPOSITE")
class CompositeEngine(VisionEngine):
    """
    Meta-engine that routes each target to its designated sub-engine.

    Each target config may include an optional ``engine`` key specifying
    which engine handles that target.  Targets without an ``engine`` key
    default to PIXEL.  This allows mixing fast template matching for
    simple icons with SIFT/ORB for scale-invariant targets, or YOLO for
    production inference — all in a single detect() pass.

    Example target config::

        "PERFECT": {"file": "template_perfect.png", "threshold": 0.65},
        "JUMP_CUE": {"file": "template_jump.png", "engine": "SIFT"},

    Frame handling:
        needs_color returns True when any sub-engine needs a BGR frame.
        detect() lazily converts BGR→GREY for sub-engines that do not need
        colour, so each sub-engine always receives the format it expects
        without redundant conversions or extra captures at the caller level.
    """

    def __init__(self) -> None:
        self._sub_engines: dict[str, VisionEngine] = {}  # engine_name → instance

    @property
    def name(self) -> str:
        return "COMPOSITE"

    @property
    def needs_color(self) -> bool:
        return any(e.needs_color for e in self._sub_engines.values())

    def load(self, targets: dict, assets_dir: str) -> None:
        self._sub_engines.clear()

        # Group targets by their designated engine.
        engine_targets: dict[str, dict] = {}
        for label, cfg in targets.items():
            engine_name = str(cfg.get("engine", _DEFAULT_ENGINE)).upper()
            if engine_name not in engine_targets:
                engine_targets[engine_name] = {}
            engine_targets[engine_name][label] = cfg

        # Instantiate each required sub-engine and load only its targets.
        for engine_name, sub_targets in engine_targets.items():
            sub_engine = create(engine_name)
            sub_engine.load(sub_targets, assets_dir)
            self._sub_engines[engine_name] = sub_engine

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run each sub-engine on the correct frame format and merge results.

        When needs_color is True the caller passes a BGR frame.  Sub-engines
        that do not need colour receive a lazily-derived greyscale conversion
        to avoid redundant cv2.cvtColor calls within a single detect() step.
        When needs_color is False the caller already passed a greyscale frame,
        so it is forwarded directly without any conversion.
        """
        grey: np.ndarray | None = None

        def _grey() -> np.ndarray:
            nonlocal grey
            if grey is None:
                # Only called when self.needs_color is True, so frame is BGR.
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return grey

        results: list[Detection] = []
        for engine in self._sub_engines.values():
            engine_frame = frame if engine.needs_color else (_grey() if self.needs_color else frame)
            results.extend(engine.detect(engine_frame))

        return results
