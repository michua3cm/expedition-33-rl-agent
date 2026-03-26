import mss
import cv2
import numpy as np
import time

from .controls import GameController
from calibration.config import TARGETS, MONITOR_INDEX, ASSETS_DIR
import vision


class GameInstance:
    """
    Represents the active game session.
    Acts as the bridge between the AI Agent and the Game Process.
    """

    def __init__(self, engine: str = "PIXEL", roi: dict | None = None):
        """
        Args:
            engine: Vision engine name ('PIXEL', 'SIFT', …). Case-insensitive.
            roi:    Optimised monitor region from calibration analysis.
                    If None, falls back to the default centre crop.
        """
        self.controller = GameController()
        self.sct = mss.mss()
        self.monitor = self._setup_monitor(roi)

        self.vision_engine = vision.registry.create(engine)
        self.vision_engine.load(TARGETS, ASSETS_DIR)

    def _setup_monitor(self, roi: dict | None) -> dict:
        if roi:
            print(f"[GameInstance] Using optimised ROI: {roi}")
            return roi
        raw = self.sct.monitors[MONITOR_INDEX]
        w, h = raw["width"], raw["height"]
        margin_h = int(h * 0.2)
        print("[GameInstance] Warning: No ROI provided. Using default crop.")
        return {
            "top": raw["top"] + margin_h,
            "left": raw["left"],
            "width": w,
            "height": h - (margin_h * 2),
            "mon": MONITOR_INDEX,
        }

    def capture_frame(self) -> np.ndarray:
        """Capture the current game screen as a greyscale ROI frame."""
        sct_img = self.sct.grab(self.monitor)
        return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2GRAY)

    def get_current_state(self, include_frame: bool = False) -> vision.GameState:
        """
        Run the vision engine on the current screen and return a GameState.

        Args:
            include_frame: If True, attaches the raw greyscale frame to the
                           GameState (useful for pixel-based / CNN policies).
        """
        frame = self.capture_frame()
        detections = self.vision_engine.detect(frame)
        return vision.GameState(
            detections=detections,
            timestamp=time.time(),
            engine_name=self.vision_engine.name,
            frame=frame if include_frame else None,
        )

    # --- Phase 1: Defensive action wrappers ---

    def dodge(self):          self.controller.dodge()
    def parry(self):          self.controller.parry()
    def gradient_parry(self): self.controller.gradient_parry()
    def jump(self):           self.controller.jump()
    def jump_attack(self):    self.controller.jump_attack()

    # --- Phase 1: Offensive action wrappers ---

    def attack(self):         self.controller.attack()
