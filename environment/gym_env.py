"""
Gymnasium environment wrapper for Clair Obscur: Expedition 33.

Observation space  — Box(20,), float32, all values in [0, 1]:
  Indices 0–9 : detection confidence for each vision target
                (order matches OBSERVATION_TARGETS below)
  Indices 10–19: normalised bounding-box centre (x, y) for each target
                 packed as [x0, y0, x1, y1, …, x9, y9]
                 0.0 when the target is not detected this frame

Action space — Discrete(7):
  See environment/actions.py for the index → action mapping.

Reward function (Phase 1):
  +10  PERFECT detected   (tight dodge timing, ≈ parry window)
  + 5  DODGE detected     (successful dodge)
  + 8  PARRIED detected   (gradient parry successful)
  + 5  JUMP detected      (successful jump)
  - 1  every step         (time penalty to discourage passivity)
  All reward signals are consumed once detected (cleared after one step).
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import actions as A
from .instance import GameInstance

# Vision targets in a fixed, stable order for the observation vector.
OBSERVATION_TARGETS = [
    "PERFECT",
    "DODGE",
    "JUMP",
    "PARRIED",
    "JUMP_CUE",
    "MOUSE",
    "BATTLE_WHEEL",
    "TURN_ALLY",
    "TURN_ENEMY",
    "GRADIENT_INCOMING",
]

# Reward granted the first step a signal is detected.
REWARD_MAP = {
    "PERFECT": 10.0,
    "DODGE":    5.0,
    "PARRIED":  8.0,
    "JUMP":     5.0,
}

STEP_PENALTY = -1.0  # applied every step to discourage inaction

OBS_DIM = len(OBSERVATION_TARGETS) * 3  # confidence + x_centre + y_centre = 30


class Expedition33Env(gym.Env):
    """
    Single-player, Phase 1 Gymnasium environment for Expedition 33.

    The environment captures the live game screen, runs the vision engine,
    builds a flat observation vector, executes the chosen action via the
    game controller, and returns the transition tuple.

    Args:
        engine:        Vision engine name ('PIXEL', 'SIFT', 'ORB', …).
        roi:           Optimised monitor region from calibration. None = default crop.
        step_delay:    Seconds to wait after executing an action before
                       capturing the next frame (allows animations to appear).
        include_frame: If True, the `info` dict includes the raw greyscale frame.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        engine: str = "PIXEL",
        roi: dict | None = None,
        step_delay: float = 0.15,
        include_frame: bool = False,
    ):
        super().__init__()

        self.game = GameInstance(engine=engine, roi=roi)
        self.step_delay = step_delay
        self.include_frame = include_frame

        # --- Spaces ---
        # OBS_DIM = 30: [conf×10, x_centre×10, y_centre×10], all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(A.NUM_ACTIONS)

        # Track which reward signals have already been collected this episode
        # so that lingering on-screen text does not grant repeated rewards.
        self._seen_signals: set[str] = set()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._seen_signals.clear()
        state = self.game.get_current_state(include_frame=self.include_frame)
        obs = self._build_obs(state)
        info = self._build_info(state)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._execute_action(action)
        time.sleep(self.step_delay)

        state = self.game.get_current_state(include_frame=self.include_frame)
        obs = self._build_obs(state)
        reward = self._compute_reward(state)
        info = self._build_info(state)

        # Phase 1: no terminal condition — episode runs until manually stopped.
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """No-op: the game window is the live render."""

    def close(self) -> None:
        """Release screen-capture resources."""
        self.game.sct.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_action(self, action: int) -> None:
        """Dispatch an integer action to the appropriate GameInstance method."""
        if action == A.NOOP:
            pass
        elif action == A.PARRY:
            self.game.parry()
        elif action == A.DODGE:
            self.game.dodge()
        elif action == A.JUMP:
            self.game.jump()
        elif action == A.GRADIENT_PARRY:
            self.game.gradient_parry()
        elif action == A.ATTACK:
            self.game.attack()
        elif action == A.JUMP_ATTACK:
            self.game.jump_attack()
        else:
            raise ValueError(f"Unknown action index: {action}")

    def _build_obs(self, state) -> np.ndarray:
        """
        Flatten detections into a fixed-length float32 vector.

        Layout: [conf_0..conf_9, x_0..x_9, y_0..y_9]
        All values normalised to [0, 1]. Missing detections → 0.0.
        """
        # Index detections by label for O(1) lookup
        det_map: dict[str, Any] = {}
        for d in state.detections:
            # Keep the highest-confidence detection if duplicates exist
            if d.label not in det_map or d.confidence > det_map[d.label].confidence:
                det_map[d.label] = d

        # ROI dimensions for normalisation (fall back to 1 to avoid div-by-zero)
        roi_w = max(self.game.monitor.get("width", 1), 1)
        roi_h = max(self.game.monitor.get("height", 1), 1)

        confidences = np.zeros(len(OBSERVATION_TARGETS), dtype=np.float32)
        x_centres   = np.zeros(len(OBSERVATION_TARGETS), dtype=np.float32)
        y_centres   = np.zeros(len(OBSERVATION_TARGETS), dtype=np.float32)

        for i, label in enumerate(OBSERVATION_TARGETS):
            if label in det_map:
                d = det_map[label]
                confidences[i] = float(d.confidence)
                x_centres[i]   = float(d.x + d.w / 2) / roi_w
                y_centres[i]   = float(d.y + d.h / 2) / roi_h

        return np.concatenate([confidences, x_centres, y_centres])

    def _compute_reward(self, state) -> float:
        """
        Return the total reward for this step.

        Reward signals are only granted once per appearance — once a label
        enters _seen_signals it is suppressed until the signal disappears
        from screen (detection drops to 0) and re-appears.
        """
        detected_labels = {d.label for d in state.detections}
        reward = STEP_PENALTY

        for label, value in REWARD_MAP.items():
            if label in detected_labels:
                if label not in self._seen_signals:
                    reward += value
                    self._seen_signals.add(label)
            else:
                # Signal left the screen — allow it to trigger again
                self._seen_signals.discard(label)

        return reward

    def _build_info(self, state) -> dict:
        info: dict = {
            "timestamp":   state.timestamp,
            "engine":      state.engine_name,
            "detections":  [d.label for d in state.detections],
        }
        if self.include_frame and state.frame is not None:
            info["frame"] = state.frame
        return info
