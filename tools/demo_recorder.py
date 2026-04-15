"""
Human Demonstration Recorder for Clair Obscur: Expedition 33.

Records human gameplay as (observation, action, timestamp) trajectories
that can later be used for imitation learning or reward shaping.

How it works
------------
1. A background capture thread samples the game screen at a fixed rate
   (default 20 Hz) and runs the vision engine to produce an observation
   vector matching the Gym environment's observation space.
2. A pynput keyboard + mouse listener runs concurrently, mapping key
   presses to action indices (same mapping as the Gym environment).
3. Each captured frame is paired with the most recent action pressed
   since the last capture tick (NOOP if no key was pressed).
4. On stop(), the trajectory is saved as a compressed .npz file.

Key → Action mapping (Phase 1)
-------------------------------
  E         → PARRY          (1)
  Q         → DODGE          (2)
  SPACE     → JUMP           (3)
  W         → GRADIENT_PARRY (4)
  F         → ATTACK         (5)
  Left click → JUMP_ATTACK   (6)
  (nothing)  → NOOP          (0)

Output format  (saved as <DEMO_DIR>/<session_name>.npz)
-------------------------------------------------------
  observations : float32 (N, 30)  — same layout as Expedition33Env obs
  actions      : int32   (N,)     — action index per step
  timestamps   : float64 (N,)     — Unix timestamp of each capture

Usage
-----
    from tools.demo_recorder import DemoRecorder
    from environment.instance import GameInstance

    game = GameInstance(engine="PIXEL")
    rec  = DemoRecorder(game, session_name="demo_01", poll_hz=20)
    rec.start()
    # ... play the game ...
    rec.stop()   # saves the .npz automatically

CLI
---
    python -m tools.demo_recorder --session demo_01 --engine PIXEL --hz 20
"""

import argparse
import os
import threading
import time

import numpy as np
from pynput import keyboard, mouse

from calibration.config import DEMO_DIR
from environment.actions import (
    ATTACK,
    DODGE,
    GRADIENT_PARRY,
    JUMP,
    JUMP_ATTACK,
    NOOP,
    PARRY,
)
from environment.instance import GameInstance

# Vision targets — must match the gym env's OBSERVATION_TARGETS order
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

_N = len(OBSERVATION_TARGETS)  # obs vector length per slot (conf + x + y)

# Keyboard character → action index
_KEY_MAP: dict[str, int] = {
    "e": PARRY,
    "q": DODGE,
    "w": GRADIENT_PARRY,
    "f": ATTACK,
}


class DemoRecorder:
    """
    Records human gameplay demonstrations from the live game window.

    Args:
        game:         Initialised GameInstance (screen capture + vision).
        session_name: Filename stem for the saved .npz (no extension).
        poll_hz:      Capture rate in Hz (frames per second recorded).
        save_dir:     Directory to save .npz files (default: DEMO_DIR).
    """

    def __init__(
        self,
        game: GameInstance,
        session_name: str = "demo",
        poll_hz: float = 20.0,
        save_dir: str = DEMO_DIR,
    ):
        self._game = game
        self._session_name = session_name
        self._interval = 1.0 / max(poll_hz, 1.0)
        self._save_dir = save_dir

        # Buffers — appended by the capture thread
        self._obs_buf:  list[np.ndarray] = []
        self._act_buf:  list[int]        = []
        self._ts_buf:   list[float]      = []

        # Most recent action pressed (consumed once per tick → NOOP)
        self._pending_action: int = NOOP
        self._action_lock = threading.Lock()

        self._stop_event = threading.Event()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="DemoRecorder-capture",
            daemon=True,
        )
        self._kb_listener  = keyboard.Listener(on_press=self._on_key_press)
        self._mouse_listener = mouse.Listener(on_click=self._on_click)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capture and input listeners."""
        print(f"[DemoRecorder] Recording '{self._session_name}' at "
              f"{1/self._interval:.0f} Hz. Press Ctrl+C to stop.")
        self._stop_event.clear()
        self._kb_listener.start()
        self._mouse_listener.start()
        self._capture_thread.start()

    def stop(self) -> str | None:
        """
        Stop recording, save the trajectory, and return the save path.

        Returns:
            Path to the saved .npz, or None if no frames were recorded.
        """
        self._stop_event.set()
        self._capture_thread.join(timeout=2.0)
        self._kb_listener.stop()
        self._mouse_listener.stop()
        return self._save()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return len(self._ts_buf)

    # ------------------------------------------------------------------
    # Input listeners
    # ------------------------------------------------------------------

    def _on_key_press(self, key: keyboard.Key) -> None:
        try:
            char = key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            char = None

        action: int | None = None

        if char and char in _KEY_MAP:
            action = _KEY_MAP[char]
        elif key == keyboard.Key.space:
            action = JUMP

        if action is not None:
            with self._action_lock:
                self._pending_action = action

    def _on_click(
        self, x: int, y: int, button: mouse.Button, pressed: bool
    ) -> None:
        if pressed and button == mouse.Button.left:
            with self._action_lock:
                self._pending_action = JUMP_ATTACK

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while True:
            tick_start = time.perf_counter()

            try:
                state = self._game.get_current_state(include_frame=False)

                with self._action_lock:
                    action = self._pending_action
                    self._pending_action = NOOP  # consume

                obs = self._build_obs(state)
                self._obs_buf.append(obs)
                self._act_buf.append(action)
                self._ts_buf.append(state.timestamp)

            except Exception as exc:  # noqa: BLE001
                print(f"[DemoRecorder] Capture error: {exc}")

            if self._stop_event.is_set():
                break

            elapsed = time.perf_counter() - tick_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Observation builder  (mirrors Expedition33Env._build_obs)
    # ------------------------------------------------------------------

    def _build_obs(self, state) -> np.ndarray:
        det_map = {}
        for d in state.detections:
            if d.label not in det_map or d.confidence > det_map[d.label].confidence:
                det_map[d.label] = d

        roi_w = max(self._game.monitor.get("width", 1), 1)
        roi_h = max(self._game.monitor.get("height", 1), 1)

        confidences = np.zeros(_N, dtype=np.float32)
        x_centres   = np.zeros(_N, dtype=np.float32)
        y_centres   = np.zeros(_N, dtype=np.float32)

        for i, label in enumerate(OBSERVATION_TARGETS):
            if label in det_map:
                d = det_map[label]
                confidences[i] = float(d.confidence)
                x_centres[i]   = float(d.x + d.w / 2) / roi_w
                y_centres[i]   = float(d.y + d.h / 2) / roi_h

        return np.concatenate([confidences, x_centres, y_centres])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> str | None:
        if not self._ts_buf:
            print("[DemoRecorder] No frames recorded — nothing saved.")
            return None

        os.makedirs(self._save_dir, exist_ok=True)
        save_path = os.path.join(self._save_dir, f"{self._session_name}.npz")

        np.savez_compressed(
            save_path,
            observations=np.array(self._obs_buf, dtype=np.float32),
            actions=np.array(self._act_buf,      dtype=np.int32),
            timestamps=np.array(self._ts_buf,    dtype=np.float64),
        )
        print(f"[DemoRecorder] Saved {len(self._ts_buf)} frames → {save_path}")
        return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record human Expedition 33 gameplay demonstrations."
    )
    p.add_argument("--session", default="demo", help="Output filename stem")
    p.add_argument("--engine",  default="PIXEL", help="Vision engine (PIXEL/SIFT/ORB)")
    p.add_argument("--hz",      type=float, default=20.0, help="Capture rate in Hz")
    p.add_argument("--save-dir", default=DEMO_DIR, help="Directory to save .npz files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    game = GameInstance(engine=args.engine)
    rec  = DemoRecorder(
        game,
        session_name=args.session,
        poll_hz=args.hz,
        save_dir=args.save_dir,
    )

    rec.start()
    try:
        while True:
            time.sleep(0.5)
            print(f"\r[DemoRecorder] {rec.frame_count} frames captured...", end="", flush=True)
    except KeyboardInterrupt:
        print()
        rec.stop()
