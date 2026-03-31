"""
Async state buffer for Expedition 33.

Runs screen capture + vision inference on a background thread so that
the RL policy loop (step / compute gradient) is never stalled waiting
for a frame.  The policy always reads the *latest* complete GameState
without blocking.

Usage::

    buf = StateBuffer(game_instance, poll_hz=30)
    buf.start()

    state = buf.latest()          # non-blocking, returns None until first frame
    state = buf.wait_for_state()  # blocks until at least one frame is ready

    buf.stop()
"""

from __future__ import annotations

import threading
import time

import vision

from .instance import GameInstance


class StateBuffer:
    """
    Background-thread capture loop that keeps the latest GameState fresh.

    Args:
        game:          Initialised GameInstance to capture from.
        poll_hz:       Target capture rate in frames per second.
        include_frame: If True, attaches the raw greyscale frame to each
                       GameState (higher memory use).
    """

    def __init__(
        self,
        game: GameInstance,
        poll_hz: float = 30.0,
        include_frame: bool = False,
    ):
        self._game = game
        self._interval = 1.0 / max(poll_hz, 1.0)
        self._include_frame = include_frame

        self._state: vision.GameState | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # set once first frame is captured

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="StateBuffer-capture",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def latest(self) -> vision.GameState | None:
        """
        Return the most recent GameState without blocking.

        Returns None if no frame has been captured yet.
        """
        with self._lock:
            return self._state

    def wait_for_state(self, timeout: float = 5.0) -> vision.GameState | None:
        """
        Block until the first frame is ready, then return the latest state.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            The latest GameState, or None if timeout expired.
        """
        if not self._ready_event.wait(timeout=timeout):
            return None
        return self.latest()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                state = self._game.get_current_state(
                    include_frame=self._include_frame
                )
                with self._lock:
                    self._state = state
                self._ready_event.set()
            except Exception as exc:  # noqa: BLE001
                # Log but do not crash the capture thread on transient errors
                print(f"[StateBuffer] Capture error: {exc}")

            elapsed = time.perf_counter() - loop_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
