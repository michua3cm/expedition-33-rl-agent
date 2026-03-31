"""
Unit tests for environment/state_buffer.py — StateBuffer.

GameInstance is fully mocked. Real threads are used to verify
concurrency behaviour; timeouts are kept short (<1 s) so the suite
runs fast.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from environment.state_buffer import StateBuffer
from vision.engine import GameState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_game(state: GameState | None = None) -> MagicMock:
    game = MagicMock()
    game.get_current_state.return_value = state or GameState(
        detections=[], timestamp=1.0, engine_name="MOCK"
    )
    return game


def _make_buffer(game=None, poll_hz: float = 200.0) -> StateBuffer:
    """Create a StateBuffer at 200 Hz so tests don't have to wait long."""
    return StateBuffer(game or _make_mock_game(), poll_hz=poll_hz)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_latest_returns_none_before_start(self):
        # Arrange
        buf = _make_buffer()

        # Assert — no frames captured yet
        assert buf.latest() is None

    def test_is_running_is_false_before_start(self):
        # Arrange
        buf = _make_buffer()

        # Assert
        assert buf.is_running is False


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_is_running_is_true_after_start(self):
        # Arrange
        buf = _make_buffer()

        # Act
        buf.start()

        try:
            # Assert
            assert buf.is_running is True
        finally:
            buf.stop()

    def test_is_running_is_false_after_stop(self):
        # Arrange
        buf = _make_buffer()
        buf.start()

        # Act
        buf.stop()

        # Assert
        assert buf.is_running is False

    def test_stop_is_idempotent_when_never_started(self):
        # Arrange
        buf = _make_buffer()

        # Act & Assert — must not raise
        buf.stop()


# ---------------------------------------------------------------------------
# latest() and wait_for_state()
# ---------------------------------------------------------------------------

class TestStateAccess:
    def test_latest_returns_game_state_after_first_capture(self):
        # Arrange
        fake_state = GameState(detections=[], timestamp=42.0, engine_name="MOCK")
        game = _make_mock_game(state=fake_state)
        buf = _make_buffer(game=game)

        # Act
        buf.start()
        state = buf.wait_for_state(timeout=1.0)
        buf.stop()

        # Assert
        assert state is not None
        assert state.timestamp == 42.0

    def test_wait_for_state_returns_none_on_timeout(self):
        # Arrange — game always raises to prevent any frame from being captured
        game = MagicMock()
        game.get_current_state.side_effect = RuntimeError("no capture")
        buf = _make_buffer(game=game)

        # Act
        buf.start()
        state = buf.wait_for_state(timeout=0.1)
        buf.stop()

        # Assert
        assert state is None

    def test_latest_always_returns_most_recent_state(self):
        # Arrange — game returns states with incrementing timestamps
        call_count = [0]
        def _next_state(**_):
            call_count[0] += 1
            return GameState([], float(call_count[0]), "MOCK")

        game = MagicMock()
        game.get_current_state.side_effect = _next_state

        buf = _make_buffer(game=game, poll_hz=200.0)

        # Act
        buf.start()
        buf.wait_for_state(timeout=1.0)
        time.sleep(0.05)   # let a few more frames capture
        state = buf.latest()
        buf.stop()

        # Assert — timestamp should be > 1 (multiple frames captured)
        assert state.timestamp > 1.0


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_capture_error_does_not_crash_thread(self):
        # Arrange — first call raises, subsequent calls succeed
        fake_state = GameState([], 1.0, "MOCK")
        call_count = [0]

        def _flaky(**_):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            return fake_state

        game = MagicMock()
        game.get_current_state.side_effect = _flaky

        buf = _make_buffer(game=game, poll_hz=200.0)

        # Act
        buf.start()
        state = buf.wait_for_state(timeout=1.0)
        buf.stop()

        # Assert — thread recovered and eventually captured a frame
        assert state is not None
        assert buf.is_running is False
