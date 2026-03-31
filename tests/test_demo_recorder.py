"""
Unit tests for tools/demo_recorder.py — DemoRecorder.

All external dependencies (GameInstance, pynput, file I/O) are fully
mocked. No screen capture, no keyboard listener, and no real files are
used in any test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pynput import keyboard, mouse

from environment.actions import (
    ATTACK,
    DODGE,
    GRADIENT_PARRY,
    JUMP,
    JUMP_ATTACK,
    NOOP,
    PARRY,
)
from tools.demo_recorder import OBSERVATION_TARGETS, DemoRecorder
from vision.engine import Detection, GameState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_game() -> MagicMock:
    game = MagicMock()
    game.monitor = {"width": 100, "height": 100}
    game.get_current_state.return_value = GameState(
        detections=[], timestamp=1.0, engine_name="MOCK"
    )
    return game


def _make_recorder(tmp_path, game=None) -> DemoRecorder:
    return DemoRecorder(
        game=game or _make_mock_game(),
        session_name="test_session",
        poll_hz=200.0,
        save_dir=str(tmp_path),
    )


def _key(char: str) -> keyboard.KeyCode:
    return keyboard.KeyCode.from_char(char)


# ---------------------------------------------------------------------------
# Key → Action mapping
# ---------------------------------------------------------------------------

class TestKeyMapping:
    @pytest.mark.parametrize("char, expected_action", [
        ("e", PARRY),
        ("E", PARRY),
        ("q", DODGE),
        ("Q", DODGE),
        ("w", GRADIENT_PARRY),
        ("W", GRADIENT_PARRY),
        ("f", ATTACK),
        ("F", ATTACK),
    ])
    def test_character_key_maps_to_correct_action(self, tmp_path, char, expected_action):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act
        rec._on_key_press(_key(char))

        # Assert
        with rec._action_lock:
            assert rec._pending_action == expected_action

    def test_space_key_maps_to_jump(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act
        rec._on_key_press(keyboard.Key.space)

        # Assert
        with rec._action_lock:
            assert rec._pending_action == JUMP

    def test_unmapped_key_does_not_change_pending_action(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act
        rec._on_key_press(keyboard.Key.shift)

        # Assert
        with rec._action_lock:
            assert rec._pending_action == NOOP

    def test_left_click_maps_to_jump_attack(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act
        rec._on_click(0, 0, mouse.Button.left, pressed=True)

        # Assert
        with rec._action_lock:
            assert rec._pending_action == JUMP_ATTACK

    def test_right_click_does_not_change_pending_action(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act
        rec._on_click(0, 0, mouse.Button.right, pressed=True)

        # Assert
        with rec._action_lock:
            assert rec._pending_action == NOOP

    def test_mouse_release_does_not_change_pending_action(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        # Act — pressed=False (button release)
        rec._on_click(0, 0, mouse.Button.left, pressed=False)

        # Assert
        with rec._action_lock:
            assert rec._pending_action == NOOP


# ---------------------------------------------------------------------------
# Capture loop — observation and action recording
# ---------------------------------------------------------------------------

class TestCaptureLoop:
    def test_capture_loop_appends_obs_action_and_timestamp(self, tmp_path):
        # Arrange
        game = _make_mock_game()
        rec  = _make_recorder(tmp_path, game=game)
        rec._pending_action = DODGE

        # Act — run one tick of the capture loop manually
        rec._stop_event.set()   # stop after first iteration
        rec._capture_loop()

        # Assert
        assert len(rec._obs_buf) == 1
        assert rec._act_buf[0] == DODGE
        assert len(rec._ts_buf) == 1

    def test_pending_action_resets_to_noop_after_capture(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)
        rec._pending_action = PARRY

        # Act
        rec._stop_event.set()
        rec._capture_loop()

        # Assert
        with rec._action_lock:
            assert rec._pending_action == NOOP

    def test_capture_error_does_not_crash_loop(self, tmp_path):
        # Arrange
        game = _make_mock_game()
        game.get_current_state.side_effect = RuntimeError("capture failed")
        rec  = _make_recorder(tmp_path, game=game)

        # Act — must not raise
        rec._stop_event.set()
        rec._capture_loop()

        # Assert — nothing was buffered, but no crash
        assert len(rec._obs_buf) == 0


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

class TestBuildObs:
    def test_empty_detections_produces_all_zeros(self, tmp_path):
        # Arrange
        rec   = _make_recorder(tmp_path)
        state = GameState(detections=[], timestamp=1.0, engine_name="MOCK")

        # Act
        obs = rec._build_obs(state)

        # Assert
        assert obs.shape == (len(OBSERVATION_TARGETS) * 3,)
        assert np.all(obs == 0.0)

    def test_detected_label_sets_correct_index(self, tmp_path):
        # Arrange
        rec   = _make_recorder(tmp_path)
        label = "DODGE"
        idx   = OBSERVATION_TARGETS.index(label)
        det   = Detection(label=label, x=0, y=0, w=10, h=10, confidence=0.75)
        state = GameState(detections=[det], timestamp=1.0, engine_name="MOCK")

        # Act
        obs = rec._build_obs(state)

        # Assert
        assert pytest.approx(obs[idx], abs=1e-5) == 0.75

    def test_duplicate_label_keeps_highest_confidence(self, tmp_path):
        # Arrange
        rec   = _make_recorder(tmp_path)
        label = "PERFECT"
        idx   = OBSERVATION_TARGETS.index(label)
        dets  = [
            Detection(label, 0, 0, 10, 10, 0.5),
            Detection(label, 0, 0, 10, 10, 0.95),
        ]
        state = GameState(detections=dets, timestamp=1.0, engine_name="MOCK")

        # Act
        obs = rec._build_obs(state)

        # Assert
        assert pytest.approx(obs[idx], abs=1e-5) == 0.95


# ---------------------------------------------------------------------------
# save / stop
# ---------------------------------------------------------------------------

class TestSave:
    def test_stop_saves_npz_with_correct_arrays(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)
        obs = np.zeros(len(OBSERVATION_TARGETS) * 3, dtype=np.float32)
        rec._obs_buf  = [obs, obs]
        rec._act_buf  = [DODGE, NOOP]
        rec._ts_buf   = [1.0, 2.0]

        with patch.object(rec._kb_listener, "stop"), \
             patch.object(rec._mouse_listener, "stop"), \
             patch.object(rec._capture_thread, "join"):

            # Act
            save_path = rec.stop()

        # Assert
        assert save_path is not None
        data = np.load(save_path)
        assert data["observations"].shape == (2, len(OBSERVATION_TARGETS) * 3)
        assert data["actions"].tolist()   == [DODGE, NOOP]
        assert data["timestamps"].tolist() == [1.0, 2.0]

    def test_stop_returns_none_when_no_frames_recorded(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)

        with patch.object(rec._kb_listener, "stop"), \
             patch.object(rec._mouse_listener, "stop"), \
             patch.object(rec._capture_thread, "join"):

            # Act
            result = rec.stop()

        # Assert
        assert result is None

    def test_frame_count_matches_buffer_length(self, tmp_path):
        # Arrange
        rec = _make_recorder(tmp_path)
        rec._ts_buf = [1.0, 2.0, 3.0]

        # Assert
        assert rec.frame_count == 3
