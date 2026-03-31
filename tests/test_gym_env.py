"""
Unit tests for environment/gym_env.py — Expedition33Env.

GameInstance is fully mocked — no screen capture, no controller, no
real vision engine is involved in any test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environment import actions as A
from environment.gym_env import (
    OBS_DIM,
    OBSERVATION_TARGETS,
    REWARD_MAP,
    STEP_PENALTY,
    Expedition33Env,
)
from vision.engine import Detection, GameState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(label: str, x: int = 0, y: int = 0,
                    w: int = 10, h: int = 10, confidence: float = 0.8) -> Detection:
    return Detection(label=label, x=x, y=y, w=w, h=h, confidence=confidence)


def _make_state(detections: list[Detection] | None = None) -> GameState:
    return GameState(
        detections=detections or [],
        timestamp=1.0,
        engine_name="MOCK",
    )


def _make_env() -> tuple[Expedition33Env, MagicMock]:
    """Build an Expedition33Env with a mocked GameInstance."""
    mock_game = MagicMock()
    mock_game.monitor = {"width": 100, "height": 100}
    mock_game.get_current_state.return_value = _make_state()

    with patch("environment.gym_env.GameInstance", return_value=mock_game):
        env = Expedition33Env(engine="PIXEL", step_delay=0.0)

    return env, mock_game


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self):
        env, _ = _make_env()
        assert env.observation_space.shape == (OBS_DIM,)
        assert OBS_DIM == 30

    def test_observation_space_dtype(self):
        env, _ = _make_env()
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds(self):
        env, _ = _make_env()
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 1.0

    def test_action_space_size(self):
        env, _ = _make_env()
        assert env.action_space.n == A.NUM_ACTIONS


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_with_correct_shape(self):
        # Arrange
        env, _ = _make_env()

        # Act
        obs, info = env.reset()

        # Assert
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_reset_returns_info_with_required_keys(self):
        # Arrange
        env, _ = _make_env()

        # Act
        _, info = env.reset()

        # Assert
        assert "timestamp" in info
        assert "engine" in info
        assert "detections" in info

    def test_reset_clears_seen_signals(self):
        # Arrange
        env, _ = _make_env()
        env._seen_signals.add("PERFECT")

        # Act
        env.reset()

        # Assert
        assert len(env._seen_signals) == 0


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self):
        # Arrange
        env, _ = _make_env()
        env.reset()

        # Act
        result = env.step(A.NOOP)

        # Assert
        assert len(result) == 5

    def test_step_terminated_and_truncated_are_always_false(self):
        # Arrange
        env, _ = _make_env()
        env.reset()

        # Act
        _, _, terminated, truncated, _ = env.step(A.NOOP)

        # Assert
        assert terminated is False
        assert truncated is False

    def test_step_obs_has_correct_shape(self):
        # Arrange
        env, _ = _make_env()
        env.reset()

        # Act
        obs, *_ = env.step(A.NOOP)

        # Assert
        assert obs.shape == (OBS_DIM,)


# ---------------------------------------------------------------------------
# _build_obs()
# ---------------------------------------------------------------------------

class TestBuildObs:
    def test_no_detections_produces_all_zeros(self):
        # Arrange
        env, mock_game = _make_env()
        state = _make_state(detections=[])

        # Act
        obs = env._build_obs(state)

        # Assert
        assert np.all(obs == 0.0)

    def test_detected_label_sets_confidence_at_correct_index(self):
        # Arrange
        env, _ = _make_env()
        label = "DODGE"
        idx   = OBSERVATION_TARGETS.index(label)
        state = _make_state(detections=[_make_detection(label, confidence=0.75)])

        # Act
        obs = env._build_obs(state)

        # Assert
        assert pytest.approx(obs[idx], abs=1e-5) == 0.75

    def test_detected_label_sets_normalised_bbox_centre(self):
        # Arrange — 100×100 ROI, detection at x=40, y=50, w=20, h=10
        env, _ = _make_env()
        env.game.monitor = {"width": 100, "height": 100}
        label = "PERFECT"
        idx   = OBSERVATION_TARGETS.index(label)
        det   = _make_detection(label, x=40, y=50, w=20, h=10, confidence=0.9)
        state = _make_state(detections=[det])

        # Act
        obs = env._build_obs(state)

        # Assert — x_centre = (40 + 10) / 100 = 0.5, y_centre = (50 + 5) / 100 = 0.55
        x_centre = obs[len(OBSERVATION_TARGETS) + idx]
        y_centre = obs[len(OBSERVATION_TARGETS) * 2 + idx]
        assert pytest.approx(x_centre, abs=1e-5) == 0.5
        assert pytest.approx(y_centre, abs=1e-5) == 0.55

    def test_duplicate_label_keeps_highest_confidence(self):
        # Arrange — two detections for the same label
        env, _ = _make_env()
        label = "JUMP"
        idx   = OBSERVATION_TARGETS.index(label)
        dets  = [
            _make_detection(label, confidence=0.6),
            _make_detection(label, confidence=0.9),
        ]
        state = _make_state(detections=dets)

        # Act
        obs = env._build_obs(state)

        # Assert — highest confidence wins
        assert pytest.approx(obs[idx], abs=1e-5) == 0.9

    def test_unknown_label_is_ignored(self):
        # Arrange
        env, _ = _make_env()
        state = _make_state(detections=[_make_detection("UNKNOWN_LABEL")])

        # Act
        obs = env._build_obs(state)

        # Assert — all zeros because label is not in OBSERVATION_TARGETS
        assert np.all(obs == 0.0)


# ---------------------------------------------------------------------------
# _compute_reward()
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_no_detections_gives_only_step_penalty(self):
        # Arrange
        env, _ = _make_env()
        state = _make_state(detections=[])

        # Act
        reward = env._compute_reward(state)

        # Assert
        assert pytest.approx(reward) == STEP_PENALTY

    def test_first_detection_of_signal_gives_reward_plus_penalty(self):
        # Arrange
        env, _ = _make_env()
        state = _make_state(detections=[_make_detection("PERFECT")])

        # Act
        reward = env._compute_reward(state)

        # Assert
        assert pytest.approx(reward) == REWARD_MAP["PERFECT"] + STEP_PENALTY

    def test_already_seen_signal_gives_only_step_penalty(self):
        # Arrange
        env, _ = _make_env()
        env._seen_signals.add("PERFECT")
        state = _make_state(detections=[_make_detection("PERFECT")])

        # Act
        reward = env._compute_reward(state)

        # Assert — signal already collected, no bonus
        assert pytest.approx(reward) == STEP_PENALTY

    def test_signal_resets_after_disappearing(self):
        # Arrange — PERFECT was seen, then disappeared, then reappears
        env, _ = _make_env()
        env._seen_signals.add("PERFECT")

        # Signal disappears
        env._compute_reward(_make_state(detections=[]))
        assert "PERFECT" not in env._seen_signals

        # Signal reappears — should grant reward again
        reward = env._compute_reward(
            _make_state(detections=[_make_detection("PERFECT")])
        )

        # Assert
        assert pytest.approx(reward) == REWARD_MAP["PERFECT"] + STEP_PENALTY

    def test_multiple_new_signals_accumulate_rewards(self):
        # Arrange
        env, _ = _make_env()
        dets = [_make_detection("PERFECT"), _make_detection("DODGE")]
        state = _make_state(detections=dets)

        # Act
        reward = env._compute_reward(state)

        # Assert
        expected = REWARD_MAP["PERFECT"] + REWARD_MAP["DODGE"] + STEP_PENALTY
        assert pytest.approx(reward) == expected


# ---------------------------------------------------------------------------
# _execute_action()
# ---------------------------------------------------------------------------

class TestExecuteAction:
    @pytest.mark.parametrize("action, method", [
        (A.NOOP,           None),
        (A.PARRY,          "parry"),
        (A.DODGE,          "dodge"),
        (A.JUMP,           "jump"),
        (A.GRADIENT_PARRY, "gradient_parry"),
        (A.ATTACK,         "attack"),
        (A.JUMP_ATTACK,    "jump_attack"),
    ])
    def test_action_dispatches_to_correct_game_method(self, action, method):
        # Arrange
        env, mock_game = _make_env()

        # Act
        env._execute_action(action)

        # Assert
        if method is None:
            mock_game.parry.assert_not_called()
            mock_game.dodge.assert_not_called()
        else:
            getattr(mock_game, method).assert_called_once()

    def test_unknown_action_raises_value_error(self):
        # Arrange
        env, _ = _make_env()

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown action index"):
            env._execute_action(99)
