"""
Unit tests for environment/ue4ss_reader.py and environment/ue4ss_env.py.

No live game required — all game state is injected via temp files or mocks.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environment.ue4ss_reader import GameState, StateReader, _DEFAULT_STATE, _merge_defaults


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f)


def _full_state(**overrides) -> dict:
    base = {
        "player_hp": 80.0, "player_hp_max": 100.0,
        "enemy_hp": 50.0,  "enemy_hp_max":  100.0,
        "player_ap": 5,
        "enemy_break": 30.0, "enemy_break_max": 60.0,
        "in_battle": True,
        "is_offensive_phase": True,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# TestStateReader
# ─────────────────────────────────────────────────────────────────────────────


class TestStateReader:
    def test_read_returns_default_when_file_missing(self, tmp_path):
        reader = StateReader(path=str(tmp_path / "nonexistent.json"))
        state = reader.read()
        assert state == dict(_DEFAULT_STATE)

    def test_read_parses_valid_json(self, tmp_path):
        path = str(tmp_path / "state.json")
        _write_state(path, _full_state())
        reader = StateReader(path=path)

        state = reader.read()
        assert state["player_hp"] == 80.0
        assert state["player_ap"] == 5
        assert state["in_battle"] is True

    def test_read_returns_default_on_malformed_json(self, tmp_path):
        path = str(tmp_path / "state.json")
        with open(path, "w") as f:
            f.write("not json {{{")
        reader = StateReader(path=path)

        state = reader.read()
        assert state == dict(_DEFAULT_STATE)

    def test_is_available_false_when_missing(self, tmp_path):
        reader = StateReader(path=str(tmp_path / "nope.json"))
        assert reader.is_available() is False

    def test_is_available_true_when_file_exists(self, tmp_path):
        path = str(tmp_path / "state.json")
        _write_state(path, _full_state())
        reader = StateReader(path=path)
        assert reader.is_available() is True

    def test_merge_defaults_fills_missing_keys(self):
        partial = {"player_hp": 42.0}
        result = _merge_defaults(partial)
        assert result["player_hp"] == 42.0
        assert result["player_hp_max"] == _DEFAULT_STATE["player_hp_max"]
        assert result["in_battle"] is False

    def test_merge_defaults_ignores_unknown_keys(self):
        data = {**_full_state(), "unknown_key": 999}
        result = _merge_defaults(data)
        assert "unknown_key" not in result


# ─────────────────────────────────────────────────────────────────────────────
# TestUE4SSExpedition33Env
# ─────────────────────────────────────────────────────────────────────────────


class TestUE4SSExpedition33Env:
    @pytest.fixture()
    def state_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        _write_state(path, _full_state())
        return path

    @pytest.fixture()
    def env(self, state_file):
        from environment.ue4ss_env import UE4SSExpedition33Env
        reader = StateReader(path=state_file)
        return UE4SSExpedition33Env(reader=reader, game=None, step_delay=0.0)

    def test_obs_space_shape(self, env):
        assert env.observation_space.shape == (9,)

    def test_action_space_size(self, env):
        from environment.actions import NUM_ACTIONS
        assert env.action_space.n == NUM_ACTIONS

    def test_reset_returns_9_dim_obs(self, env):
        obs, info = env.reset()
        assert obs.shape == (9,)
        assert obs.dtype == np.float32

    def test_reset_deltas_are_zero(self, env):
        obs, _ = env.reset()
        assert obs[6] == 0.0  # player_hp_delta
        assert obs[7] == 0.0  # enemy_hp_delta
        assert obs[8] == 0.0  # player_ap_delta

    def test_reset_hp_normalised(self, env):
        obs, _ = env.reset()
        assert pytest.approx(obs[0], abs=1e-5) == 0.80  # 80/100
        assert pytest.approx(obs[1], abs=1e-5) == 0.50  # 50/100

    def test_reset_ap_normalised(self, env):
        obs, _ = env.reset()
        assert pytest.approx(obs[2], abs=1e-5) == 5 / 9.0

    def test_in_battle_flag(self, env):
        obs, _ = env.reset()
        assert obs[4] == 1.0

    def test_offensive_phase_flag(self, env):
        obs, _ = env.reset()
        assert obs[5] == 1.0

    def test_step_returns_5tuple(self, env, state_file):
        env.reset()
        _write_state(state_file, _full_state(enemy_hp=40.0))  # simulate damage dealt
        result = env.step(0)  # NOOP
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

    def test_enemy_hp_delta_negative_when_damage_dealt(self, env, state_file):
        env.reset()
        _write_state(state_file, _full_state(enemy_hp=40.0))  # 50→40 = -0.1 ratio
        obs, reward, *_ = env.step(0)
        assert obs[7] < 0  # enemy_hp_delta

    def test_reward_positive_when_damage_dealt(self, env, state_file):
        env.reset()
        _write_state(state_file, _full_state(enemy_hp=40.0))
        _, reward, *_ = env.step(0)
        assert reward > -0.1  # higher than plain step penalty

    def test_reward_negative_when_player_damaged(self, env, state_file):
        env.reset()
        _write_state(state_file, _full_state(player_hp=60.0))  # 80→60
        _, reward, *_ = env.step(0)
        assert reward < -0.1

    def test_terminated_when_battle_ends(self, env, state_file):
        env.reset()
        env.step(0)  # step 1
        _write_state(state_file, _full_state(in_battle=False))
        _, _, terminated, _, _ = env.step(0)  # step 2 — battle gone
        assert terminated is True

    def test_truncated_at_max_steps(self, state_file):
        from environment.ue4ss_env import UE4SSExpedition33Env
        reader = StateReader(path=state_file)
        env = UE4SSExpedition33Env(reader=reader, game=None, step_delay=0.0, max_steps=2)
        env.reset()
        env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

    def test_obs_clipped_at_zero_max_hp(self, tmp_path):
        path = str(tmp_path / "state.json")
        _write_state(path, _full_state(player_hp_max=0.0))
        from environment.ue4ss_env import UE4SSExpedition33Env
        env = UE4SSExpedition33Env(reader=StateReader(path=path), game=None, step_delay=0.0)
        obs, _ = env.reset()
        assert obs[0] == 0.0  # player_hp_ratio should be 0, not NaN

    def test_no_game_instance_skips_action_execution(self, env):
        env.reset()
        # Should not raise even though no GameInstance is provided
        obs, _, _, _, _ = env.step(1)  # PARRY with no game → silently skipped
        assert obs.shape == (9,)
