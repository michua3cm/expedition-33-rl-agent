"""
Unit tests for rl/train_sac.py and the SAC additions to rl/policy.py.

All heavy deps (SB3, torch) are mocked via sys.modules injection — tests run
in CI with only the dev group installed (no cuda/il groups required).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Shared stubs ──────────────────────────────────────────────────────────────


def _make_sb3_stubs():
    sb3_mock = MagicMock()
    mock_model = MagicMock()
    sb3_mock.SAC.return_value = mock_model
    sb3_mock.PPO.return_value = mock_model

    stubs = {
        "stable_baselines3": sb3_mock,
        "stable_baselines3.common": MagicMock(),
        "stable_baselines3.common.vec_env": MagicMock(),
    }
    return stubs, sb3_mock, mock_model


# ── TestLoadSacCheckpoint — rl/policy.load_sac_checkpoint() ──────────────────


class TestLoadSacCheckpoint:
    def test_sac_load_called_with_checkpoint_and_env(self):
        stubs, sb3_mock, mock_model = _make_sb3_stubs()
        mock_env = MagicMock()

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.policy", None)
            from rl.policy import load_sac_checkpoint
            result = load_sac_checkpoint(mock_env, "data/models/sac_test.zip")

        sb3_mock.SAC.load.assert_called_once_with(
            "data/models/sac_test.zip", env=mock_env, device="auto"
        )
        assert result is sb3_mock.SAC.load.return_value

    def test_device_forwarded_to_sac_load(self):
        stubs, sb3_mock, _ = _make_sb3_stubs()
        mock_env = MagicMock()

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.policy", None)
            from rl.policy import load_sac_checkpoint
            load_sac_checkpoint(mock_env, "checkpoint.zip", device="cpu")

        sb3_mock.SAC.load.assert_called_once_with(
            "checkpoint.zip", env=mock_env, device="cpu"
        )

    def test_load_gail_weights_still_works(self):
        """Ensure existing PPO function was not broken."""
        stubs, sb3_mock, _ = _make_sb3_stubs()
        mock_env = MagicMock()

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.policy", None)
            from rl.policy import load_gail_weights
            load_gail_weights(mock_env, "gail.zip")

        sb3_mock.PPO.load.assert_called_once()


# ── TestTrainSac — rl/train_sac.train_sac() ──────────────────────────────────


def _make_sac_train_stubs():
    sb3_mock = MagicMock()
    mock_model = MagicMock()
    sb3_mock.SAC.return_value = mock_model

    mock_env_instance = MagicMock()

    stubs = {
        "stable_baselines3": sb3_mock,
        "stable_baselines3.common": MagicMock(),
        "stable_baselines3.common.vec_env": MagicMock(),
    }
    return stubs, sb3_mock.SAC, mock_model, mock_env_instance


class TestTrainSac:
    @pytest.fixture(autouse=True)
    def _stubs(self):
        stubs, MockSAC, mock_model, mock_env = _make_sac_train_stubs()
        self.MockSAC = MockSAC
        self.mock_model = mock_model
        self.mock_env = mock_env

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.train_sac", None)
            yield

    def test_returns_zip_path(self, tmp_path):
        from rl.train_sac import train_sac
        result = train_sac(self.mock_env, out_dir=str(tmp_path), total_timesteps=1)
        assert result.endswith(".zip")

    def test_model_save_called_with_sac_prefix(self, tmp_path):
        from rl.train_sac import train_sac
        train_sac(self.mock_env, out_dir=str(tmp_path), total_timesteps=1)

        self.mock_model.save.assert_called_once()
        saved_path = self.mock_model.save.call_args[0][0]
        assert "sac_" in saved_path

    def test_model_learn_called_with_timesteps(self, tmp_path):
        from rl.train_sac import train_sac
        train_sac(self.mock_env, out_dir=str(tmp_path), total_timesteps=42)
        self.mock_model.learn.assert_called_once_with(total_timesteps=42)

    def test_env_close_called_on_completion(self, tmp_path):
        from rl.train_sac import train_sac
        train_sac(self.mock_env, out_dir=str(tmp_path), total_timesteps=1)
        self.mock_env.close.assert_called_once()

    def test_output_dir_created_if_missing(self, tmp_path):
        import os

        from rl.train_sac import train_sac
        new_dir = str(tmp_path / "new" / "models")
        train_sac(self.mock_env, out_dir=new_dir, total_timesteps=1)
        assert os.path.isdir(new_dir)

    def test_sac_kwargs_forwarded(self, tmp_path):
        from rl.train_sac import train_sac
        train_sac(
            self.mock_env,
            out_dir=str(tmp_path),
            total_timesteps=1,
            learning_rate=1e-3,
        )
        call_kwargs = self.MockSAC.call_args[1]
        assert call_kwargs["learning_rate"] == 1e-3

    def test_no_dp_warmstart_when_checkpoint_none(self, tmp_path):
        from rl.train_sac import train_sac
        train_sac(self.mock_env, dp_checkpoint=None, out_dir=str(tmp_path), total_timesteps=1)
        # model.replay_buffer.add should NOT have been called
        self.mock_model.replay_buffer.add.assert_not_called()
