"""
Unit tests for rl/policy.py and rl/train.py.

All heavy deps (SB3, torch) are mocked via sys.modules injection — tests run
in CI with only the dev group installed (no cuda/il groups required).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------

def _make_sb3_stubs() -> tuple[dict, MagicMock]:
    """Return (stubs_dict, MockPPO) for stable_baselines3."""
    sb3_mock = MagicMock()
    stubs = {
        "stable_baselines3": sb3_mock,
        "stable_baselines3.common": MagicMock(),
        "stable_baselines3.common.vec_env": MagicMock(),
    }
    return stubs, sb3_mock.PPO


# ===========================================================================
# TestLoadGailWeights — rl/policy.load_gail_weights()
# ===========================================================================

class TestLoadGailWeights:
    def test_ppo_load_called_with_checkpoint_and_env(self):
        # Arrange
        stubs, MockPPO = _make_sb3_stubs()
        mock_env = MagicMock()

        # Act
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.policy", None)
            from rl.policy import load_gail_weights
            result = load_gail_weights(mock_env, "data/models/gail_test.zip")

        # Assert
        MockPPO.load.assert_called_once_with(
            "data/models/gail_test.zip", env=mock_env, device="auto"
        )
        assert result is MockPPO.load.return_value

    def test_device_forwarded_to_ppo_load(self):
        # Arrange
        stubs, MockPPO = _make_sb3_stubs()
        mock_env = MagicMock()

        # Act
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.policy", None)
            from rl.policy import load_gail_weights
            load_gail_weights(mock_env, "checkpoint.zip", device="cpu")

        # Assert
        MockPPO.load.assert_called_once_with("checkpoint.zip", env=mock_env, device="cpu")


# ===========================================================================
# TestRlTrain — rl/train.train()
# ===========================================================================

def _make_train_stubs() -> tuple[dict, MagicMock, MagicMock, MagicMock]:
    """
    Return (stubs, MockPPO, mock_lgw, mock_model) for rl.train tests.

    Stubs out stable_baselines3, rl.policy (load_gail_weights), and
    environment.gym_env (Expedition33Env) so no game or SB3 install required.
    """
    sb3_mock = MagicMock()
    mock_model = MagicMock()
    sb3_mock.PPO.return_value = mock_model
    sb3_mock.PPO.load.return_value = mock_model

    mock_lgw = MagicMock(return_value=mock_model)
    rl_policy_stub = MagicMock()
    rl_policy_stub.load_gail_weights = mock_lgw

    mock_env_instance = MagicMock()
    gym_env_stub = MagicMock()
    gym_env_stub.Expedition33Env.return_value = mock_env_instance

    stubs = {
        "stable_baselines3": sb3_mock,
        "stable_baselines3.common": MagicMock(),
        "stable_baselines3.common.vec_env": MagicMock(),
        "rl.policy": rl_policy_stub,
        "environment.gym_env": gym_env_stub,
    }
    return stubs, sb3_mock.PPO, mock_lgw, mock_model, mock_env_instance


class TestRlTrain:
    @pytest.fixture(autouse=True)
    def _stubs(self):
        stubs, MockPPO, mock_lgw, mock_model, mock_env = _make_train_stubs()
        self.MockPPO = MockPPO
        self.mock_lgw = mock_lgw
        self.mock_model = mock_model
        self.mock_env = mock_env

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("rl.train", None)
            yield

    def test_returns_zip_path(self, tmp_path):
        # Arrange / Act
        from rl.train import train
        result = train(out_dir=str(tmp_path), total_timesteps=1)

        # Assert
        assert result.endswith(".zip")

    def test_model_save_called_with_ppo_prefix(self, tmp_path):
        from rl.train import train
        train(out_dir=str(tmp_path), total_timesteps=1)

        self.mock_model.save.assert_called_once()
        saved_path = self.mock_model.save.call_args[0][0]
        assert "ppo_" in saved_path

    def test_model_learn_called_with_correct_timesteps(self, tmp_path):
        from rl.train import train
        train(out_dir=str(tmp_path), total_timesteps=42)

        self.mock_model.learn.assert_called_once_with(total_timesteps=42)

    def test_load_gail_weights_called_when_checkpoint_given(self, tmp_path):
        from rl.train import train
        train(
            gail_checkpoint="data/models/gail_test.zip",
            out_dir=str(tmp_path),
            total_timesteps=1,
        )

        self.mock_lgw.assert_called_once()
        assert self.mock_lgw.call_args[0][1] == "data/models/gail_test.zip"

    def test_load_gail_weights_skipped_when_no_checkpoint(self, tmp_path):
        from rl.train import train
        train(gail_checkpoint=None, out_dir=str(tmp_path), total_timesteps=1)

        self.mock_lgw.assert_not_called()

    def test_env_close_called_on_completion(self, tmp_path):
        from rl.train import train
        train(out_dir=str(tmp_path), total_timesteps=1)

        self.mock_env.close.assert_called_once()

    def test_output_dir_created_if_missing(self, tmp_path):
        import os

        from rl.train import train

        new_dir = str(tmp_path / "new" / "subdir")
        train(out_dir=new_dir, total_timesteps=1)

        assert os.path.isdir(new_dir)
