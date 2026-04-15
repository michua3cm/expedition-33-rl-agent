"""
Unit tests for rl/policy.py and rl/train.py.

All external dependencies are fully mocked:
  - Expedition33Env  — no screen capture, no game window
  - stable_baselines3.PPO — no actual RL training
  - BCPolicy.load    — uses real torch layers so weight comparisons are real

Weight-transfer tests use real nn.Linear instances with known weights so we
can verify the exact parameter values before and after transfer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed (cuda group required)")
pytest.importorskip("stable_baselines3", reason="stable_baselines3 not installed (rl group required)")

import torch.nn as nn  # noqa: E402

from imitation.policy import BCPolicy  # noqa: E402
from rl.policy import BC_NET_ARCH, load_bc_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — minimal PPO policy structure
# ---------------------------------------------------------------------------

def _make_ppo_policy() -> SimpleNamespace:
    """
    Return a minimal object that mirrors the structure PPO.policy exposes.

    We use real nn.Linear instances so load_state_dict() works as expected
    and we can compare actual parameter tensors in assertions.
    """
    policy_net = nn.Sequential(
        nn.Linear(30, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
    )
    value_net = nn.Sequential(
        nn.Linear(30, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
    )
    action_net = nn.Linear(64, 7)
    value_head = nn.Linear(64, 1)

    extractor = SimpleNamespace(policy_net=policy_net, value_net=value_net)
    pol = SimpleNamespace(
        mlp_extractor=extractor,
        action_net=action_net,
        value_net=value_head,
    )
    return pol


def _make_mock_ppo(policy: SimpleNamespace | None = None) -> MagicMock:
    """Return a mock PPO model with the given policy attribute."""
    model = MagicMock()
    model.policy = policy or _make_ppo_policy()
    return model


def _bc_with_known_weights() -> BCPolicy:
    """
    Return a BCPolicy whose weights are filled with a known constant (0.5).
    This makes it easy to assert the exact values after transfer.
    """
    bc = BCPolicy()
    with torch.no_grad():
        for param in bc.parameters():
            param.fill_(0.5)
    return bc


# ===========================================================================
# load_bc_weights — weight transfer correctness
# ===========================================================================

class TestLoadBcWeights:
    def test_bc_net_arch_has_correct_shape(self):
        # BC_NET_ARCH must match the actor hidden layers in BCPolicy exactly.
        assert BC_NET_ARCH == {"pi": [128, 64], "vf": [128, 64]}

    def test_actor_hidden_layer_0_weights_match(self, tmp_path):
        # Arrange — save a BC checkpoint with known weights
        bc = _bc_with_known_weights()
        ckpt = str(tmp_path / "bc.pt")
        bc.save(ckpt)

        pol = _make_ppo_policy()
        model = _make_mock_ppo(policy=pol)

        # Act
        load_bc_weights(model, ckpt)

        # Assert — policy_net[0] must equal bc.net[0] exactly
        bc_w  = bc.net[0].weight
        ppo_w = pol.mlp_extractor.policy_net[0].weight
        assert torch.allclose(ppo_w, bc_w), "policy_net[0] weight mismatch"

        bc_b  = bc.net[0].bias
        ppo_b = pol.mlp_extractor.policy_net[0].bias
        assert torch.allclose(ppo_b, bc_b), "policy_net[0] bias mismatch"

    def test_actor_hidden_layer_2_weights_match(self, tmp_path):
        # Arrange
        bc = _bc_with_known_weights()
        ckpt = str(tmp_path / "bc.pt")
        bc.save(ckpt)

        pol = _make_ppo_policy()
        model = _make_mock_ppo(policy=pol)

        # Act
        load_bc_weights(model, ckpt)

        # Assert — policy_net[2] must equal bc.net[2] exactly
        assert torch.allclose(
            pol.mlp_extractor.policy_net[2].weight,
            bc.net[2].weight,
        )
        assert torch.allclose(
            pol.mlp_extractor.policy_net[2].bias,
            bc.net[2].bias,
        )

    def test_action_net_weights_match(self, tmp_path):
        # Arrange
        bc = _bc_with_known_weights()
        ckpt = str(tmp_path / "bc.pt")
        bc.save(ckpt)

        pol = _make_ppo_policy()
        model = _make_mock_ppo(policy=pol)

        # Act
        load_bc_weights(model, ckpt)

        # Assert — action_net must equal bc.net[4] exactly
        assert torch.allclose(pol.action_net.weight, bc.net[4].weight)
        assert torch.allclose(pol.action_net.bias,   bc.net[4].bias)

    def test_critic_value_net_is_not_modified(self, tmp_path):
        """
        The critic (value_net) must remain at its original random values.
        BC never trained a value function — copying its weights would be wrong.
        """
        # Arrange
        bc = _bc_with_known_weights()  # all weights = 0.5
        ckpt = str(tmp_path / "bc.pt")
        bc.save(ckpt)

        pol = _make_ppo_policy()
        # Record critic weights before transfer
        critic_w_before = pol.mlp_extractor.value_net[0].weight.clone()

        model = _make_mock_ppo(policy=pol)

        # Act
        load_bc_weights(model, ckpt)

        # Assert — critic weights unchanged
        assert torch.allclose(
            pol.mlp_extractor.value_net[0].weight,
            critic_w_before,
        ), "Critic weights were modified — they must stay random."

    def test_transfer_is_independent_of_bc_random_init(self, tmp_path):
        """
        Two different BCPolicy checkpoints produce different PPO actor weights.
        Verifies that load_bc_weights reflects the specific checkpoint loaded.
        """
        # Arrange — two BCs with different weights
        bc_a = BCPolicy()
        bc_b = BCPolicy()
        # Re-init bc_b to ensure it differs from bc_a
        nn.init.constant_(bc_b.net[0].weight, 0.1)

        ckpt_a = str(tmp_path / "bc_a.pt")
        ckpt_b = str(tmp_path / "bc_b.pt")
        bc_a.save(ckpt_a)
        bc_b.save(ckpt_b)

        pol_a = _make_ppo_policy()
        pol_b = _make_ppo_policy()

        # Act
        load_bc_weights(_make_mock_ppo(policy=pol_a), ckpt_a)
        load_bc_weights(_make_mock_ppo(policy=pol_b), ckpt_b)

        # Assert — the two PPO policies have different actor weights
        assert not torch.allclose(
            pol_a.mlp_extractor.policy_net[0].weight,
            pol_b.mlp_extractor.policy_net[0].weight,
        )


# ===========================================================================
# rl.train.train() — training loop interactions
# ===========================================================================

class TestRlTrain:
    """
    Tests for rl/train.py.  PPO is fully mocked — we test the orchestration
    logic (warm-start, env creation, model save, return value) without
    running any real RL.
    """

    def _make_mock_env(self) -> MagicMock:
        env = MagicMock()
        env.observation_space = MagicMock()
        env.action_space = MagicMock()
        return env

    def test_train_returns_zip_path(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            result = train(out_dir=str(tmp_path), total_timesteps=1)

        # Assert — always ends with .zip
        assert result.endswith(".zip")

    def test_train_calls_model_save(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            train(out_dir=str(tmp_path), total_timesteps=1)

        # Assert — model.save() was called once with a path containing "ppo_"
        mock_model.save.assert_called_once()
        saved_path = mock_model.save.call_args[0][0]
        assert "ppo_" in saved_path

    def test_train_calls_model_learn(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            train(out_dir=str(tmp_path), total_timesteps=42)

        # Assert — learn called with the right timesteps
        mock_model.learn.assert_called_once_with(total_timesteps=42)

    def test_train_warm_starts_from_bc_when_checkpoint_given(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()
        ckpt = str(tmp_path / "bc_best.pt")
        open(ckpt, "w").close()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights") as mock_lbw:

            from rl.train import train

            train(bc_checkpoint=ckpt, out_dir=str(tmp_path), total_timesteps=1)

        # Assert — load_bc_weights was called with the model and checkpoint
        mock_lbw.assert_called_once_with(mock_model, ckpt)

    def test_train_skips_bc_warmstart_when_no_checkpoint(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights") as mock_lbw:

            from rl.train import train

            train(bc_checkpoint=None, out_dir=str(tmp_path), total_timesteps=1)

        # Assert — load_bc_weights was NOT called
        mock_lbw.assert_not_called()

    def test_train_closes_env_on_completion(self, tmp_path):
        # Arrange
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            train(out_dir=str(tmp_path), total_timesteps=1)

        # Assert
        mock_env.close.assert_called_once()

    def test_train_passes_bc_net_arch_to_ppo(self, tmp_path):
        # The PPO model must use BC_NET_ARCH so actor layers match BCPolicy.
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model) as MockPPO, \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            train(out_dir=str(tmp_path), total_timesteps=1)

        # Assert — PPO was constructed with policy_kwargs containing BC_NET_ARCH
        _, ppo_kwargs = MockPPO.call_args
        assert ppo_kwargs["policy_kwargs"] == {"net_arch": BC_NET_ARCH}

    def test_train_creates_output_directory_if_missing(self, tmp_path):
        # Arrange — use a subdirectory that does not yet exist
        new_dir    = str(tmp_path / "new" / "subdir")
        mock_env   = self._make_mock_env()
        mock_model = MagicMock()

        with patch("rl.train.Expedition33Env", return_value=mock_env), \
             patch("rl.train.PPO",             return_value=mock_model), \
             patch("rl.train.load_bc_weights"):

            from rl.train import train

            train(out_dir=new_dir, total_timesteps=1)

        import os
        assert os.path.isdir(new_dir)
