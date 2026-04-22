"""
Tests for the GAIL imitation learning pipeline.

All external dependencies (imitation library, SB3, screen capture) are fully
mocked via sys.modules injection — no live training, no disk models, no game
window required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level sys.modules stubs for heavy optional deps.
# These must be installed before any il.* import to prevent ImportError when
# the il group (torch, SB3, imitation) is not in the current environment.
# ---------------------------------------------------------------------------

def _make_imitation_stubs() -> dict:
    """Return a dict of module-name → MagicMock for the imitation/SB3 tree."""
    torch_mock = MagicMock()
    sb3_mock = MagicMock()
    sb3_ppo_mock = MagicMock()
    sb3_vec_mock = MagicMock()

    # DummyVecEnv must be a real class so isinstance() in gail.py doesn't
    # raise TypeError. We track construction calls via a class-level MagicMock.
    dvec_calls = MagicMock()

    class _FakeDummyVecEnv:
        def __init__(self, fns):
            dvec_calls(fns)
            self.observation_space = MagicMock()
            self.action_space = MagicMock()

    _FakeDummyVecEnv.dvec_calls = dvec_calls  # accessible from tests
    sb3_vec_mock.DummyVecEnv = _FakeDummyVecEnv

    imitation_mock = MagicMock()
    imitation_data_mock = MagicMock()
    imitation_data_types_mock = MagicMock()
    imitation_algo_mock = MagicMock()
    imitation_algo_adv_mock = MagicMock()
    imitation_algo_adv_gail_mock = MagicMock()
    imitation_rewards_mock = MagicMock()
    imitation_rewards_nets_mock = MagicMock()
    imitation_util_mock = MagicMock()
    imitation_util_networks_mock = MagicMock()

    return {
        "torch": torch_mock,
        "torchvision": MagicMock(),
        "stable_baselines3": sb3_mock,
        "stable_baselines3.common": MagicMock(),
        "stable_baselines3.common.vec_env": sb3_vec_mock,
        "stable_baselines3.PPO": sb3_ppo_mock,
        "imitation": imitation_mock,
        "imitation.data": imitation_data_mock,
        "imitation.data.types": imitation_data_types_mock,
        "imitation.algorithms": imitation_algo_mock,
        "imitation.algorithms.adversarial": imitation_algo_adv_mock,
        "imitation.algorithms.adversarial.gail": imitation_algo_adv_gail_mock,
        "imitation.rewards": imitation_rewards_mock,
        "imitation.rewards.reward_nets": imitation_rewards_nets_mock,
        "imitation.util": imitation_util_mock,
        "imitation.util.networks": imitation_util_networks_mock,
        "_FakeDummyVecEnv": _FakeDummyVecEnv,
    }


OBS_DIM = 30


def _write_npz(path: Path, T: int = 50) -> None:
    """Write a minimal valid .npz demo file with T timesteps."""
    np.savez(
        path,
        obs=np.random.rand(T, OBS_DIM).astype(np.float32),
        actions=np.random.randint(0, 7, size=(T,)).astype(np.int64),
        timestamps=np.arange(T, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# il/dataset.py — load_transitions()
# ---------------------------------------------------------------------------

class TestLoadTransitions:
    """Tests for il.dataset.load_transitions — imitation.data.types is mocked."""

    @pytest.fixture(autouse=True)
    def _stub_modules(self):
        stubs = _make_imitation_stubs()
        with patch.dict(sys.modules, stubs):
            yield stubs

    def test_returns_correct_shapes_for_single_file(self, tmp_path):
        # Arrange
        _write_npz(tmp_path / "demo_01.npz", T=50)

        # Act — re-import inside the patched context
        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions
        transitions_cls = sys.modules["imitation.data.types"].Transitions
        load_transitions(str(tmp_path))

        # Assert — Transitions was constructed with correct array shapes
        _, kwargs = transitions_cls.call_args
        assert kwargs["obs"].shape == (50, OBS_DIM)
        assert kwargs["acts"].shape == (50,)
        assert kwargs["next_obs"].shape == (50, OBS_DIM)
        assert kwargs["dones"].shape == (50,)

    def test_concatenates_multiple_files(self, tmp_path):
        # Arrange — two demos: 30 + 20 = 50 total steps
        _write_npz(tmp_path / "demo_01.npz", T=30)
        _write_npz(tmp_path / "demo_02.npz", T=20)

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions
        transitions_cls = sys.modules["imitation.data.types"].Transitions
        load_transitions(str(tmp_path))

        # Assert
        _, kwargs = transitions_cls.call_args
        assert kwargs["obs"].shape == (50, OBS_DIM)
        assert kwargs["acts"].shape == (50,)

    def test_next_obs_is_shifted_by_one(self, tmp_path):
        # Arrange — deterministic obs to verify the shift
        demo_path = tmp_path / "demo_01.npz"
        obs = np.arange(5 * OBS_DIM, dtype=np.float32).reshape(5, OBS_DIM)
        np.savez(demo_path, obs=obs, actions=np.zeros(5, dtype=np.int64))

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions
        transitions_cls = sys.modules["imitation.data.types"].Transitions
        load_transitions(str(tmp_path))

        # Assert
        _, kwargs = transitions_cls.call_args
        next_obs = kwargs["next_obs"]
        assert np.allclose(next_obs[0], obs[1])       # shifted by one
        assert np.allclose(next_obs[-1], np.zeros(OBS_DIM))  # final row padded

    def test_dones_are_all_false(self, tmp_path):
        # Arrange
        _write_npz(tmp_path / "demo_01.npz", T=10)

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions
        transitions_cls = sys.modules["imitation.data.types"].Transitions
        load_transitions(str(tmp_path))

        _, kwargs = transitions_cls.call_args
        assert not kwargs["dones"].any()

    def test_obs_cast_to_float32(self, tmp_path):
        # Arrange — save obs as float64 to exercise the dtype cast
        demo_path = tmp_path / "demo_01.npz"
        np.savez(
            demo_path,
            obs=np.ones((5, OBS_DIM), dtype=np.float64),
            actions=np.zeros(5, dtype=np.int64),
        )

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions
        transitions_cls = sys.modules["imitation.data.types"].Transitions
        load_transitions(str(tmp_path))

        _, kwargs = transitions_cls.call_args
        assert kwargs["obs"].dtype == np.float32

    def test_raises_when_no_npz_files_found(self, tmp_path):
        # Arrange — empty directory

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="No .npz demo files found"):
            load_transitions(str(tmp_path))

    def test_raises_for_nonexistent_directory(self, tmp_path):
        # Arrange
        missing = str(tmp_path / "does_not_exist")

        if "il.dataset" in sys.modules:
            del sys.modules["il.dataset"]
        from il.dataset import load_transitions

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_transitions(missing)


# ---------------------------------------------------------------------------
# il/gail.py — train_gail()
# ---------------------------------------------------------------------------

class TestTrainGail:
    """
    All external library imports (imitation, SB3) are intercepted via
    sys.modules stubs so no heavy dependencies need to be installed.
    load_transitions is also mocked so no real demo files are needed.
    """

    @pytest.fixture(autouse=True)
    def _stub_modules(self):
        self.stubs = _make_imitation_stubs()
        # Mock load_transitions at the il.dataset module level so that
        # the `from .dataset import load_transitions` inside train_gail()
        # resolves to our stub regardless of sys.modules cache state.
        self.mock_load = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, self.stubs):
            for mod in ("il.gail", "il.dataset", "il"):
                sys.modules.pop(mod, None)
            with patch("il.dataset.load_transitions", self.mock_load):
                yield

    def _make_env(self):
        env = MagicMock()
        env.observation_space = MagicMock()
        env.action_space = MagicMock()
        return env

    def test_calls_gail_train_with_correct_timesteps(self, tmp_path):
        # Arrange
        mock_gail_instance = MagicMock()
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = (
            mock_gail_instance
        )
        self.stubs["stable_baselines3"].PPO.return_value = MagicMock()
        from il.gail import train_gail

        # Act
        train_gail(env=self._make_env(), demos_dir="data/demos",
                   total_timesteps=7777, checkpoint_dir=str(tmp_path))

        # Assert
        mock_gail_instance.train.assert_called_once_with(7777)

    def test_saves_checkpoint_returns_zip_path(self, tmp_path):
        # Arrange
        mock_ppo = MagicMock()
        self.stubs["stable_baselines3"].PPO.return_value = mock_ppo
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = MagicMock()
        from il.gail import train_gail

        # Act
        result = train_gail(env=self._make_env(), demos_dir="data/demos",
                            total_timesteps=100, checkpoint_dir=str(tmp_path))

        # Assert
        mock_ppo.save.assert_called_once()
        assert str(tmp_path) in result
        assert result.endswith(".zip")
        assert "gail_" in result

    def test_creates_checkpoint_dir_if_missing(self, tmp_path):
        # Arrange
        self.stubs["stable_baselines3"].PPO.return_value = MagicMock()
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = MagicMock()
        new_dir = str(tmp_path / "models" / "nested")
        from il.gail import train_gail

        # Act
        train_gail(env=self._make_env(), demos_dir="data/demos",
                   total_timesteps=100, checkpoint_dir=new_dir)

        # Assert
        assert Path(new_dir).exists()

    def test_wraps_raw_env_in_dummy_vec_env(self, tmp_path):
        # Arrange — raw env (not a DummyVecEnv instance)
        self.stubs["stable_baselines3"].PPO.return_value = MagicMock()
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = MagicMock()
        FakeDVE = self.stubs["_FakeDummyVecEnv"]
        FakeDVE.dvec_calls.reset_mock()
        env = self._make_env()  # plain MagicMock, not a FakeDVE instance
        from il.gail import train_gail

        # Act
        train_gail(env=env, demos_dir="data/demos",
                   total_timesteps=100, checkpoint_dir=str(tmp_path))

        # Assert — DummyVecEnv.__init__ was called once to wrap the raw env
        FakeDVE.dvec_calls.assert_called_once()

    def test_skips_wrapping_when_already_dummy_vec_env(self, tmp_path):
        # Arrange — env is already an instance of the stubbed DummyVecEnv class
        self.stubs["stable_baselines3"].PPO.return_value = MagicMock()
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = MagicMock()
        FakeDVE = self.stubs["_FakeDummyVecEnv"]
        FakeDVE.dvec_calls.reset_mock()

        env = FakeDVE([])  # already a DummyVecEnv — dvec_calls invoked once here
        FakeDVE.dvec_calls.reset_mock()  # clear the construction call above
        from il.gail import train_gail

        # Act
        train_gail(env=env, demos_dir="data/demos",
                   total_timesteps=100, checkpoint_dir=str(tmp_path))

        # Assert — DummyVecEnv NOT constructed again (env passed through as-is)
        FakeDVE.dvec_calls.assert_not_called()

    def test_passes_device_to_ppo(self, tmp_path):
        # Arrange
        self.stubs["stable_baselines3"].PPO.return_value = MagicMock()
        self.stubs["imitation.algorithms.adversarial.gail"].GAIL.return_value = MagicMock()
        from il.gail import train_gail

        # Act
        train_gail(env=self._make_env(), demos_dir="data/demos",
                   total_timesteps=100, checkpoint_dir=str(tmp_path), device="cpu")

        # Assert
        _, ppo_kwargs = self.stubs["stable_baselines3"].PPO.call_args
        assert ppo_kwargs["device"] == "cpu"


# ---------------------------------------------------------------------------
# environment/gym_env.py — max_steps truncation
# ---------------------------------------------------------------------------

class TestGymEnvTruncation:
    """Tests for the max_steps episode-truncation feature added to Expedition33Env."""

    def _make_env(self, max_steps: int = 0):
        from vision.engine import GameState

        mock_game = MagicMock()
        mock_game.monitor = {"width": 100, "height": 100}
        mock_game.get_current_state.return_value = GameState(
            detections=[], timestamp=0.0, engine_name="MOCK"
        )
        with patch("environment.gym_env.GameInstance", return_value=mock_game):
            from environment.gym_env import Expedition33Env
            env = Expedition33Env(engine="PIXEL", step_delay=0.0, max_steps=max_steps)
        return env

    def test_truncated_true_exactly_at_max_steps(self):
        # Arrange
        env = self._make_env(max_steps=3)
        env.reset()

        # Act — first two steps must not truncate
        _, _, _, truncated, _ = env.step(0)
        assert truncated is False
        _, _, _, truncated, _ = env.step(0)
        assert truncated is False

        # Third step hits the limit
        _, _, _, truncated, _ = env.step(0)

        # Assert
        assert truncated is True

    def test_not_truncated_before_max_steps(self):
        # Arrange
        env = self._make_env(max_steps=5)
        env.reset()

        # Act — 4 steps, all below the limit
        for _ in range(4):
            _, _, _, truncated, _ = env.step(0)
            assert truncated is False

    def test_no_truncation_when_max_steps_is_zero(self):
        # Arrange — 0 means unlimited (live play mode)
        env = self._make_env(max_steps=0)
        env.reset()

        # Act
        for _ in range(30):
            _, _, _, truncated, _ = env.step(0)

        # Assert
        assert truncated is False

    def test_reset_resets_step_counter(self):
        # Arrange — truncates after 2 steps
        env = self._make_env(max_steps=2)
        env.reset()
        env.step(0)
        env.step(0)  # episode truncated here

        # Act — reset and take one step
        env.reset()
        _, _, _, truncated, _ = env.step(0)

        # Assert — counter was reset, so not truncated yet
        assert truncated is False

    def test_terminated_always_false(self):
        # Phase 1 has no terminal game-over condition
        env = self._make_env(max_steps=1)
        env.reset()

        _, _, terminated, _, _ = env.step(0)

        assert terminated is False
