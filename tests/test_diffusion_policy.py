"""
Unit tests for il/diffusion_policy.py and il/dataset.py (DemoDataset).

All torch/cuda deps are mocked via sys.modules injection — tests run under
CI with only the dev group installed (pytest + numpy, no torch required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_demo_npz(path: Path, n: int = 100, obs_dim: int = 9, num_actions: int = 7) -> None:
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    acts = np.random.randint(0, num_actions, size=n).astype(np.int32)
    np.savez_compressed(str(path), observations=obs, actions=acts)


def _make_demos(tmp_path: Path, n_files: int = 2, **kw) -> str:
    d = tmp_path / "demos"
    d.mkdir()
    for i in range(n_files):
        _make_demo_npz(d / f"demo_{i:02d}.npz", **kw)
    return str(d)


# ─────────────────────────────────────────────────────────────────────────────
# TestDemoDataset — il/dataset.py DemoDataset
# ─────────────────────────────────────────────────────────────────────────────


def _torch_stub():
    """Return a minimal torch stub that supports tensor operations needed by DemoDataset."""
    import torch as _real_torch  # may not be available; skip if not
    return _real_torch


_torch_available = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)


class TestDemoDataset:
    """Tests that require real torch (skip if not installed)."""

    @_torch_available
    def test_length_correct(self, tmp_path):
        from il.dataset import DemoDataset

        demos = _make_demos(tmp_path, n_files=1, n=50, obs_dim=9)
        ds = DemoDataset(demos, obs_horizon=2, pred_horizon=16)
        # valid range: [1, 50-16] = [1, 34] → 34 samples
        assert len(ds) == 50 - 2 + 1 - 16 + 1

    @_torch_available
    def test_item_shapes(self, tmp_path):
        import torch

        from il.dataset import DemoDataset

        demos = _make_demos(tmp_path, n_files=2, n=100, obs_dim=9)
        ds = DemoDataset(demos, obs_horizon=2, pred_horizon=16)
        obs_seq, act_seq = ds[0]

        assert obs_seq.shape == (2, 9)
        assert act_seq.shape == (16, 1)
        assert obs_seq.dtype == torch.float32
        assert act_seq.dtype == torch.float32

    @_torch_available
    def test_actions_normalised(self, tmp_path):
        from il.dataset import DemoDataset

        demos = _make_demos(tmp_path, n_files=1, n=100, obs_dim=9, num_actions=7)
        ds = DemoDataset(demos, obs_horizon=2, pred_horizon=16, num_actions=7)

        for i in range(len(ds)):
            _, act_seq = ds[i]
            assert act_seq.min() >= -1.0 - 1e-6
            assert act_seq.max() <=  1.0 + 1e-6

    @_torch_available
    def test_obs_dim_attribute(self, tmp_path):
        from il.dataset import DemoDataset

        demos = _make_demos(tmp_path, n_files=1, n=50, obs_dim=9)
        ds = DemoDataset(demos, obs_horizon=2, pred_horizon=16)
        assert ds.obs_dim == 9

    @_torch_available
    def test_raises_on_empty_dir(self, tmp_path):
        from il.dataset import DemoDataset

        with pytest.raises(FileNotFoundError):
            DemoDataset(str(tmp_path / "empty"), obs_horizon=2, pred_horizon=16)


# ─────────────────────────────────────────────────────────────────────────────
# TestDiffusionPolicyMocked — il/diffusion_policy.py with torch mocked
# Tests structural correctness without running any actual torch ops.
# ─────────────────────────────────────────────────────────────────────────────


def _make_torch_stubs():
    """Build a minimal torch mock sufficient for DiffusionPolicy import."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False

    # Minimal tensor mock
    fake_tensor = MagicMock()
    fake_tensor.__getitem__ = MagicMock(return_value=fake_tensor)
    fake_tensor.shape = (1, 16, 1)
    fake_tensor.squeeze.return_value = MagicMock(cpu=MagicMock(
        return_value=MagicMock(numpy=MagicMock(return_value=np.zeros(16)))
    ))

    torch_mock.from_numpy.return_value = fake_tensor
    torch_mock.zeros.return_value = fake_tensor
    torch_mock.randn.return_value = fake_tensor
    torch_mock.cat.return_value = fake_tensor
    torch_mock.randint.return_value = fake_tensor
    torch_mock.linspace.return_value = fake_tensor
    torch_mock.arange.return_value = fake_tensor
    torch_mock.tensor.return_value = fake_tensor
    torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # nn stubs — Module base class must have .to() so _build_unet works
    class _FakeModule:
        def __init__(self, *a, **kw):
            pass

        def to(self, *a, **kw):
            return self

        def parameters(self):
            return iter([])

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def __call__(self, *a, **kw):
            return fake_tensor

        def state_dict(self):
            return {}

        def load_state_dict(self, *a, **kw):
            pass

    nn_mock = MagicMock()
    torch_mock.nn = nn_mock
    nn_mock.Module = _FakeModule
    nn_mock.functional.mse_loss.return_value = fake_tensor

    torch_mock.optim = MagicMock()
    torch_mock.save = MagicMock()
    torch_mock.load = MagicMock(return_value={
        "net": {},
        "obs_dim": 9,
        "num_actions": 7,
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_horizon": 8,
    })

    stubs = {
        "torch": torch_mock,
        "torch.nn": nn_mock,
        "torch.nn.functional": nn_mock.functional,
        "torch.optim": torch_mock.optim,
        "torch.utils": MagicMock(),
        "torch.utils.data": MagicMock(),
    }
    return stubs, torch_mock


class TestDiffusionPolicyMocked:
    @pytest.fixture(autouse=True)
    def _patch_torch(self):
        stubs, self.torch_mock = _make_torch_stubs()
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("il.diffusion_policy", None)
            sys.modules.pop("il.dataset", None)
            yield

    def test_init_sets_attributes(self):
        from il.diffusion_policy import DiffusionPolicy

        dp = DiffusionPolicy(obs_dim=9, num_actions=7)
        assert dp.obs_dim == 9
        assert dp.num_actions == 7
        assert dp.obs_horizon == 2
        assert dp.pred_horizon == 16
        assert dp.action_horizon == 8

    def test_init_with_custom_params(self):
        from il.diffusion_policy import DiffusionPolicy

        dp = DiffusionPolicy(obs_dim=30, num_actions=7, obs_horizon=4, pred_horizon=8)
        assert dp.obs_dim == 30
        assert dp.obs_horizon == 4
        assert dp.pred_horizon == 8


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadTransitions — il/dataset.py load_transitions (GAIL compat)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadTransitions:
    def _make_imitation_stubs(self):
        imitation_mock = MagicMock()
        transitions_cls = MagicMock(return_value=MagicMock())
        imitation_mock.data.types.Transitions = transitions_cls

        stubs = {
            "imitation": imitation_mock,
            "imitation.data": imitation_mock.data,
            "imitation.data.types": imitation_mock.data.types,
        }
        return stubs, transitions_cls

    def test_raises_on_missing_demos(self, tmp_path):
        stubs, _ = self._make_imitation_stubs()
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("il.dataset", None)
            from il.dataset import load_transitions

            with pytest.raises(FileNotFoundError):
                load_transitions(str(tmp_path / "empty"))

    def test_constructs_transitions_from_npz(self, tmp_path):
        _make_demo_npz(tmp_path / "d.npz", n=20, obs_dim=9)
        stubs, transitions_cls = self._make_imitation_stubs()

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("il.dataset", None)
            from il.dataset import load_transitions
            load_transitions(str(tmp_path))

        transitions_cls.assert_called_once()
        call_kwargs = transitions_cls.call_args[1]
        assert call_kwargs["obs"].shape == (20, 9)
        assert call_kwargs["acts"].shape == (20,)

    def test_concatenates_multiple_files(self, tmp_path):
        _make_demo_npz(tmp_path / "a.npz", n=30, obs_dim=9)
        _make_demo_npz(tmp_path / "b.npz", n=20, obs_dim=9)
        stubs, transitions_cls = self._make_imitation_stubs()

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("il.dataset", None)
            from il.dataset import load_transitions
            load_transitions(str(tmp_path))

        call_kwargs = transitions_cls.call_args[1]
        assert call_kwargs["obs"].shape[0] == 50
