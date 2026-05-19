"""
Demo dataset utilities for imitation learning.

Provides two interfaces:
  - load_transitions(): for GAIL — loads .npz demos into imitation.Transitions
  - DemoDataset:        for Diffusion Policy — sliding-window PyTorch Dataset

Both expect .npz files with:
  observations : float32 (T, obs_dim)
  actions      : int32   (T,)
"""

import glob
import os

import numpy as np

# ── GAIL interface ────────────────────────────────────────────────────────────


def load_transitions(demos_dir: str):
    """
    Load all .npz demo files and return an imitation.data.types.Transitions
    object suitable for GAIL training.

    Raises FileNotFoundError if no .npz files are found.
    """
    from imitation.data.types import Transitions

    paths = sorted(glob.glob(os.path.join(demos_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz demo files found in {demos_dir!r}")

    all_obs: list[np.ndarray] = []
    all_acts: list[np.ndarray] = []

    for path in paths:
        data = np.load(path)
        all_obs.append(data["observations"].astype(np.float32))
        all_acts.append(data["actions"].astype(np.int64))

    obs = np.concatenate(all_obs, axis=0)
    acts = np.concatenate(all_acts, axis=0)
    next_obs = np.concatenate([obs[1:], np.zeros_like(obs[:1])], axis=0)
    N = len(obs)

    return Transitions(
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=np.zeros(N, dtype=bool),
        infos=np.array([{}] * N),
    )


# ── Diffusion Policy interface ────────────────────────────────────────────────


class DemoDataset:
    """
    Sliding-window dataset for Diffusion Policy training.

    Each sample is a (obs_seq, action_seq) pair:
      obs_seq    : (obs_horizon, obs_dim) — recent observation context
      action_seq : (pred_horizon, 1)      — target actions, normalised to [-1, 1]

    Actions are normalised as: a_norm = (a / (num_actions - 1)) * 2 - 1
    so that action 0 maps to -1 and the maximum action maps to +1.

    Args:
        demos_dir:    Directory containing .npz demo files.
        obs_horizon:  Number of past observations fed as context (default: 2).
        pred_horizon: Number of future actions to predict (default: 16).
        num_actions:  Total number of discrete actions.  Auto-detected from
                      data if not specified.
    """

    def __init__(
        self,
        demos_dir: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_actions: int | None = None,
    ):
        import torch  # noqa: F401 — verify torch is available before doing work

        paths = sorted(glob.glob(os.path.join(demos_dir, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"No .npz demo files found in {demos_dir!r}")

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        for path in paths:
            data = np.load(path)
            all_obs.append(data["observations"].astype(np.float32))
            all_acts.append(data["actions"].astype(np.int64))

        obs = np.concatenate(all_obs, axis=0)    # (N, obs_dim)
        acts = np.concatenate(all_acts, axis=0)  # (N,)

        n_actions = num_actions if num_actions is not None else int(acts.max()) + 1
        acts_norm = (acts.astype(np.float32) / (n_actions - 1)) * 2 - 1  # → [-1, 1]

        self._obs = torch.from_numpy(obs)                  # (N, obs_dim)
        self._acts = torch.from_numpy(acts_norm[:, None])  # (N, 1)
        self.obs_dim: int = obs.shape[1]
        self.action_dim: int = 1

        # Valid start indices: need obs_horizon history and pred_horizon future
        pad = obs_horizon - 1
        self._indices = list(range(pad, len(obs) - pred_horizon + 1))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        start = self._indices[idx]
        obs_seq = self._obs[start - self.obs_horizon + 1 : start + 1]  # (obs_h, obs_dim)
        act_seq = self._acts[start : start + self.pred_horizon]          # (pred_h, 1)
        return obs_seq, act_seq
