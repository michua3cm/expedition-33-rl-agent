"""Load human-recorded .npz demo files into imitation.data.types.Transitions."""

from __future__ import annotations

import glob
import os

import numpy as np


def load_transitions(demos_dir: str):
    """
    Load all .npz demo files from *demos_dir* and return an
    ``imitation.data.types.Transitions`` object suitable for GAIL training.

    Each .npz is expected to contain:
      - ``obs``     : float32 array of shape (T, OBS_DIM)
      - ``actions`` : int32/int64 array of shape (T,)

    Returns a Transitions object with N = sum of all timesteps across files.
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
        all_obs.append(data["obs"].astype(np.float32))
        all_acts.append(data["actions"].astype(np.int64))

    obs = np.concatenate(all_obs, axis=0)    # (N, OBS_DIM)
    acts = np.concatenate(all_acts, axis=0)  # (N,)

    # next_obs: shift by 1; the final step's next_obs is a zero-padded row
    next_obs = np.concatenate([obs[1:], np.zeros_like(obs[:1])], axis=0)

    N = len(obs)
    dones = np.zeros(N, dtype=bool)
    infos = np.array([{}] * N)

    return Transitions(
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=dones,
        infos=infos,
    )
