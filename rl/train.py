"""
PPO training loop for Phase 1 RL fine-tuning.

What this file does
-------------------
1. Creates Expedition33Env (live game environment).
2. Creates an SB3 PPO model with the same actor architecture as BCPolicy
   (net_arch=BC_NET_ARCH, so weight transfer is possible).
3. Optionally warm-starts the actor from a BC checkpoint via load_bc_weights().
4. Calls model.learn() for the requested number of timesteps.
5. Saves the final policy to data/models/ppo_<timestamp>.zip and returns the path.

Why warm-start from BC?
-----------------------
PPO started from random weights will spend a long time learning the basics
(dodge when told to dodge, attack during battle).  Starting from a competent
BC policy means RL only needs to correct the distribution-shift mistakes BC
makes — reducing training time significantly.

The critic is left random on purpose.  BC only trained an actor; there is no
sensible value function to copy.  SB3 initialises it from scratch, which is
correct.

PPO hyperparameters — Phase 1 defaults
---------------------------------------
n_steps=2048      Roll out 2 048 steps of experience before each update.
batch_size=64     Mini-batch size for PPO gradient updates.
n_epochs=10       Number of passes over the rollout buffer per update.
learning_rate=3e-4  Standard PPO learning rate.
gamma=0.99        Discount factor — rewards up to ~100 steps ahead matter.
gae_lambda=0.95   GAE lambda — trades off bias vs variance in advantage estimates.
clip_range=0.2    PPO clipping — prevents too-large policy updates.
ent_coef=0.01     Small entropy bonus to discourage premature collapse to
                  one action (e.g., always NOOP).

Usage
-----
    # From the project root (requires rl dependency group):
    python -m rl.train

    # With BC warm-start:
    python -m rl.train --bc-checkpoint data/models/bc_best.pt --timesteps 200000

CLI arguments
-------------
    --bc-checkpoint  str    Path to a BC checkpoint for actor warm-start (optional)
    --timesteps      int    Total environment steps to train for  (default: 100_000)
    --engine         str    Vision engine name                    (default: PIXEL)
    --n-steps        int    PPO rollout steps per update          (default: 2048)
    --batch-size     int    Mini-batch size                       (default: 64)
    --n-epochs       int    PPO gradient epochs per rollout       (default: 10)
    --lr             float  Learning rate                         (default: 3e-4)
    --gamma          float  Discount factor                       (default: 0.99)
    --gae-lambda     float  GAE lambda                            (default: 0.95)
    --clip-range     float  PPO clip range                        (default: 0.2)
    --ent-coef       float  Entropy coefficient                   (default: 0.01)
    --out-dir        str    Directory for saved checkpoints       (default: data/models)
    --no-cuda               Disable GPU even if available
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO

from environment.gym_env import Expedition33Env
from rl.policy import BC_NET_ARCH, load_bc_weights

# Default output directory for PPO checkpoints
PPO_MODEL_DIR = os.path.join("data", "models")


def train(
    bc_checkpoint: str | None    = None,
    engine: str                  = "PIXEL",
    total_timesteps: int         = 100_000,
    n_steps: int                 = 2048,
    batch_size: int              = 64,
    n_epochs: int                = 10,
    lr: float                    = 3e-4,
    gamma: float                 = 0.99,
    gae_lambda: float            = 0.95,
    clip_range: float            = 0.2,
    ent_coef: float              = 0.01,
    out_dir: str                 = PPO_MODEL_DIR,
    use_cuda: bool               = True,
) -> str:
    """
    Train a PPO policy on Expedition33Env, optionally warm-started from BC.

    Args:
        bc_checkpoint:   Path to a BCPolicy checkpoint for actor warm-start.
                         Pass None to train from random initialisation.
        engine:          Vision engine name ('PIXEL', 'SIFT', 'ORB', 'YOLO').
        total_timesteps: Total number of environment steps to collect.
        n_steps:         Steps per PPO rollout buffer fill.
        batch_size:      Mini-batch size for PPO gradient updates.
        n_epochs:        Gradient epochs per rollout update.
        lr:              Adam learning rate.
        gamma:           Discount factor for future rewards.
        gae_lambda:      GAE lambda for advantage estimation.
        clip_range:      PPO objective clipping range.
        ent_coef:        Entropy regularisation coefficient.
        out_dir:         Directory to write the final checkpoint into.
        use_cuda:        If True, use GPU when available.

    Returns:
        Path to the saved PPO checkpoint (.zip).
    """
    device = "cuda" if use_cuda else "cpu"
    print(f"[RL Train] Device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Environment                                                       #
    # ------------------------------------------------------------------ #
    env = Expedition33Env(engine=engine)
    print(f"[RL Train] Environment: Expedition33Env (engine={engine})")

    # ------------------------------------------------------------------ #
    # 2. Model                                                             #
    # ------------------------------------------------------------------ #
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": BC_NET_ARCH},
        device=device,
        verbose=1,
    )
    print("[RL Train] PPO model created.")

    # ------------------------------------------------------------------ #
    # 3. Optional BC warm-start                                            #
    # ------------------------------------------------------------------ #
    if bc_checkpoint is not None:
        load_bc_weights(model, bc_checkpoint)
    else:
        print("[RL Train] No BC checkpoint — starting from random initialisation.")

    # ------------------------------------------------------------------ #
    # 4. Training                                                          #
    # ------------------------------------------------------------------ #
    print(f"[RL Train] Learning for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps)
    print("[RL Train] Learning complete.")

    # ------------------------------------------------------------------ #
    # 5. Save                                                              #
    # ------------------------------------------------------------------ #
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(out_dir, f"ppo_{timestamp}")
    model.save(checkpoint_path)
    print(f"[RL Train] Checkpoint saved → {checkpoint_path}.zip")

    env.close()
    return f"{checkpoint_path}.zip"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PPO policy on Expedition 33 (Phase 1)."
    )
    p.add_argument("--bc-checkpoint", type=str,   default=None,
                   help="BC actor warm-start checkpoint (optional).")
    p.add_argument("--timesteps",     type=int,   default=100_000)
    p.add_argument("--engine",        type=str,   default="PIXEL")
    p.add_argument("--n-steps",       type=int,   default=2048)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--n-epochs",      type=int,   default=10)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-range",    type=float, default=0.2)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--out-dir",       type=str,   default=PPO_MODEL_DIR)
    p.add_argument("--no-cuda",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        bc_checkpoint   = args.bc_checkpoint,
        engine          = args.engine,
        total_timesteps = args.timesteps,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        n_epochs        = args.n_epochs,
        lr              = args.lr,
        gamma           = args.gamma,
        gae_lambda      = args.gae_lambda,
        clip_range      = args.clip_range,
        ent_coef        = args.ent_coef,
        out_dir         = args.out_dir,
        use_cuda        = not args.no_cuda,
    )
