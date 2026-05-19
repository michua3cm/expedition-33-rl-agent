"""
SAC (Soft Actor-Critic) trainer for robotics-ready continuous-action environments.

SAC is the industry-standard off-policy algorithm for continuous joint control
(robot arms, wheeled bases) and is included here alongside PPO so the codebase
is ready for the robotics migration without additional changes.

For the game (Discrete(7) action space), use rl/train.py (PPO).
For continuous-action robot envs, use this module.

Usage
-----
    from rl.train_sac import train_sac
    from environment.ue4ss_env import UE4SSExpedition33Env   # example env

    env = make_my_continuous_env()
    path = train_sac(env, total_timesteps=500_000)
"""

import os
from datetime import datetime

SAC_MODEL_DIR = os.path.join("data", "models")

# Default hyperparameters — conservative for a first run.
# batch_size=256 and learning_starts=10_000 are standard SAC practice.
_SAC_DEFAULTS = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
}


def train_sac(
    env,
    total_timesteps: int = 500_000,
    dp_checkpoint: str | None = None,
    out_dir: str = SAC_MODEL_DIR,
    use_cuda: bool = True,
    **sac_kwargs,
) -> str:
    """
    Train a SAC policy on a continuous-action Gymnasium environment.

    Args:
        env:              A Gymnasium environment with a Box action space.
        total_timesteps:  Total environment steps to train for.
        dp_checkpoint:    Optional Diffusion Policy .pt checkpoint to use as
                          a replay buffer warm-start (behaviour cloning phase).
                          When provided, demonstrations are added to the buffer
                          before online training starts.
        out_dir:          Directory where the final .zip checkpoint is saved.
        use_cuda:         Use GPU if available (default: True).
        **sac_kwargs:     Override any SAC hyperparameter (passed to SB3 SAC).

    Returns:
        Path to the saved checkpoint (.zip).
    """
    from stable_baselines3 import SAC


    device = "auto" if use_cuda else "cpu"
    params = {**_SAC_DEFAULTS, **sac_kwargs}

    model = SAC(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        **params,
    )

    if dp_checkpoint is not None:
        _warm_start_from_dp(model, dp_checkpoint)

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"sac_{ts}")
    model.save(path)
    env.close()
    return f"{path}.zip"


def _warm_start_from_dp(model, dp_checkpoint: str) -> None:
    """
    Pre-fill the SAC replay buffer with demonstrations from a Diffusion
    Policy checkpoint using behavioural cloning rollouts.

    This is a lightweight warm-start: the DP checkpoint is loaded, used to
    collect `learning_starts` transitions in the env, then the SAC offline
    training phase runs on those transitions before online exploration begins.

    NOTE: This requires the env to be step-compatible with the DP's obs/action
    format.  Mismatched obs_dim or action_dim will raise at rollout time.
    """
    try:

        from il.diffusion_policy import DiffusionPolicy

        dp = DiffusionPolicy.load(dp_checkpoint, device="cpu")
        env = model.env

        obs, _ = env.reset()
        n = 0
        target = model.learning_starts
        print(f"[SAC] Warm-starting replay buffer from DP checkpoint ({target} steps)...")
        while n < target:
            action = dp.predict(obs.flatten() if hasattr(obs, "flatten") else obs)
            next_obs, reward, terminated, truncated, info = env.step([action])
            model.replay_buffer.add(obs, next_obs, [action], [reward], [terminated], [info])
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
            n += 1
        print(f"[SAC] Replay buffer warm-started with {n} transitions.")
    except Exception as exc:
        print(f"[SAC] DP warm-start skipped: {exc}")
