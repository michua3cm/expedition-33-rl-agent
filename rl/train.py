import os
from datetime import datetime

PPO_MODEL_DIR = os.path.join("data", "models")

# Architecture for fresh-start PPO (no GAIL checkpoint).
# Must match the GAIL policy's hidden layers so fine-tuning is consistent.
GAIL_NET_ARCH = {"pi": [128, 64], "vf": [128, 64]}


def train(
    gail_checkpoint: str | None = None,
    engine: str = "PIXEL",
    total_timesteps: int = 100_000,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    out_dir: str = PPO_MODEL_DIR,
    use_cuda: bool = True,
) -> str:
    """
    Train a PPO policy on Expedition33Env, optionally warm-started from GAIL.

    Returns:
        Path to the saved PPO checkpoint (.zip).
    """
    from stable_baselines3 import PPO

    from environment.gym_env import Expedition33Env
    from rl.policy import load_gail_weights

    device = "auto" if use_cuda else "cpu"
    env = Expedition33Env(engine=engine)

    if gail_checkpoint is not None:
        model = load_gail_weights(env, gail_checkpoint, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": GAIL_NET_ARCH},
            device=device,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(out_dir, f"ppo_{timestamp}")
    model.save(checkpoint_path)
    env.close()
    return f"{checkpoint_path}.zip"
