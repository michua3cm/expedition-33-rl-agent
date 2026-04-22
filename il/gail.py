"""GAIL trainer for Expedition 33."""

import os
from datetime import datetime


def train_gail(
    env,
    demos_dir: str,
    total_timesteps: int = 200_000,
    checkpoint_dir: str = "data/models",
    n_steps: int = 2048,
    batch_size: int = 64,
    device: str = "auto",
) -> str:
    """
    Train a GAIL agent and save the resulting policy checkpoint.

    Args:
        env:              A wrapped Gymnasium environment (DummyVecEnv).
        demos_dir:        Path to the directory containing .npz demo files.
        total_timesteps:  Total environment steps to train for.
        checkpoint_dir:   Directory where the final .zip checkpoint is saved.
        n_steps:          PPO rollout steps per update.
        batch_size:       Minibatch size for discriminator + PPO updates.
        device:           PyTorch device string ('auto', 'cpu', 'cuda').

    Returns:
        Path to the saved checkpoint file.
    """
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .dataset import load_transitions

    transitions = load_transitions(demos_dir)

    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])

    ppo = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        device=device,
        policy_kwargs={"net_arch": {"pi": [128, 64], "vf": [128, 64]}},
    )

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=batch_size,
        gen_algo=ppo,
        reward_net=reward_net,
    )

    gail_trainer.train(total_timesteps)

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"gail_{timestamp}.zip")
    ppo.save(checkpoint_path)

    return checkpoint_path
