def load_gail_weights(env, checkpoint: str, device: str = "auto"):
    """Load a GAIL .zip checkpoint (SB3 PPO) and return it warm-started on env."""
    from stable_baselines3 import PPO
    return PPO.load(checkpoint, env=env, device=device)


def load_sac_checkpoint(env, checkpoint: str, device: str = "auto"):
    """Load a SAC .zip checkpoint and return it ready to continue training on env."""
    from stable_baselines3 import SAC
    return SAC.load(checkpoint, env=env, device=device)
