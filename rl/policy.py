def load_gail_weights(env, checkpoint: str, device: str = "auto"):
    """Load a GAIL .zip checkpoint (SB3 PPO) and return it warm-started on env."""
    from stable_baselines3 import PPO
    return PPO.load(checkpoint, env=env, device=device)
