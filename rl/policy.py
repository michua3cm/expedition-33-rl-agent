"""
BC → PPO actor weight transfer.

Why this file exists
--------------------
Stable-Baselines3's PPO uses an ActorCritic architecture:

    Shared MLP extractor
    ├── policy_net  (actor path)  → action_net  → action logits
    └── value_net   (critic path) → value_net   → V(s)

Our BCPolicy is a pure actor (no value head):

    net: Linear(30,128) → ReLU → Linear(128,64) → ReLU → Linear(64,7)

The actor weights are identical in structure to SB3's policy path when we
configure PPO with net_arch=dict(pi=[128,64], vf=[128,64]).  This file
provides a single function that copies those trained BC weights into the
freshly initialised PPO policy so RL fine-tuning starts from a competent
policy instead of random noise.

Only the actor (policy_net + action_net) is warm-started.  The critic
(value_net) is intentionally left random — it has never been trained, so
bootstrapping it from BC weights would be meaningless.

Mapping
-------
    BC net[0]  (Linear 30→128)  →  policy.mlp_extractor.policy_net[0]
    BC net[2]  (Linear 128→64)  →  policy.mlp_extractor.policy_net[2]
    BC net[4]  (Linear 64→7)    →  policy.action_net

Usage
-----
    from stable_baselines3 import PPO
    from rl.policy import BC_NET_ARCH, load_bc_weights

    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": BC_NET_ARCH})
    load_bc_weights(model, "data/models/bc_best.pt")
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

from stable_baselines3 import PPO

from imitation.policy import BCPolicy

# Matches BCPolicy hidden layers exactly.
# Pass this as policy_kwargs={"net_arch": BC_NET_ARCH} when creating PPO
# so the actor path has the same shape as our trained BCPolicy.
BC_NET_ARCH: dict[str, list[int]] = {"pi": [128, 64], "vf": [128, 64]}


def load_bc_weights(model: PPO, checkpoint: str) -> None:
    """
    Copy BCPolicy actor weights into the actor path of an SB3 PPO model.

    The PPO model must have been created with BC_NET_ARCH so the layer
    dimensions match.  The critic (value) path is left unchanged.

    Args:
        model:       An SB3 PPO instance whose policy_kwargs used BC_NET_ARCH.
        checkpoint:  Path to a BCPolicy checkpoint (bc_best.pt).
    """
    bc  = BCPolicy.load(checkpoint)
    pol = model.policy

    # ── Actor hidden layers ────────────────────────────────────────────────
    # policy_net is nn.Sequential:
    #   [0] Linear(30, 128)   ← bc.net[0]
    #   [1] ReLU
    #   [2] Linear(128, 64)   ← bc.net[2]
    #   [3] ReLU
    pol.mlp_extractor.policy_net[0].load_state_dict(bc.net[0].state_dict())
    pol.mlp_extractor.policy_net[2].load_state_dict(bc.net[2].state_dict())

    # ── Action head ────────────────────────────────────────────────────────
    # action_net: Linear(64, 7)   ← bc.net[4]
    pol.action_net.load_state_dict(bc.net[4].state_dict())

    print(f"[RL Policy] BC weights loaded from: {checkpoint}")
    print("[RL Policy] Actor warm-started. Critic initialised randomly.")
