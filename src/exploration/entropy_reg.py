"""
entropy_reg.py — Entropy regularization utility.

Provides the entropy_bonus function used by PPO + Entropy training.
The bonus is simply alpha * H(pi), which is subtracted from the loss
to *encourage* higher-entropy (more exploratory) policies.

USED BY: src/training/train_ppo_entropy.py
"""

import jax.numpy as jnp


def entropy_bonus(entropy_value: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Scale policy entropy by the entropy coefficient.

    Args:
        entropy_value: scalar H(pi) — the policy entropy
        alpha:         entropy coefficient (e.g. 0.01 or 0.05)

    Returns:
        alpha * entropy_value  (added to reward / subtracted from loss
        to encourage exploration)
    """
    return alpha * entropy_value
