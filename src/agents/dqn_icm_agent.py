"""
DQN + ICM Agent — Student B.

Extends vanilla DQN with curiosity-driven exploration from the Intrinsic Curiosity Module.
The ICM computes an intrinsic reward based on forward model prediction error:
    total_reward = extrinsic_reward + η * ||f_fwd(φ(s), a) - φ(s')||²

This bonus is added to the reward stored in the replay buffer, so the DQN
learns Q-values that incorporate both extrinsic and curiosity-driven signals.

Key design decisions:
  - ICM networks are updated on the same mini-batches as the Q-network
  - Intrinsic reward is computed at transition time and stored in the buffer
  - η (curiosity scale) controls the intrinsic reward magnitude
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, NamedTuple

from src.agents.dqn_agent import DQNConfig


class DQNICMConfig(NamedTuple):
    """DQN + ICM hyperparameters."""
    # DQN params
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 100
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 5000
    # ICM params
    icm_lr: float = 1e-3
    eta: float = 1.0           # Curiosity reward scale (from proposal: {0.1, 1.0})
    beta_icm: float = 0.2     # Forward vs inverse loss weight
    feature_dim: int = 64
    icm_hidden_dim: int = 64

    def to_dqn_config(self) -> DQNConfig:
        """Extract the base DQN config."""
        return DQNConfig(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            gamma=self.gamma,
            target_update_freq=self.target_update_freq,
            grad_clip_norm=self.grad_clip_norm,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )
