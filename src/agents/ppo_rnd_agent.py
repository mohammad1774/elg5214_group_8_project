"""
ppo_rnd_agent.py — PPO + Random Network Distillation agent.

WHAT IT DOES:
    Wraps policy network + value network + RND networks (target + predictor)
    into one agent. The RND logic lives in the training loop; the agent
    just exposes the helpers the loop needs.

ACTION SELECTION:
    Stochastic: jax.random.categorical(key, logits)
    Greedy:     argmax over logits

USED BY: src/training/train_ppo_rnd.py
"""

import jax
import jax.numpy as jnp

from src.networks.policy_network import (
    init_policy_params,
    policy_forward,
    log_prob as policy_log_prob,
    entropy as policy_entropy,
)
from src.networks.value_network import init_value_params, value_forward
from src.exploration.rnd import compute_rnd_reward


class PPORNDAgent:
    """Agent that carries policy, value, AND RND parameters for PPO+RND."""

    def __init__(self, policy_params, value_params, rnd_params=None):
        self.policy_params = policy_params
        self.value_params = value_params
        self.rnd_params = rnd_params   # {"target": ..., "predictor": ...}

    # ── action selection ──────────────────────────────────────

    def act(self, key, obs):
        """Stochastic action: sample from the categorical policy."""
        logits = policy_forward(self.policy_params, obs)
        return jax.random.categorical(key, logits)

    def greedy_action(self, obs):
        """Deterministic action: argmax over policy logits."""
        logits = policy_forward(self.policy_params, obs)
        return jnp.argmax(logits)

    # ── helpers consumed by the training loop ─────────────────

    def log_prob(self, obs, action):
        """Log-probability of *action* under the current policy."""
        return policy_log_prob(self.policy_params, obs, action)

    def entropy(self, obs):
        """Policy entropy H(pi) at the given observation."""
        return policy_entropy(self.policy_params, obs)

    def get_value(self, obs):
        """State value V(s) from the value network."""
        return value_forward(self.value_params, obs)

    def intrinsic_reward(self, obs):
        """RND intrinsic reward = MSE(target(s), predictor(s))."""
        if self.rnd_params is None:
            return jnp.float32(0.0)
        return compute_rnd_reward(self.rnd_params, obs)
