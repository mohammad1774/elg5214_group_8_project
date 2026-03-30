"""
ppo_entropy_agent.py — PPO + Entropy Regularization agent.

WHAT IT DOES:
    Wraps both a policy network (action probabilities) and a value network
    (state value baseline) into a single agent interface.  The entropy
    regularization lives in the *training loss*, not here — the agent just
    exposes the helper methods the training loop needs (log_prob, entropy,
    get_value).

ACTION SELECTION:
    Stochastic: sample from the categorical distribution defined by the
    policy logits — jax.random.categorical(key, logits).
    Greedy:     argmax over logits (deterministic, for evaluation).

USED BY: src/training/train_ppo_entropy.py
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


class PPOEntropyAgent:
    """Agent that carries both policy and value parameters for PPO + Entropy."""

    def __init__(self, policy_params, value_params):
        """
        Args:
            policy_params: dict of policy network weights
            value_params:  dict of value network weights
        """
        self.policy_params = policy_params
        self.value_params = value_params

    # ── action selection ──────────────────────────────────────

    def act(self, key, obs):
        """Stochastic action: sample from the categorical policy."""
        logits = policy_forward(self.policy_params, obs)
        action = jax.random.categorical(key, logits)
        return action

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
