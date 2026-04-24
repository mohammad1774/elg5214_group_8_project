"""
ppo_entropy_agent.py — PPO + entropy regularization agent.

Student C (Md Mosarraf)
"""

import jax
import jax.numpy as jnp

from src.networks.policy_network import policy_forward


class PPOEntropyAgent:
    """PPO agent for entropy-regularized training."""

    def __init__(self, params: dict, num_actions: int = 2):
        self.params = params
        self.num_actions = num_actions

    def act(self, key: jax.Array, obs: jnp.ndarray) -> jnp.ndarray:
        logits = policy_forward(self.params, obs)
        action = jax.random.categorical(key, logits)
        return action.astype(jnp.int32)

    def greedy_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        logits = policy_forward(self.params, obs)
        return jnp.argmax(logits).astype(jnp.int32)

    def log_prob(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        logits = policy_forward(self.params, obs)
        log_probs = jax.nn.log_softmax(logits)
        return log_probs[action]

    def entropy(self, obs: jnp.ndarray) -> jnp.ndarray:
        logits = policy_forward(self.params, obs)
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs)

    def update_params(self, new_params: dict) -> None:
        self.params = new_params
