"""
dqn_entropy_agent.py — DQN + Entropy Regularization agent.

Same as DQN baseline agent — entropy regularization doesn't change
action selection, it changes the LOSS function in the training loop.

The agent still uses epsilon-greedy. The entropy bonus is computed
from Q-values inside train_dqn_entropy.py and added to the loss.

Student A (Mohammad)
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.networks.q_network import q_forward


class DQNEntropyAgent:
    """DQN agent — identical to baseline. Entropy is in the loss, not here."""

    def __init__(self, params: Dict):
        self.params = params

    def greedy_action(self, obs: jnp.ndarray) -> int:
        q_values = q_forward(self.params, obs)
        return int(jnp.argmax(q_values))

    def act(self, key: jax.Array, obs: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        q_values = q_forward(self.params, obs)
        greedy_act = jnp.argmax(q_values)

        key1, key2 = jax.random.split(key)
        random_act = jax.random.randint(key1, shape=(), minval=0, maxval=q_values.shape[0])
        should_explore = jax.random.uniform(key2) < epsilon
        action = jnp.where(should_explore, random_act, greedy_act)

        return action

    def update_params(self, new_params: dict) -> None:
        self.params = new_params
