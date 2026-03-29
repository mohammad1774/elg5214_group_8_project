"""
dqn_rnd_agent.py — DQN + RND agent.

Same epsilon-greedy action selection as baseline DQN.
RND doesn't change the agent — it changes the REWARD in the training loop.

The training loop (train_dqn_rnd.py) computes:
    r_total = r_extrinsic + beta * r_intrinsic
and stores r_total in the replay buffer. The agent sees augmented rewards
but selects actions the same way as vanilla DQN.

Student A (Mohammad)
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.networks.q_network import q_forward


class DQNRndAgent:
    """DQN agent — identical to baseline. RND augments the reward, not the agent."""

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
