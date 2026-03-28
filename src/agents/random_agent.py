"""
random_agent.py — Random action baseline agent.

WHY WE NEED THIS:
    The proposal says "A random-action agent serves as a lower bound."
    If our trained agents can't beat random, something is broken.
    If RND/ICM only match random under sparse rewards, entropy alone
    might be sufficient.

    This agent picks a uniformly random action every step, regardless
    of the observation. It's the simplest possible policy.

USED BY: src/test/test_random_agent.py, src/evaluate/evaluate_random.py
"""

import jax
import jax.numpy as jnp


class RandomAgent:
    def __init__(self, n_actions: int = 2):
        self.n_actions = n_actions

    def act(self, key: jax.Array, obs: jnp.ndarray) -> jnp.ndarray:
        """Pick a random action uniformly."""
        del obs  # random agent ignores the observation
        action = jax.random.randint(
            key,
            shape=(),
            minval=0,
            maxval=self.n_actions,
        )
        return action

    def greedy_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Random agent has no greedy policy — always returns action 0."""
        return jnp.int32(0)
