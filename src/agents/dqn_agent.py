"""
dqn_agent.py — DQN baseline agent.

HOW DQN ACTION SELECTION WORKS:
    Training: epsilon-greedy
        With probability epsilon → pick a random action (explore)
        With probability 1-epsilon → pick argmax Q(s,a) (exploit)
        epsilon starts high (1.0 = fully random) and decays to low (0.01)

    Evaluation: greedy
        Always pick argmax Q(s,a) — no randomness

    This is fundamentally different from PPO, which samples from a
    learned probability distribution. DQN's exploration is forced
    (epsilon), while PPO's exploration is natural (stochastic policy).

WHY THIS IS IN THE SHARED FOUNDATION:
    Student B owns the full implementation and training loop, but
    this agent class is needed by:
    - src/evaluate/evaluate_dqn.py (shared by all DQN variants)
    - Student A's DQN+entropy and DQN+RND as a reference pattern
    - Student B's DQN+ICM as the base to extend

    The DQN+entropy/RND/ICM agents can either:
    (a) Import and use this class directly (if exploration only changes the reward/loss)
    (b) Copy and extend this class (if exploration changes action selection)

USED BY: Students A and B (all DQN variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.networks.q_network import q_forward


class DQNAgent:
    """Agent that uses a Q-network to select actions.

    Supports:
        - Greedy action selection (for evaluation)
        - Epsilon-greedy action selection (for training)
    """

    def __init__(self, params: Dict):
        self.params = params

    def greedy_action(self, obs: jnp.ndarray) -> int:
        """Pick the action with the highest Q-value."""
        q_values = q_forward(self.params, obs)
        return int(jnp.argmax(q_values))

    def act(self, key: jax.Array, obs: jnp.ndarray, epsilon: float = 0.0) -> jnp.ndarray:
        """Select action using epsilon-greedy strategy.

        Args:
            key:     JAX random key
            obs:     current observation
            epsilon: exploration probability (0 = fully greedy, 1 = fully random)

        Returns:
            Selected action (int32 scalar)
        """
        q_values = q_forward(self.params, obs)
        greedy_act = jnp.argmax(q_values)

        key1, key2 = jax.random.split(key)

        random_act = jax.random.randint(
            key1,
            shape=(),
            minval=0,
            maxval=q_values.shape[0],
        )

        should_explore = jax.random.uniform(key2) < epsilon
        action = jnp.where(should_explore, random_act, greedy_act)

        return action

    def update_params(self, new_params: dict) -> None:
        """Replace Q-network weights after a gradient update."""
        self.params = new_params
