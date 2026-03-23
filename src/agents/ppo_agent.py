"""
ppo_agent.py — PPO baseline agent.

HOW THIS DIFFERS FROM DQN AGENT (from Assignment 2):

    DQN Agent:
        - Stores Q-network params
        - act() uses epsilon-greedy: with prob epsilon pick random, else argmax Q
        - greedy_action() always picks argmax Q
        - Learns by minimizing TD error on (s, a, r, s') transitions

    PPO Agent:
        - Stores POLICY network params (not Q-network)
        - act() SAMPLES from the policy distribution: action ~ pi(a|s)
        - greedy_action() picks argmax of policy logits (deterministic)
        - Also provides log_prob() and entropy() — needed for PPO's loss function
        - Learns by maximizing a clipped surrogate objective

    Key insight: DQN explores via epsilon (forced randomness).
    PPO explores via stochastic policy (natural randomness in the distribution).

WHY THE AGENT CLASS EXISTS (vs just calling network functions directly):
    The rollout.py scan function calls agent.act(key, obs). By wrapping
    the network in an agent class, the same rollout code works for all agents.
    The training loop creates an agent, passes it to the rollout, and gets
    back trajectories. The agent is a thin wrapper — all the math lives
    in the network functions.

USED BY: Student D (PPO baseline), and as reference for PPO+entropy/RND/ICM
"""

import jax
import jax.numpy as jnp

from src.networks.policy_network import policy_forward


class PPOAgent:
    """Agent that samples actions from a categorical policy.

    Compatible with run_one_episode_scan_simple() from rollout.py.
    """

    def __init__(self, params: dict, num_actions: int = 2):
        self.params = params
        self.num_actions = num_actions

    def act(self, key: jax.Array, obs: jnp.ndarray) -> jnp.ndarray:
        """Sample an action from the policy distribution.

        Unlike DQN's epsilon-greedy, this is ALWAYS stochastic during training.
        The policy itself learns to be more or less exploratory over time.
        """
        logits = policy_forward(self.params, obs)
        action = jax.random.categorical(key, logits)
        return action.astype(jnp.int32)

    def greedy_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Pick the most likely action (for evaluation only)."""
        logits = policy_forward(self.params, obs)
        return jnp.argmax(logits).astype(jnp.int32)

    def get_logits(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Raw logits — needed by training loop for loss computation."""
        return policy_forward(self.params, obs)

    def get_action_probs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Action probabilities — useful for debugging/logging."""
        logits = policy_forward(self.params, obs)
        return jax.nn.softmax(logits)

    def log_prob(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Log-probability of a specific action.

        This is THE key quantity for PPO's loss:
            ratio = exp(new_log_prob - old_log_prob)
            loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        """
        logits = policy_forward(self.params, obs)
        log_probs = jax.nn.log_softmax(logits)
        return log_probs[action]

    def entropy(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Policy entropy H(pi) = -sum(pi * log(pi)).

        Logged every episode to track exploration behavior.
        PPO+entropy adds alpha * H(pi) to the objective to encourage exploration.
        """
        logits = policy_forward(self.params, obs)
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs)

    def update_params(self, new_params: dict) -> None:
        """Replace network weights after a gradient update."""
        self.params = new_params
