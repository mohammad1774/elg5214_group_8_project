"""
policy_network.py — Categorical policy network for all PPO variants.

WHAT IT DOES:
    Takes an observation and outputs LOGITS (unnormalized scores) for each action.
    To get actual probabilities, apply softmax: probs = softmax(logits).
    To sample an action: action = jax.random.categorical(key, logits).

HOW IT DIFFERS FROM q_network.py:
    - Q-network outputs values (how much reward per action)
    - Policy network outputs probabilities (how likely each action is)

    The Q-network picks the BEST action (argmax).
    The policy network SAMPLES from a distribution (stochastic).
    This stochasticity is what makes PPO on-policy — it explores naturally.

WHY LOGITS INSTEAD OF PROBABILITIES:
    Numerically safer. softmax can overflow/underflow with raw probs.
    JAX provides log_softmax and categorical that work directly with logits.

EXTRA FUNCTIONS:
    - log_prob(params, obs, action): needed for the PPO loss function
      (how likely was the action I took under my current policy?)
    - entropy(params, obs): H(pi) = -sum(p * log(p))
      (how uncertain is my policy? used for entropy regularization)

ARCHITECTURE: same 2×64 ReLU MLP as q_network

USED BY: Students C and D (all PPO variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_policy_params(key, obs_dim: int, hidden_dim: int = 64, num_actions: int = 2):
    """Initialize policy network weights.

    Same structure as init_q_params — the only difference is how
    the output is interpreted (logits vs Q-values).
    """
    k1, k2, k3 = jax.random.split(key, 3)

    params = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1,
        "b2": jnp.zeros((hidden_dim,)),
        "W3": jax.random.normal(k3, (hidden_dim, num_actions)) * 0.1,
        "b3": jnp.zeros((num_actions,)),
    }

    return params


def policy_forward(params, obs):
    """Forward pass: observation → logits.

    These logits are NOT probabilities yet. Apply softmax to get probs,
    or use jax.random.categorical(key, logits) to sample an action.
    """
    x = jnp.asarray(obs, dtype=jnp.float32)

    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])

    logits = h2 @ params["W3"] + params["b3"]

    return logits


def action_probs(params, obs):
    """Get action probabilities (softmax over logits)."""
    logits = policy_forward(params, obs)
    return jax.nn.softmax(logits)


def log_prob(params: dict, obs: jnp.ndarray, action: int) -> jnp.ndarray:
    """Log-probability of a specific action under the current policy.

    This is the key quantity for the PPO loss:
        ratio = exp(log_prob_new - log_prob_old)
        loss = -min(ratio * advantage, clip(ratio) * advantage)
    """
    logits = policy_forward(params, obs)
    log_probs = jax.nn.log_softmax(logits)  # numerically stable
    return log_probs[action]


def entropy(params: Dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    """Policy entropy: H(pi) = -sum(pi * log(pi)).

    High entropy = uncertain policy (explores more).
    Low entropy = confident policy (exploits more).

    Used for:
        - PPO+entropy: added as a bonus to encourage exploration
        - Logging: track how policy confidence evolves during training
    """
    logits = policy_forward(params, obs)
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs)
