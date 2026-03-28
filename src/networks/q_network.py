"""
q_network.py — Q-network for all DQN variants.

WHAT IT DOES:
    Takes an observation (e.g. [cart_pos, cart_vel, pole_angle, pole_vel])
    and outputs one Q-value per action: Q(s, a) for each a.

    The agent picks the action with the highest Q-value (greedy) or
    sometimes picks randomly (epsilon-greedy for exploration).

ARCHITECTURE (from proposal):
    obs → Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, num_actions)

    This is the same 2-hidden-layer MLP the proposal specifies.
    The reference assignment used tanh; we use ReLU per the proposal.

WHY PURE FUNCTIONS (not a class):
    JAX works best with pure functions + explicit parameter dicts.
    We pass `params` as a dict of weight matrices. This makes it easy to:
    - Use jax.jit for fast GPU execution
    - Use jax.value_and_grad for automatic differentiation
    - Copy params for the target network (target_params = q_params)

USED BY: Students A and B (all DQN variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_q_params(key: jax.Array,
                  obs_dim: int,
                  hidden_dim: int = 64,
                  num_actions: int = 2,
                  init_scale: float = 0.1) -> Dict[str, jnp.ndarray]:
    """Initialize Q-network weights randomly.

    Args:
        key:         JAX random key
        obs_dim:     size of observation vector (4 for CartPole, 2 for MountainCar)
        hidden_dim:  hidden layer size (64 per proposal)
        num_actions: number of actions (2 for CartPole, 3 for MountainCar)
        init_scale:  multiply random weights by this (small = stable start)

    Returns:
        Dict with keys W1, b1, W2, b2, W3, b3
    """
    k1, k2, k3 = jax.random.split(key, 3)

    params = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * init_scale,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * init_scale,
        "b2": jnp.zeros((hidden_dim,)),
        "W3": jax.random.normal(k3, (hidden_dim, num_actions)) * init_scale,
        "b3": jnp.zeros((num_actions,)),
    }

    return params


def q_forward(params: Dict[str, jnp.ndarray],
              obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: single observation → Q-values for all actions.

    Args:
        params: weight dict from init_q_params
        obs:    single observation, shape (obs_dim,)

    Returns:
        Q-values, shape (num_actions,)
    """
    x = jnp.asarray(obs, dtype=jnp.float32)

    # Layer 1: obs → hidden (ReLU)
    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])

    # Layer 2: hidden → hidden (ReLU)
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])

    # Output: hidden → Q-values (no activation — Q-values can be any real number)
    q_values = h2 @ params["W3"] + params["b3"]

    return q_values


def q_forward_batch(params: Dict[str, jnp.ndarray],
                    obs_batch: jnp.ndarray) -> jnp.ndarray:
    """Forward pass over a batch of observations.

    Args:
        params:    weight dict
        obs_batch: shape (batch_size, obs_dim)

    Returns:
        Q-values, shape (batch_size, num_actions)
    """
    return jax.vmap(lambda obs: q_forward(params, obs))(obs_batch)


def greedy_action(params: Dict[str, jnp.ndarray],
                  obs: jnp.ndarray) -> jnp.ndarray:
    """Pick the action with the highest Q-value."""
    q_values = q_forward(params, obs)
    return jnp.argmax(q_values).astype(jnp.int32)
