"""
Q-Network for DQN variants.
Pure JAX functional style: init_q_params(key, obs_dim, act_dim) → params dict
                           q_forward(params, obs) → Q-values (act_dim,)
                           q_forward_batch(params, obs_batch) → Q-values (batch, act_dim)

Architecture: 2 hidden layers × 64 units, ReLU activations (per proposal spec).
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple


def _init_layer(key: jax.random.PRNGKey, in_dim: int, out_dim: int) -> Dict:
    """Xavier uniform init for a single dense layer."""
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    w_key, b_key = jax.random.split(key)
    W = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros(out_dim)
    return {"W": W, "b": b}


def init_q_params(
    key: jax.random.PRNGKey,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int = 64,
) -> Dict:
    """Initialize Q-network parameters.

    Args:
        key: JAX PRNG key.
        obs_dim: Observation space dimensionality.
        act_dim: Number of discrete actions.
        hidden_dim: Hidden layer width (default 64 per proposal).

    Returns:
        Nested dict of parameters: {layer1, layer2, output}.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "layer1": _init_layer(k1, obs_dim, hidden_dim),
        "layer2": _init_layer(k2, hidden_dim, hidden_dim),
        "output": _init_layer(k3, hidden_dim, act_dim),
    }


def q_forward(params: Dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for a single observation.

    Args:
        params: Q-network parameters from init_q_params.
        obs: Single observation, shape (obs_dim,).

    Returns:
        Q-values for each action, shape (act_dim,).
    """
    x = obs
    # Hidden layer 1
    x = jnp.dot(x, params["layer1"]["W"]) + params["layer1"]["b"]
    x = jax.nn.relu(x)
    # Hidden layer 2
    x = jnp.dot(x, params["layer2"]["W"]) + params["layer2"]["b"]
    x = jax.nn.relu(x)
    # Output layer (no activation — raw Q-values)
    x = jnp.dot(x, params["output"]["W"]) + params["output"]["b"]
    return x


def q_forward_batch(params: Dict, obs_batch: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for a batch of observations.

    Args:
        params: Q-network parameters.
        obs_batch: Batch of observations, shape (batch_size, obs_dim).

    Returns:
        Q-values, shape (batch_size, act_dim).
    """
    return jax.vmap(q_forward, in_axes=(None, 0))(params, obs_batch)
