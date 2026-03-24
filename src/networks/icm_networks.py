"""
Intrinsic Curiosity Module (ICM) Networks — Student B + Student D shared.

Three components (Pathak et al., 2017):
  1. Feature encoder φ(s): obs → learned feature embedding
  2. Inverse model: (φ(s), φ(s')) → predicted action (trains the encoder)
  3. Forward model: (φ(s), a) → predicted φ(s') (prediction error = curiosity bonus)

Architecture: 1 hidden layer × 64 units each (lightweight, per proposal).
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


def init_icm_params(
    key: jax.random.PRNGKey,
    obs_dim: int,
    act_dim: int,
    feature_dim: int = 64,
    hidden_dim: int = 64,
) -> Dict:
    """Initialize all ICM network parameters.

    Args:
        key: JAX PRNG key.
        obs_dim: Observation dimensionality.
        act_dim: Number of discrete actions.
        feature_dim: Feature embedding dimensionality.
        hidden_dim: Hidden layer width.

    Returns:
        Dict with keys: encoder, inverse, forward.
    """
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    params = {
        # Feature encoder: obs → feature_dim
        "encoder": {
            "hidden": _init_layer(k1, obs_dim, hidden_dim),
            "output": _init_layer(k2, hidden_dim, feature_dim),
        },
        # Inverse model: (φ(s), φ(s')) concatenated → action prediction
        "inverse": {
            "hidden": _init_layer(k3, feature_dim * 2, hidden_dim),
            "output": _init_layer(k4, hidden_dim, act_dim),
        },
        # Forward model: (φ(s), one_hot(a)) concatenated → predicted φ(s')
        "forward": {
            "hidden": _init_layer(k5, feature_dim + act_dim, hidden_dim),
            "output": _init_layer(k6, hidden_dim, feature_dim),
        },
    }
    return params


def encode(params: Dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Feature encoder φ(s): obs → feature embedding.

    Args:
        params: Encoder parameters (params["encoder"]).
        obs: Observation, shape (obs_dim,).

    Returns:
        Feature embedding, shape (feature_dim,).
    """
    enc = params["encoder"]
    x = jnp.dot(obs, enc["hidden"]["W"]) + enc["hidden"]["b"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, enc["output"]["W"]) + enc["output"]["b"]
    return x


def inverse_model(params: Dict, phi_s: jnp.ndarray, phi_s_next: jnp.ndarray) -> jnp.ndarray:
    """Inverse model: (φ(s), φ(s')) → action logits.

    Args:
        params: ICM parameters.
        phi_s: Feature embedding of current state, shape (feature_dim,).
        phi_s_next: Feature embedding of next state, shape (feature_dim,).

    Returns:
        Action logits, shape (act_dim,).
    """
    inv = params["inverse"]
    x = jnp.concatenate([phi_s, phi_s_next])
    x = jnp.dot(x, inv["hidden"]["W"]) + inv["hidden"]["b"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, inv["output"]["W"]) + inv["output"]["b"]
    return x


def forward_model(
    params: Dict, phi_s: jnp.ndarray, action: int, act_dim: int
) -> jnp.ndarray:
    """Forward model: (φ(s), a) → predicted φ(s').

    Args:
        params: ICM parameters.
        phi_s: Feature embedding of current state, shape (feature_dim,).
        action: Action taken (integer).
        act_dim: Number of discrete actions (for one-hot encoding).

    Returns:
        Predicted next feature embedding, shape (feature_dim,).
    """
    fwd = params["forward"]
    action_onehot = jax.nn.one_hot(action, act_dim)
    x = jnp.concatenate([phi_s, action_onehot])
    x = jnp.dot(x, fwd["hidden"]["W"]) + fwd["hidden"]["b"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, fwd["output"]["W"]) + fwd["output"]["b"]
    return x


def icm_forward(
    params: Dict,
    obs: jnp.ndarray,
    action: int,
    next_obs: jnp.ndarray,
    act_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full ICM forward pass for a single transition.

    Args:
        params: ICM parameters.
        obs: Current observation, shape (obs_dim,).
        action: Action taken (integer).
        next_obs: Next observation, shape (obs_dim,).
        act_dim: Number of discrete actions.

    Returns:
        (phi_s, phi_s_next, predicted_phi_s_next, action_logits)
    """
    phi_s = encode(params, obs)
    phi_s_next = encode(params, next_obs)
    pred_phi_s_next = forward_model(params, phi_s, action, act_dim)
    action_logits = inverse_model(params, phi_s, phi_s_next)
    return phi_s, phi_s_next, pred_phi_s_next, action_logits


def icm_forward_batch(
    params: Dict,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    act_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full ICM forward pass for a batch of transitions."""
    return jax.vmap(
        lambda o, a, no: icm_forward(params, o, a, no, act_dim),
        in_axes=(0, 0, 0),
    )(obs_batch, action_batch, next_obs_batch)
