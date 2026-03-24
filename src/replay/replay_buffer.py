"""
Experience Replay Buffer for DQN variants.
Uses pre-allocated JAX arrays with a circular write pointer.

Usage:
    buffer = init_buffer(capacity=10000, obs_dim=4)
    buffer = add_transition(buffer, obs, action, reward, next_obs, done)
    batch = sample_batch(key, buffer, batch_size=32)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple


def init_buffer(capacity: int, obs_dim: int) -> Dict:
    """Create an empty replay buffer with pre-allocated arrays.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimensionality.

    Returns:
        Buffer state dict with arrays and metadata.
    """
    return {
        "obs": jnp.zeros((capacity, obs_dim)),
        "actions": jnp.zeros(capacity, dtype=jnp.int32),
        "rewards": jnp.zeros(capacity),
        "next_obs": jnp.zeros((capacity, obs_dim)),
        "dones": jnp.zeros(capacity, dtype=jnp.bool_),
        "ptr": 0,       # Next write position
        "size": 0,       # Current number of valid transitions
        "capacity": capacity,
    }


def add_transition(
    buffer: Dict,
    obs: jnp.ndarray,
    action: int,
    reward: float,
    next_obs: jnp.ndarray,
    done: bool,
) -> Dict:
    """Add a single transition to the buffer (circular overwrite).

    Args:
        buffer: Current buffer state.
        obs: Observation, shape (obs_dim,).
        action: Action taken (integer).
        reward: Reward received (extrinsic + any intrinsic bonus).
        next_obs: Next observation, shape (obs_dim,).
        done: Whether episode terminated.

    Returns:
        Updated buffer state.
    """
    idx = buffer["ptr"]
    cap = buffer["capacity"]

    buffer = {
        **buffer,
        "obs": buffer["obs"].at[idx].set(obs),
        "actions": buffer["actions"].at[idx].set(action),
        "rewards": buffer["rewards"].at[idx].set(reward),
        "next_obs": buffer["next_obs"].at[idx].set(next_obs),
        "dones": buffer["dones"].at[idx].set(done),
        "ptr": (idx + 1) % cap,
        "size": min(buffer["size"] + 1, cap),
    }
    return buffer


def sample_batch(
    key: jax.random.PRNGKey, buffer: Dict, batch_size: int
) -> Dict:
    """Sample a uniformly random batch of transitions.

    Args:
        key: JAX PRNG key.
        buffer: Current buffer state.
        batch_size: Number of transitions to sample.

    Returns:
        Dict with keys: obs, actions, rewards, next_obs, dones.
        Each has leading dimension batch_size.
    """
    indices = jax.random.randint(key, (batch_size,), 0, buffer["size"])
    return {
        "obs": buffer["obs"][indices],
        "actions": buffer["actions"][indices],
        "rewards": buffer["rewards"][indices],
        "next_obs": buffer["next_obs"][indices],
        "dones": buffer["dones"][indices],
    }


def can_sample(buffer: Dict, batch_size: int) -> bool:
    """Check if buffer has enough transitions to sample a batch."""
    return buffer["size"] >= batch_size
