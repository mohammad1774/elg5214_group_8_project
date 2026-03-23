"""
replay_buffer.py — Experience replay buffer using JAX arrays.

All fields (including ptr, size, capacity) are jnp arrays so the
buffer can be used inside lax.fori_loop / lax.scan without pulling
scalars to CPU.

Used by: DQN baseline, DQN+entropy, DQN+RND, DQN+ICM
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_buffer(capacity: int, obs_dim: int) -> Dict:
    """Create an empty replay buffer. All metadata are JAX scalars."""
    buffer = {
        "obs":      jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        "actions":  jnp.zeros((capacity,), dtype=jnp.int32),
        "rewards":  jnp.zeros((capacity,), dtype=jnp.float32),
        "next_obs": jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        "dones":    jnp.zeros((capacity,), dtype=jnp.bool_),
        "size":     jnp.int32(0),
        "ptr":      jnp.int32(0),
        "capacity": jnp.int32(capacity),
    }
    return buffer


def add_transition(
    buffer: Dict,
    obs: jnp.ndarray,
    action: int,
    reward: float,
    next_obs: jnp.ndarray,
    done: bool,
) -> Dict:
    """Add a single transition (works in Python loops and inside lax transforms)."""
    ptr = buffer["ptr"]

    buffer = {
        "obs":      buffer["obs"].at[ptr].set(obs),
        "actions":  buffer["actions"].at[ptr].set(action),
        "rewards":  buffer["rewards"].at[ptr].set(reward),
        "next_obs": buffer["next_obs"].at[ptr].set(next_obs),
        "dones":    buffer["dones"].at[ptr].set(done),
        "ptr":      (ptr + 1) % buffer["capacity"],
        "size":     jnp.minimum(buffer["size"] + 1, buffer["capacity"]),
        "capacity": buffer["capacity"],
    }

    return buffer


def sample_batch(
    buffer: Dict,
    key: jax.Array,
    batch_size: int,
) -> Dict:
    """Sample a random mini-batch of transitions."""
    size = buffer["size"]
    indices = jax.random.randint(key, (batch_size,), 0, size)

    return {
        "obs":      buffer["obs"][indices],
        "actions":  buffer["actions"][indices],
        "rewards":  buffer["rewards"][indices],
        "next_obs": buffer["next_obs"][indices],
        "dones":    buffer["dones"][indices],
    }


def add_transitions_batch(buffer, rollout, n_valid):
    """Insert n_valid transitions from rollout arrays into the buffer.

    Uses lax.fori_loop so pointer arithmetic stays on-device (GPU).
    """
    from jax import lax

    def body_fn(t, buf):
        ptr = buf["ptr"]
        buf = {
            "obs":      buf["obs"].at[ptr].set(rollout["obs"][t]),
            "actions":  buf["actions"].at[ptr].set(rollout["actions"][t]),
            "rewards":  buf["rewards"].at[ptr].set(rollout["rewards"][t]),
            "next_obs": buf["next_obs"].at[ptr].set(rollout["next_obs"][t]),
            "dones":    buf["dones"].at[ptr].set(rollout["dones"][t]),
            "ptr":      (ptr + 1) % buf["capacity"],
            "size":     jnp.minimum(buf["size"] + 1, buf["capacity"]),
            "capacity": buf["capacity"],
        }
        return buf

    return lax.fori_loop(0, n_valid, body_fn, buffer)
