"""
value_network.py — State-value network V(s) for PPO.

WHAT IT DOES:
    Takes an observation and outputs a SINGLE NUMBER: V(s).
    This estimates "how much total future reward can I expect from this state?"

WHY PPO NEEDS THIS (but DQN doesn't):
    PPO's loss uses the "advantage": A(s,a) = actual_return - V(s)
    This tells the agent "was this action better or worse than average?"

    - If A > 0: this action was better than expected → increase its probability
    - If A < 0: this action was worse than expected → decrease its probability

    Without V(s), PPO would use raw returns, which have high variance
    and make training unstable. The value baseline reduces variance.

    DQN doesn't need this because DQN learns Q(s,a) directly — it already
    knows the value of each specific action, not just the state.

WHY A SEPARATE NETWORK (not shared with policy):
    Sharing weights between policy and value networks can cause interference —
    a gradient update that improves value estimation might hurt the policy.
    Separate networks are simpler and more stable for our comparison.

ARCHITECTURE: same 2×64 ReLU MLP, but output is 1 scalar (not num_actions)

USED BY: Students C and D (all PPO variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_value_params(key, obs_dim: int, hidden_dim: int = 64):
    """Initialize value network weights.

    Output layer is (hidden_dim, 1) — one scalar per state.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    params = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1,
        "b2": jnp.zeros((hidden_dim,)),
        "W3": jax.random.normal(k3, (hidden_dim, 1)) * 0.1,
        "b3": jnp.zeros((1,)),
    }

    return params


def value_forward(params: Dict[str, jnp.ndarray],
                  obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: observation → scalar value V(s).

    Returns a scalar (not an array of shape (1,)) because squeeze(-1)
    removes the trailing dimension.
    """
    x = jnp.asarray(obs, dtype=jnp.float32)

    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])

    value = h2 @ params["W3"] + params["b3"]

    return value.squeeze(-1)  # shape () — a scalar


def value_forward_batch(params: Dict[str, jnp.ndarray],
                        obs_batch: jnp.ndarray) -> jnp.ndarray:
    """Forward pass over a batch: (batch_size, obs_dim) → (batch_size,)."""
    return jax.vmap(lambda obs: value_forward(params, obs))(obs_batch)
