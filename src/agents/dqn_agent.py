"""
Vanilla DQN Agent — Student B.

Implements:
  - ε-greedy action selection with linear decay
  - Target network with periodic hard updates
  - Gradient clipping (norm 10) per proposal risk mitigation
  - Huber loss for stable TD learning

Interface:
  agent = DQNAgent(params, target_params, config)
  action = agent.act(key, obs, epsilon)
  action = agent.greedy_action(obs)
  params, target_params, opt_state, loss = agent.update(...)
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, NamedTuple

from src.networks.q_network import q_forward, q_forward_batch, init_q_params


class DQNConfig(NamedTuple):
    """DQN hyperparameters."""
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 100      # Steps between hard target updates
    grad_clip_norm: float = 10.0       # Per proposal risk mitigation
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 5000


def create_dqn_agent(key: jax.random.PRNGKey, config: DQNConfig):
    """Initialize DQN agent: params, target params, optimizer state.

    Args:
        key: JAX PRNG key.
        config: DQN hyperparameters.

    Returns:
        (params, target_params, opt_state, optimizer)
    """
    params = init_q_params(key, config.obs_dim, config.act_dim, config.hidden_dim)
    target_params = jax.tree.map(lambda x: x.copy(), params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(params)

    return params, target_params, opt_state, optimizer


def get_epsilon(step: int, config: DQNConfig) -> float:
    """Linear epsilon decay schedule."""
    frac = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def act(
    key: jax.random.PRNGKey,
    params: Dict,
    obs: jnp.ndarray,
    epsilon: float,
    act_dim: int,
) -> int:
    """Epsilon-greedy action selection.

    Args:
        key: JAX PRNG key.
        params: Q-network parameters.
        obs: Current observation, shape (obs_dim,).
        epsilon: Current exploration rate.
        act_dim: Number of discrete actions.

    Returns:
        Selected action (integer).
    """
    k1, k2 = jax.random.split(key)
    q_values = q_forward(params, obs)
    greedy_action = jnp.argmax(q_values)
    random_action = jax.random.randint(k1, (), 0, act_dim)
    # Choose random with probability epsilon, greedy otherwise
    use_random = jax.random.uniform(k2) < epsilon
    return jnp.where(use_random, random_action, greedy_action)


def greedy_action(params: Dict, obs: jnp.ndarray) -> int:
    """Greedy action selection (for evaluation).

    Args:
        params: Q-network parameters.
        obs: Current observation, shape (obs_dim,).

    Returns:
        Greedy action (integer).
    """
    q_values = q_forward(params, obs)
    return jnp.argmax(q_values)


def dqn_loss_fn(
    params: Dict,
    target_params: Dict,
    batch: Dict,
    gamma: float,
) -> jnp.ndarray:
    """Compute DQN Huber loss on a batch of transitions.

    Uses the standard DQN target: r + γ * max_a' Q_target(s', a') * (1 - done)

    Args:
        params: Online Q-network parameters.
        target_params: Target Q-network parameters.
        batch: Dict with obs, actions, rewards, next_obs, dones.
        gamma: Discount factor.

    Returns:
        Scalar mean Huber loss.
    """
    # Current Q-values for taken actions
    q_values = q_forward_batch(params, batch["obs"])  # (batch, act_dim)
    q_taken = q_values[jnp.arange(q_values.shape[0]), batch["actions"]]  # (batch,)

    # Target Q-values (no gradient)
    next_q_values = q_forward_batch(target_params, batch["next_obs"])  # (batch, act_dim)
    next_q_max = jnp.max(next_q_values, axis=-1)  # (batch,)

    # TD target
    targets = batch["rewards"] + gamma * next_q_max * (1.0 - batch["dones"].astype(jnp.float32))

    # Huber loss (more stable than MSE for RL)
    td_error = targets - q_taken
    loss = jnp.mean(optax.huber_loss(q_taken, targets, delta=1.0))

    return loss


def update_dqn(
    params: Dict,
    target_params: Dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: Dict,
    gamma: float,
    step: int,
    target_update_freq: int,
) -> Tuple[Dict, Dict, optax.OptState, float]:
    """Single DQN update step.

    Args:
        params: Online Q-network parameters.
        target_params: Target Q-network parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer (with grad clipping).
        batch: Sampled batch from replay buffer.
        gamma: Discount factor.
        step: Current training step (for target network update).
        target_update_freq: Steps between hard target network updates.

    Returns:
        (updated_params, updated_target_params, updated_opt_state, loss_value)
    """
    loss, grads = jax.value_and_grad(dqn_loss_fn)(params, target_params, batch, gamma)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Hard target update
    target_params = jax.lax.cond(
        step % target_update_freq == 0,
        lambda _: jax.tree.map(lambda x: x.copy(), params),
        lambda _: target_params,
        operand=None,
    )

    return params, target_params, opt_state, loss
