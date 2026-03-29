from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from src.networks.q_network import init_q_params, q_forward, greedy_action as q_greedy_action


@struct.dataclass
class DQNAgent:
    q_params: Any
    target_q_params: Any
    num_actions: int = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False)


def create_dqn_agent(
    key: jax.Array,
    obs_dim: int,
    num_actions: int,
    hidden_dim: int = 64,
    epsilon: float = 0.1,
) -> DQNAgent:
    """
    Create a vanilla DQN agent with:
    - online Q-network parameters
    - target Q-network parameters
    - epsilon for epsilon-greedy exploration
    """
    q_params = init_q_params(
        key=key,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    )

    target_q_params = jax.tree_util.tree_map(lambda x: x.copy(), q_params)

    return DQNAgent(
        q_params=q_params,
        target_q_params=target_q_params,
        num_actions=num_actions,
        epsilon=epsilon,
    )


def act(
    agent: DQNAgent,
    key: jax.Array,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Epsilon-greedy action selection.

    With probability epsilon:
        sample a random action
    Otherwise:
        take greedy argmax action from the online Q-network
    """
    explore = jax.random.uniform(key) < agent.epsilon

    random_action = jax.random.randint(
        key,
        shape=(),
        minval=0,
        maxval=agent.num_actions,
        dtype=jnp.int32,
    )

    greedy_action = q_greedy_action(agent.q_params, obs)

    return jnp.where(explore, random_action, greedy_action).astype(jnp.int32)


def greedy_action(
    agent: DQNAgent,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Pure greedy action for evaluation.
    """
    return q_greedy_action(agent.q_params, obs)


def q_values(
    agent: DQNAgent,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convenience helper for debugging/logging.
    """
    return q_forward(agent.q_params, obs)


def target_q_values(
    agent: DQNAgent,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convenience helper for TD target computation / debugging.
    """
    return q_forward(agent.target_q_params, obs)


def update_q_params(
    agent: DQNAgent,
    new_q_params: Any,
) -> DQNAgent:
    """
    Return a new agent with updated online network params.
    """
    return DQNAgent(
        q_params=new_q_params,
        target_q_params=agent.target_q_params,
        num_actions=agent.num_actions,
        epsilon=agent.epsilon,
    )


def update_epsilon(
    agent: DQNAgent,
    new_epsilon: float,
) -> DQNAgent:
    """
    Return a new agent with updated epsilon.
    """
    return DQNAgent(
        q_params=agent.q_params,
        target_q_params=agent.target_q_params,
        num_actions=agent.num_actions,
        epsilon=float(new_epsilon),
    )


def sync_target_network(
    agent: DQNAgent,
) -> DQNAgent:
    """
    Hard update: target network becomes an exact copy of online network.
    """
    new_target_q_params = jax.tree_util.tree_map(lambda x: x.copy(), agent.q_params)

    return DQNAgent(
        q_params=agent.q_params,
        target_q_params=new_target_q_params,
        num_actions=agent.num_actions,
        epsilon=agent.epsilon,
    )