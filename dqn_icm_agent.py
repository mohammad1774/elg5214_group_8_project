from typing import Any

from flax import struct
import jax
import jax.numpy as jnp

from src.networks.q_network import init_q_params, q_forward, greedy_action as q_greedy_action
from src.networks.icm_networks import init_icm_params


@struct.dataclass
class DQNICMAgent:
    q_params: Any
    target_q_params: Any
    icm_params: Any
    num_actions: int = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False)


def create_dqn_icm_agent(
    key: jax.Array,
    obs_dim: int,
    num_actions: int,
    q_hidden_dim: int = 64,
    icm_hidden_dim: int = 64,
    icm_feat_dim: int = 64,
    epsilon: float = 0.1,
) -> DQNICMAgent:
    k_q, k_icm = jax.random.split(key)

    q_params = init_q_params(
        key=k_q,
        obs_dim=obs_dim,
        hidden_dim=q_hidden_dim,
        num_actions=num_actions,
    )

    target_q_params = jax.tree_util.tree_map(lambda x: x.copy(), q_params)

    icm_params = init_icm_params(
        key=k_icm,
        obs_dim=obs_dim,
        num_actions=num_actions,
        feat_dim=icm_feat_dim,
        hidden_dim=icm_hidden_dim,
    )

    return DQNICMAgent(
        q_params=q_params,
        target_q_params=target_q_params,
        icm_params=icm_params,
        num_actions=num_actions,
        epsilon=epsilon,
    )


def act(
    agent: DQNICMAgent,
    key: jax.Array,
    obs: jnp.ndarray,
) -> jnp.ndarray:
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
    agent: DQNICMAgent,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    return q_greedy_action(agent.q_params, obs)


def q_values(
    agent: DQNICMAgent,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    return q_forward(agent.q_params, obs)


def update_q_params(
    agent: DQNICMAgent,
    new_q_params: Any,
) -> DQNICMAgent:
    return DQNICMAgent(
        q_params=new_q_params,
        target_q_params=agent.target_q_params,
        icm_params=agent.icm_params,
        num_actions=agent.num_actions,
        epsilon=agent.epsilon,
    )


def update_icm_params(
    agent: DQNICMAgent,
    new_icm_params: Any,
) -> DQNICMAgent:
    return DQNICMAgent(
        q_params=agent.q_params,
        target_q_params=agent.target_q_params,
        icm_params=new_icm_params,
        num_actions=agent.num_actions,
        epsilon=agent.epsilon,
    )


def update_epsilon(
    agent: DQNICMAgent,
    new_epsilon: float,
) -> DQNICMAgent:
    return DQNICMAgent(
        q_params=agent.q_params,
        target_q_params=agent.target_q_params,
        icm_params=agent.icm_params,
        num_actions=agent.num_actions,
        epsilon=float(new_epsilon),
    )


def sync_target_network(
    agent: DQNICMAgent,
) -> DQNICMAgent:
    new_target_q_params = jax.tree_util.tree_map(lambda x: x.copy(), agent.q_params)

    return DQNICMAgent(
        q_params=agent.q_params,
        target_q_params=new_target_q_params,
        icm_params=agent.icm_params,
        num_actions=agent.num_actions,
        epsilon=agent.epsilon,
    )