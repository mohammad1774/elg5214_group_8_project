"""
rollout.py — Shared episode collection via lax.scan.

Used by ALL agent variants (DQN, PPO, and their exploration extensions).
Provides both a Python-loop rollout (for debugging) and a fully
on-device lax.scan rollout (for training).
"""

from typing import Dict, List, Any

import jax
import jax.numpy as jnp
from jax import lax


def run_one_episode(env, env_params, agent, key: jax.Array, max_steps: int = 100) -> Dict[str, List[Any]]:
    """Run one episode with a Python while-loop (for debugging only)."""
    key, reset_key = jax.random.split(key)
    obs, state = env.reset_env(reset_key, env_params)

    observations: List[jnp.ndarray] = [obs]
    actions: List[int] = []
    rewards: List[float] = []

    total_reward = 0.0
    done = False
    step = 0

    while (not bool(done) and step < max_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        action = agent.act(act_key, obs)
        next_obs, next_state, reward, done, info = env.step_env(step_key, state, action, env_params)

        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)

        total_reward += float(reward)
        state = next_state
        step += 1

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "total_reward": total_reward,
        "episode_length": step,
        "final_done": bool(done),
    }


def run_one_episode_scan_simple(env, env_params, agent, key, max_steps=100, greedy: bool = False) -> Dict[str, jnp.ndarray]:
    """Run one episode fully on-device via lax.scan.

    Returns dict with arrays of shape (max_steps, ...) plus scalar summaries.
    Used by REINFORCE, PPO, and evaluation functions.
    """
    key, reset_key = jax.random.split(key)
    obs0, state0 = env.reset_env(reset_key, env_params)

    def step_fn(carry, _):
        key, obs, state, done = carry
        current_obs = obs

        key, act_key, step_key = jax.random.split(key, 3)

        def select_action(act_key, obs):
            if greedy:
                return agent.greedy_action(obs)
            else:
                return agent.act(act_key, obs)

        action = select_action(act_key, current_obs)

        next_obs, next_state, reward, next_done, _ = env.step_env(
            step_key, state, action, env_params
        )

        reward = jnp.where(done, 0.0, reward)
        next_done = jnp.logical_or(done, next_done)

        updated_obs = jnp.where(done, current_obs, next_obs)
        updated_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(done, old, new),
            state,
            next_state,
        )

        carry = (key, updated_obs, updated_state, next_done)

        output = {
            "observations": current_obs,
            "actions": action,
            "rewards": reward,
            "dones": next_done,
            "next_observations": updated_obs,
        }

        return carry, output

    init_carry = (key, obs0, state0, jnp.array(False))
    final_carry, outputs = lax.scan(step_fn, init_carry, xs=None, length=max_steps)

    total_reward = jnp.sum(outputs["rewards"])

    done_cumsum = jnp.cumsum(outputs["dones"].astype(jnp.int32))
    valid_mask = done_cumsum <= 1
    episode_length = jnp.sum(valid_mask)

    return {
        "observations": outputs["observations"],           # (max_steps, obs_dim)
        "actions": outputs["actions"],                     # (max_steps,)
        "rewards": outputs["rewards"],                     # (max_steps,)
        "dones": outputs["dones"],                         # (max_steps,)
        "next_observations": outputs["next_observations"], # (max_steps, obs_dim)
        "total_reward": total_reward,
        "episode_length": episode_length,
        "final_done": outputs["dones"][-1],
    }
