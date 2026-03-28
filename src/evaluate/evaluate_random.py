"""
evaluate_random.py — Evaluate the random baseline agent.

Uses vmap over episodes for fast parallel evaluation, same pattern
as evaluate_ppo.py.

USED BY: src/test/test_random_agent.py
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.training.rollout import run_one_episode_scan_simple


def evaluate_random_agent(env, env_params, agent, num_episodes: int = 100,
                          max_steps: int = 200, seed: int = 0) -> Dict[str, float]:
    """Evaluate the random agent over multiple episodes.

    Args:
        env:          Gymnax environment
        env_params:   environment parameters
        agent:        RandomAgent instance
        num_episodes: number of episodes to run
        max_steps:    max steps per episode
        seed:         random seed

    Returns:
        Dict with average_reward, average_length, success_rate, std_reward, all_rewards
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_episodes)

    def run_episode(ep_key):
        result = run_one_episode_scan_simple(
            env=env,
            env_params=env_params,
            agent=agent,
            key=ep_key,
            max_steps=max_steps,
        )
        return result["total_reward"], result["episode_length"]

    rewards, lengths = jax.vmap(run_episode)(keys)

    return {
        "average_reward": float(jnp.mean(rewards)),
        "average_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(jnp.where(rewards > 0, 1.0, 0.0))),
        "std_reward": float(jnp.std(rewards)),
        "all_rewards": rewards,
    }
