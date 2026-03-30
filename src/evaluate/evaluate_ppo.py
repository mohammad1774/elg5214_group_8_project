"""
evaluate_ppo.py — Greedy evaluation for PPO-based agents.

Runs multiple greedy episodes and returns the success rate.
An episode is "successful" if its return exceeds a fraction of the
environment's maximum possible return (threshold_fraction from sweep.yaml).

USED BY: src/training/train_ppo_entropy.py, src/test/test_ppo_entropy_agent.py
"""

import jax
import jax.numpy as jnp

from src.training.rollout import run_one_episode_scan_simple


def evaluate_ppo(env, env_params, agent, key, num_episodes=100, max_steps=200,
                 threshold_fraction=0.9, max_return=None):
    """Run greedy evaluation episodes and compute success rate.

    Args:
        env, env_params: Gymnax environment
        agent:           PPOEntropyAgent (or any agent with .greedy_action)
        key:             PRNG key
        num_episodes:    how many greedy episodes to run
        max_steps:       max steps per episode
        threshold_fraction: fraction of max_return required for "success"
        max_return:      env-specific max return (auto-detected if None)

    Returns:
        success_rate: float in [0, 1]
    """
    total_successes = 0.0

    for i in range(num_episodes):
        key, ep_key = jax.random.split(key)
        rollout = run_one_episode_scan_simple(
            env, env_params, agent, ep_key,
            max_steps=max_steps, greedy=True,
        )
        total_reward = float(rollout["total_reward"])
        ep_len = int(rollout["episode_length"])

        # Heuristic success: episode lasted ≥ 90% of max_steps (CartPole-like)
        # or total reward is positive (MountainCar sparse)
        if max_return is not None:
            success = total_reward >= threshold_fraction * max_return
        else:
            # Auto-detect: if max_steps <= 200, use length-based heuristic
            success = (ep_len >= int(threshold_fraction * max_steps)) or (
                total_reward > 0.0 and ep_len < max_steps
            )
        total_successes += float(success)

    return total_successes / num_episodes
