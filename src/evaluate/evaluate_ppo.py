"""
evaluate_ppo.py — Evaluation for all PPO agent variants.

WHAT IT DOES:
    Takes trained policy params, creates a PPOAgent, runs episodes
    using run_one_episode_scan_simple (with greedy=True for deterministic
    evaluation), and reports metrics.

WHY IT USES lax.scan (unlike evaluate_dqn.py):
    PPO evaluation uses run_one_episode_scan_simple which is already
    lax.scan-based. We can vmap over multiple episodes for fast parallel
    evaluation on GPU. DQN's eval uses a Python loop because the DQN agent
    class takes epsilon as an argument to act(), which doesn't fit the
    simple agent.act(key, obs) interface that rollout.py expects.

CALLED FROM:
    - Every PPO training loop (train_ppo.py, train_ppo_entropy.py, etc.)
    - Every PPO test script (test_ppo_agent.py, etc.)

USED BY: Students C and D (all PPO variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.agents.ppo_agent import PPOAgent
from src.training.rollout import run_one_episode_scan_simple


def evaluate_ppo(
    env,
    env_params,
    policy_params: dict,
    num_episodes: int = 50,
    max_steps: int = 200,
    seed: int = 0,
    greedy: bool = True,
    num_actions: int = 2,
) -> Dict[str, float]:
    """Evaluate a trained policy network.

    Args:
        env:            Gymnax environment (or sparse wrapper)
        env_params:     environment parameters
        policy_params:  trained policy network weights
        num_episodes:   number of evaluation episodes
        max_steps:      max steps per episode
        seed:           random seed
        greedy:         True = argmax (deterministic), False = sample from policy
        num_actions:    action space size (2 for CartPole, 3 for MountainCar)

    Returns:
        Dict with mean_reward, std_reward, mean_length, success_rate, all_rewards
    """
    key = jax.random.PRNGKey(seed)
    agent = PPOAgent(policy_params, num_actions=num_actions)

    keys = jax.random.split(key, num_episodes)

    def run_episode(ep_key):
        rollout = run_one_episode_scan_simple(
            env=env,
            env_params=env_params,
            agent=agent,
            key=ep_key,
            max_steps=max_steps,
            greedy=greedy,
        )
        return rollout["total_reward"], rollout["episode_length"], rollout["observations"][-1]

    # vmap across all episodes for fast parallel evaluation
    rewards, lengths, last_obs = jax.vmap(run_episode)(keys)

    # Success based on environment:
    # CartPole (obs_dim=4): survived full episode (length >= max_steps)
    # MountainCar (obs_dim=2): reached goal (position >= 0.5)
    if num_actions == 2:  # CartPole
        successes = jnp.where(lengths >= max_steps, 1.0, 0.0)
    else:  # MountainCar
        successes = jnp.where(last_obs[:, 0] >= 0.5, 1.0, 0.0)

    return {
        "mean_reward": float(jnp.mean(rewards)),
        "std_reward": float(jnp.std(rewards)),
        "mean_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(successes)),
        "all_rewards": rewards,
    }
