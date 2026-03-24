"""
Greedy evaluation for DQN variants.
Runs episodes with greedy (argmax Q) action selection, no exploration noise.

Works for vanilla DQN, DQN+entropy, DQN+RND, DQN+ICM — all share the same
Q-network interface for greedy action selection.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from src.networks.q_network import q_forward


def evaluate_dqn_greedy(
    key: jax.random.PRNGKey,
    params: Dict,
    env,
    env_params,
    num_episodes: int = 10,
    max_steps: int = 500,
) -> Tuple[float, float, float]:
    """Run greedy evaluation episodes and return statistics.

    Args:
        key: JAX PRNG key.
        params: Q-network parameters.
        env: Gymnax environment (or sparse wrapper).
        env_params: Environment parameters.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.

    Returns:
        (mean_reward, std_reward, success_rate)
    """
    total_rewards = []
    successes = 0

    for ep in range(num_episodes):
        key, reset_key, step_key = jax.random.split(key, 3)
        obs, state = env.reset(reset_key, env_params)
        episode_reward = 0.0
        done = False

        for t in range(max_steps):
            if done:
                break
            q_values = q_forward(params, obs)
            action = jnp.argmax(q_values).item()

            step_key, key = jax.random.split(key)
            obs, state, reward, done, info = env.step(step_key, state, action, env_params)
            episode_reward += float(reward)
            done = bool(done)

        total_rewards.append(episode_reward)

        # Success heuristic: CartPole survived long, MountainCar reached goal
        if episode_reward > 0:
            successes += 1

    rewards_arr = jnp.array(total_rewards)
    mean_reward = float(jnp.mean(rewards_arr))
    std_reward = float(jnp.std(rewards_arr))
    success_rate = successes / num_episodes

    return mean_reward, std_reward, success_rate
