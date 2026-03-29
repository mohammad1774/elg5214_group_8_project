"""
evaluate_dqn.py — Greedy evaluation for all DQN agent variants.

Works with: DQN baseline, DQN+entropy, DQN+RND, DQN+ICM
Any agent with a greedy_action(obs) method can be evaluated here.

SUCCESS DEFINITION:
    CartPole:     survived the full max_steps (pole never fell)
    MountainCar:  reached goal position (>= 0.5)

    We detect which env we're in by checking obs_dim:
        obs_dim=4 → CartPole
        obs_dim=2 → MountainCar

USED BY: Students A and B (all DQN variants)
"""

from typing import Dict

import jax
import jax.numpy as jnp

from src.networks.q_network import q_forward


def evaluate_dqn_greedy(
    env,
    env_params,
    q_params: Dict,
    num_episodes: int = 50,
    max_steps: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate a trained Q-network with greedy action selection.

    Args:
        env:          Gymnax environment (or sparse wrapper)
        env_params:   environment parameters
        q_params:     trained Q-network weights
        num_episodes: number of evaluation episodes
        max_steps:    max steps per episode
        seed:         random seed for env resets

    Returns:
        Dict with mean_reward, std_reward, mean_length, success_rate
    """
    key = jax.random.PRNGKey(seed)

    rewards = []
    lengths = []
    successes = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, env_params)

        total_reward = 0.0
        done = False
        step = 0
        last_obs = obs

        while (not bool(done)) and (step < max_steps):
            # Greedy: always pick argmax Q
            q_values = q_forward(q_params, obs)
            action = int(jnp.argmax(q_values))

            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )

            total_reward += float(reward)
            last_obs = next_obs
            obs = next_obs
            state = next_state
            step += 1

        rewards.append(total_reward)
        lengths.append(step)

        # Determine success based on environment type
        obs_dim = last_obs.shape[0]
        if obs_dim == 4:
            # CartPole: success = survived full episode (pole never fell)
            success = (not bool(done)) or (step >= max_steps)
        elif obs_dim == 2:
            # MountainCar: success = reached goal (position >= 0.5)
            success = float(last_obs[0]) >= 0.5
        else:
            # Fallback
            success = step >= max_steps

        successes.append(float(success))

    rewards_arr = jnp.array(rewards, dtype=jnp.float32)
    lengths_arr = jnp.array(lengths, dtype=jnp.float32)
    successes_arr = jnp.array(successes, dtype=jnp.float32)

    return {
        "mean_reward": float(jnp.mean(rewards_arr)),
        "std_reward": float(jnp.std(rewards_arr)),
        "mean_length": float(jnp.mean(lengths_arr)),
        "success_rate": float(jnp.mean(successes_arr)),
    }
