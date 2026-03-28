"""
evaluate_dqn.py — Greedy evaluation for all DQN agent variants.

WHAT IT DOES:
    Takes trained Q-network params, runs episodes with greedy action
    selection (always pick argmax Q), and reports:
        - mean_reward:   average total return per episode
        - std_reward:    standard deviation of returns
        - mean_length:   average episode length
        - success_rate:  fraction of episodes that "succeeded"

HOW SUCCESS IS DEFINED:
    CartPole:     success = survived to max_steps (pole didn't fall)
    MountainCar:  success = reached the goal (position >= 0.5)

    Since we run both envs, we use a general heuristic:
    success = episode lasted max_steps OR total reward > 0.

WHY A PYTHON LOOP (not lax.scan):
    Evaluation doesn't need to be fast — we only run 50-100 episodes.
    A Python loop is clearer and easier to debug. The training loop
    uses lax.scan because it runs thousands of episodes.

CALLED FROM:
    - Every DQN training loop (train_dqn.py, train_dqn_entropy.py, etc.)
      at intervals during training (every log_every episodes)
    - Every DQN test script (test_dqn_agent.py, etc.) for final evaluation

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

        while (not bool(done)) and (step < max_steps):
            # Greedy: always pick argmax Q
            q_values = q_forward(q_params, obs)
            action = int(jnp.argmax(q_values))

            key, step_key = jax.random.split(key)
            next_obs, next_state, reward, done, _ = env.step_env(
                step_key, state, action, env_params
            )

            total_reward += float(reward)
            obs = next_obs
            state = next_state
            step += 1

        rewards.append(total_reward)
        lengths.append(step)
        # Success heuristic: survived max_steps OR got positive reward
        successes.append(float(step >= max_steps or total_reward > 0))

    rewards_arr = jnp.array(rewards, dtype=jnp.float32)
    lengths_arr = jnp.array(lengths, dtype=jnp.float32)
    successes_arr = jnp.array(successes, dtype=jnp.float32)

    return {
        "mean_reward": float(jnp.mean(rewards_arr)),
        "std_reward": float(jnp.std(rewards_arr)),
        "mean_length": float(jnp.mean(lengths_arr)),
        "success_rate": float(jnp.mean(successes_arr)),
    }
