"""
test_random_agent.py — Run random baseline evaluation.

Evaluates the random agent on all 4 env variants and saves metrics.
This establishes the lower bound that all trained agents should beat.

Usage:
    python -m src.test.test_random_agent

Output:
    metrics/RandomBaseline_dataset_metrics_summary.csv
"""

import jax

from src.envs.env_utils import get_env
from src.agents.random_agent import RandomAgent
from src.evaluate.evaluate_random import evaluate_random_agent
from src.utils.reusable import RLMetricsDataset, setup_logger


def test_random_agent(seed: int = 42, num_episodes: int = 1000,
                      max_steps: int = 200, env_name: str = "cartpole",
                      reward_type: str = "dense",
                      met_df: RLMetricsDataset = None):
    """Evaluate random agent on a single env variant.

    Args:
        seed:         random seed
        num_episodes: number of eval episodes
        max_steps:    max steps per episode
        env_name:     "cartpole" or "mountaincar"
        reward_type:  "dense" or "sparse"
        met_df:       RLMetricsDataset to log results to
    """
    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)
    agent = RandomAgent(n_actions=num_actions)

    run_id = hash(f"random_{env_name}_{reward_type}_{seed}") % 100000
    logger = setup_logger(run_id, path="./logs/random_agent/")
    logger.info(f"JAX DEVICES: {jax.devices()}")
    logger.info(f"Evaluating RandomAgent on {env_name}/{reward_type} "
                f"seed={seed}, num_episodes={num_episodes}")

    stats = evaluate_random_agent(
        env, env_params, agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
    )

    logger.info(f"Results: {stats}")
    print(f"[Random | {env_name}/{reward_type}] "
          f"avg_reward={stats['average_reward']:.3f}, "
          f"success={stats['success_rate']:.3f}")

    if met_df is not None:
        met_df.add_summary(
            seed=seed, algorithm="RandomAgent",
            lr=0.0, gamma=0.0,
            final_mean_reward=stats["average_reward"],
            final_success_rate=stats["success_rate"],
            backend="JAX", devices=jax.devices(),
            env_name=env_name, reward_type=reward_type,
        )

    return stats


def main():
    """Run random baseline on all 4 env variants."""
    met_df = RLMetricsDataset("RandomBaseline")

    for env_name in ["cartpole", "mountaincar"]:
        for reward_type in ["dense", "sparse"]:
            test_random_agent(
                seed=42, num_episodes=1000, max_steps=200,
                env_name=env_name, reward_type=reward_type,
                met_df=met_df,
            )

    paths = met_df.save()
    print(f"\nSaved to: {paths['summary']}")


if __name__ == "__main__":
    main()
