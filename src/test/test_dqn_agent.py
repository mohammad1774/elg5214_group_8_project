"""
DQN Baseline — Test / Run Script — Student B.

Entry point that sets up the environment, hyperparameters, and runs the
DQN baseline training + evaluation for a specific configuration.

Usage:
    python -m src.test.test_dqn_agent
    python -m src.test.test_dqn_agent --env CartPole-v1 --sparse --lr 1e-3 --gamma 0.99 --seed 0

For full sweep:
    See scripts/run_sweep.sh which loops over all combinations.
"""

import argparse
import jax
import logging
import os
import sys

from src.agents.dqn_agent import DQNConfig
from src.envs.env_utils import get_env, get_env_config
from src.training.train_dqn import train_dqn


def setup_logger(name: str, log_dir: str = "logs/dqn") -> logging.Logger:
    """Create a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    return logger


def main():
    parser = argparse.ArgumentParser(description="DQN Baseline Training")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "MountainCar-v0"])
    parser.add_argument("--sparse", action="store_true", help="Use sparse rewards")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=50)
    args = parser.parse_args()

    # Environment setup
    env_config = get_env_config(args.env)
    env, env_params = get_env(args.env, sparse=args.sparse)
    reward_type = "sparse" if args.sparse else "dense"

    # Run identifier
    run_name = f"dqn_baseline_{args.env}_{reward_type}_lr{args.lr}_g{args.gamma}_seed{args.seed}"
    logger = setup_logger(run_name)
    logger.info(f"Starting: {run_name}")
    logger.info(f"Device: {jax.devices()}")

    # DQN config
    config = DQNConfig(
        obs_dim=env_config["obs_dim"],
        act_dim=env_config["act_dim"],
        lr=args.lr,
        gamma=args.gamma,
    )

    # Train
    key = jax.random.PRNGKey(args.seed)
    results = train_dqn(
        key=key,
        env=env,
        env_params=env_params,
        config=config,
        num_episodes=args.num_episodes,
        max_steps_per_episode=env_config["max_steps"],
        eval_interval=args.eval_interval,
        logger=logger,
    )

    # Save results CSV
    env_name = args.env.replace("-", "").lower()
    result_dir = f"results/dqn_baseline/{env_name}_{reward_type}"
    os.makedirs(result_dir, exist_ok=True)

    import csv
    csv_path = os.path.join(result_dir, f"lr{args.lr}_g{args.gamma}_seed{args.seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "loss"])
        for ep, (r, l) in enumerate(
            zip(results["episode_rewards"], results["loss_history"]), 1
        ):
            writer.writerow([ep, r, l])

    logger.info(f"Results saved to {csv_path}")
    logger.info(f"Wall time: {results['wall_time_s']:.1f}s | Global steps: {results['global_steps']}")

    # Print final eval
    if results["eval_history"]:
        final = results["eval_history"][-1]
        logger.info(
            f"Final eval: mean_reward={final['mean_reward']:.2f} | "
            f"success_rate={final['success_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
