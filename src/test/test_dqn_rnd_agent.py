"""
test_dqn_rnd_agent.py — Run a single DQN+RND training run.

Device selection is intentionally left to the shell environment so the same
entry point can be used on CPU or GPU. Bash/Slurm scripts should set
JAX_PLATFORMS / CUDA visibility as needed before invoking this module.

Called by the bash sweep script with CLI arguments:
    python -m src.test.test_dqn_rnd_agent \
        --seed 0 --lr 0.001 --gamma 0.99 --beta 0.1 \
        --env cartpole --reward sparse \
        --config configs/dqn_rnd.yaml

The YAML provides fixed hyperparameters (buffer size, batch size, etc.).
The CLI provides sweep variables (seed, lr, gamma, beta, env, reward).

Student A (Mohammad)
"""

import argparse
import yaml
import jax

from src.envs.env_utils import get_env
from src.networks.q_network import init_q_params
from src.training.train_dqn_rnd import train_dqn_rnd
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy
from src.utils.reusable import RLMetricsDataset, setup_logger, Timer, log_device_info


def run(args):
    # Load fixed config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config.get("log_dir", "./logs/dqn_rnd")
    results_dir = config.get("results_dir", "results/dqn_rnd")
    project_name = config.get("project_name", "DQN_RND")
    final_eval_episodes = config.get("final_eval_episodes", 100)

    # Setup environment
    env, env_params, obs_dim, num_actions = get_env(args.env, args.reward)

    # Init Q-network
    key = jax.random.PRNGKey(args.seed)
    init_params = init_q_params(
        key=key,
        obs_dim=obs_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_actions=num_actions,
    )

    # Logger
    run_id = f"s{args.seed}_lr{args.lr}_g{args.gamma}_b{args.beta}_{args.env}_{args.reward}"
    logger = setup_logger(run_id, path=log_dir)
    log_device_info(logger)
    logger.info(f"DQN+RND | seed={args.seed}, lr={args.lr}, gamma={args.gamma}, beta={args.beta}")
    logger.info(f"Env: {args.env}/{args.reward}, obs_dim={obs_dim}, num_actions={num_actions}")

    # Metrics
    met_df = RLMetricsDataset(project_name)

    # Train
    timer = Timer("DQN+RND training")
    with timer:
        results = train_dqn_rnd(
            env=env,
            env_params=env_params,
            init_q_params=init_params,
            num_episodes=config["num_episodes"],
            max_steps=config["max_steps"],
            learning_rate=args.lr,
            gamma=args.gamma,
            seed=args.seed,
            buffer_capacity=config["buffer_capacity"],
            batch_size=config["batch_size"],
            warmup_steps=config["warmup_steps"],
            target_update_freq=config["target_update_freq"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay_episodes=config["epsilon_decay_episodes"],
            updates_per_episode=config["updates_per_episode"],
            log_every=config["log_every"],
            beta=args.beta,
            predictor_lr=config["predictor_lr"],
            obs_dim=obs_dim,
            num_actions=num_actions,
            env_name=args.env,
            reward_type=args.reward,
            logger=logger,
            met_df=met_df,
        )

    logger.info(timer.report())

    # Final greedy evaluation
    eval_stats = evaluate_dqn_greedy(
        env=env, env_params=env_params,
        q_params=results["final_q_params"],
        num_episodes=final_eval_episodes,
        max_steps=config["max_steps"],
        seed=args.seed + 99999,
    )

    logger.info(f"[Final Eval] success={eval_stats['success_rate']:.3f}, "
                f"mean_reward={eval_stats['mean_reward']:.3f}")
    print(f"[DQN+RND | {args.env}/{args.reward} | s={args.seed} lr={args.lr} "
          f"g={args.gamma} b={args.beta}] "
          f"success={eval_stats['success_rate']:.3f}, "
          f"reward={eval_stats['mean_reward']:.3f}, "
          f"time={timer.report()}")

    met_df.add_summary(
        seed=args.seed, algorithm="DQN_RND",
        lr=args.lr, gamma=args.gamma,
        final_mean_reward=eval_stats["mean_reward"],
        final_success_rate=eval_stats["success_rate"],
        backend="JAX", devices=jax.devices(),
        action="greedy",
        mean_length=eval_stats["mean_length"],
        wall_time_s=timer.elapsed,
        env_name=args.env, reward_type=args.reward,
    )

    # Save per-run CSV
    output_dir = f"{results_dir}/{args.env}_{args.reward}"
    filename = f"lr{args.lr}_g{args.gamma}_b{args.beta}_seed{args.seed}.csv"
    met_df.save(output_dir=output_dir, filename=filename)

    return results["final_q_params"], eval_stats


def main():
    parser = argparse.ArgumentParser(description="DQN + RND single run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--env", type=str, required=True, choices=["cartpole", "mountaincar"])
    parser.add_argument("--reward", type=str, required=True, choices=["dense", "sparse"])
    parser.add_argument("--config", type=str, default="configs/dqn_rnd.yaml")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
