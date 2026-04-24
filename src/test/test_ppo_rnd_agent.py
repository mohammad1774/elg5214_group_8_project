"""
test_ppo_rnd_agent.py — Run a single PPO+RND training run.

Student C (Md Mosarraf)
"""

import argparse
import yaml
import jax

from src.envs.env_utils import get_env
from src.networks.policy_network import init_policy_params
from src.networks.value_network import init_value_params
from src.networks.rnd_networks import init_rnd_params
from src.training.train_ppo_rnd import train_ppo_rnd
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.utils.reusable import RLMetricsDataset, setup_logger, Timer, log_device_info


def run(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config.get("log_dir", "./logs/ppo_rnd")
    results_dir = config.get("results_dir", "results/ppo_rnd")
    project_name = config.get("project_name", "PPO_RND")
    final_eval_episodes = config.get("final_eval_episodes", 100)

    env, env_params, obs_dim, num_actions = get_env(args.env, args.reward)

    key = jax.random.PRNGKey(args.seed)
    key, policy_key, value_key, rnd_key = jax.random.split(key, 4)
    policy_params = init_policy_params(
        key=policy_key,
        obs_dim=obs_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_actions=num_actions,
    )
    value_params = init_value_params(
        key=value_key,
        obs_dim=obs_dim,
        hidden_dim=config["model"]["hidden_dim"],
    )
    rnd_params = init_rnd_params(
        rnd_key,
        obs_dim=obs_dim,
        hidden_dim=config["rnd"]["hidden_dim"],
    )

    run_id = f"s{args.seed}_lr{args.lr}_g{args.gamma}_b{args.beta}_{args.env}_{args.reward}"
    logger = setup_logger(run_id, path=log_dir)
    log_device_info(logger)
    logger.info(
        f"PPO+RND | seed={args.seed}, lr={args.lr}, "
        f"gamma={args.gamma}, beta={args.beta}"
    )
    logger.info(f"Env: {args.env}/{args.reward}, obs_dim={obs_dim}, num_actions={num_actions}")

    met_df = RLMetricsDataset(project_name)

    timer = Timer("PPO+RND training")
    with timer:
        results = train_ppo_rnd(
            env=env,
            env_params=env_params,
            init_policy_params=policy_params,
            init_value_params=value_params,
            num_episodes=config["num_episodes"],
            max_steps=config["max_steps"],
            learning_rate=args.lr,
            gamma=args.gamma,
            seed=args.seed,
            log_every=config["log_every"],
            clip_epsilon=config["ppo"]["clip_epsilon"],
            gae_lambda=config["ppo"]["gae_lambda"],
            value_coef=config.get("value_coef", 0.5),
            entropy_coeff=config.get("entropy_coeff", 0.0),
            beta=args.beta,
            predictor_lr=config["rnd"]["predictor_lr"],
            obs_dim=obs_dim,
            num_actions=num_actions,
            env_name=args.env,
            reward_type=args.reward,
            logger=logger,
            met_df=met_df,
        )

    logger.info(timer.report())

    eval_stats = evaluate_ppo(
        env=env,
        env_params=env_params,
        policy_params=results["final_policy_params"],
        num_episodes=final_eval_episodes,
        max_steps=config["max_steps"],
        seed=args.seed + 99999,
        greedy=True,
        num_actions=num_actions,
    )

    logger.info(
        f"[Final Eval] success={eval_stats['success_rate']:.3f}, "
        f"mean_reward={eval_stats['mean_reward']:.3f}"
    )
    print(
        f"[PPO+RND | {args.env}/{args.reward} | s={args.seed} lr={args.lr} "
        f"g={args.gamma} b={args.beta}] success={eval_stats['success_rate']:.3f}, "
        f"reward={eval_stats['mean_reward']:.3f}, time={timer.report()}"
    )

    met_df.add_summary(
        seed=args.seed,
        algorithm="PPO_RND",
        lr=args.lr,
        gamma=args.gamma,
        final_mean_reward=eval_stats["mean_reward"],
        final_success_rate=eval_stats["success_rate"],
        backend="JAX",
        devices=jax.devices(),
        action="stochastic",
        mean_length=eval_stats["mean_length"],
        wall_time_s=timer.elapsed,
        env_name=args.env,
        reward_type=args.reward,
    )

    output_dir = f"{results_dir}/{args.env}_{args.reward}"
    filename = f"lr{args.lr}_g{args.gamma}_b{args.beta}_seed{args.seed}.csv"
    met_df.save(output_dir=output_dir, filename=filename)

    return results["final_policy_params"], eval_stats


def main():
    parser = argparse.ArgumentParser(description="PPO + RND single run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["cartpole", "mountaincar"],
    )
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        choices=["dense", "sparse"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_rnd.yaml",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
