"""
test_ppo_agent.py — PPO Baseline sweep script.

WHAT THIS FILE DOES:
    Orchestrates the full hyperparameter sweep for PPO baseline:
        3 learning rates × 2 gammas × 5 seeds × 2 envs × 2 reward types
        = 120 runs total

    For each combination it:
        1. Creates the environment
        2. Initializes network params
        3. Sets up a file logger
        4. Calls train_ppo()
        5. Runs a final greedy evaluation
        6. Saves summary metrics to CSV

HOW TO RUN:
    Full sweep (120 runs — slow on CPU):
        python -m src.test.test_ppo_agent

    Single quick run for testing:
        python -m src.test.test_ppo_agent --quick 
"""

import argparse

import jax
import yaml

from src.envs.env_utils import get_env
from src.networks.policy_network import init_policy_params, policy_forward, log_prob, entropy
from src.networks.value_network import init_value_params, value_forward
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.training.train_ppo import train_ppo
from src.utils.reusable import RLMetricsDataset, setup_logger, Timer

def test_ppo(
    seed: int,
    gamma: float,
    lr: float,
    config: dict,
    config_env_params: dict,
    met_df: RLMetricsDataset,
) -> dict:

    env_name    = config_env_params["env_name"]
    reward_type = config_env_params["reward_type"]

    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    run_id = abs(hash(f"ppo_{env_name}_{reward_type}_lr{lr}_g{gamma}_s{seed}")) % 100000
    logger = setup_logger(run_id, path="./logs/ppo")
    logger.info(f"JAX devices : {jax.devices()}")
    logger.info(f"JAX backend : {jax.default_backend()}")
    logger.info(
        f"Run config  : env={env_name}/{reward_type} "
        f"seed={seed} lr={lr} gamma={gamma}"
    )

    #Training 
    result = train_ppo(
        env            = env,
        env_params     = env_params,
        obs_dim        = obs_dim,
        num_actions    = num_actions,
        config         = config,
        seed           = seed,
        lr             = lr,
        gamma          = gamma,
        logger         = logger,
        met_df         = met_df,
        algorithm_name = "PPO",
        env_name       = env_name,
        reward_type    = reward_type,
    )

    policy_params = result["policy_params"]
    value_params  = result["value_params"]
    wall_time_s   = result["wall_time_s"]

    eval_stats = evaluate_ppo(
        env           = env,
        env_params    = env_params,
        policy_params = policy_params,
        num_episodes  = config.get("eval_episodes", 100),
        max_steps     = config.get("max_steps", 200),
        seed          = seed + 77777,
        greedy        = True,
        num_actions   = num_actions,
    )

    final_reward = eval_stats["mean_reward"]
    success_rate = eval_stats["success_rate"]
    mean_length  = eval_stats["mean_length"]

    logger.info(
        f"FINAL EVAL | mean_reward={final_reward:.2f} | "
        f"success_rate={success_rate:.3f} | "
        f"mean_length={mean_length:.1f} | "
        f"wall_time={wall_time_s:.1f}s"
    )

    met_df.add_summary(
        seed               = seed,
        algorithm          = "PPO",
        lr                 = lr,
        gamma              = gamma,
        final_mean_reward  = final_reward,
        final_success_rate = success_rate,
        backend            = jax.default_backend(),
        devices            = str(jax.devices()),
        action             = "stochastic",
        mean_length        = mean_length,
        wall_time_s        = wall_time_s,
        env_name           = env_name,
        reward_type        = reward_type,
    )

    print(
        f"  → reward={final_reward:.2f} | "
        f"success={success_rate:.3f} | "
        f"time={wall_time_s:.1f}s"
    )

    return {
        "policy_params": policy_params,
        "value_params":  value_params,
        "eval_stats":    eval_stats,
    }

#  Full sweep — loops over all combinations

def run_sweep(config: dict, quick: bool = False):
    if quick:
        learning_rates = [0.001]
        gammas         = [0.99]
        seeds          = [0]
        env_configs    = [{"env_name": "cartpole", "reward_type": "dense"}]
        print("QUICK MODE: 1 run (cartpole/dense, lr=0.001, γ=0.99, seed=0)\n")
    else:
        learning_rates = config["sweep"]["learning_rates"]
        gammas         = config["sweep"]["gammas"]
        seeds          = config["sweep"]["seeds"]
        env_configs    = [
            {"env_name": env_name, "reward_type": reward_type}
            for env_name    in config["sweep"]["environments"]
            for reward_type in config["sweep"]["reward_types"]
        ]

    met_df     = RLMetricsDataset("PPOBaseline")
    total_runs = len(learning_rates) * len(gammas) * len(seeds) * len(env_configs)
    completed  = 0

    print(f"PPO Baseline sweep — {total_runs} total runs")
    print(f"Backend: {jax.default_backend()} | Devices: {jax.devices()}\n")

    with Timer("Full PPO sweep") as sweep_timer:
        for env_cfg in env_configs:
            for lr in learning_rates:
                for gamma in gammas:
                    for seed in seeds:
                        completed += 1
                        print(
                            f"[{completed:3d}/{total_runs}] "
                            f"{env_cfg['env_name']}/{env_cfg['reward_type']} "
                            f"lr={lr} γ={gamma} seed={seed}"
                        )

                        try:
                            test_ppo(
                                seed              = seed,
                                gamma             = gamma,
                                lr                = lr,
                                config            = config,
                                config_env_params = env_cfg,
                                met_df            = met_df,
                            )
                        except Exception as e:
                            # Log the failure but continue the sweep —
                            # one bad run should not kill all 120
                            print(f"  ERROR in run {completed}: {e}")
                            continue

    paths = met_df.save()
    print(f"\nSweep complete in {sweep_timer.elapsed:.1f}s")
    print(f"Episode CSV : {paths['iteration']}")
    print(f"Summary CSV : {paths['summary']}")


def main():
    parser = argparse.ArgumentParser(description="PPO Baseline hyperparameter sweep")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a single quick test instead of the full 120-run sweep"
    )
    args = parser.parse_args()

    with open("configs/sweep.yaml") as f:
        config = yaml.safe_load(f)

    run_sweep(config, quick=args.quick)


if __name__ == "__main__":
    main()