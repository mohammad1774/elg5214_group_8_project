"""
test_ppo_icm_agent.py — PPO + ICM sweep script.

Sweep grid adds eta (ICM curiosity scale) on top of the base PPO sweep:
    3 LR × 2 gamma × 5 seeds × 2 envs × 2 reward types × 2 eta values
    = 240 runs total

HOW TO RUN:
    Full sweep:
        python -m src.test.test_ppo_icm_agent

    Quick test (1 run):
        python -m src.test.test_ppo_icm_agent --quick
"""

import argparse

import jax
import yaml

from src.envs.env_utils import get_env
from src.networks.policy_network import init_policy_params, policy_forward, log_prob, entropy
from src.networks.value_network import init_value_params, value_forward
from src.networks.icm_networks import init_icm_params
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.training.train_ppo_icm import train_ppo_icm
from src.utils.reusable import RLMetricsDataset, setup_logger, Timer

#  Single-run function
def test_ppo_icm(
    seed: int,
    gamma: float,
    lr: float,
    eta: float,
    config: dict,
    config_env_params: dict,
    met_df: RLMetricsDataset,
) -> dict:

    env_name    = config_env_params["env_name"]
    reward_type = config_env_params["reward_type"]

    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    run_id = abs(hash(
        f"ppo_icm_{env_name}_{reward_type}_lr{lr}_g{gamma}_eta{eta}_s{seed}"
    )) % 100000
    logger = setup_logger(run_id, path="./logs/ppo_icm")
    logger.info(f"JAX devices : {jax.devices()}")
    logger.info(f"JAX backend : {jax.default_backend()}")
    logger.info(
        f"Run config  : env={env_name}/{reward_type} "
        f"seed={seed} lr={lr} gamma={gamma} eta={eta}"
    )

    result = train_ppo_icm(
        env            = env,
        env_params     = env_params,
        obs_dim        = obs_dim,
        num_actions    = num_actions,
        config         = config,
        seed           = seed,
        lr             = lr,
        gamma          = gamma,
        eta            = eta,
        logger         = logger,
        met_df         = met_df,
        algorithm_name = "PPO_ICM",
        env_name       = env_name,
        reward_type    = reward_type,
    )

    policy_params = result["policy_params"]
    value_params  = result["value_params"]
    icm_params    = result["icm_params"]
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
        algorithm          = "PPO_ICM",
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
        "icm_params":    icm_params,
        "eval_stats":    eval_stats,
    }

#  Full sweep
def run_sweep(config: dict, quick: bool = False):
    """Loop over all hyperparameter combinations.

    Sweep grid:
        learning_rates : [0.0001, 0.0005, 0.001]
        gammas         : [0.99, 0.95]
        seeds          : [0, 1, 2, 3, 4]
        environments   : [cartpole, mountaincar]
        reward_types   : [dense, sparse]
        eta (ICM scale): [0.1, 1.0]

    Full sweep: 3 × 2 × 5 × 2 × 2 × 2 = 240 runs
    """
    if quick:
        learning_rates = [0.001]
        gammas         = [0.99]
        seeds          = [0]
        eta_values     = [1.0]
        env_configs    = [{"env_name": "cartpole", "reward_type": "dense"}]
        print("QUICK MODE: 1 run (cartpole/dense, lr=0.001, γ=0.99, eta=1.0, seed=0)\n")
    else:
        learning_rates = config["sweep"]["learning_rates"]
        gammas         = config["sweep"]["gammas"]
        seeds          = config["sweep"]["seeds"]
        eta_values     = config.get("icm", {}).get("eta", [0.1, 1.0])
        env_configs    = [
            {"env_name": env_name, "reward_type": reward_type}
            for env_name    in config["sweep"]["environments"]
            for reward_type in config["sweep"]["reward_types"]
        ]

    met_df     = RLMetricsDataset("PPO_ICM")
    total_runs = (
        len(learning_rates) * len(gammas) * len(seeds)
        * len(env_configs) * len(eta_values)
    )
    completed = 0

    print(f"PPO+ICM sweep — {total_runs} total runs")
    print(f"Backend: {jax.default_backend()} | Devices: {jax.devices()}\n")

    with Timer("Full PPO+ICM sweep") as sweep_timer:
        for env_cfg in env_configs:
            for eta in eta_values:
                for lr in learning_rates:
                    for gamma in gammas:
                        for seed in seeds:
                            completed += 1
                            print(
                                f"[{completed:3d}/{total_runs}] "
                                f"{env_cfg['env_name']}/{env_cfg['reward_type']} "
                                f"lr={lr} γ={gamma} eta={eta} seed={seed}"
                            )

                            try:
                                test_ppo_icm(
                                    seed              = seed,
                                    gamma             = gamma,
                                    lr                = lr,
                                    eta               = eta,
                                    config            = config,
                                    config_env_params = env_cfg,
                                    met_df            = met_df,
                                )
                            except Exception as e:
                                print(f"  ERROR in run {completed}: {e}")
                                continue

    paths = met_df.save()
    print(f"\nSweep complete in {sweep_timer.elapsed:.1f}s")
    print(f"Episode CSV : {paths['iteration']}")
    print(f"Summary CSV : {paths['summary']}")

def main():
    parser = argparse.ArgumentParser(description="PPO+ICM hyperparameter sweep")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run 1 test run (cartpole/dense, lr=0.001, γ=0.99, eta=1.0, seed=0)"
    )
    args = parser.parse_args()

    with open("configs/sweep.yaml") as f:
        config = yaml.safe_load(f)

    run_sweep(config, quick=args.quick)


if __name__ == "__main__":
    main()