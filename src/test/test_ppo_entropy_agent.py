"""
test_ppo_entropy_agent.py — Sweep runner for PPO + Entropy Regularization.

Iterates over all (env, reward_type, lr, gamma, alpha, seed) combos defined
in sweep.yaml, trains a PPO+Entropy agent for each, and logs results via
RLMetricsDataset.

USAGE:
    python -m src.test.test_ppo_entropy_agent
"""

import jax
import jax.numpy as jnp

from src.envs.env_utils import get_env
from src.networks.policy_network import init_policy_params
from src.networks.value_network import init_value_params
from src.training.train_ppo_entropy import train_ppo_entropy
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.utils.reusable import (
    RLMetricsDataset,
    setup_logger,
    Timer,
    log_device_info,
    force_jax_gpu_or_warn,
    warmup_jit,
)


# ──────────────────────────────────────────────
#  Single-run function
# ──────────────────────────────────────────────

def test_ppo_entropy(seed, gamma, lr, config, config_env_params, met_df):
    """Train and evaluate one PPO+Entropy configuration.

    Args:
        seed:              random seed
        gamma:             discount factor
        lr:                learning rate
        config:            full sweep config dict
        config_env_params: dict with 'env_name' and 'reward_type'
        met_df:            shared RLMetricsDataset

    Returns:
        (trained_policy_params, trained_value_params, eval_stats)
    """
    env_name = config_env_params["env_name"]
    reward_type = config_env_params["reward_type"]
    alpha = config.get("alpha", 0.01)

    run_id = (
        f"ppo_entropy_{env_name}_{reward_type}_lr{lr}_g{gamma}"
        f"_a{alpha}_s{seed}"
    )
    logger = setup_logger(run_id, path="./logs/ppo_entropy")
    log_device_info(logger)
    backend = force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    logger.info("=" * 60)
    logger.info(f"RUN: {run_id}")
    logger.info(f"  env={env_name}  reward={reward_type}")
    logger.info(f"  lr={lr}  gamma={gamma}  alpha={alpha}  seed={seed}")
    logger.info("=" * 60)

    # ---- environment ----
    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    # ---- initialise networks ----
    key = jax.random.PRNGKey(seed)
    key, pk, vk = jax.random.split(key, 3)

    hidden_dim = config.get("hidden_dim", 64)
    policy_params = init_policy_params(
        pk, obs_dim=obs_dim, hidden_dim=hidden_dim, num_actions=num_actions
    )
    value_params = init_value_params(
        vk, obs_dim=obs_dim, hidden_dim=hidden_dim
    )

    # ---- training config ----
    train_config = {
        "num_episodes": config.get("num_episodes", 2000),
        "max_steps": config.get("max_steps", 200),
        "lr": lr,
        "gamma": gamma,
        "log_every": config.get("log_every", 50),
        "eval_episodes": config.get("eval_episodes", 100),
        "algorithm": "ppo_entropy",
        "env_name": env_name,
        "reward_type": reward_type,
        # PPO-specific
        "n_epochs": config.get("n_epochs", 4),
        "mini_batch_size": config.get("mini_batch_size", 64),
        "clip_epsilon": config.get("clip_epsilon", 0.2),
        "gae_lambda": config.get("gae_lambda", 0.95),
        "alpha": alpha,
    }

    # ---- train ----
    with Timer("Training") as t:
        results = train_ppo_entropy(
            env=env,
            env_params=env_params,
            policy_params=policy_params,
            value_params=value_params,
            config=train_config,
            logger=logger,
            met_df=met_df,
            evaluate_fn=evaluate_ppo,
            seed=seed,
        )

    logger.info(t.report())

    # ---- final greedy evaluation ----
    from src.agents.ppo_entropy_agent import PPOEntropyAgent

    key, eval_key = jax.random.split(key)
    final_agent = PPOEntropyAgent(
        results["policy_params"], results["value_params"]
    )
    final_sr = evaluate_ppo(
        env, env_params, final_agent, eval_key,
        num_episodes=config.get("eval_episodes", 100),
        max_steps=config.get("max_steps", 200),
    )

    last_rewards = results["episode_rewards"][-100:]
    final_mean_reward = sum(last_rewards) / len(last_rewards) if last_rewards else 0.0
    last_lengths = results["episode_lengths"][-100:]
    mean_length = sum(last_lengths) / len(last_lengths) if last_lengths else 0.0

    logger.info(
        f"FINAL | mean_reward(last100)={final_mean_reward:.2f} | "
        f"eval_sr={final_sr:.3f} | wall={t.elapsed:.1f}s"
    )

    # ---- summary ----
    met_df.add_summary(
        seed=seed,
        algorithm="ppo_entropy",
        lr=lr,
        gamma=gamma,
        final_mean_reward=final_mean_reward,
        final_success_rate=final_sr,
        backend=backend,
        devices=str(jax.devices()),
        action="stochastic",
        mean_length=mean_length,
        wall_time_s=t.elapsed,
        env_name=env_name,
        reward_type=reward_type,
    )

    return results["policy_params"], results["value_params"], {
        "final_mean_reward": final_mean_reward,
        "final_success_rate": final_sr,
    }


# ──────────────────────────────────────────────
#  Full sweep
# ──────────────────────────────────────────────

def run_full_sweep():
    """Run the complete PPO+Entropy sweep over all hyperparameter combos."""
    import yaml, os

    # Load sweep config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "sweep.yaml"
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)
    else:
        # Fallback defaults matching the uploaded sweep.yaml
        full_cfg = {
            "sweep": {
                "learning_rates": [0.0001, 0.0005, 0.001],
                "gammas": [0.99, 0.95],
                "seeds": [0, 1, 2, 3, 4],
                "environments": ["cartpole", "mountaincar"],
                "reward_types": ["dense", "sparse"],
            },
            "model": {"hidden_dim": 64},
            "num_episodes": 2000,
            "max_steps": 200,
            "eval_episodes": 100,
            "log_every": 50,
            "ppo": {
                "n_epochs": 4,
                "mini_batch_size": 64,
                "clip_epsilon": 0.2,
                "gae_lambda": 0.95,
            },
            "entropy": {"alpha": [0.01, 0.05]},
        }

    sweep = full_cfg["sweep"]
    ppo_cfg = full_cfg.get("ppo", {})
    entropy_cfg = full_cfg.get("entropy", {})
    alphas = entropy_cfg.get("alpha", [0.01, 0.05])
    if not isinstance(alphas, list):
        alphas = [alphas]

    base_config = {
        "num_episodes": full_cfg.get("num_episodes", 2000),
        "max_steps": full_cfg.get("max_steps", 200),
        "eval_episodes": full_cfg.get("eval_episodes", 100),
        "log_every": full_cfg.get("log_every", 50),
        "hidden_dim": full_cfg.get("model", {}).get("hidden_dim", 64),
        "n_epochs": ppo_cfg.get("n_epochs", 4),
        "mini_batch_size": ppo_cfg.get("mini_batch_size", 64),
        "clip_epsilon": ppo_cfg.get("clip_epsilon", 0.2),
        "gae_lambda": ppo_cfg.get("gae_lambda", 0.95),
    }

    met_df = RLMetricsDataset(proj_name="ppo_entropy")

    total_runs = (
        len(sweep["environments"])
        * len(sweep["reward_types"])
        * len(sweep["learning_rates"])
        * len(sweep["gammas"])
        * len(alphas)
        * len(sweep["seeds"])
    )
    run_idx = 0

    for env_name in sweep["environments"]:
        for reward_type in sweep["reward_types"]:
            for lr in sweep["learning_rates"]:
                for gamma in sweep["gammas"]:
                    for alpha in alphas:
                        for seed in sweep["seeds"]:
                            run_idx += 1
                            print(
                                f"\n{'='*60}\n"
                                f"Run {run_idx}/{total_runs}: "
                                f"env={env_name} rw={reward_type} "
                                f"lr={lr} g={gamma} a={alpha} s={seed}\n"
                                f"{'='*60}"
                            )
                            config = {**base_config, "alpha": alpha}
                            env_params = {
                                "env_name": env_name,
                                "reward_type": reward_type,
                            }
                            test_ppo_entropy(
                                seed=seed,
                                gamma=gamma,
                                lr=lr,
                                config=config,
                                config_env_params=env_params,
                                met_df=met_df,
                            )

    saved = met_df.save(output_dir="metrics", filename="ppo_entropy_metrics.csv")
    print(f"\nMetrics saved → {saved}")


if __name__ == "__main__":
    run_full_sweep()
