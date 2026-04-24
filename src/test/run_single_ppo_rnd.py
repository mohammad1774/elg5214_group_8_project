#!/usr/bin/env python3
"""
run_single_ppo_rnd.py — Run ONE PPO+RND configuration from CLI args.

Usage:
    python -m src.test.run_single_ppo_rnd \
        --env cartpole --reward dense \
        --lr 0.005 --gamma 0.99 --beta 0.1 --seed 1
"""

import argparse
import os

# Default CPU — overridden by Colab notebook when GPU is available
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

from src.envs.env_utils import get_env                       # noqa: E402
from src.networks.policy_network import init_policy_params   # noqa: E402
from src.networks.value_network import init_value_params     # noqa: E402
from src.networks.rnd_networks import init_rnd_params        # noqa: E402
from src.training.train_ppo_rnd import train_ppo_rnd         # noqa: E402
from src.evaluate.evaluate_ppo import evaluate_ppo           # noqa: E402
from src.utils.reusable import (                             # noqa: E402
    RLMetricsDataset,
    setup_logger,
    Timer,
    log_device_info,
    force_jax_gpu_or_warn,
    warmup_jit,
)


def parse_args():
    p = argparse.ArgumentParser(description="Single PPO+RND run")
    p.add_argument("--env",     type=str,   required=True,
                   choices=["cartpole", "mountaincar"])
    p.add_argument("--reward",  type=str,   required=True,
                   choices=["dense", "sparse"])
    p.add_argument("--lr",      type=float, required=True)
    p.add_argument("--gamma",   type=float, required=True)
    p.add_argument("--beta",    type=float, required=True)
    p.add_argument("--seed",    type=int,   required=True)

    # optional overrides
    p.add_argument("--num_episodes",  type=int,   default=2000)
    p.add_argument("--max_steps",     type=int,   default=200)
    p.add_argument("--eval_episodes", type=int,   default=50)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--n_epochs",      type=int,   default=4)
    p.add_argument("--mini_batch",    type=int,   default=64)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--gae_lambda",    type=float, default=0.95)
    p.add_argument("--predictor_lr",  type=float, default=0.001)
    p.add_argument("--embed_dim",     type=int,   default=32)
    p.add_argument("--metrics_dir",   type=str,   default="metrics")
    return p.parse_args()


def main():
    args = parse_args()

    run_id = (
        f"ppo_rnd_{args.env}_{args.reward}"
        f"_lr{args.lr}_g{args.gamma}_b{args.beta}_s{args.seed}"
    )

    logger = setup_logger(run_id, path="./logs/ppo_rnd")
    log_device_info(logger)
    backend = force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    logger.info("=" * 60)
    logger.info(f"RUN: {run_id}")
    logger.info(f"  env={args.env}  reward={args.reward}")
    logger.info(f"  lr={args.lr}  gamma={args.gamma}  beta={args.beta}  seed={args.seed}")
    logger.info(f"  predictor_lr={args.predictor_lr}  embed_dim={args.embed_dim}")
    logger.info(f"  JAX backend={jax.default_backend()}")
    logger.info("=" * 60)

    # ── environment ──
    env, env_params, obs_dim, num_actions = get_env(args.env, args.reward)

    # ── initialise networks ──
    key = jax.random.PRNGKey(args.seed)
    key, pk, vk, rk = jax.random.split(key, 4)
    policy_params = init_policy_params(pk, obs_dim, args.hidden_dim, num_actions)
    value_params = init_value_params(vk, obs_dim, args.hidden_dim)
    rnd_params = init_rnd_params(rk, obs_dim, args.hidden_dim, args.embed_dim)

    train_config = {
        "num_episodes":    args.num_episodes,
        "max_steps":       args.max_steps,
        "lr":              args.lr,
        "gamma":           args.gamma,
        "log_every":       args.log_every,
        "eval_episodes":   args.eval_episodes,
        "algorithm":       "ppo_rnd",
        "env_name":        args.env,
        "reward_type":     args.reward,
        "n_epochs":        args.n_epochs,
        "mini_batch_size": args.mini_batch,
        "clip_epsilon":    args.clip_eps,
        "gae_lambda":      args.gae_lambda,
        "beta":            args.beta,
        "predictor_lr":    args.predictor_lr,
    }

    met_df = RLMetricsDataset(proj_name="ppo_rnd")

    # ── train ──
    with Timer("Training") as t:
        results = train_ppo_rnd(
            env=env,
            env_params=env_params,
            policy_params=policy_params,
            value_params=value_params,
            rnd_params=rnd_params,
            config=train_config,
            logger=logger,
            met_df=met_df,
            evaluate_fn=evaluate_ppo,
            seed=args.seed,
            num_actions=num_actions,
        )
    logger.info(t.report())

    # ── final greedy eval ──
    final_eval = evaluate_ppo(
        env, env_params, results["policy_params"],
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed + 9999,
        greedy=True,
        num_actions=num_actions,
    )
    final_sr = float(final_eval["success_rate"])

    last_100 = results["episode_rewards"][-100:]
    final_mean_reward = sum(last_100) / len(last_100) if last_100 else 0.0
    last_lens = results["episode_lengths"][-100:]
    mean_length = sum(last_lens) / len(last_lens) if last_lens else 0.0

    last_intr = results["intrinsic_rewards"][-100:]
    mean_intrinsic = sum(last_intr) / len(last_intr) if last_intr else 0.0

    logger.info(
        f"FINAL | mean_rew(last100)={final_mean_reward:.2f} | "
        f"eval_sr={final_sr:.3f} | mean_intr={mean_intrinsic:.4f} | "
        f"wall={t.elapsed:.1f}s"
    )

    met_df.add_summary(
        seed=args.seed,
        algorithm="ppo_rnd",
        lr=args.lr,
        gamma=args.gamma,
        final_mean_reward=final_mean_reward,
        final_success_rate=final_sr,
        backend=backend,
        devices=str(jax.devices()),
        action="stochastic",
        mean_length=mean_length,
        wall_time_s=t.elapsed,
        env_name=args.env,
        reward_type=args.reward,
    )

    os.makedirs(args.metrics_dir, exist_ok=True)
    met_df.save(output_dir=args.metrics_dir, filename=f"ppo_rnd_{run_id}.csv")

    print(
        f"[DONE] {run_id}  ext_rew={final_mean_reward:.2f}  "
        f"sr={final_sr:.3f}  intr={mean_intrinsic:.4f}  "
        f"wall={t.elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
