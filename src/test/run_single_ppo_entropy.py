#!/usr/bin/env python3
"""
run_single_ppo_entropy.py — Run ONE PPO+Entropy configuration from CLI args.

Called by the parallel bash script. Each invocation is its own process with
JAX pinned to CPU, so dozens can run side-by-side.

Usage:
    python -m src.test.run_single_ppo_entropy \
        --env cartpole --reward dense \
        --lr 0.0005 --gamma 0.99 --alpha 0.01 --seed 0
"""

import argparse
import os
import sys
import time

# Force CPU before JAX initialises (the bash script also sets this,
# but belt-and-suspenders never hurts).
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax                                          # noqa: E402 (after env vars)
import jax.numpy as jnp                             # noqa: E402

from src.envs.env_utils import get_env              # noqa: E402
from src.networks.policy_network import init_policy_params   # noqa: E402
from src.networks.value_network import init_value_params     # noqa: E402
from src.training.train_ppo_entropy import train_ppo_entropy # noqa: E402
from src.evaluate.evaluate_ppo import evaluate_ppo           # noqa: E402
from src.agents.ppo_entropy_agent import PPOEntropyAgent     # noqa: E402
from src.utils.reusable import (                             # noqa: E402
    RLMetricsDataset,
    setup_logger,
    Timer,
    log_device_info,
    force_jax_gpu_or_warn,
    warmup_jit,
)


def parse_args():
    p = argparse.ArgumentParser(description="Single PPO+Entropy run")
    p.add_argument("--env",         type=str,   required=True, choices=["cartpole", "mountaincar"])
    p.add_argument("--reward",      type=str,   required=True, choices=["dense", "sparse"])
    p.add_argument("--lr",          type=float, required=True)
    p.add_argument("--gamma",       type=float, required=True)
    p.add_argument("--alpha",       type=float, required=True)
    p.add_argument("--seed",        type=int,   required=True)
    # optional overrides
    p.add_argument("--num_episodes", type=int,   default=2000)
    p.add_argument("--max_steps",    type=int,   default=200)
    p.add_argument("--eval_episodes",type=int,   default=100)
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--hidden_dim",   type=int,   default=64)
    p.add_argument("--n_epochs",     type=int,   default=4)
    p.add_argument("--mini_batch",   type=int,   default=64)
    p.add_argument("--clip_eps",     type=float, default=0.2)
    p.add_argument("--gae_lambda",   type=float, default=0.95)
    p.add_argument("--metrics_dir",  type=str,   default="metrics")
    return p.parse_args()


def main():
    args = parse_args()

    run_id = (
        f"ppo_entropy_{args.env}_{args.reward}"
        f"_lr{args.lr}_g{args.gamma}_a{args.alpha}_s{args.seed}"
    )

    logger = setup_logger(run_id, path="./logs/ppo_entropy")
    log_device_info(logger)
    backend = force_jax_gpu_or_warn(logger)
    warmup_jit(logger)

    logger.info("=" * 60)
    logger.info(f"RUN: {run_id}")
    logger.info(f"  env={args.env}  reward={args.reward}")
    logger.info(f"  lr={args.lr}  gamma={args.gamma}  alpha={args.alpha}  seed={args.seed}")
    logger.info(f"  JAX backend={jax.default_backend()}  devices={jax.devices()}")
    logger.info("=" * 60)

    # ── environment ──
    env, env_params, obs_dim, num_actions = get_env(args.env, args.reward)

    # ── initialise networks ──
    key = jax.random.PRNGKey(args.seed)
    key, pk, vk = jax.random.split(key, 3)
    policy_params = init_policy_params(pk, obs_dim, args.hidden_dim, num_actions)
    value_params  = init_value_params(vk, obs_dim, args.hidden_dim)

    train_config = {
        "num_episodes":   args.num_episodes,
        "max_steps":      args.max_steps,
        "lr":             args.lr,
        "gamma":          args.gamma,
        "log_every":      args.log_every,
        "eval_episodes":  args.eval_episodes,
        "algorithm":      "ppo_entropy",
        "env_name":       args.env,
        "reward_type":    args.reward,
        "n_epochs":       args.n_epochs,
        "mini_batch_size":args.mini_batch,
        "clip_epsilon":   args.clip_eps,
        "gae_lambda":     args.gae_lambda,
        "alpha":          args.alpha,
    }

    met_df = RLMetricsDataset(proj_name="ppo_entropy")

    # ── train ──
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
            seed=args.seed,
        )
    logger.info(t.report())

    # ── final greedy evaluation ──
    key, eval_key = jax.random.split(key)
    final_agent = PPOEntropyAgent(results["policy_params"], results["value_params"])
    final_sr = evaluate_ppo(
        env, env_params, final_agent, eval_key,
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )

    last_100 = results["episode_rewards"][-100:]
    final_mean_reward = sum(last_100) / len(last_100) if last_100 else 0.0
    last_len = results["episode_lengths"][-100:]
    mean_length = sum(last_len) / len(last_len) if last_len else 0.0

    logger.info(
        f"FINAL | mean_reward(last100)={final_mean_reward:.2f} | "
        f"eval_sr={final_sr:.3f} | wall={t.elapsed:.1f}s"
    )

    met_df.add_summary(
        seed=args.seed,
        algorithm="ppo_entropy",
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

    # ── save per-run metrics (one CSV per run, merged later) ──
    os.makedirs(args.metrics_dir, exist_ok=True)
    met_df.save(
        output_dir=args.metrics_dir,
        filename=f"ppo_entropy_{run_id}.csv",
    )

    print(f"[DONE] {run_id}  reward={final_mean_reward:.2f}  sr={final_sr:.3f}  wall={t.elapsed:.1f}s")


if __name__ == "__main__":
    main()
