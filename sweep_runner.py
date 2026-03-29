"""
sweep_runner.py — Run a single DQN or DQN+ICM experiment with CLI args.
Saves episode-level CSV to results/ directory.

Usage:
    python sweep_runner.py --agent dqn --env cartpole --reward dense --lr 0.001 --gamma 0.99 --seed 0
    python sweep_runner.py --agent dqn_icm --env cartpole --reward sparse --lr 0.001 --gamma 0.99 --seed 0 --eta 1.0
"""

import argparse
import os
import json
import csv
import time
import sys

import jax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=["dqn", "dqn_icm"])
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--reward", type=str, required=True, choices=["dense", "sparse"])
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    # ── Build run ID and output path ─────────────────────
    if args.agent == "dqn":
        run_id = f"dqn_{args.env}_{args.reward}_lr{args.lr}_g{args.gamma}_s{args.seed}"
        agent_dir = f"dqn_baseline/{args.env}_{args.reward}"
    else:
        run_id = f"dqn_icm_{args.env}_{args.reward}_lr{args.lr}_g{args.gamma}_s{args.seed}_eta{args.eta}"
        agent_dir = f"dqn_icm/{args.env}_{args.reward}"

    out_dir = os.path.join(args.results_dir, agent_dir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{run_id}.csv")

    # Skip if already completed
    if os.path.exists(csv_path):
        print(f"SKIP (already exists): {csv_path}")
        return

    print(f"START: {run_id}")
    t0 = time.time()

    # ── Run training ─────────────────────────────────────
    if args.agent == "dqn":
        from src.training.train_dqn import train_dqn

        agent, history = train_dqn(
            env_name=args.env,
            reward_type=args.reward,
            seed=args.seed,
            num_episodes=args.num_episodes,
            max_steps_per_episode=500,
            hidden_dim=64,
            learning_rate=args.lr,
            gamma=args.gamma,
            buffer_capacity=50000,
            batch_size=64,
            min_buffer_size_before_training=500,
            train_freq=4,
            target_update_freq=500,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=300,
            num_eval_episodes=10,
            eval_every=50,
        )
    else:
        from src.training.train_dqn_icm import train_dqn_icm

        agent, history = train_dqn_icm(
            env_name=args.env,
            reward_type=args.reward,
            seed=args.seed,
            num_episodes=args.num_episodes,
            max_steps_per_episode=500,
            q_hidden_dim=64,
            icm_hidden_dim=64,
            icm_feat_dim=64,
            q_learning_rate=args.lr,
            icm_learning_rate=args.lr,
            gamma=args.gamma,
            buffer_capacity=50000,
            batch_size=64,
            min_buffer_size_before_training=500,
            train_freq=4,
            target_update_freq=500,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=300,
            num_eval_episodes=10,
            eval_every=50,
            icm_eta=args.eta,
            icm_beta=0.2,
        )

    elapsed = time.time() - t0

    # ── Save episode-level CSV ───────────────────────────
    num_eps = len(history["episode_return"])

    # Build eval lookup for quick access
    eval_lookup = {}
    if "eval_episode_idx" in history:
        for i, ep_idx in enumerate(history["eval_episode_idx"]):
            eval_lookup[ep_idx] = {
                "eval_return_mean": history["eval_return_mean"][i],
                "eval_return_std": history.get("eval_return_std", [0.0] * len(history["eval_return_mean"]))[i],
                "eval_length_mean": history.get("eval_length_mean", [0.0] * len(history["eval_return_mean"]))[i],
            }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "return", "length", "epsilon",
            "loss", "q_mean", "td_error_mean",
            "icm_loss", "intrinsic_reward_mean",
            "eval_return_mean", "eval_return_std", "eval_length_mean",
            "agent", "env", "reward_type", "lr", "gamma", "seed", "eta",
        ])
        writer.writeheader()

        for ep in range(num_eps):
            ep_num = ep + 1
            eval_data = eval_lookup.get(ep_num, {})

            row = {
                "episode": ep_num,
                "return": history["episode_return"][ep],
                "length": history["episode_length"][ep],
                "epsilon": history["epsilon"][ep],
                "loss": history.get("train_loss", history.get("train_q_loss", [float("nan")] * num_eps))[ep],
                "q_mean": history.get("train_q_mean", [float("nan")] * num_eps)[ep],
                "td_error_mean": history.get("train_td_error_mean", [float("nan")] * num_eps)[ep],
                "icm_loss": history.get("train_icm_loss", [float("nan")] * num_eps)[ep] if args.agent == "dqn_icm" else "",
                "intrinsic_reward_mean": history.get("train_intrinsic_reward_mean", [float("nan")] * num_eps)[ep] if args.agent == "dqn_icm" else "",
                "eval_return_mean": eval_data.get("eval_return_mean", ""),
                "eval_return_std": eval_data.get("eval_return_std", ""),
                "eval_length_mean": eval_data.get("eval_length_mean", ""),
                "agent": args.agent,
                "env": args.env,
                "reward_type": args.reward,
                "lr": args.lr,
                "gamma": args.gamma,
                "seed": args.seed,
                "eta": args.eta if args.agent == "dqn_icm" else "",
            }
            writer.writerow(row)

    print(f"DONE: {run_id} | {elapsed:.0f}s | saved to {csv_path}")


if __name__ == "__main__":
    main()
