#!/usr/bin/env python3
"""Plot MountainCar 8000 training progress from log files."""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

LOG_ROOT = Path("logs")
OUT_DIR = Path("visualizations")
OUT_DIR.mkdir(exist_ok=True)

PATTERN = re.compile(
    r"\[Episode\s+(\d+)\].*?avg_length=\s*([0-9.]+).*?eval_success=\s*([0-9.]+)"
)


def parse_log(log_path):
    records = []
    line = None
    for line in log_path.read_text().splitlines():
        m = PATTERN.search(line)
        if m:
            episode = int(m.group(1))
            avg_length = float(m.group(2))
            eval_success = float(m.group(3))
            records.append({"episode": episode, "avg_length": avg_length, "eval_success": eval_success})
    return pd.DataFrame(records)


def load_data():
    rows = []
    for algo_dir in ["dqn_entropy_mc_8000", "dqn_rnd_mc_8000"]:
        for log_path in (LOG_ROOT / algo_dir).glob("*.log"):
            name = log_path.stem
            parts = name.split("_")
            if len(parts) < 5:
                continue
            seed = parts[1]
            lr = parts[2]
            gamma = parts[3]
            mech = parts[4]
            reward = parts[-1]
            algorithm = "DQN_Entropy" if algo_dir.startswith("dqn_entropy") else "DQN_RND"
            df = parse_log(log_path)
            if df.empty:
                continue
            df["algorithm"] = algorithm
            df["reward"] = reward
            df["setting"] = f"{algorithm} {reward}"
            df["log_file"] = str(log_path)
            rows.append(df)
    if not rows:
        raise SystemExit("No log data found.")
    return pd.concat(rows, ignore_index=True)


def plot_algorithm(df, algorithm):
    subset = df[df["algorithm"] == algorithm]
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for reward_type, group in subset.groupby("reward"):
        axes[0].plot(group["episode"], group["eval_success"], marker="o", label=reward_type)
        axes[1].plot(group["episode"], group["avg_length"], marker="o", label=reward_type)

    axes[0].set_ylabel("Eval Success Rate")
    axes[0].set_title(f"{algorithm}: Eval Success over Training")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_ylabel("Avg Episode Length")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_title(f"{algorithm}: Avg Episode Length over Training")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    out_path = OUT_DIR / f"mountaincar_8000_{algorithm.lower()}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


def main():
    df = load_data()
    for algo in df["algorithm"].unique():
        plot_algorithm(df, algo)
    combined = df.copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for (algo, reward), group in combined.groupby(["algorithm", "reward"]):
        label = f"{algo} {reward}"
        axes[0].plot(group["episode"], group["eval_success"], marker="o", label=label)
        axes[1].plot(group["episode"], group["avg_length"], marker="o", label=label)
    axes[0].set_ylabel("Eval Success Rate")
    axes[0].set_title("MountainCar 8000: Eval Success over Training")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].set_ylabel("Avg Episode Length")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_title("MountainCar 8000: Avg Length over Training")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    out_path = OUT_DIR / "mountaincar_8000_all.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved combined plot: {out_path}")


if __name__ == "__main__":
    main()
