"""
plot_results.py — Generate all figures from sweep results.

Reads CSVs from results/ and produces:
  1. Learning curves (avg return vs episodes) per condition
  2. Episode length curves (key for sparse where return is binary)
  3. Eval return comparison across hyperparameters
  4. ICM intrinsic reward over training
  5. Summary bar charts
  6. Heatmaps (LR × γ)

Usage:
    python plot_results.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

RESULTS_DIR = "results"
PLOT_DIR = "visualizations"
WINDOW = 50  # smoothing window for learning curves


def load_all_csvs():
    """Load all CSV files from results/ into one DataFrame."""
    all_files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True)
    if not all_files:
        print(f"No CSV files found in {RESULTS_DIR}/")
        return None

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_files)} CSV files, {len(combined)} total rows")
    print(f"Agents: {combined['agent'].unique()}")
    print(f"Reward types: {combined['reward_type'].unique()}")
    print(f"LRs: {sorted(combined['lr'].unique())}")
    print(f"Gammas: {sorted(combined['gamma'].unique())}")
    print(f"Seeds: {sorted(combined['seed'].unique())}")
    if 'eta' in combined.columns:
        etas = combined.loc[combined['eta'].notna() & (combined['eta'] != ''), 'eta'].unique()
        print(f"ETAs: {sorted(etas)}")
    return combined


def smooth(series, window=WINDOW):
    """Rolling mean with min_periods=1 to avoid NaN at start."""
    return series.rolling(window=window, min_periods=1).mean()


def get_condition_label(agent, eta=None):
    """Human-readable label for plotting."""
    if agent == "dqn":
        return "DQN"
    elif agent == "dqn_icm":
        if eta is not None and eta != "":
            return f"DQN+ICM (η={eta})"
        return "DQN+ICM"
    return agent


def plot_learning_curves(df):
    """Plot average return vs episode for each condition, split by reward type."""
    os.makedirs(f"{PLOT_DIR}/learning_curves", exist_ok=True)

    for reward_type in df["reward_type"].unique():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        df_r = df[df["reward_type"] == reward_type]

        # Get all unique conditions
        conditions = []
        for agent in sorted(df_r["agent"].unique()):
            if agent == "dqn":
                conditions.append(("dqn", None))
            elif agent == "dqn_icm":
                etas = df_r.loc[(df_r["agent"] == agent) & df_r["eta"].notna() & (df_r["eta"] != ""), "eta"].unique()
                for eta in sorted(etas):
                    conditions.append(("dqn_icm", float(eta)))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Left plot: Return
        ax = axes[0]
        for i, (agent, eta) in enumerate(conditions):
            if eta is not None:
                mask = (df_r["agent"] == agent) & (df_r["eta"].astype(float) == eta)
            else:
                mask = (df_r["agent"] == agent)
            sub = df_r[mask]

            grouped = sub.groupby("episode")["return"].agg(["mean", "std"]).reset_index()
            grouped["smooth_mean"] = smooth(grouped["mean"])

            label = get_condition_label(agent, eta)
            color = colors[i % len(colors)]
            ax.plot(grouped["episode"], grouped["smooth_mean"], label=label, color=color, linewidth=2)
            ax.fill_between(
                grouped["episode"],
                grouped["smooth_mean"] - grouped["std"] / 2,
                grouped["smooth_mean"] + grouped["std"] / 2,
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.set_title(f"CartPole {reward_type.capitalize()} — Return", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Right plot: Episode Length
        ax = axes[1]
        for i, (agent, eta) in enumerate(conditions):
            if eta is not None:
                mask = (df_r["agent"] == agent) & (df_r["eta"].astype(float) == eta)
            else:
                mask = (df_r["agent"] == agent)
            sub = df_r[mask]

            grouped = sub.groupby("episode")["length"].agg(["mean", "std"]).reset_index()
            grouped["smooth_mean"] = smooth(grouped["mean"])

            label = get_condition_label(agent, eta)
            color = colors[i % len(colors)]
            ax.plot(grouped["episode"], grouped["smooth_mean"], label=label, color=color, linewidth=2)
            ax.fill_between(
                grouped["episode"],
                grouped["smooth_mean"] - grouped["std"] / 2,
                grouped["smooth_mean"] + grouped["std"] / 2,
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Average Episode Length", fontsize=12)
        ax.set_title(f"CartPole {reward_type.capitalize()} — Episode Length", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{PLOT_DIR}/learning_curves/cartpole_{reward_type}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_learning_curves_by_lr(df):
    """Plot learning curves split by learning rate."""
    os.makedirs(f"{PLOT_DIR}/learning_curves", exist_ok=True)

    for reward_type in df["reward_type"].unique():
        for lr in sorted(df["lr"].unique()):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            df_sub = df[(df["reward_type"] == reward_type) & (df["lr"] == lr)]

            if len(df_sub) == 0:
                plt.close()
                continue

            conditions = []
            for agent in sorted(df_sub["agent"].unique()):
                if agent == "dqn":
                    conditions.append(("dqn", None))
                elif agent == "dqn_icm":
                    etas = df_sub.loc[(df_sub["agent"] == agent) & df_sub["eta"].notna() & (df_sub["eta"] != ""), "eta"].unique()
                    for eta in sorted(etas):
                        conditions.append(("dqn_icm", float(eta)))

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

            for plot_idx, (col, ylabel) in enumerate([("return", "Return"), ("length", "Episode Length")]):
                ax = axes[plot_idx]
                for i, (agent, eta) in enumerate(conditions):
                    if eta is not None:
                        mask = (df_sub["agent"] == agent) & (df_sub["eta"].astype(float) == eta)
                    else:
                        mask = (df_sub["agent"] == agent)
                    sub = df_sub[mask]

                    grouped = sub.groupby("episode")[col].agg(["mean", "std"]).reset_index()
                    grouped["smooth_mean"] = smooth(grouped["mean"])

                    label = get_condition_label(agent, eta)
                    color = colors[i % len(colors)]
                    ax.plot(grouped["episode"], grouped["smooth_mean"], label=label, color=color, linewidth=2)
                    ax.fill_between(
                        grouped["episode"],
                        grouped["smooth_mean"] - grouped["std"] / 2,
                        grouped["smooth_mean"] + grouped["std"] / 2,
                        alpha=0.15, color=color,
                    )

                ax.set_xlabel("Episode", fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"CartPole {reward_type.capitalize()} — {ylabel} (lr={lr})", fontsize=13)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = f"{PLOT_DIR}/learning_curves/cartpole_{reward_type}_lr{lr}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")


def plot_intrinsic_rewards(df):
    """Plot ICM intrinsic reward magnitude over training."""
    os.makedirs(f"{PLOT_DIR}/intrinsic_rewards", exist_ok=True)

    df_icm = df[df["agent"] == "dqn_icm"].copy()
    df_icm["intrinsic_reward_mean"] = pd.to_numeric(df_icm["intrinsic_reward_mean"], errors="coerce")

    for reward_type in df_icm["reward_type"].unique():
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sub = df_icm[df_icm["reward_type"] == reward_type]

        etas = sorted(df_sub.loc[df_sub["eta"].notna() & (df_sub["eta"] != ""), "eta"].astype(float).unique())
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]

        for i, eta in enumerate(etas):
            sub = df_sub[df_sub["eta"].astype(float) == eta]
            grouped = sub.groupby("episode")["intrinsic_reward_mean"].agg(["mean", "std"]).reset_index()
            grouped["smooth_mean"] = smooth(grouped["mean"])

            color = colors[i % len(colors)]
            ax.plot(grouped["episode"], grouped["smooth_mean"],
                    label=f"η={eta}", color=color, linewidth=2)
            ax.fill_between(
                grouped["episode"],
                grouped["smooth_mean"] - grouped["std"] / 2,
                grouped["smooth_mean"] + grouped["std"] / 2,
                alpha=0.15, color=color,
            )

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Mean Intrinsic Reward", fontsize=12)
        ax.set_title(f"CartPole {reward_type.capitalize()} — ICM Intrinsic Reward", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{PLOT_DIR}/intrinsic_rewards/cartpole_{reward_type}_intrinsic.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_heatmaps(df):
    """Heatmap of final eval return for each LR × γ combination."""
    os.makedirs(f"{PLOT_DIR}/heatmaps", exist_ok=True)

    # Get eval data (last eval point per run)
    eval_rows = df[df["eval_return_mean"].notna() & (df["eval_return_mean"] != "")].copy()
    eval_rows["eval_return_mean"] = pd.to_numeric(eval_rows["eval_return_mean"], errors="coerce")

    # Get the last eval per run
    last_eval = eval_rows.groupby(
        ["agent", "reward_type", "lr", "gamma", "seed", "eta"]
    )["eval_return_mean"].last().reset_index()

    conditions = []
    for agent in sorted(last_eval["agent"].unique()):
        if agent == "dqn":
            conditions.append(("dqn", None, "DQN"))
        elif agent == "dqn_icm":
            etas = last_eval.loc[
                (last_eval["agent"] == agent) & last_eval["eta"].notna() & (last_eval["eta"] != ""),
                "eta"
            ].astype(float).unique()
            for eta in sorted(etas):
                conditions.append(("dqn_icm", eta, f"DQN+ICM (η={eta})"))

    for reward_type in sorted(last_eval["reward_type"].unique()):
        n_conditions = len(conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 5))
        if n_conditions == 1:
            axes = [axes]

        lrs = sorted(last_eval["lr"].unique())
        gammas = sorted(last_eval["gamma"].unique())

        for ax, (agent, eta, label) in zip(axes, conditions):
            if eta is not None:
                mask = (last_eval["agent"] == agent) & \
                       (last_eval["reward_type"] == reward_type) & \
                       (last_eval["eta"].astype(float) == eta)
            else:
                mask = (last_eval["agent"] == agent) & \
                       (last_eval["reward_type"] == reward_type)

            sub = last_eval[mask]

            heatmap_data = np.full((len(gammas), len(lrs)), np.nan)
            for gi, g in enumerate(gammas):
                for li, lr in enumerate(lrs):
                    vals = sub[(sub["gamma"] == g) & (sub["lr"] == lr)]["eval_return_mean"]
                    if len(vals) > 0:
                        heatmap_data[gi, li] = vals.mean()

            im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(lrs)))
            ax.set_xticklabels([str(lr) for lr in lrs])
            ax.set_yticks(range(len(gammas)))
            ax.set_yticklabels([str(g) for g in gammas])
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("γ")
            ax.set_title(label)

            # Annotate cells
            for gi in range(len(gammas)):
                for li in range(len(lrs)):
                    val = heatmap_data[gi, li]
                    if not np.isnan(val):
                        ax.text(li, gi, f"{val:.1f}", ha="center", va="center",
                                fontsize=11, fontweight="bold")

            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"CartPole {reward_type.capitalize()} — Final Eval Return (LR × γ)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = f"{PLOT_DIR}/heatmaps/cartpole_{reward_type}_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_summary_bars(df):
    """Bar chart comparing final performance across all conditions."""
    os.makedirs(f"{PLOT_DIR}/comparison_tables", exist_ok=True)

    # Use best LR/gamma per condition (highest final eval return averaged over seeds)
    eval_rows = df[df["eval_return_mean"].notna() & (df["eval_return_mean"] != "")].copy()
    eval_rows["eval_return_mean"] = pd.to_numeric(eval_rows["eval_return_mean"], errors="coerce")

    last_eval = eval_rows.groupby(
        ["agent", "reward_type", "lr", "gamma", "seed", "eta"]
    )["eval_return_mean"].last().reset_index()

    # Average over seeds, then pick best LR/gamma
    avg_over_seeds = last_eval.groupby(
        ["agent", "reward_type", "lr", "gamma", "eta"]
    )["eval_return_mean"].mean().reset_index()

    # For each (agent, reward_type, eta), pick best (lr, gamma)
    best_configs = avg_over_seeds.loc[
        avg_over_seeds.groupby(["agent", "reward_type", "eta"])["eval_return_mean"].idxmax()
    ]

    for reward_type in sorted(best_configs["reward_type"].unique()):
        sub = best_configs[best_configs["reward_type"] == reward_type]

        labels = []
        means = []
        for _, row in sub.iterrows():
            label = get_condition_label(row["agent"], row["eta"] if row["eta"] != "" else None)
            labels.append(f"{label}\n(lr={row['lr']}, γ={row['gamma']})")
            means.append(row["eval_return_mean"])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(labels)]
        bars = ax.bar(labels, means, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel("Best Final Eval Return", fontsize=12)
        ax.set_title(f"CartPole {reward_type.capitalize()} — Best Config Comparison", fontsize=14)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = f"{PLOT_DIR}/comparison_tables/cartpole_{reward_type}_bars.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def generate_summary_csv(df):
    """Generate a summary CSV with best config per condition."""
    eval_rows = df[df["eval_return_mean"].notna() & (df["eval_return_mean"] != "")].copy()
    eval_rows["eval_return_mean"] = pd.to_numeric(eval_rows["eval_return_mean"], errors="coerce")

    last_eval = eval_rows.groupby(
        ["agent", "reward_type", "lr", "gamma", "seed", "eta"]
    )["eval_return_mean"].last().reset_index()

    summary = last_eval.groupby(
        ["agent", "reward_type", "lr", "gamma", "eta"]
    ).agg(
        mean_return=("eval_return_mean", "mean"),
        std_return=("eval_return_mean", "std"),
        num_seeds=("eval_return_mean", "count"),
    ).reset_index()

    summary = summary.sort_values(["reward_type", "agent", "eta", "mean_return"], ascending=[True, True, True, False])

    path = f"{RESULTS_DIR}/summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved summary: {path}")

    # Print top configs
    print("\n" + "=" * 70)
    print("TOP CONFIGURATIONS (by mean final eval return)")
    print("=" * 70)
    for reward_type in sorted(summary["reward_type"].unique()):
        print(f"\n  {reward_type.upper()}:")
        sub = summary[summary["reward_type"] == reward_type].head(10)
        for _, row in sub.iterrows():
            label = get_condition_label(row["agent"], row["eta"] if row["eta"] != "" else None)
            print(f"    {label:25s} lr={row['lr']:<6} γ={row['gamma']:<5} "
                  f"return={row['mean_return']:>7.2f} ± {row['std_return']:.2f} "
                  f"({int(row['num_seeds'])} seeds)")


def main():
    print("=" * 60)
    print("GENERATING PLOTS FROM SWEEP RESULTS")
    print("=" * 60)

    df = load_all_csvs()
    if df is None:
        print("No data found. Run the sweep first.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\n--- Learning Curves (aggregated) ---")
    plot_learning_curves(df)

    print("\n--- Learning Curves (by LR) ---")
    plot_learning_curves_by_lr(df)

    print("\n--- Intrinsic Reward Curves ---")
    plot_intrinsic_rewards(df)

    print("\n--- Heatmaps (LR × γ) ---")
    plot_heatmaps(df)

    print("\n--- Summary Bar Charts ---")
    plot_summary_bars(df)

    print("\n--- Summary CSV ---")
    generate_summary_csv(df)

    print("\n" + "=" * 60)
    print(f"All plots saved to {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
