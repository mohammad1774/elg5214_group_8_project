#!/usr/bin/env python3
"""
generate_plots.py

Unified plotting script for:
- dqn_entropy
- dqn_rnd
- ppo_baseline

Usage:
    python scripts/generate_plots.py --model dqn_entropy
    python scripts/generate_plots.py --model dqn_rnd
    python scripts/generate_plots.py --model ppo_baseline

Optional:
    python scripts/generate_plots.py --model ppo_baseline \
        --episode-csv metrics/PPOBaseline_dataset_metrics.csv \
        --summary-csv metrics/PPOBaseline_dataset_metrics_summary.csv \
        --out-dir visualizations/ppo_baseline
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VALID_MODELS = {"dqn_entropy", "dqn_rnd", "ppo_baseline"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def standard_error(series: pd.Series) -> float:
    series = series.dropna()
    n = len(series)
    if n <= 1:
        return 0.0
    return float(series.std(ddof=1) / math.sqrt(n))


def smooth_series(series: pd.Series, window: int = 10) -> pd.Series:
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def prettify_model(model: str) -> str:
    mapping = {
        "dqn_entropy": "DQN + Entropy",
        "dqn_rnd": "DQN + RND",
        "ppo_baseline": "PPO Baseline",
    }
    return mapping.get(model, model.replace("_", " ").upper())


def mech_symbol(model: str) -> str:
    if model == "dqn_entropy":
        return "α"
    if model == "dqn_rnd":
        return "β"
    return ""


def mech_col(model: str) -> Optional[str]:
    if model == "dqn_entropy":
        return "alpha"
    if model == "dqn_rnd":
        return "beta"
    return None


def metric_dir_name(model: str) -> str:
    if model == "dqn_entropy":
        return "policy_entropy"
    if model == "dqn_rnd":
        return "intrinsic_rewards"
    return "policy_entropy"


def default_paths(model: str) -> tuple[Path, Path, Path]:
    if model == "ppo_baseline":
        return (
            Path("metrics/PPOBaseline_dataset_metrics.csv"),
            Path("metrics/PPOBaseline_dataset_metrics_summary.csv"),
            Path("visualizations/ppo_baseline"),
        )

    return (
        Path("metrics") / f"{model}_all_episodes.csv",
        Path("metrics") / f"{model}_all_summary.csv",
        Path("visualizations") / model,
    )


def validate_episode_df(df: pd.DataFrame, model: str) -> None:
    required = [
        "seed",
        "episode",
        "reward",
        "episode_length",
        "loss",
        "learning_rate",
        "gamma",
        "env_name",
        "reward_type",
    ]

    if model == "dqn_entropy":
        required += ["alpha", "policy_entropy"]
    elif model == "dqn_rnd":
        required += ["beta", "intrinsic_reward"]
    elif model == "ppo_baseline":
        pass

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Episode CSV missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def validate_summary_df(df: pd.DataFrame, model: str) -> None:
    required = {
        "seed",
        "algorithm",
        "learning_rate",
        "gamma",
        "final_mean_reward",
        "final_success_rate",
        "mean_length",
        "wall_time_s",
        "env_name",
        "reward_type",
    }

    mcol = mech_col(model)
    if mcol is not None:
        required.add(mcol)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Summary CSV missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )


def aggregate_episode_metric(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols + ["episode"], dropna=False)[metric]
        .agg(mean="mean", se=standard_error)
        .reset_index()
    )
    return agg


def config_group_cols(model: str) -> list[str]:
    cols = ["env_name", "reward_type", "learning_rate", "gamma"]
    mcol = mech_col(model)
    if mcol is not None:
        cols.append(mcol)
    return cols


def config_curve_cols(model: str) -> list[str]:
    cols = ["learning_rate", "gamma"]
    mcol = mech_col(model)
    if mcol is not None:
        cols.append(mcol)
    return cols


def format_cfg_label(model: str, keys) -> str:
    mcol = mech_col(model)
    if mcol is not None:
        lr, gamma, mech = keys
        return f"lr={lr}, γ={gamma}, {mech_symbol(model)}={mech}"
    lr, gamma = keys
    return f"lr={lr}, γ={gamma}"


def plot_learning_curves(
    episodes_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    agg = aggregate_episode_metric(
        episodes_df,
        metric="reward",
        group_cols=config_group_cols(model),
    )

    for (env_name, reward_type), sub in agg.groupby(["env_name", "reward_type"], dropna=False):
        fig, ax = plt.subplots(figsize=(10, 6))

        for keys, curve in sub.groupby(config_curve_cols(model), dropna=False):
            curve = curve.sort_values("episode")
            mean_y = smooth_series(curve["mean"], window=10)
            se_y = curve["se"]

            ax.plot(curve["episode"], mean_y, label=format_cfg_label(model, keys))
            ax.fill_between(
                curve["episode"],
                curve["mean"] - se_y,
                curve["mean"] + se_y,
                alpha=0.15,
            )

        ax.set_title(f"{prettify_model(model)} — Learning Curves — {env_name} ({reward_type})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        save_plot(fig, out_dir / f"learning_curve_{model}_{env_name}_{reward_type}.png")


def plot_seed_overlays(
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    cfg_cols = config_group_cols(model)
    best_cfg = (
        summary_df.groupby(cfg_cols, dropna=False)["final_mean_reward"]
        .mean()
        .reset_index()
    )

    mcol = mech_col(model)

    for (env_name, reward_type), sub in best_cfg.groupby(["env_name", "reward_type"], dropna=False):
        best_row = sub.sort_values("final_mean_reward", ascending=False).iloc[0]

        mask = (
            (episodes_df["env_name"] == env_name) &
            (episodes_df["reward_type"] == reward_type) &
            (episodes_df["learning_rate"] == best_row["learning_rate"]) &
            (episodes_df["gamma"] == best_row["gamma"])
        )
        if mcol is not None:
            mask &= (episodes_df[mcol] == best_row[mcol])

        plot_df = episodes_df.loc[mask].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        for seed, sdf in plot_df.groupby("seed", dropna=False):
            sdf = sdf.sort_values("episode")
            ax.plot(
                sdf["episode"],
                smooth_series(sdf["reward"], window=10),
                label=f"seed={seed}",
                alpha=0.9,
            )

        if mcol is not None:
            title_cfg = (
                f"lr={best_row['learning_rate']}, "
                f"γ={best_row['gamma']}, "
                f"{mech_symbol(model)}={best_row[mcol]}"
            )
        else:
            title_cfg = (
                f"lr={best_row['learning_rate']}, "
                f"γ={best_row['gamma']}"
            )

        ax.set_title(
            f"{prettify_model(model)} — Seed Overlays — {env_name} ({reward_type})\n"
            f"Best config: {title_cfg}"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        save_plot(fig, out_dir / f"seed_overlay_{model}_{env_name}_{reward_type}.png")


def plot_heatmaps(
    summary_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    mcol = mech_col(model)

    for metric in ["final_mean_reward", "final_success_rate"]:
        group_cols = ["env_name", "reward_type", "learning_rate", "gamma"]
        if mcol is not None:
            group_cols.insert(2, mcol)

        grouped = (
            summary_df.groupby(group_cols, dropna=False)[metric]
            .mean()
            .reset_index()
        )

        if mcol is not None:
            grouped_iter = grouped.groupby(["env_name", "reward_type", mcol], dropna=False)
        else:
            grouped_iter = grouped.groupby(["env_name", "reward_type"], dropna=False)

        for keys, sub in grouped_iter:
            if mcol is not None:
                env_name, reward_type, mech_val = keys
            else:
                env_name, reward_type = keys
                mech_val = None

            lr_vals = sorted(sub["learning_rate"].unique().tolist())
            gamma_vals = sorted(sub["gamma"].unique().tolist())

            matrix = np.full((len(gamma_vals), len(lr_vals)), np.nan)

            for _, row in sub.iterrows():
                i = gamma_vals.index(row["gamma"])
                j = lr_vals.index(row["learning_rate"])
                matrix[i, j] = row[metric]

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(matrix, aspect="auto")

            ax.set_xticks(range(len(lr_vals)))
            ax.set_xticklabels([str(x) for x in lr_vals])
            ax.set_yticks(range(len(gamma_vals)))
            ax.set_yticklabels([str(x) for x in gamma_vals])

            for i in range(len(gamma_vals)):
                for j in range(len(lr_vals)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

            metric_name = "reward" if metric == "final_mean_reward" else "success"

            if mcol is not None:
                ax.set_title(
                    f"{prettify_model(model)} — {metric_name.capitalize()} Heatmap\n"
                    f"{env_name} ({reward_type}), {mech_symbol(model)}={mech_val}"
                )
                out_name = f"heatmap_{metric_name}_{model}_{env_name}_{reward_type}_{mech_val}.png"
            else:
                ax.set_title(
                    f"{prettify_model(model)} — {metric_name.capitalize()} Heatmap\n"
                    f"{env_name} ({reward_type})"
                )
                out_name = f"heatmap_{metric_name}_{model}_{env_name}_{reward_type}.png"

            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Gamma")
            fig.colorbar(im, ax=ax)

            save_plot(fig, out_dir / out_name)


def plot_model_specific_metric(
    episodes_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    if model == "dqn_entropy":
        metric = "policy_entropy"
        ylabel = "Policy Entropy"
        prefix = "entropy_curve"
        title_metric = "Entropy"
    elif model == "dqn_rnd":
        metric = "intrinsic_reward"
        ylabel = "Intrinsic Reward"
        prefix = "intrinsic_reward_curve"
        title_metric = "Intrinsic Reward"
    else:
        metric = "policy_entropy"
        ylabel = "Policy Entropy"
        prefix = "entropy_curve"
        title_metric = "Entropy"

    agg = aggregate_episode_metric(
        episodes_df,
        metric=metric,
        group_cols=config_group_cols(model),
    )

    for (env_name, reward_type), sub in agg.groupby(["env_name", "reward_type"], dropna=False):
        fig, ax = plt.subplots(figsize=(10, 6))

        for keys, curve in sub.groupby(config_curve_cols(model), dropna=False):
            curve = curve.sort_values("episode")
            ax.plot(
                curve["episode"],
                smooth_series(curve["mean"], window=10),
                label=format_cfg_label(model, keys),
            )
            ax.fill_between(
                curve["episode"],
                curve["mean"] - curve["se"],
                curve["mean"] + curve["se"],
                alpha=0.15,
            )

        ax.set_title(f"{prettify_model(model)} — {title_metric} — {env_name} ({reward_type})")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        save_plot(fig, out_dir / f"{prefix}_{model}_{env_name}_{reward_type}.png")


def plot_summary_bars(
    summary_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    cfg_cols = config_group_cols(model)

    grouped_reward = (
        summary_df.groupby(cfg_cols, dropna=False)["final_mean_reward"]
        .mean()
        .reset_index()
    )

    best_reward_rows = []
    for (_, _), sub in grouped_reward.groupby(["env_name", "reward_type"], dropna=False):
        best_reward_rows.append(sub.sort_values("final_mean_reward", ascending=False).iloc[0])
    best_reward_df = pd.DataFrame(best_reward_rows)

    labels = [f"{r.env_name}\n{r.reward_type}" for r in best_reward_df.itertuples()]
    values = best_reward_df["final_mean_reward"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values)
    ax.set_title(f"{prettify_model(model)} — Best Final Reward by Condition")
    ax.set_ylabel("Final Mean Reward")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    save_plot(fig, out_dir / f"bar_final_reward_{model}.png")

    grouped_success = (
        summary_df.groupby(cfg_cols, dropna=False)["final_success_rate"]
        .mean()
        .reset_index()
    )

    best_success_rows = []
    for (_, _), sub in grouped_success.groupby(["env_name", "reward_type"], dropna=False):
        best_success_rows.append(sub.sort_values("final_success_rate", ascending=False).iloc[0])
    best_success_df = pd.DataFrame(best_success_rows)

    labels = [f"{r.env_name}\n{r.reward_type}" for r in best_success_df.itertuples()]
    values = best_success_df["final_success_rate"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values)
    ax.set_title(f"{prettify_model(model)} — Best Final Success Rate by Condition")
    ax.set_ylabel("Final Success Rate")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    save_plot(fig, out_dir / f"bar_final_success_{model}.png")


def plot_dense_vs_sparse(
    summary_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    cfg_cols = config_group_cols(model)

    grouped = (
        summary_df.groupby(cfg_cols, dropna=False)["final_mean_reward"]
        .mean()
        .reset_index()
    )

    best_rows = []
    for (_, _), sub in grouped.groupby(["env_name", "reward_type"], dropna=False):
        best_rows.append(sub.sort_values("final_mean_reward", ascending=False).iloc[0])

    best_df = pd.DataFrame(best_rows)
    envs = sorted(best_df["env_name"].unique().tolist())

    dense_vals = []
    sparse_vals = []
    for env in envs:
        drow = best_df[(best_df["env_name"] == env) & (best_df["reward_type"] == "dense")]
        srow = best_df[(best_df["env_name"] == env) & (best_df["reward_type"] == "sparse")]

        dense_vals.append(float(drow.iloc[0]["final_mean_reward"]) if not drow.empty else np.nan)
        sparse_vals.append(float(srow.iloc[0]["final_mean_reward"]) if not srow.empty else np.nan)

    x = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - width / 2, dense_vals, width, label="dense")
    b2 = ax.bar(x + width / 2, sparse_vals, width, label="sparse")

    ax.set_title(f"{prettify_model(model)} — Dense vs Sparse")
    ax.set_ylabel("Best Final Mean Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=9)

    save_plot(fig, out_dir / f"dense_vs_sparse_{model}.png")


def plot_mechanism_strength_comparison(
    summary_df: pd.DataFrame,
    model: str,
    out_dir: Path,
) -> None:
    mcol = mech_col(model)
    if mcol is None:
        return

    grouped = (
        summary_df.groupby(["env_name", "reward_type", mcol], dropna=False)["final_mean_reward"]
        .mean()
        .reset_index()
    )

    for (env_name, reward_type), sub in grouped.groupby(["env_name", "reward_type"], dropna=False):
        sub = sub.sort_values(mcol)

        labels = [str(v) for v in sub[mcol].tolist()]
        vals = sub["final_mean_reward"].tolist()

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, vals)

        ax.set_title(
            f"{prettify_model(model)} — {mech_symbol(model)} Comparison — {env_name} ({reward_type})"
        )
        ax.set_xlabel(mech_symbol(model))
        ax.set_ylabel("Mean Final Reward")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)

        save_plot(fig, out_dir / f"mech_comparison_{model}_{env_name}_{reward_type}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for RL experiment models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(VALID_MODELS),
        help="Which model to plot: dqn_entropy, dqn_rnd, or ppo_baseline",
    )
    parser.add_argument(
        "--episode-csv",
        type=str,
        default=None,
        help="Path to episode CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to summary CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path to output visualization directory.",
    )
    args = parser.parse_args()

    model = args.model
    default_episode_csv, default_summary_csv, default_out_dir = default_paths(model)

    episode_csv = Path(args.episode_csv) if args.episode_csv else default_episode_csv
    summary_csv = Path(args.summary_csv) if args.summary_csv else default_summary_csv
    out_root = Path(args.out_dir) if args.out_dir else default_out_dir

    if not episode_csv.exists():
        raise FileNotFoundError(f"Episode CSV not found: {episode_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    episodes_df = pd.read_csv(episode_csv)
    summary_df = pd.read_csv(summary_csv)

    validate_episode_df(episodes_df, model)
    validate_summary_df(summary_df, model)

    learning_dir = out_root / "learning_curves"
    heatmap_dir = out_root / "heatmaps"
    metric_dir = out_root / metric_dir_name(model)
    compare_dir = out_root / "comparison_tables"

    for d in [learning_dir, heatmap_dir, metric_dir, compare_dir]:
        ensure_dir(d)

    print(f"[INFO] model       : {model}")
    print(f"[INFO] episode csv : {episode_csv}")
    print(f"[INFO] summary csv : {summary_csv}")
    print(f"[INFO] output dir  : {out_root}")

    plot_learning_curves(episodes_df, model, learning_dir)
    plot_seed_overlays(episodes_df, summary_df, model, learning_dir)
    plot_heatmaps(summary_df, model, heatmap_dir)
    plot_model_specific_metric(episodes_df, model, metric_dir)
    plot_summary_bars(summary_df, model, compare_dir)
    plot_dense_vs_sparse(summary_df, model, compare_dir)
    plot_mechanism_strength_comparison(summary_df, model, compare_dir)

    print("[INFO] All plots generated successfully.")


if __name__ == "__main__":
    main()