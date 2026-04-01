"""
aggregate_results.py — Combine all per-run CSVs into master episode + summary files.

Scans results/<agent_type>/<env_reward>/ for individual CSVs,
parses alpha/beta from filenames, and outputs:
    metrics/<agent_type>_all_episodes.csv
    metrics/<agent_type>_all_summary.csv

Usage:
    python scripts/aggregate_results.py --agent dqn_entropy --results-dir results/dqn_entropy
    python scripts/aggregate_results.py --agent dqn_rnd --results-dir results/dqn_rnd

Student A (Mohammad)
"""

import os
import re
import glob
import argparse
import pandas as pd


def parse_filename(filepath):
    """Extract hyperparameters from filename.

    Handles patterns like:
        lr0.01_g0.99_a0.05_seed3.csv        → alpha=0.05
        lr0.01_g0.99_b0.1_seed3.csv         → beta=0.1
        lr0.01_g0.99_a0.05_seed3_summary.csv → alpha=0.05, is_summary=True
    """
    fname = os.path.basename(filepath)
    is_summary = "_summary" in fname
    fname_clean = fname.replace("_summary", "").replace(".csv", "")

    parts = fname_clean.split("_")
    params = {}

    for p in parts:
        if p.startswith("lr"):
            params["lr"] = float(p[2:])
        elif p.startswith("g"):
            try:
                params["gamma"] = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("a") and not p.startswith("al"):
            try:
                params["alpha"] = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("b"):
            try:
                params["beta"] = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("seed"):
            params["seed"] = int(p[4:])

    return params, is_summary


def aggregate(agent_type, results_dir, output_dir="metrics"):
    """Combine all CSVs under results_dir into two master files."""

    os.makedirs(output_dir, exist_ok=True)

    all_episodes = []
    all_summaries = []
    file_count = 0

    for env_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(env_dir):
            continue

        env_reward = os.path.basename(env_dir)  # e.g. "cartpole_dense"

        for csv_file in sorted(glob.glob(os.path.join(env_dir, "*.csv"))):
            params, is_summary = parse_filename(csv_file)

            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"  WARN: skipping {csv_file}: {e}")
                continue

            if len(df) == 0:
                continue

            # Add parsed params as columns (fills in alpha/beta from filename)
            df["env_reward"] = env_reward
            if "alpha" in params:
                df["alpha"] = params["alpha"]
            if "beta" in params:
                df["beta"] = params["beta"]

            if is_summary:
                all_summaries.append(df)
            else:
                all_episodes.append(df)
                file_count += 1

    if not all_episodes:
        print(f"No episode CSVs found in {results_dir}")
        return None, None

    episodes_df = pd.concat(all_episodes, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    # Save
    ep_path = os.path.join(output_dir, f"{agent_type}_all_episodes.csv")
    sum_path = os.path.join(output_dir, f"{agent_type}_all_summary.csv")

    episodes_df.to_csv(ep_path, index=False)
    if len(summary_df) > 0:
        summary_df.to_csv(sum_path, index=False)

    print(f"Aggregated {file_count} episode files from {results_dir}")
    print(f"  Episodes: {len(episodes_df)} rows → {ep_path}")
    print(f"  Summary:  {len(summary_df)} rows → {sum_path}")
    print(f"  Envs:     {sorted(episodes_df['env_reward'].unique())}")
    print(f"  LRs:      {sorted(episodes_df['learning_rate'].unique())}")
    print(f"  Gammas:   {sorted(episodes_df['gamma'].unique())}")

    if "alpha" in episodes_df.columns:
        print(f"  Alphas:   {sorted(episodes_df['alpha'].dropna().unique())}")
    if "beta" in episodes_df.columns:
        print(f"  Betas:    {sorted(episodes_df['beta'].dropna().unique())}")

    return ep_path, sum_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run CSVs into master files")
    parser.add_argument("--agent", type=str, required=True,
                        help="Agent type: dqn_entropy, dqn_rnd, etc.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to results directory, e.g. results/dqn_entropy")
    parser.add_argument("--output-dir", type=str, default="metrics",
                        help="Output directory for combined CSVs")
    args = parser.parse_args()

    aggregate(args.agent, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
