"""Utilities for notebook-based reward comparison analysis.

This module builds a filtered analysis dataset from the project results tree,
exports merged CSVs, and generates comparison plots where:
    x-axis = episode
    y-axis = mean raw reward
    dense and sparse conditions are kept separate

The selected scope matches the notebook plan:
- CartPole: DQN Entropy + DQN RND
- MountainCar: DQN Entropy MC 8000 + DQN RND MC 8000 only
- Incomplete hyperparameter groups are excluded from the analysis outputs
"""

from __future__ import annotations

import argparse
import math
from numbers import Real
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SEED_COUNT = 10
EXPECTED_COUNTS = {
    "raw_files": 580,
    "summary_files": 580,
    "episode_rows": 1_157_250,
    "summary_rows": 580,
    "included_configs": 58,
    "excluded_configs": 3,
}

RESULT_FILE_RE = re.compile(
    r"^lr(?P<learning_rate>[^_]+)_g(?P<gamma>[^_]+)_(?P<mechanism_code>[ab])"
    r"(?P<mechanism_value>[^_]+)_seed(?P<seed>\d+)(?P<summary>_summary)?\.csv$"
)

EVAL_SUCCESS_LOG_RE = re.compile(
    r"\[Episode\s+(?P<episode>\d+)\].*?eval_success=(?P<eval_success_rate>[-+]?\d*\.?\d+)"
)

MECHANISM_BY_CODE = {
    "a": ("entropy", "alpha"),
    "b": ("rnd", "beta"),
}


@dataclass(frozen=True)
class ScopeEntry:
    source_family: str
    env_reward: str
    analysis_cohort: str

    @property
    def relative_dir(self) -> str:
        return f"{self.source_family}/{self.env_reward}"


SELECTED_SCOPE: tuple[ScopeEntry, ...] = (
    ScopeEntry("dqn_entropy", "cartpole_dense", "cartpole"),
    ScopeEntry("dqn_entropy", "cartpole_sparse", "cartpole"),
    ScopeEntry("dqn_rnd", "cartpole_dense", "cartpole"),
    ScopeEntry("dqn_rnd", "cartpole_sparse", "cartpole"),
    ScopeEntry("dqn_entropy_mc_8000", "mountaincar_dense", "mountaincar_mc8000"),
    ScopeEntry("dqn_entropy_mc_8000", "mountaincar_sparse", "mountaincar_mc8000"),
    ScopeEntry("dqn_rnd_mc_8000", "mountaincar_dense", "mountaincar_mc8000"),
    ScopeEntry("dqn_rnd_mc_8000", "mountaincar_sparse", "mountaincar_mc8000"),
)


@dataclass
class AnalysisArtifacts:
    manifest_df: pd.DataFrame
    coverage_df: pd.DataFrame
    episodes_df: pd.DataFrame
    summary_df: pd.DataFrame
    eval_success_df: pd.DataFrame
    plot_paths: list[Path]
    counts: dict[str, int]


def parse_result_file(file_path: Path, results_root: Path) -> dict[str, object]:
    """Parse metadata from a single result CSV path."""

    match = RESULT_FILE_RE.match(file_path.name)
    if match is None:
        raise ValueError(f"Unsupported results filename: {file_path.name}")

    relative_parts = file_path.relative_to(results_root).parts
    if len(relative_parts) != 3:
        raise ValueError(f"Expected results/<family>/<env_reward>/<file>.csv, got {file_path}")

    source_family, env_reward, _ = relative_parts
    mechanism_code = match.group("mechanism_code")
    mechanism, mechanism_param_name = MECHANISM_BY_CODE[mechanism_code]

    group_stem = file_path.name.replace("_summary.csv", ".csv").replace(".csv", "")
    group_stem = re.sub(r"_seed\d+$", "", group_stem)

    analysis_cohort = infer_analysis_cohort(source_family, env_reward)
    config_key = f"{source_family}/{env_reward}/{group_stem}"

    return {
        "path": str(file_path),
        "source_family": source_family,
        "env_reward": env_reward,
        "analysis_cohort": analysis_cohort,
        "file_name": file_path.name,
        "group_stem": group_stem,
        "config_key": config_key,
        "seed": int(match.group("seed")),
        "learning_rate_token": match.group("learning_rate"),
        "learning_rate": float(match.group("learning_rate")),
        "gamma_token": match.group("gamma"),
        "gamma": float(match.group("gamma")),
        "mechanism": mechanism,
        "mechanism_param_name": mechanism_param_name,
        "mechanism_param_token": match.group("mechanism_value"),
        "mechanism_param_value": float(match.group("mechanism_value")),
        "is_summary": bool(match.group("summary")),
    }


def infer_analysis_cohort(source_family: str, env_reward: str) -> str:
    """Map a source directory to the notebook's analysis cohort."""

    if env_reward.startswith("cartpole"):
        return "cartpole"
    if source_family.endswith("_mc_8000") and env_reward.startswith("mountaincar"):
        return "mountaincar_mc8000"
    raise ValueError(f"Unsupported cohort mapping for {source_family}/{env_reward}")


def slugify_algorithm(algorithm: str) -> str:
    """Convert algorithm labels like DQN_Entropy to file-safe names."""

    return algorithm.strip().lower().replace("+", "_plus_").replace(" ", "_")


def prettify_cohort(cohort: str) -> str:
    if cohort == "cartpole":
        return "CartPole"
    if cohort == "mountaincar_mc8000":
        return "MountainCar MC 8000"
    return cohort.replace("_", " ").title()


def prettify_reward_type(reward_type: str) -> str:
    return reward_type.replace("_", " ").title()


def build_log_path(record: pd.Series | object, logs_root: Path = Path("logs")) -> Path:
    """Infer the training log path for a selected run from parsed results metadata."""

    env_name, reward_type = str(record.env_reward).split("_", maxsplit=1)
    mechanism_code = "a" if str(record.mechanism_param_name) == "alpha" else "b"
    log_name = (
        f"runs{int(record.seed)}_lr{record.learning_rate_token}_g{record.gamma_token}_"
        f"{mechanism_code}{record.mechanism_param_token}_{env_name}_{reward_type}.log"
    )
    return logs_root / str(record.source_family) / log_name


def infer_algorithm_label(source_family: str, mechanism: str) -> str:
    """Infer the user-facing algorithm label used in the CSVs/logs."""

    prefix = "DQN"
    suffix = "Entropy" if mechanism == "entropy" else "RND"
    if not source_family.startswith("dqn"):
        raise ValueError(f"Unsupported source family for algorithm label: {source_family}")
    return f"{prefix}_{suffix}"


def build_scope_manifest(results_root: Path) -> pd.DataFrame:
    """Enumerate all CSVs in the selected analysis scope."""

    rows: list[dict[str, object]] = []
    for entry in SELECTED_SCOPE:
        result_dir = results_root / entry.relative_dir
        if not result_dir.exists():
            raise FileNotFoundError(f"Missing selected results directory: {result_dir}")

        for file_path in sorted(result_dir.glob("*.csv")):
            rows.append(parse_result_file(file_path, results_root))

    manifest_df = pd.DataFrame(rows)
    if manifest_df.empty:
        raise ValueError("No CSV files were found in the selected analysis scope.")

    manifest_df = manifest_df.sort_values(
        ["analysis_cohort", "source_family", "env_reward", "group_stem", "seed", "is_summary"]
    ).reset_index(drop=True)
    return manifest_df


def build_coverage_audit(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Count raw and summary files per hyperparameter group and flag completeness."""

    rows: list[dict[str, object]] = []
    group_cols = [
        "analysis_cohort",
        "source_family",
        "env_reward",
        "mechanism",
        "mechanism_param_name",
        "mechanism_param_value",
        "config_key",
        "group_stem",
        "learning_rate",
        "gamma",
    ]

    for keys, sub_df in manifest_df.groupby(group_cols, sort=True, dropna=False):
        raw_df = sub_df.loc[~sub_df["is_summary"]]
        summary_df = sub_df.loc[sub_df["is_summary"]]

        raw_file_count = int(len(raw_df))
        summary_file_count = int(len(summary_df))
        raw_seed_count = int(raw_df["seed"].nunique())
        summary_seed_count = int(summary_df["seed"].nunique())

        include_in_analysis = (
            raw_file_count == EXPECTED_SEED_COUNT
            and summary_file_count == EXPECTED_SEED_COUNT
            and raw_seed_count == EXPECTED_SEED_COUNT
            and summary_seed_count == EXPECTED_SEED_COUNT
        )

        exclusion_reason = ""
        if not include_in_analysis:
            exclusion_reason = (
                f"incomplete_seed_coverage raw_files={raw_file_count} "
                f"summary_files={summary_file_count} raw_seeds={raw_seed_count} "
                f"summary_seeds={summary_seed_count}"
            )

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "expected_seed_count": EXPECTED_SEED_COUNT,
                "raw_file_count": raw_file_count,
                "summary_file_count": summary_file_count,
                "raw_seed_count": raw_seed_count,
                "summary_seed_count": summary_seed_count,
                "include_in_analysis": include_in_analysis,
                "exclusion_reason": exclusion_reason,
            }
        )
        rows.append(row)

    coverage_df = pd.DataFrame(rows).sort_values(
        ["analysis_cohort", "source_family", "env_reward", "group_stem"]
    ).reset_index(drop=True)
    return coverage_df


def load_analysis_data(
    manifest_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load filtered raw + summary datasets and return the flagged manifest."""

    include_cols = ["config_key", "include_in_analysis"]
    flagged_manifest = manifest_df.merge(
        coverage_df[include_cols],
        on="config_key",
        how="left",
        validate="many_to_one",
    )
    flagged_manifest["include_in_analysis"] = flagged_manifest["include_in_analysis"].fillna(False)

    selected_manifest = flagged_manifest.loc[flagged_manifest["include_in_analysis"]].copy()
    if selected_manifest.empty:
        raise ValueError("Coverage audit excluded every file in the selected scope.")

    episode_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for record in selected_manifest.itertuples(index=False):
        df = pd.read_csv(record.path)
        if df.empty:
            continue

        df = df.copy()
        df["source_family"] = record.source_family
        df["analysis_cohort"] = record.analysis_cohort
        df["env_reward"] = record.env_reward
        df["mechanism"] = record.mechanism
        df["mechanism_param_name"] = record.mechanism_param_name
        df["mechanism_param_value"] = record.mechanism_param_value
        df["config_key"] = record.config_key
        df["alpha"] = pd.NA
        df["beta"] = pd.NA
        df[record.mechanism_param_name] = record.mechanism_param_value

        if record.is_summary:
            summary_frames.append(df)
        else:
            episode_frames.append(df)

    if not episode_frames or not summary_frames:
        raise ValueError("Failed to load both episode and summary analysis datasets.")

    episodes_df = pd.concat(episode_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    numeric_cols = [
        "seed",
        "learning_rate",
        "gamma",
        "mechanism_param_value",
        "reward",
        "episode",
        "episode_length",
        "loss",
        "eval_success_rate",
        "intrinsic_reward",
        "policy_entropy",
    ]
    for col in numeric_cols:
        if col in episodes_df.columns:
            episodes_df[col] = pd.to_numeric(episodes_df[col], errors="coerce")

    summary_numeric_cols = [
        "seed",
        "learning_rate",
        "gamma",
        "mechanism_param_value",
        "final_mean_reward",
        "final_success_rate",
        "mean_length",
        "wall_time_s",
    ]
    for col in summary_numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    return flagged_manifest, episodes_df, summary_df


def load_eval_success_data(
    flagged_manifest: pd.DataFrame,
    logs_root: str | Path = "logs",
) -> pd.DataFrame:
    """Load per-eval success checkpoints from training logs for selected raw runs."""

    logs_root = Path(logs_root)

    selected_runs = (
        flagged_manifest.loc[
            flagged_manifest["include_in_analysis"] & (~flagged_manifest["is_summary"])
        ]
        .drop_duplicates(["config_key", "seed"])
        .copy()
    )

    eval_frames: list[pd.DataFrame] = []
    for record in selected_runs.itertuples(index=False):
        log_path = build_log_path(record, logs_root=logs_root)
        if not log_path.exists():
            raise FileNotFoundError(f"Expected training log for eval-success plot: {log_path}")

        _, reward_type = str(record.env_reward).split("_", maxsplit=1)
        algorithm = infer_algorithm_label(str(record.source_family), str(record.mechanism))

        rows: list[dict[str, object]] = []
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = EVAL_SUCCESS_LOG_RE.search(line)
                if match is None:
                    continue
                rows.append(
                    {
                        "analysis_cohort": record.analysis_cohort,
                        "reward_type": reward_type,
                        "source_family": record.source_family,
                        "algorithm": algorithm,
                        "config_key": record.config_key,
                        "seed": int(record.seed),
                        "learning_rate": float(record.learning_rate),
                        "gamma": float(record.gamma),
                        "mechanism": record.mechanism,
                        "mechanism_param_name": record.mechanism_param_name,
                        "mechanism_param_value": float(record.mechanism_param_value),
                        "episode": int(match.group("episode")),
                        "eval_success_rate": float(match.group("eval_success_rate")),
                    }
                )

        if not rows:
            raise ValueError(f"No eval-success checkpoints were parsed from {log_path}")

        eval_frames.append(pd.DataFrame(rows))

    if not eval_frames:
        raise ValueError("No eval-success log data was found for the selected analysis scope.")

    eval_success_df = pd.concat(eval_frames, ignore_index=True)
    return eval_success_df.sort_values(
        ["analysis_cohort", "reward_type", "seed", "episode"]
    ).reset_index(drop=True)


def summarize_counts(
    flagged_manifest: pd.DataFrame,
    coverage_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, int]:
    """Compute validation counts for the filtered analysis scope."""

    selected_manifest = flagged_manifest.loc[flagged_manifest["include_in_analysis"]]
    counts = {
        "raw_files": int((~selected_manifest["is_summary"]).sum()),
        "summary_files": int(selected_manifest["is_summary"].sum()),
        "episode_rows": int(len(episodes_df)),
        "summary_rows": int(len(summary_df)),
        "included_configs": int(coverage_df["include_in_analysis"].sum()),
        "excluded_configs": int((~coverage_df["include_in_analysis"]).sum()),
    }
    return counts


def validate_expected_counts(counts: dict[str, int]) -> None:
    """Raise an informative error if the selected scope changed unexpectedly."""

    mismatches = {
        key: (counts[key], expected)
        for key, expected in EXPECTED_COUNTS.items()
        if counts.get(key) != expected
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{key}=observed {observed} expected {expected}"
            for key, (observed, expected) in mismatches.items()
        )
        raise ValueError(f"Analysis counts did not match the expected scope: {mismatch_text}")


def export_analysis_data(
    coverage_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metrics_dir: Path,
) -> None:
    """Write merged CSVs for the notebook analysis."""

    metrics_dir.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(metrics_dir / "config_coverage_audit.csv", index=False)
    episodes_df.to_csv(metrics_dir / "selected_scope_episodes.csv", index=False)
    summary_df.to_csv(metrics_dir / "selected_scope_summary.csv", index=False)


def aggregate_metric_curves(
    df: pd.DataFrame,
    metric: str,
    analysis_cohort: str,
    reward_type: str,
    compare_by: str,
    algorithm: str | None = None,
) -> pd.DataFrame:
    """Aggregate an episode-indexed metric for a cohort, reward type, and comparison dimension."""

    cohort_df = df.loc[
        (df["analysis_cohort"] == analysis_cohort)
        & (df["reward_type"] == reward_type)
    ].copy()
    if algorithm is not None:
        cohort_df = cohort_df.loc[cohort_df["algorithm"] == algorithm].copy()

    if cohort_df.empty:
        raise ValueError(
            "No episode rows found for "
            f"cohort={analysis_cohort!r} reward_type={reward_type!r} algorithm={algorithm!r}."
        )

    if compare_by not in cohort_df.columns:
        raise KeyError(f"Comparison column {compare_by!r} not found in metric dataframe.")
    if metric not in cohort_df.columns:
        raise KeyError(f"Metric column {metric!r} not found in metric dataframe.")

    agg_df = (
        cohort_df.groupby([compare_by, "episode"], dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values([compare_by, "episode"])
    )
    return agg_df


def _sorted_unique(values: Iterable[object]) -> list[object]:
    def sort_key(value: object) -> tuple[int, object]:
        if isinstance(value, str):
            return (1, value)
        if isinstance(value, Real) and not math.isnan(float(value)):
            return (0, float(value))
        return (2, str(value))

    return sorted(pd.Series(list(values)).dropna().unique().tolist(), key=sort_key)


def _format_line_label(compare_by: str, value: object) -> str:
    if compare_by == "learning_rate":
        return f"lr={float(value):g}" if isinstance(value, Real) else f"lr={value}"
    if compare_by == "gamma":
        return f"gamma={float(value):g}" if isinstance(value, Real) else f"gamma={value}"
    if compare_by == "seed":
        return f"seed={int(value)}"
    if compare_by == "mechanism_param_value":
        return f"value={float(value):g}" if isinstance(value, Real) else f"value={value}"
    return str(value)


def plot_metric_curves(
    agg_df: pd.DataFrame,
    metric: str,
    compare_by: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    """Render and save an episode-indexed comparison line chart."""

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for line_value in _sorted_unique(agg_df[compare_by]):
        line_df = agg_df.loc[agg_df[compare_by] == line_value].sort_values("episode")
        ax.plot(
            line_df["episode"],
            line_df[metric],
            label=_format_line_label(compare_by, line_value),
            linewidth=2.0,
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_comparison_plots(
    episodes_df: pd.DataFrame,
    eval_success_df: pd.DataFrame,
    visualizations_root: Path,
) -> list[Path]:
    """Generate all notebook comparison plots and return their paths."""

    plot_paths: list[Path] = []
    compare_dims = ["seed", "learning_rate", "gamma", "algorithm"]

    for old_plot in visualizations_root.rglob("*.png"):
        old_plot.unlink()

    for cohort in _sorted_unique(episodes_df["analysis_cohort"]):
        cohort_dir = visualizations_root / str(cohort)
        cohort_title = prettify_cohort(str(cohort))
        cohort_df = episodes_df.loc[episodes_df["analysis_cohort"] == cohort].copy()

        for reward_type in _sorted_unique(cohort_df["reward_type"]):
            reward_title = prettify_reward_type(str(reward_type))
            reward_slug = str(reward_type)

            for compare_by in compare_dims:
                agg_df = aggregate_metric_curves(
                    df=episodes_df,
                    metric="reward",
                    analysis_cohort=str(cohort),
                    reward_type=reward_slug,
                    compare_by=compare_by,
                )
                plot_path = cohort_dir / f"{cohort}_{reward_slug}_compare_by_{compare_by}.png"
                plot_paths.append(
                    plot_metric_curves(
                        agg_df,
                        metric="reward",
                        compare_by=compare_by,
                        output_path=plot_path,
                        title=(
                            f"{cohort_title} ({reward_title}): Mean Reward by "
                            f"{compare_by.replace('_', ' ').title()}"
                        ),
                        ylabel="Mean Reward",
                    )
                )

            eval_seed_df = aggregate_metric_curves(
                eval_success_df,
                metric="eval_success_rate",
                analysis_cohort=str(cohort),
                reward_type=reward_slug,
                compare_by="seed",
            )
            eval_plot_path = cohort_dir / f"{cohort}_{reward_slug}_eval_success_compare_by_seed.png"
            plot_paths.append(
                plot_metric_curves(
                    eval_seed_df,
                    metric="eval_success_rate",
                    compare_by="seed",
                    output_path=eval_plot_path,
                    title=f"{cohort_title} ({reward_title}): Eval Success Rate by Seed",
                    ylabel="Mean Eval Success Rate",
                )
            )

            reward_df = cohort_df.loc[cohort_df["reward_type"] == reward_type].copy()
            for algorithm in _sorted_unique(reward_df["algorithm"]):
                algorithm_df = reward_df.loc[reward_df["algorithm"] == algorithm].copy()
                mechanism_names = _sorted_unique(algorithm_df["mechanism_param_name"])
                if len(mechanism_names) != 1:
                    raise ValueError(
                        "Expected one mechanism parameter for "
                        f"{cohort}/{reward_type}/{algorithm}, got {mechanism_names}"
                    )

                mechanism_param_name = str(mechanism_names[0])
                agg_df = aggregate_metric_curves(
                    df=episodes_df,
                    metric="reward",
                    analysis_cohort=str(cohort),
                    reward_type=reward_slug,
                    compare_by="mechanism_param_value",
                    algorithm=str(algorithm),
                )
                plot_path = cohort_dir / (
                    f"{cohort}_{reward_slug}_compare_by_{mechanism_param_name}_"
                    f"{slugify_algorithm(str(algorithm))}.png"
                )
                plot_paths.append(
                    plot_metric_curves(
                        agg_df,
                        metric="reward",
                        compare_by="mechanism_param_value",
                        output_path=plot_path,
                        title=(
                            f"{cohort_title} ({reward_title}): Mean Reward by "
                            f"{mechanism_param_name.title()} for {algorithm}"
                        ),
                        ylabel="Mean Reward",
                    )
                )

    return plot_paths


def build_summary_tables(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare compact summary tables for notebook display."""

    overall = (
        summary_df.groupby(["analysis_cohort", "reward_type", "algorithm"], dropna=False)
        .agg(
            runs=("seed", "count"),
            mean_final_reward=("final_mean_reward", "mean"),
            mean_success_rate=("final_success_rate", "mean"),
            mean_wall_time_s=("wall_time_s", "mean"),
        )
        .reset_index()
        .sort_values(
            ["analysis_cohort", "reward_type", "mean_final_reward"],
            ascending=[True, True, False],
        )
    )

    top_configs = (
        summary_df.groupby(
            [
                "analysis_cohort",
                "algorithm",
                "reward_type",
                "learning_rate",
                "gamma",
                "mechanism_param_name",
                "mechanism_param_value",
                "config_key",
            ],
            dropna=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            mean_final_reward=("final_mean_reward", "mean"),
            mean_success_rate=("final_success_rate", "mean"),
        )
        .reset_index()
        .sort_values(
            ["analysis_cohort", "reward_type", "algorithm", "mean_final_reward"],
            ascending=[True, True, True, False],
        )
    )

    return {"overall_summary": overall, "top_configs": top_configs}


def run_full_analysis(
    results_root: str | Path = "results",
    metrics_dir: str | Path = "metrics/analysis",
    visualizations_root: str | Path = "visualizations/analysis",
    logs_root: str | Path = "logs",
    save_outputs: bool = True,
    validate_counts: bool = True,
) -> AnalysisArtifacts:
    """Build the selected-scope analysis datasets and comparison plots."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    results_root = Path(results_root)
    metrics_dir = Path(metrics_dir)
    visualizations_root = Path(visualizations_root)
    logs_root = Path(logs_root)

    manifest_df = build_scope_manifest(results_root)
    coverage_df = build_coverage_audit(manifest_df)
    flagged_manifest, episodes_df, summary_df = load_analysis_data(manifest_df, coverage_df)
    eval_success_df = load_eval_success_data(flagged_manifest, logs_root=logs_root)
    counts = summarize_counts(flagged_manifest, coverage_df, episodes_df, summary_df)

    if validate_counts:
        validate_expected_counts(counts)

    if save_outputs:
        export_analysis_data(coverage_df, episodes_df, summary_df, metrics_dir)

    plot_paths = generate_comparison_plots(episodes_df, eval_success_df, visualizations_root)

    return AnalysisArtifacts(
        manifest_df=flagged_manifest,
        coverage_df=coverage_df,
        episodes_df=episodes_df,
        summary_df=summary_df,
        eval_success_df=eval_success_df,
        plot_paths=plot_paths,
        counts=counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the notebook reward comparison datasets and plots."
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--metrics-dir", type=str, default="metrics/analysis")
    parser.add_argument("--visualizations-root", type=str, default="visualizations/analysis")
    parser.add_argument("--logs-root", type=str, default="logs")
    parser.add_argument("--no-validate-counts", action="store_true")
    args = parser.parse_args()

    artifacts = run_full_analysis(
        results_root=args.results_root,
        metrics_dir=args.metrics_dir,
        visualizations_root=args.visualizations_root,
        logs_root=args.logs_root,
        save_outputs=True,
        validate_counts=not args.no_validate_counts,
    )

    print("Analysis complete.")
    for key, value in artifacts.counts.items():
        print(f"  {key}: {value}")
    print(f"  plots: {len(artifacts.plot_paths)}")


if __name__ == "__main__":
    main()
