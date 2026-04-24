"""Tests for the notebook reward comparison helper module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import results_analysis_helpers as helpers


class ResultsAnalysisHelpersTest(unittest.TestCase):
    maxDiff = None

    def test_parse_result_file_for_summary_csv(self) -> None:
        path = (
            REPO_ROOT
            / "results"
            / "dqn_rnd_mc_8000"
            / "mountaincar_sparse"
            / "lr0.001_g0.99_b0.01_seed3_summary.csv"
        )

        parsed = helpers.parse_result_file(path, REPO_ROOT / "results")

        self.assertEqual(parsed["source_family"], "dqn_rnd_mc_8000")
        self.assertEqual(parsed["env_reward"], "mountaincar_sparse")
        self.assertEqual(parsed["analysis_cohort"], "mountaincar_mc8000")
        self.assertEqual(parsed["mechanism"], "rnd")
        self.assertEqual(parsed["mechanism_param_name"], "beta")
        self.assertAlmostEqual(parsed["mechanism_param_value"], 0.01)
        self.assertEqual(parsed["seed"], 3)
        self.assertTrue(parsed["is_summary"])
        self.assertEqual(
            parsed["config_key"],
            "dqn_rnd_mc_8000/mountaincar_sparse/lr0.001_g0.99_b0.01",
        )

    def test_parse_result_file_for_raw_csv(self) -> None:
        path = (
            REPO_ROOT
            / "results"
            / "dqn_entropy"
            / "cartpole_dense"
            / "lr0.0005_g0.95_a0.01_seed0.csv"
        )

        parsed = helpers.parse_result_file(path, REPO_ROOT / "results")

        self.assertEqual(parsed["analysis_cohort"], "cartpole")
        self.assertEqual(parsed["mechanism"], "entropy")
        self.assertEqual(parsed["mechanism_param_name"], "alpha")
        self.assertAlmostEqual(parsed["mechanism_param_value"], 0.01)
        self.assertFalse(parsed["is_summary"])

    def test_scope_manifest_and_coverage_match_expected_counts(self) -> None:
        manifest_df = helpers.build_scope_manifest(REPO_ROOT / "results")
        coverage_df = helpers.build_coverage_audit(manifest_df)

        self.assertEqual(len(manifest_df), 1168)
        self.assertEqual(int(coverage_df["include_in_analysis"].sum()), 58)
        self.assertEqual(int((~coverage_df["include_in_analysis"]).sum()), 3)

        excluded_keys = set(
            coverage_df.loc[~coverage_df["include_in_analysis"], "config_key"].tolist()
        )
        self.assertEqual(
            excluded_keys,
            {
                "dqn_entropy/cartpole_dense/lr0.1_g0.99_a0.01",
                "dqn_entropy_mc_8000/mountaincar_dense/lr0.001_g0.99_a0.05",
                "dqn_entropy_mc_8000/mountaincar_sparse/lr0.001_g0.99_a0.05",
            },
        )

    def test_build_log_path_and_eval_regex(self) -> None:
        class Record:
            env_reward = "mountaincar_sparse"
            mechanism_param_name = "beta"
            mechanism_param_value = 0.01
            mechanism_param_token = "0.01"
            seed = 3
            learning_rate = 0.001
            learning_rate_token = "0.001"
            gamma = 0.99
            gamma_token = "0.99"
            source_family = "dqn_rnd_mc_8000"

        log_path = helpers.build_log_path(Record())
        self.assertEqual(
            str(log_path),
            "logs/dqn_rnd_mc_8000/runs3_lr0.001_g0.99_b0.01_mountaincar_sparse.log",
        )

        line = (
            "2026-04-22 01:24:07 | INFO | [Episode  100] eps=0.901 "
            "avg_reward=  -1.000  avg_length=200.00  avg_loss=  0.0053 "
            "avg_intrinsic=0.0003  eval_success=0.000"
        )
        match = helpers.EVAL_SUCCESS_LOG_RE.search(line)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(int(match.group("episode")), 100)
        self.assertAlmostEqual(float(match.group("eval_success_rate")), 0.0)
        self.assertEqual(
            helpers.infer_algorithm_label("dqn_rnd_mc_8000", "rnd"),
            "DQN_RND",
        )

    def test_aggregate_reward_curves_keeps_reward_types_separate(self) -> None:
        episodes_df = pd.DataFrame(
            [
                {
                    "analysis_cohort": "cartpole",
                    "reward_type": "dense",
                    "algorithm": "DQN_Entropy",
                    "seed": 0,
                    "episode": 1,
                    "reward": 10.0,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "mechanism_param_value": 0.01,
                },
                {
                    "analysis_cohort": "cartpole",
                    "reward_type": "dense",
                    "algorithm": "DQN_Entropy",
                    "seed": 1,
                    "episode": 1,
                    "reward": 14.0,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "mechanism_param_value": 0.01,
                },
                {
                    "analysis_cohort": "cartpole",
                    "reward_type": "sparse",
                    "algorithm": "DQN_Entropy",
                    "seed": 0,
                    "episode": 1,
                    "reward": 2.0,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "mechanism_param_value": 0.01,
                },
                {
                    "analysis_cohort": "cartpole",
                    "reward_type": "sparse",
                    "algorithm": "DQN_Entropy",
                    "seed": 1,
                    "episode": 1,
                    "reward": 6.0,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "mechanism_param_value": 0.01,
                },
            ]
        )

        dense_curve = helpers.aggregate_metric_curves(
            df=episodes_df,
            metric="reward",
            analysis_cohort="cartpole",
            reward_type="dense",
            compare_by="seed",
        )
        sparse_curve = helpers.aggregate_metric_curves(
            df=episodes_df,
            metric="reward",
            analysis_cohort="cartpole",
            reward_type="sparse",
            compare_by="seed",
        )

        self.assertEqual(dense_curve["reward"].tolist(), [10.0, 14.0])
        self.assertEqual(sparse_curve["reward"].tolist(), [2.0, 6.0])


if __name__ == "__main__":
    unittest.main()
