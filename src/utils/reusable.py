"""
reusable.py — Shared utilities for ELG5214 Group 8 project.

Provides:
    - RLMetricsDataset: Episode + summary CSV logging
    - setup_logger: Per-run file logger
    - Timer: Wall-clock context manager (JAX-aware)
    - Device info utilities
"""

import os
import time
import logging
import pandas as pd
import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────
#  GPU / Device Utilities
# ──────────────────────────────────────────────

def get_device_info() -> dict:
    backend = jax.default_backend()
    devices = jax.devices()
    device_strs = [str(d) for d in devices]
    info = {
        "backend": backend.upper(),
        "num_devices": len(devices),
        "devices": device_strs,
        "platform": backend,
    }
    if backend == "gpu":
        try:
            info["gpu_name"] = devices[0].device_kind
        except Exception:
            info["gpu_name"] = "unknown"
    return info


def log_device_info(logger: logging.Logger) -> dict:
    info = get_device_info()
    logger.info("=" * 60)
    logger.info("DEVICE / COMPUTE INFORMATION")
    logger.info(f"  JAX backend       : {info['backend']}")
    logger.info(f"  Platform          : {info['platform']}")
    logger.info(f"  Num devices       : {info['num_devices']}")
    for d in info["devices"]:
        logger.info(f"  Device            : {d}")
    if "gpu_name" in info:
        logger.info(f"  GPU name          : {info['gpu_name']}")
    logger.info("=" * 60)
    return info


def force_jax_gpu_or_warn(logger: logging.Logger) -> str:
    backend = jax.default_backend()
    if backend != "gpu":
        logger.warning(
            "JAX is running on %s, NOT GPU. "
            "Install jax[cuda12] for GPU acceleration.",
            backend.upper(),
        )
    else:
        logger.info("JAX GPU backend confirmed.")
    return backend


class Timer:
    """Context manager that times a block (JAX-aware)."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        jax.effects_barrier()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        jax.effects_barrier()
        self.elapsed = time.perf_counter() - self._start

    def report(self) -> str:
        mins, secs = divmod(self.elapsed, 60)
        if mins > 0:
            return f"{self.label}: {int(mins)}m {secs:.2f}s"
        return f"{self.label}: {self.elapsed:.2f}s"


def warmup_jit(logger: logging.Logger):
    logger.info("Warming up JAX JIT compiler...")
    x = jnp.ones((2, 2))

    @jax.jit
    def _warmup(x):
        return x @ x + x

    _warmup(x).block_until_ready()
    logger.info("JIT warmup complete.")


# ──────────────────────────────────────────────
#  Metrics Dataset
# ──────────────────────────────────────────────

class RLMetricsDataset:
    """Collects per-episode and summary metrics, saves to CSV.

    Extended from Assignment 2 with intrinsic_reward and policy_entropy
    fields for RND/ICM/entropy tracking, plus env_name and reward_type.
    """

    def __init__(self, proj_name: str):
        self.proj_name = proj_name
        self.episode_records = []
        self.summary_records = []

    def add_episode(
            self,
            seed: int,
            episode: int,
            reward: float,
            episode_length: int,
            algorithm: str,
            lr: float,
            gamma: float,
            loss: float = 0.0,
            eval_success_rate: float = -1.0,
            intrinsic_reward: float = 0.0,
            policy_entropy: float = 0.0,
            env_name: str = "",
            reward_type: str = "",
    ):
        self.episode_records.append({
            "seed": seed,
            "episode": episode,
            "reward": reward,
            "episode_length": episode_length,
            "loss": loss,
            "eval_success_rate": eval_success_rate,
            "intrinsic_reward": intrinsic_reward,
            "policy_entropy": policy_entropy,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma,
            "env_name": env_name,
            "reward_type": reward_type,
        })

    def add_summary(
            self,
            seed: int,
            algorithm: str,
            lr: float,
            gamma: float,
            final_mean_reward: float,
            final_success_rate: float,
            backend: str,
            devices: str,
            action: str = "stochastic",
            mean_length: float = 0,
            wall_time_s: float = 0.0,
            env_name: str = "",
            reward_type: str = "",
    ):
        self.summary_records.append({
            "seed": seed,
            "algorithm": algorithm,
            "learning_rate": lr,
            "gamma": gamma,
            "final_mean_reward": final_mean_reward,
            "final_success_rate": final_success_rate,
            "backend": str(devices),
            "devices": str(devices),
            "action": action,
            "mean_length": mean_length,
            "wall_time_s": wall_time_s,
            "env_name": env_name,
            "reward_type": reward_type,
        })

    def save(self, output_dir: str = "metrics", filename: str | None = None):
        if not os.path.isabs(output_dir):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            output_dir = os.path.join(project_root, output_dir)

        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename_iters = f"{self.proj_name}_dataset_metrics.csv"
            filename_summ = f"{self.proj_name}_dataset_metrics_summary.csv"
        else:
            filename_iters = filename
            filename_summ = filename.replace(".csv", "_summary.csv")

        path_iter = os.path.join(output_dir, filename_iters)
        path_summ = os.path.join(output_dir, filename_summ)

        pd.DataFrame(self.episode_records).to_csv(path_iter, index=False)
        pd.DataFrame(self.summary_records).to_csv(path_summ, index=False)
        return {"iteration": path_iter, "summary": path_summ}


# ──────────────────────────────────────────────
#  Logger Setup
# ──────────────────────────────────────────────

def setup_logger(run_id, path: str):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(f"run{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = f"{path}/run{run_id}.log"
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
