import os
import sys
import json
import time
import yaml
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

ALGO_INFO = {
    "dqn_entropy": {
        "module": "src.test.test_dqn_entropy_agent",
        "config": "configs/dqn_entropy.yaml",
        "mechanism_key": "alphas",
        "cli_flag": "--alpha",
        "result_dir": "results/dqn_entropy",
        "result_token": "a",
    },
    "dqn_rnd": {
        "module": "src.test.test_dqn_rnd_agent",
        "config": "configs/dqn_rnd.yaml",
        "mechanism_key": "betas",
        "cli_flag": "--beta",
        "result_dir": "results/dqn_rnd",
        "result_token": "b",
    },
}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def result_path_for_job(job: dict) -> str:
    info = ALGO_INFO[job["algo"]]
    subdir = f"{job['env_name']}_{job['reward_type']}"
    token = info["result_token"]

    filename = (
        f"lr{job['lr']}_g{job['gamma']}_{token}{job['mechanism_value']}_seed{job['seed']}.csv"
    )
    return os.path.join(info["result_dir"], subdir, filename)


def file_complete(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def build_jobs_for_algo(algo_name: str) -> list[dict]:
    info = ALGO_INFO[algo_name]
    cfg = load_yaml(info["config"])
    sweep = cfg["sweep"]

    seeds = sweep["seeds"]
    lrs = sweep["learning_rates"]
    gammas = sweep["gammas"]
    environments = sweep["environments"]
    mechanism_values = sweep[info["mechanism_key"]]

    jobs = []
    for env_item in environments:
        env_name = env_item["name"]
        reward_type = env_item["reward"]

        for seed in seeds:
            for lr in lrs:
                for gamma in gammas:
                    for mech_value in mechanism_values:
                        job = {
                            "algo": algo_name,
                            "env_name": env_name,
                            "reward_type": reward_type,
                            "seed": seed,
                            "lr": lr,
                            "gamma": gamma,
                            "mechanism_value": mech_value,
                            "config_path": info["config"],
                        }
                        job["result_path"] = result_path_for_job(job)
                        jobs.append(job)

    return jobs


def format_job(job: dict) -> str:
    mech_name = "alpha" if job["algo"] == "dqn_entropy" else "beta"
    return (
        f"{job['algo']} {job['env_name']} {job['reward_type']} "
        f"lr={job['lr']} gamma={job['gamma']} seed={job['seed']} "
        f"{mech_name}={job['mechanism_value']}"
    )


def run_one_job(job: dict) -> dict:
    info = ALGO_INFO[job["algo"]]

    cmd = [
        sys.executable,
        "-m",
        info["module"],
        "--seed", str(job["seed"]),
        "--lr", str(job["lr"]),
        "--gamma", str(job["gamma"]),
        "--env", job["env_name"],
        "--reward", job["reward_type"],
        info["cli_flag"], str(job["mechanism_value"]),
        "--config", job["config_path"],
    ]

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

    start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wall_time = time.time() - start

    return {
        "job": job,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_time_s": round(wall_time, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Student A sweep runner")

    parser.add_argument(
        "--algos",
        nargs="*",
        default=["dqn_entropy", "dqn_rnd"],
        choices=["dqn_entropy", "dqn_rnd"],
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--results-log", type=str, default="logs/studentA_sweep_results.jsonl")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    all_jobs = []
    for algo in args.algos:
        all_jobs.extend(build_jobs_for_algo(algo))

    skipped = []
    pending = []

    if args.rerun_completed:
        pending = all_jobs
    else:
        for job in all_jobs:
            if file_complete(job["result_path"]):
                skipped.append(job)
            else:
                pending.append(job)

    os.makedirs(os.path.dirname(args.results_log) or ".", exist_ok=True)

    print(f"Total jobs discovered: {len(all_jobs)}")
    print(f"Skipped existing jobs: {len(skipped)}")
    print(f"Pending jobs: {len(pending)}")
    print(f"Using max_workers={args.max_workers}")

    for job in skipped[:20]:
        print(f"[SKIP] {format_job(job)} -> {job['result_path']}")
    if len(skipped) > 20:
        print(f"... and {len(skipped) - 20} more skipped jobs")

    if not pending:
        print("Nothing to run.")
        return

    ok = 0
    fail = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor, open(
        args.results_log, "a", encoding="utf-8"
    ) as f:
        futures = {executor.submit(run_one_job, job): job for job in pending}

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job_str = format_job(result["job"])

            f.write(json.dumps(result) + "\n")
            f.flush()

            if result["returncode"] == 0:
                ok += 1
                print(f"[{i}/{len(pending)}] OK   {job_str} ({result['wall_time_s']}s)")
            else:
                fail += 1
                print(f"[{i}/{len(pending)}] FAIL {job_str} ({result['wall_time_s']}s)")
                print(result["stderr"][:1200])

                if args.fail_fast:
                    print("Stopping because --fail-fast was set.")
                    break

    print("\nSweep finished.")
    print(f"Successful jobs: {ok}")
    print(f"Failed jobs: {fail}")
    print(f"Skipped existing jobs: {len(skipped)}")


if __name__ == "__main__":
    main()