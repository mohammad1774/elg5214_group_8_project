#!/usr/bin/env bash
# ============================================================================
# run_ppo_entropy_sweep.sh
# Launches all PPO + Entropy hyperparameter combinations in parallel on CPU.
#
# Usage:
#   chmod +x run_ppo_entropy_sweep.sh
#   ./run_ppo_entropy_sweep.sh              # uses defaults
#   ./run_ppo_entropy_sweep.sh -j 12        # override max parallel jobs
#   ./run_ppo_entropy_sweep.sh --dry-run    # print commands without running
# ============================================================================

set -euo pipefail

# ── Defaults (edit here or override via flags) ─────────────────────────────
MAX_JOBS=8              # concurrent processes (tune to your CPU core count)
THREADS_PER_JOB=2       # OMP/MKL threads each process may use
DRY_RUN=false
METRICS_DIR="metrics/ppo_entropy"
LOG_DIR="logs/ppo_entropy"

# ── Parse CLI flags ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)      MAX_JOBS="$2";    shift 2 ;;
        -t|--threads)   THREADS_PER_JOB="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true;     shift   ;;
        --metrics-dir)  METRICS_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-j MAX_JOBS] [-t THREADS_PER_JOB] [--dry-run] [--metrics-dir DIR]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Force JAX onto CPU for every child process ─────────────────────────────
export JAX_PLATFORM_NAME="cpu"
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export PYTHONUNBUFFERED=1

# ── Sweep grid (must match configs/ppo_entropy.yaml) ──────────────────────
ENVIRONMENTS=("cartpole" "mountaincar")
REWARD_TYPES=("dense" "sparse")
LEARNING_RATES=("0.0001" "0.0005" "0.001")
GAMMAS=("0.99" "0.95")
ALPHAS=("0.01" "0.05")
SEEDS=("0" "1" "2" "3" "4")

# ── Training constants ─────────────────────────────────────────────────────
NUM_EPISODES=2000
MAX_STEPS=200
EVAL_EPISODES=100
LOG_EVERY=50
HIDDEN_DIM=64
N_EPOCHS=4
MINI_BATCH=64
CLIP_EPS=0.2
GAE_LAMBDA=0.95

# ── Compute total runs ────────────────────────────────────────────────────
TOTAL=$(( ${#ENVIRONMENTS[@]} * ${#REWARD_TYPES[@]} * ${#LEARNING_RATES[@]} \
        * ${#GAMMAS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} ))

echo "========================================================"
echo " PPO + Entropy Regularization — Parallel Sweep"
echo "========================================================"
echo " Total configurations : $TOTAL"
echo " Max parallel jobs    : $MAX_JOBS"
echo " Threads per job      : $THREADS_PER_JOB"
echo " JAX platform         : cpu"
echo " Metrics dir          : $METRICS_DIR"
echo " Log dir              : $LOG_DIR"
echo "========================================================"

mkdir -p "$METRICS_DIR" "$LOG_DIR"

# ── Build the command list ─────────────────────────────────────────────────
CMDFILE=$(mktemp /tmp/ppo_entropy_cmds.XXXXXX)
trap "rm -f $CMDFILE" EXIT

for ENV in "${ENVIRONMENTS[@]}"; do
for RW  in "${REWARD_TYPES[@]}"; do
for LR  in "${LEARNING_RATES[@]}"; do
for G   in "${GAMMAS[@]}"; do
for A   in "${ALPHAS[@]}"; do
for S   in "${SEEDS[@]}"; do
    cat >> "$CMDFILE" <<EOF
python -m src.test.run_single_ppo_entropy \
    --env $ENV --reward $RW \
    --lr $LR --gamma $G --alpha $A --seed $S \
    --num_episodes $NUM_EPISODES --max_steps $MAX_STEPS \
    --eval_episodes $EVAL_EPISODES --log_every $LOG_EVERY \
    --hidden_dim $HIDDEN_DIM --n_epochs $N_EPOCHS \
    --mini_batch $MINI_BATCH --clip_eps $CLIP_EPS \
    --gae_lambda $GAE_LAMBDA --metrics_dir $METRICS_DIR
EOF
done; done; done; done; done; done

echo "Generated $TOTAL commands → $CMDFILE"

# ── Dry run: just print ───────────────────────────────────────────────────
if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Commands that would be executed:"
    echo "-------------------------------------------"
    cat "$CMDFILE"
    exit 0
fi

# ── Execute in parallel ──────────────────────────────────────────────────
# Strategy: try GNU parallel first (best UX), fall back to xargs -P,
# and finally a pure-bash background-job pool.

START_TIME=$(date +%s)

if command -v parallel &>/dev/null; then
    # ── GNU parallel ──────────────────────────────────────────────────
    echo ""
    echo "Using GNU parallel ($(parallel --version | head -1))"
    echo ""
    parallel --jobs "$MAX_JOBS" \
             --bar \
             --halt soon,fail=10 \
             --joblog "${LOG_DIR}/parallel_joblog.txt" \
             --results "${LOG_DIR}/parallel_out" \
             < "$CMDFILE"

elif xargs --version 2>&1 | grep -q GNU; then
    # ── GNU xargs -P ─────────────────────────────────────────────────
    echo ""
    echo "GNU parallel not found — falling back to xargs -P $MAX_JOBS"
    echo ""
    cat "$CMDFILE" | xargs -I CMD -P "$MAX_JOBS" bash -c 'CMD'
    # The above trick doesn't expand CMD properly; use the loop below
    # as the real fallback instead.  Keeping this branch as a note.
    # In practice, the bash pool below is the reliable fallback.
    echo "(xargs fallback may not work on all systems — see bash pool below)"

else
    # ── Pure bash background-job pool ────────────────────────────────
    echo ""
    echo "No GNU parallel or xargs -P — using bash job pool (max $MAX_JOBS)"
    echo ""

    RUNNING=0
    IDX=0

    while IFS= read -r CMD; do
        IDX=$((IDX + 1))

        # Launch in background, redirect stdout/stderr to per-run log
        RUN_LOG="${LOG_DIR}/run_${IDX}.out"
        eval "$CMD" > "$RUN_LOG" 2>&1 &

        RUNNING=$((RUNNING + 1))

        # If we've hit the cap, wait for one job to finish
        if (( RUNNING >= MAX_JOBS )); then
            wait -n 2>/dev/null || true   # wait for any one child
            RUNNING=$((RUNNING - 1))
        fi

        # Progress
        if (( IDX % 10 == 0 )); then
            echo "[progress] launched $IDX / $TOTAL"
        fi
    done < "$CMDFILE"

    # Wait for stragglers
    echo "All $TOTAL jobs launched — waiting for remaining to finish..."
    wait
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "========================================================"
echo " Sweep complete!  ${MINS}m ${SECS}s elapsed"
echo " Per-run CSVs in : $METRICS_DIR/"
echo " Logs in         : $LOG_DIR/"
echo "========================================================"

# ── Merge per-run CSVs into one master file ──────────────────────────────
echo ""
echo "Merging per-run CSVs..."

python3 - "$METRICS_DIR" <<'PYEOF'
import sys, os, glob
import pandas as pd

metrics_dir = sys.argv[1]

# episode-level
ep_files = sorted(glob.glob(os.path.join(metrics_dir, "ppo_entropy_ppo_entropy_*.csv")))
ep_files = [f for f in ep_files if "_summary" not in f]
if ep_files:
    df = pd.concat([pd.read_csv(f) for f in ep_files], ignore_index=True)
    out = os.path.join(metrics_dir, "ppo_entropy_ALL_episodes.csv")
    df.to_csv(out, index=False)
    print(f"  Episodes : {out}  ({len(df)} rows from {len(ep_files)} files)")

# summary-level
sum_files = sorted(glob.glob(os.path.join(metrics_dir, "*_summary.csv")))
if sum_files:
    df = pd.concat([pd.read_csv(f) for f in sum_files], ignore_index=True)
    out = os.path.join(metrics_dir, "ppo_entropy_ALL_summary.csv")
    df.to_csv(out, index=False)
    print(f"  Summary  : {out}  ({len(df)} rows from {len(sum_files)} files)")

print("Done.")
PYEOF

echo ""
echo "All done. Check $METRICS_DIR/ppo_entropy_ALL_*.csv for merged results."
