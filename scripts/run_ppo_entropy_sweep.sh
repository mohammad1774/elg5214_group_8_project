#!/bin/bash
# ============================================================================
# PPO + Entropy Regularization — Full Sweep (CPU)
# Student A (Mohammad)
#
# Reads sweep grid from configs/ppo_entropy.yaml
# Skips runs that already have output CSVs (resume-safe)
#
# Usage:
#   bash scripts/run_ppo_entropy_sweep.sh
#   bash scripts/run_ppo_entropy_sweep.sh --parallel 2
#   bash scripts/run_ppo_entropy_sweep.sh --parallel 4
# ============================================================================

set -u

# Parse --parallel flag
PARALLEL=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
    shift
done

# ----------------------------------------------------------------------
# Force CPU
# ----------------------------------------------------------------------
export JAX_PLATFORMS=cpu

# Remove GPU-specific settings if present
unset XLA_PYTHON_CLIENT_PREALLOCATE
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset CUDA_VISIBLE_DEVICES

# ----------------------------------------------------------------------
# Control CPU threading to avoid oversubscription
# ----------------------------------------------------------------------
TOTAL_CORES=$(nproc)
THREADS_PER_JOB=$(( TOTAL_CORES / PARALLEL ))
if [ "$THREADS_PER_JOB" -lt 1 ]; then
    THREADS_PER_JOB=1
fi

export OMP_NUM_THREADS=$THREADS_PER_JOB
export OPENBLAS_NUM_THREADS=$THREADS_PER_JOB
export MKL_NUM_THREADS=$THREADS_PER_JOB
export NUMEXPR_NUM_THREADS=$THREADS_PER_JOB

CONFIG="configs/ppo_entropy.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "============================================"
echo "  PPO + Entropy Regularization Sweep (CPU)"
echo "  Config: $CONFIG"
echo "  Parallel jobs: $PARALLEL"
echo "  Total CPU cores: $TOTAL_CORES"
echo "  Threads per job: $THREADS_PER_JOB"
echo "============================================"
echo ""

# Parse sweep grid from YAML
read -r -a SEEDS <<< $(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(s) for s in c['sweep']['seeds']))
")

read -r -a LRS <<< $(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(lr) for lr in c['sweep']['learning_rates']))
")

read -r -a GAMMAS <<< $(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(g) for g in c['sweep']['gammas']))
")

read -r -a ALPHAS <<< $(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(a) for a in c['sweep']['alphas']))
")

ENVS=$(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
for e in c['sweep']['environments']:
    print(f\"{e['name']} {e['reward']}\")
")

# Collect all commands (skip completed runs)
TOTAL=0
SKIPPED=0
COMMANDS=()

while IFS=' ' read -r env_name reward_type; do
    for gamma in "${GAMMAS[@]}"; do
        for lr in "${LRS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    TOTAL=$((TOTAL + 1))

                    OUTPUT_FILE="metrics/ppo_entropy/${env_name}_${reward_type}/lr${lr}_g${gamma}_a${alpha}_seed${seed}.csv"
                    if [ -f "$OUTPUT_FILE" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    COMMANDS+=("python -m src.test.run_single_ppo_entropy --seed $seed --lr $lr --gamma $gamma --alpha $alpha --env $env_name --reward $reward_type --config $CONFIG")
                done
            done
        done
    done
done <<< "$ENVS"

REMAINING=${#COMMANDS[@]}
echo "Total: $TOTAL | Skipped: $SKIPPED | Remaining: $REMAINING"
echo ""

if [ "$REMAINING" -eq 0 ]; then
    echo "All runs already completed!"
    exit 0
fi

# ── Run ──────────────────────────────────────────────────────────

FAILS=0

if [ "$PARALLEL" -eq 1 ]; then
    # Sequential
    IDX=0
    for CMD in "${COMMANDS[@]}"; do
        IDX=$((IDX + 1))
        echo "[$IDX/$REMAINING] $CMD"
        eval "$CMD"
        STATUS=$?
        if [ "$STATUS" -ne 0 ]; then
            echo "[$IDX/$REMAINING] FAILED with exit code $STATUS"
            FAILS=$((FAILS + 1))
        fi
        echo ""
    done
else
    # Parallel batch mode
    IDX=0
    while [ $IDX -lt $REMAINING ]; do
        PIDS=()
        LABELS=()

        for ((j=0; j<PARALLEL && IDX<REMAINING; j++)); do
            CMD="${COMMANDS[$IDX]}"
            IDX=$((IDX + 1))
            echo "[$IDX/$REMAINING] START: $CMD"
            eval "$CMD" &
            PIDS+=($!)
            LABELS+=("$IDX/$REMAINING")
        done

        for k in "${!PIDS[@]}"; do
            PID="${PIDS[$k]}"
            LABEL="${LABELS[$k]}"
            wait "$PID"
            STATUS=$?
            if [ "$STATUS" -ne 0 ]; then
                echo "[$LABEL] FAILED with exit code $STATUS"
                FAILS=$((FAILS + 1))
            fi
        done

        echo "--- batch done ---"
        echo ""
    done
fi

echo "============================================"
echo "  Sweep complete!"
echo "  Total: $TOTAL | Ran: $REMAINING | Skipped: $SKIPPED | Failed: $FAILS"
echo "  Results in: metrics/ppo_entropy/"
echo "============================================"

if [ "$FAILS" -ne 0 ]; then
    exit 1
fi
