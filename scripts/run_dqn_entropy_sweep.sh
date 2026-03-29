#!/bin/bash
# ============================================================================
# DQN + Entropy Regularization — Full Sweep
# Student A (Mohammad)
#
# Reads sweep grid from configs/dqn_entropy.yaml
# Skips runs that already have output CSVs (resume-safe)
#
# Usage:
#   bash scripts/run_dqn_entropy_sweep.sh                 # sequential
#   bash scripts/run_dqn_entropy_sweep.sh --parallel 2    # 2 at a time
#   bash scripts/run_dqn_entropy_sweep.sh --parallel 3    # 3 at a time
#
# Single run:
#   python -m src.test.test_dqn_entropy_agent \
#       --seed 0 --lr 0.001 --gamma 0.99 --alpha 0.01 \
#       --env cartpole --reward sparse
# ============================================================================

# Parse --parallel flag
PARALLEL=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# JAX GPU memory — split across parallel jobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "$PARALLEL" -gt 1 ]; then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.9 / $PARALLEL, 2))")
else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
fi

CONFIG="configs/dqn_entropy.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "============================================"
echo "  DQN + Entropy Regularization Sweep"
echo "  Config: $CONFIG"
echo "  Parallel jobs: $PARALLEL"
echo "  GPU mem per job: $XLA_PYTHON_CLIENT_MEM_FRACTION"
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

                    OUTPUT_FILE="results/dqn_entropy/${env_name}_${reward_type}/lr${lr}_g${gamma}_a${alpha}_seed${seed}.csv"
                    if [ -f "$OUTPUT_FILE" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    COMMANDS+=("python -m src.test.test_dqn_entropy_agent --seed $seed --lr $lr --gamma $gamma --alpha $alpha --env $env_name --reward $reward_type --config $CONFIG")
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

if [ "$PARALLEL" -eq 1 ]; then
    # Sequential
    IDX=0
    for CMD in "${COMMANDS[@]}"; do
        IDX=$((IDX + 1))
        echo "[$IDX/$REMAINING] $CMD"
        eval "$CMD"
        echo ""
    done
else
    # Parallel: launch $PARALLEL jobs, wait for batch, launch next batch
    IDX=0
    while [ $IDX -lt $REMAINING ]; do
        PIDS=()
        for ((j=0; j<PARALLEL && IDX<REMAINING; j++)); do
            CMD="${COMMANDS[$IDX]}"
            IDX=$((IDX + 1))
            echo "[$IDX/$REMAINING] START: $CMD"
            eval "$CMD" &
            PIDS+=($!)
        done

        # Wait for this batch to finish before launching next
        for PID in "${PIDS[@]}"; do
            wait "$PID"
        done
        echo "--- batch done ---"
        echo ""
    done
fi

echo "============================================"
echo "  Sweep complete!"
echo "  Total: $TOTAL | Ran: $REMAINING | Skipped: $SKIPPED"
echo "  Results in: results/dqn_entropy/"
echo "============================================"
