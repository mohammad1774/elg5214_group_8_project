#!/bin/bash
# run_cartpole_sweep.sh — Full CartPole sweep for DQN and DQN+ICM
#
# Usage:
#   chmod +x run_cartpole_sweep.sh
#   ./run_cartpole_sweep.sh
#
# Runs 360 total experiments:
#   DQN:     2 rewards × 3 LR × 2 γ × 10 seeds = 120
#   DQN+ICM: 2 rewards × 3 LR × 2 γ × 10 seeds × 2 η = 240
#
# Each run saves a CSV to results/ — if a CSV already exists, it's skipped.
# This means you can safely re-run the script if it's interrupted.

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_triton_gemm_any=false"
export JAX_PLATFORMS=cpu

REWARDS="dense sparse"
LRS="0.1 0.01 0.001"
GAMMAS="0.99 0.9"
SEEDS="0 1 2 3 4 5 6 7 8 9"
ETAS="0.1 1.0"
NUM_EPISODES=1000

# Count total runs
TOTAL=0
for r in $REWARDS; do for lr in $LRS; do for g in $GAMMAS; do for s in $SEEDS; do
    TOTAL=$((TOTAL + 1))  # DQN
    for eta in $ETAS; do
        TOTAL=$((TOTAL + 1))  # DQN+ICM
    done
done; done; done; done

echo "============================================================"
echo "CARTPOLE FULL SWEEP — $TOTAL total runs"
echo "============================================================"
echo "DQN:     120 runs (2 rewards × 3 LR × 2 γ × 10 seeds)"
echo "DQN+ICM: 240 runs (2 rewards × 3 LR × 2 γ × 10 seeds × 2 η)"
echo "Episodes per run: $NUM_EPISODES"
echo "============================================================"
echo ""

COUNT=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

for reward in $REWARDS; do
    for lr in $LRS; do
        for gamma in $GAMMAS; do
            for seed in $SEEDS; do

                # ── DQN baseline ──────────────────────────
                COUNT=$((COUNT + 1))
                echo ""
                echo "[$COUNT/$TOTAL] DQN | reward=$reward lr=$lr γ=$gamma seed=$seed"
                python sweep_runner.py \
                    --agent dqn \
                    --env cartpole \
                    --reward "$reward" \
                    --lr "$lr" \
                    --gamma "$gamma" \
                    --seed "$seed" \
                    --num_episodes "$NUM_EPISODES"

                if [ $? -ne 0 ]; then
                    echo "  *** FAILED ***"
                    FAILED=$((FAILED + 1))
                fi

                # ── DQN + ICM (for each eta) ──────────────
                for eta in $ETAS; do
                    COUNT=$((COUNT + 1))
                    echo ""
                    echo "[$COUNT/$TOTAL] DQN+ICM | reward=$reward lr=$lr γ=$gamma seed=$seed η=$eta"
                    python sweep_runner.py \
                        --agent dqn_icm \
                        --env cartpole \
                        --reward "$reward" \
                        --lr "$lr" \
                        --gamma "$gamma" \
                        --seed "$seed" \
                        --eta "$eta" \
                        --num_episodes "$NUM_EPISODES"

                    if [ $? -ne 0 ]; then
                        echo "  *** FAILED ***"
                        FAILED=$((FAILED + 1))
                    fi
                done

            done
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
echo "  Total runs: $TOTAL"
echo "  Failed:     $FAILED"
echo "  Wall time:  ${HOURS}h ${MINS}m"
echo "============================================================"
echo ""
echo "Next step: python plot_results.py"
