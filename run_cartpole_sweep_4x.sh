#!/bin/bash
# run_cartpole_sweep.sh — Full CartPole sweep for DQN and DQN+ICM
#
# Runs 4 experiments in PARALLEL using xargs.
# Each run is fully independent — separate process, separate CSV.
#
# Usage:
#   chmod +x run_cartpole_sweep.sh
#   ./run_cartpole_sweep.sh
#
# Total: 360 runs (120 DQN + 240 DQN+ICM)
# Estimated time with 4 parallel: ~40-60 minutes on i7-9700
#
# Safe to re-run if interrupted — skips CSVs that already exist.

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS="--xla_gpu_triton_gemm_any=false"
export JAX_PLATFORMS=cpu

NUM_PARALLEL=4
NUM_EPISODES=1000

echo "============================================================"
echo "CARTPOLE FULL SWEEP — 360 total runs, ${NUM_PARALLEL} parallel"
echo "============================================================"
echo "DQN:     120 runs (2 rewards × 3 LR × 2 γ × 10 seeds)"
echo "DQN+ICM: 240 runs (2 rewards × 3 LR × 2 γ × 10 seeds × 2 η)"
echo "Episodes per run: $NUM_EPISODES"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

# Generate all 360 commands, then run NUM_PARALLEL at a time
python3 -c "
rewards = ['dense', 'sparse']
lrs = [0.1, 0.01, 0.001]
gammas = [0.99, 0.9]
seeds = range(10)
etas = [0.1, 1.0]
num_episodes = ${NUM_EPISODES}

cmds = []
for r in rewards:
    for lr in lrs:
        for g in gammas:
            for s in seeds:
                cmds.append(f'python sweep_runner.py --agent dqn --env cartpole --reward {r} --lr {lr} --gamma {g} --seed {s} --num_episodes {num_episodes}')
                for eta in etas:
                    cmds.append(f'python sweep_runner.py --agent dqn_icm --env cartpole --reward {r} --lr {lr} --gamma {g} --seed {s} --eta {eta} --num_episodes {num_episodes}')

total = len(cmds)
for i, cmd in enumerate(cmds):
    # Prefix each command with a counter for progress tracking
    print(f'echo \"[{i+1}/{total}]\" && {cmd} >> sweep_log.txt 2>&1')
" | xargs -P ${NUM_PARALLEL} -I {} bash -c '{}'

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
echo "  Wall time: ${HOURS}h ${MINS}m"
echo "  CSVs generated:"
find results/ -name "*.csv" 2>/dev/null | wc -l
echo "============================================================"
echo ""
echo "Next step: python plot_results.py"