#!/bin/bash
# ============================================================================
# ELG5214 Group 8 — Project Folder Structure Setup
# Exploration Mechanisms for On/Off-Policy RL Under Sparse Rewards
# ============================================================================
# Usage: chmod +x setup_structure.sh && ./setup_structure.sh

set -e
echo "Setting up project folder structure..."

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ─── src/agents/ ─────────────────────────────────────────────────────────────
# One file per (algorithm × exploration mechanism) condition
mkdir -p "$ROOT/src/agents"
touch "$ROOT/src/agents/dqn_agent.py"              # [Student B] DQN baseline agent (epsilon-greedy)
touch "$ROOT/src/agents/dqn_entropy_agent.py"       # [Student A] DQN + entropy reg agent
touch "$ROOT/src/agents/dqn_rnd_agent.py"           # [Student A] DQN + RND agent
touch "$ROOT/src/agents/dqn_icm_agent.py"           # [Student B] DQN + ICM agent
touch "$ROOT/src/agents/ppo_agent.py"               # [Student D] PPO baseline agent
touch "$ROOT/src/agents/ppo_entropy_agent.py"       # [Student C] PPO + entropy reg agent
touch "$ROOT/src/agents/ppo_rnd_agent.py"           # [Student C] PPO + RND agent
touch "$ROOT/src/agents/ppo_icm_agent.py"           # [Student D] PPO + ICM agent
touch "$ROOT/src/agents/random_agent.py"            # Random baseline agent

# ─── src/envs/ ───────────────────────────────────────────────────────────────
# Gymnax environment wrappers (sparse reward variants)
mkdir -p "$ROOT/src/envs"
touch "$ROOT/src/envs/cartpole_env.py"              # CartPole-v1 dense + sparse wrapper
touch "$ROOT/src/envs/mountaincar_env.py"           # MountainCar-v0 dense + sparse wrapper
touch "$ROOT/src/envs/env_utils.py"                 # Factory: get_env(name, reward_type)

# ─── src/networks/ ───────────────────────────────────────────────────────────
# Pure JAX network definitions (init_params + forward)
mkdir -p "$ROOT/src/networks"
touch "$ROOT/src/networks/q_network.py"             # Q-network: 2×64 ReLU MLP for DQN
touch "$ROOT/src/networks/policy_network.py"        # Policy network: 2×64 MLP for PPO
touch "$ROOT/src/networks/value_network.py"         # Value network for PPO baseline
touch "$ROOT/src/networks/rnd_networks.py"          # RND target (fixed) + predictor networks
touch "$ROOT/src/networks/icm_networks.py"          # ICM forward model + inverse model

# ─── src/exploration/ ────────────────────────────────────────────────────────
# Exploration mechanism modules (intrinsic reward computation)
mkdir -p "$ROOT/src/exploration"
touch "$ROOT/src/exploration/entropy_reg.py"        # Entropy bonus for policy loss
touch "$ROOT/src/exploration/rnd.py"                # RND: r_i = ||f(s) - f_hat(s)||^2
touch "$ROOT/src/exploration/icm.py"                # ICM: forward/inverse model curiosity

# ─── src/replay/ ─────────────────────────────────────────────────────────────
# Experience replay buffer (JAX arrays, lax.fori_loop insertion)
mkdir -p "$ROOT/src/replay"
touch "$ROOT/src/replay/replay_buffer.py"           # Circular buffer: init, add, sample

# ─── src/training/ ───────────────────────────────────────────────────────────
# Training loops and rollout utilities
mkdir -p "$ROOT/src/training"
touch "$ROOT/src/training/rollout.py"               # lax.scan episode collection
touch "$ROOT/src/training/train_dqn.py"             # DQN baseline training loop
touch "$ROOT/src/training/train_dqn_entropy.py"     # [Student A] DQN + entropy training
touch "$ROOT/src/training/train_dqn_rnd.py"         # [Student A] DQN + RND training
touch "$ROOT/src/training/train_dqn_icm.py"         # [Student B] DQN + ICM training
touch "$ROOT/src/training/train_ppo.py"             # PPO baseline training loop
touch "$ROOT/src/training/train_ppo_entropy.py"     # [Student C] PPO + entropy training
touch "$ROOT/src/training/train_ppo_rnd.py"         # [Student C] PPO + RND training
touch "$ROOT/src/training/train_ppo_icm.py"         # [Student D] PPO + ICM training

# ─── src/evaluate/ ───────────────────────────────────────────────────────────
# Greedy evaluation scripts per agent type
mkdir -p "$ROOT/src/evaluate"
touch "$ROOT/src/evaluate/evaluate_dqn.py"          # Greedy DQN eval (all DQN variants)
touch "$ROOT/src/evaluate/evaluate_ppo.py"          # Greedy PPO eval (all PPO variants)
touch "$ROOT/src/evaluate/evaluate_random.py"       # Random agent eval

# ─── src/test/ ───────────────────────────────────────────────────────────────
# Per-agent test/run scripts (like test_dqn_agent.py in reference)
mkdir -p "$ROOT/src/test"
touch "$ROOT/src/test/test_dqn_agent.py"            # [Student B] Run DQN baseline
touch "$ROOT/src/test/test_dqn_entropy_agent.py"    # [Student A] Run DQN + entropy
touch "$ROOT/src/test/test_dqn_rnd_agent.py"        # [Student A] Run DQN + RND
touch "$ROOT/src/test/test_dqn_icm_agent.py"        # [Student B] Run DQN + ICM
touch "$ROOT/src/test/test_ppo_agent.py"            # [Student D] Run PPO baseline
touch "$ROOT/src/test/test_ppo_entropy_agent.py"    # [Student C] Run PPO + entropy
touch "$ROOT/src/test/test_ppo_rnd_agent.py"        # [Student C] Run PPO + RND
touch "$ROOT/src/test/test_ppo_icm_agent.py"        # [Student D] Run PPO + ICM
touch "$ROOT/src/test/test_random_agent.py"         # Run random baseline
touch "$ROOT/src/test/test_env.py"                  # Env sanity check

# ─── src/utils/ ──────────────────────────────────────────────────────────────
# Shared utilities (metrics, logging, device info, timers)
mkdir -p "$ROOT/src/utils"
touch "$ROOT/src/utils/reusable.py"                 # RLMetricsDataset, setup_logger, Timer, device utils

# ─── configs/ ────────────────────────────────────────────────────────────────
mkdir -p "$ROOT/configs"
touch "$ROOT/configs/dqn_baseline.yaml"
touch "$ROOT/configs/dqn_entropy.yaml"
touch "$ROOT/configs/dqn_rnd.yaml"
touch "$ROOT/configs/dqn_icm.yaml"
touch "$ROOT/configs/ppo_baseline.yaml"
touch "$ROOT/configs/ppo_entropy.yaml"
touch "$ROOT/configs/ppo_rnd.yaml"
touch "$ROOT/configs/ppo_icm.yaml"
touch "$ROOT/configs/sweep.yaml"                    # Master sweep: LR × γ × seed grid

# ─── scripts/ ────────────────────────────────────────────────────────────────
# Orchestration and analysis scripts
mkdir -p "$ROOT/scripts"
touch "$ROOT/scripts/run_sweep.sh"                  # Bash orchestrator for full sweep
touch "$ROOT/scripts/aggregate_results.py"          # Merge per-seed CSVs → summary.csv
touch "$ROOT/scripts/generate_plots.py"             # Generate all figures

# ─── results/ ────────────────────────────────────────────────────────────────
# Per-condition × per-env × per-reward results
for algo in dqn_baseline dqn_entropy dqn_rnd dqn_icm ppo_baseline ppo_entropy ppo_rnd ppo_icm; do
    for env_reward in cartpole_dense cartpole_sparse mountaincar_dense mountaincar_sparse; do
        mkdir -p "$ROOT/results/${algo}/${env_reward}"
    done
done
mkdir -p "$ROOT/results/random_baseline"
mkdir -p "$ROOT/results/heuristic_baseline"

# ─── metrics/ ────────────────────────────────────────────────────────────────
# RLMetricsDataset CSV output (matches reference reusable.py save() default)
mkdir -p "$ROOT/metrics"

# ─── logs/ ───────────────────────────────────────────────────────────────────
# Per-agent log files (matches reference setup_logger pattern)
mkdir -p "$ROOT/logs/dqn"
mkdir -p "$ROOT/logs/dqn_entropy"
mkdir -p "$ROOT/logs/dqn_rnd"
mkdir -p "$ROOT/logs/dqn_icm"
mkdir -p "$ROOT/logs/ppo"
mkdir -p "$ROOT/logs/ppo_entropy"
mkdir -p "$ROOT/logs/ppo_rnd"
mkdir -p "$ROOT/logs/ppo_icm"
mkdir -p "$ROOT/logs/random_agent"

# ─── visualizations/ ─────────────────────────────────────────────────────────
mkdir -p "$ROOT/visualizations/learning_curves"
mkdir -p "$ROOT/visualizations/heatmaps"
mkdir -p "$ROOT/visualizations/intrinsic_rewards"
mkdir -p "$ROOT/visualizations/policy_entropy"
mkdir -p "$ROOT/visualizations/comparison_tables"

# ─── report & presentation ───────────────────────────────────────────────────
mkdir -p "$ROOT/report"
mkdir -p "$ROOT/presentation"

# ─── .gitignore ──────────────────────────────────────────────────────────────
cat > "$ROOT/.gitignore" << 'GITIGNORE'
venv/
.env
GITIGNORE

echo ""
echo "Project structure created successfully!"
echo ""
echo "Folder tree:"
find "$ROOT" -not -path '*/__pycache__/*' -not -path '*/.git/*' | head -150 | sed "s|$ROOT|.|g" | sort
