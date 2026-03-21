# Exploration Mechanisms for On and Off-Policy RL Under Sparse Rewards

**ELG5214 / CSI5340 — Introduction to Deep Reinforcement Learning**
**Group 8 — University of Ottawa — Winter 2026**

| Role | Name | Email |
|------|------|-------|
| Student A | Mohammad | fmoha077@uottawa.ca |
| Student B | Anthony Nasr | anasr014@uottawa.ca |
| Student C | Md Mosarraf | mhoss048@uottawa.ca |
| Student D | Mariana Chavez Flores | mchaz097@uottawa.ca |

---

## Overview

A **reproducible, matched-compute comparison** of three exploration mechanisms — **Entropy Regularization**, **Random Network Distillation (RND)**, and the **Intrinsic Curiosity Module (ICM)** — applied to both **PPO** (on-policy) and **DQN** (off-policy) on Gymnax CartPole-v1 and MountainCar-v0 under dense and sparse reward variants.

**Research Questions:**
1. Under sparse rewards and matched compute, do RND or ICM improve sample efficiency more for PPO than DQN, or does entropy regularization close the gap?
2. Does ICM's dynamics-based signal yield different gains than RND's prediction error, and does this interact with the on/off-policy distinction as reward density varies?

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/mohammad1774/elg5214_group_8_project.git
cd elg5214_group_8_project

# 2. Create folder structure
chmod +x setup_structure.sh && ./setup_structure.sh

# 3. Install
pip install -r requirements.txt

# 4. Set JAX memory flags (recommended for GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# 5. Run a single agent test (example: DQN + RND)
python -m src.test.test_dqn_rnd_agent

# 6. Run full sweep
bash scripts/run_sweep.sh

# 7. Aggregate + plot
python scripts/aggregate_results.py
python scripts/generate_plots.py
```

---

## Folder Structure

The layout mirrors our Assignment 2 codebase. Every directory under `src/` has one clear responsibility. Imports use `from src.X.Y import Z` throughout.

```
elg5214_group_8_project/
│
├── README.md                              ← You are here
├── setup_structure.sh                     ← Run once to create all directories
├── requirements.txt                       ← Pinned pip dependencies
├── environment.yml                        ← Conda environment spec
├── .gitignore
│
├── src/
│   ├── agents/                            ← AGENT CLASSES (act + greedy_action)
│   │   ├── random_agent.py                ← Random baseline
│   │   ├── dqn_agent.py                   ← [Student B] Vanilla DQN (ε-greedy)
│   │   ├── dqn_entropy_agent.py           ← [Student A] DQN + entropy-regularized action selection
│   │   ├── dqn_rnd_agent.py               ← [Student A] DQN + RND-augmented rewards
│   │   ├── dqn_icm_agent.py               ← [Student B] DQN + ICM curiosity rewards
│   │   ├── ppo_agent.py                   ← [Student D] Vanilla PPO (categorical policy)
│   │   ├── ppo_entropy_agent.py           ← [Student C] PPO + entropy bonus
│   │   ├── ppo_rnd_agent.py               ← [Student C] PPO + RND intrinsic reward
│   │   └── ppo_icm_agent.py               ← [Student D] PPO + ICM curiosity
│   │
│   ├── envs/                              ← ENVIRONMENT WRAPPERS
│   │   ├── cartpole_env.py                ← Gymnax CartPole-v1 + sparse reward variant
│   │   ├── mountaincar_env.py             ← Gymnax MountainCar-v0 + sparse reward variant
│   │   └── env_utils.py                   ← Factory: get_env(name, reward_type) → (env, params)
│   │
│   ├── networks/                          ← PURE JAX NETWORKS (init_params + forward)
│   │   ├── q_network.py                   ← Q-network: init_q_params, q_forward, q_forward_batch
│   │   ├── policy_network.py              ← Policy net: init_policy_params, policy_forward, log_prob
│   │   ├── value_network.py               ← Value net for PPO advantage estimation
│   │   ├── rnd_networks.py                ← RND: init_rnd_params, rnd_target_forward, rnd_predictor_forward
│   │   └── icm_networks.py                ← ICM: init_icm_params, icm_forward_model, icm_inverse_model
│   │
│   ├── exploration/                       ← EXPLORATION MECHANISMS (intrinsic reward computation)
│   │   ├── entropy_reg.py                 ← entropy_bonus(logits) → scalar bonus
│   │   ├── rnd.py                         ← rnd_intrinsic_reward(rnd_params, obs) → r_intrinsic
│   │   └── icm.py                         ← icm_intrinsic_reward(icm_params, obs, action, next_obs)
│   │
│   ├── replay/                            ← EXPERIENCE REPLAY (for DQN variants)
│   │   └── replay_buffer.py               ← init_buffer, add_transition, sample_batch (JAX arrays)
│   │
│   ├── training/                          ← TRAINING LOOPS (one file per condition)
│   │   ├── rollout.py                     ← run_one_episode_scan_simple() — lax.scan rollout
│   │   ├── train_dqn.py                   ← [Student B] DQN baseline: _run_dqn_episode_scan + update loop
│   │   ├── train_dqn_entropy.py           ← [Student A] DQN + entropy reg training
│   │   ├── train_dqn_rnd.py               ← [Student A] DQN + RND training
│   │   ├── train_dqn_icm.py              ← [Student B] DQN + ICM training
│   │   ├── train_ppo.py                   ← [Student D] PPO baseline training
│   │   ├── train_ppo_entropy.py           ← [Student C] PPO + entropy training
│   │   ├── train_ppo_rnd.py              ← [Student C] PPO + RND training
│   │   └── train_ppo_icm.py              ← [Student D] PPO + ICM training
│   │
│   ├── evaluate/                          ← GREEDY EVALUATION
│   │   ├── evaluate_dqn.py                ← evaluate_dqn_greedy() — works for all DQN variants
│   │   ├── evaluate_ppo.py                ← evaluate_ppo() — works for all PPO variants
│   │   └── evaluate_random.py             ← evaluate_random_agent()
│   │
│   ├── test/                              ← TEST/RUN SCRIPTS (one per agent, calls training + eval)
│   │   ├── test_env.py                    ← Environment sanity check
│   │   ├── test_random_agent.py           ← Run random baseline eval
│   │   ├── test_dqn_agent.py              ← [Student B] Run DQN baseline sweep
│   │   ├── test_dqn_entropy_agent.py      ← [Student A] Run DQN + entropy sweep
│   │   ├── test_dqn_rnd_agent.py          ← [Student A] Run DQN + RND sweep
│   │   ├── test_dqn_icm_agent.py          ← [Student B] Run DQN + ICM sweep
│   │   ├── test_ppo_agent.py              ← [Student D] Run PPO baseline sweep
│   │   ├── test_ppo_entropy_agent.py      ← [Student C] Run PPO + entropy sweep
│   │   ├── test_ppo_rnd_agent.py          ← [Student C] Run PPO + RND sweep
│   │   └── test_ppo_icm_agent.py          ← [Student D] Run PPO + ICM sweep
│   │
│   └── utils/                             ← SHARED UTILITIES
│       └── reusable.py                    ← RLMetricsDataset, setup_logger, Timer, device utils
│
├── configs/                               ← YAML CONFIG FILES
│   ├── dqn_baseline.yaml
│   ├── dqn_entropy.yaml
│   ├── dqn_rnd.yaml
│   ├── dqn_icm.yaml
│   ├── ppo_baseline.yaml
│   ├── ppo_entropy.yaml
│   ├── ppo_rnd.yaml
│   ├── ppo_icm.yaml
│   └── sweep.yaml                         ← Master: all LR × γ × seed combinations
│
├── scripts/                               ← ORCHESTRATION & ANALYSIS
│   ├── run_sweep.sh                       ← Loop all conditions × seeds
│   ├── aggregate_results.py               ← Merge CSVs → results/summary.csv
│   └── generate_plots.py                  ← Produce all figures → visualizations/
│
├── results/                               ← EXPERIMENT OUTPUTS
│   ├── dqn_baseline/
│   │   ├── cartpole_dense/                ← Per-seed CSVs: lr0.001_g0.99_seed0.csv
│   │   ├── cartpole_sparse/
│   │   ├── mountaincar_dense/
│   │   └── mountaincar_sparse/
│   ├── dqn_entropy/                       ← Same 4 subdirectories
│   ├── dqn_rnd/                           ← Same 4 subdirectories
│   ├── dqn_icm/                           ← Same 4 subdirectories
│   ├── ppo_baseline/                      ← Same 4 subdirectories
│   ├── ppo_entropy/                       ← Same 4 subdirectories
│   ├── ppo_rnd/                           ← Same 4 subdirectories
│   ├── ppo_icm/                           ← Same 4 subdirectories
│   ├── random_baseline/
│   └── heuristic_baseline/
│
├── metrics/                               ← RLMetricsDataset CSV output
│   ├── {proj_name}_dataset_metrics.csv            ← Per-episode data
│   └── {proj_name}_dataset_metrics_summary.csv    ← Aggregated summaries
│
├── logs/                                  ← PER-AGENT LOG FILES
│   ├── dqn/                               ← run{id}.log for DQN baseline
│   ├── dqn_entropy/
│   ├── dqn_rnd/
│   ├── dqn_icm/
│   ├── ppo/
│   ├── ppo_entropy/
│   ├── ppo_rnd/
│   ├── ppo_icm/
│   └── random_agent/
│
├── visualizations/                        ← ALL FIGURES (from scripts/generate_plots.py)
│   ├── learning_curves/                   ← avg return vs episodes per condition
│   ├── heatmaps/                          ← Hyperparameter sensitivity (LR × γ)
│   ├── intrinsic_rewards/                 ← RND/ICM reward magnitude over training
│   ├── policy_entropy/                    ← Entropy trajectories per condition
│   └── comparison_tables/                 ← Summary bar charts, crossover plots
│
├── report/                                ← FINAL REPORT
└── presentation/                          ← SLIDES
```

---

## How the Code Flows (following the reference pattern)

Each condition follows the same 4-step pipeline:

```
test/test_<agent>.py          ← Entry point: sets seed, lr, gamma, loads config
    ↓ calls
training/train_<agent>.py     ← Training loop: collects episodes (rollout.py), updates params
    ↓ uses
agents/<agent>.py             ← Agent class: act(key, obs), greedy_action(obs)
networks/<network>.py         ← Pure functions: init_params(key), forward(params, obs)
exploration/<mechanism>.py    ← Intrinsic reward: compute_intrinsic(params, obs, ...)
replay/replay_buffer.py       ← (DQN only) init_buffer, add_transition, sample_batch
    ↓ at intervals
evaluate/evaluate_<algo>.py   ← Greedy evaluation → success_rate, mean_reward
    ↓ logs to
utils/reusable.py             ← RLMetricsDataset.add_episode() + .add_summary() → CSV
```

Every `test_*.py` file is self-contained: it creates the env, initializes params, calls the training function, runs final evaluation, and saves metrics. This means each teammate can run their conditions independently.

---

## Hyperparameter Sweep

| Parameter | Values |
|-----------|--------|
| Learning rate | `1e-4`, `5e-4`, `1e-3` |
| Discount γ | `0.99`, `0.95` |
| Seeds | `0, 1, 2, 3, 4` |
| Environments | CartPole-v1, MountainCar-v0 |
| Reward type | Dense, Sparse |

**Per agent variant:** 3 LR × 2 γ × 5 seeds × 2 envs × 2 reward types = **120 runs**

Additional exploration hyperparameters (fixed per condition, tuned separately):
- Entropy coefficient α ∈ {0.01, 0.05}
- RND reward scale β ∈ {0.1, 1.0}
- ICM curiosity scale η ∈ {0.1, 1.0}

---

## Metrics Output

`RLMetricsDataset` (in `src/utils/reusable.py`) produces two CSVs per project:

**Episode-level** (`metrics/{proj}_dataset_metrics.csv`):
seed, episode, reward, episode_length, loss, eval_success_rate, algorithm, learning_rate, gamma

**Summary-level** (`metrics/{proj}_dataset_metrics_summary.csv`):
seed, algorithm, learning_rate, gamma, final_mean_reward, final_success_rate, backend, devices, action, mean_length, wall_time_s

---

## Work Distribution & Deadlines

### Phase 1 — Shared Foundation (Mar 21–22)
All members coordinate on shared modules. Student A scaffolds the repo.

| File | Owner | Status |
|------|-------|--------|
| `src/envs/*` | All | Shared |
| `src/networks/q_network.py` | A + B | Shared by DQN variants |
| `src/networks/policy_network.py` | C + D | Shared by PPO variants |
| `src/networks/value_network.py` | C + D | PPO baseline estimator |
| `src/networks/rnd_networks.py` | A + C | Shared by RND variants |
| `src/networks/icm_networks.py` | B + D | Shared by ICM variants |
| `src/replay/replay_buffer.py` | A + B | DQN replay buffer |
| `src/training/rollout.py` | All | Shared lax.scan rollout |
| `src/utils/reusable.py` | A | Metrics, logging, timers |
| `src/exploration/*` | By mechanism owner | Module per mechanism |

### Phase 2 — Agent Implementation + Training (Mar 22–24)

| Agent Variant | Agent File | Training File | Test File | Owner |
|---------------|-----------|--------------|-----------|-------|
| DQN baseline | `dqn_agent.py` | `train_dqn.py` | `test_dqn_agent.py` | **Student B** |
| DQN + entropy | `dqn_entropy_agent.py` | `train_dqn_entropy.py` | `test_dqn_entropy_agent.py` | **Student A** |
| DQN + RND | `dqn_rnd_agent.py` | `train_dqn_rnd.py` | `test_dqn_rnd_agent.py` | **Student A** |
| DQN + ICM | `dqn_icm_agent.py` | `train_dqn_icm.py` | `test_dqn_icm_agent.py` | **Student B** |
| PPO baseline | `ppo_agent.py` | `train_ppo.py` | `test_ppo_agent.py` | **Student D** |
| PPO + entropy | `ppo_entropy_agent.py` | `train_ppo_entropy.py` | `test_ppo_entropy_agent.py` | **Student C** |
| PPO + RND | `ppo_rnd_agent.py` | `train_ppo_rnd.py` | `test_ppo_rnd_agent.py` | **Student C** |
| PPO + ICM | `ppo_icm_agent.py` | `train_ppo_icm.py` | `test_ppo_icm_agent.py` | **Student D** |

**Code review:** A ↔ B, C ↔ D

### Phase 3 — Visualizations & Analysis (Mar 24–25)
| Task | Owner |
|------|-------|
| `scripts/aggregate_results.py` | Student A |
| `scripts/generate_plots.py` | Student A |
| Significance tests (Welch's t-test) | Student A |

### Phase 4 — Report & Presentation (Mar 25–26)
| Task | Owner |
|------|-------|
| Report draft | Students B, C, D |
| Slides + Q&A prep | Students B, C, D |
| Final review + merge | Student A |

**Deadline: March 26, 2026**

---

## Naming Conventions

| What | Pattern | Example |
|------|---------|---------|
| Agent file | `{algo}_{mechanism}_agent.py` | `dqn_rnd_agent.py` |
| Training file | `train_{algo}_{mechanism}.py` | `train_dqn_rnd.py` |
| Test file | `test_{algo}_{mechanism}_agent.py` | `test_dqn_rnd_agent.py` |
| Config file | `{algo}_{mechanism}.yaml` | `dqn_rnd.yaml` |
| Result CSV | `lr{LR}_g{GAMMA}_seed{SEED}.csv` | `lr0.001_g0.99_seed3.csv` |
| Log file | `logs/{algo}_{mechanism}/run{id}.log` | `logs/dqn_rnd/run342.log` |
| Plot file | `{metric}_{algo}_{env}_{reward}.png` | `learning_curve_dqn_rnd_cartpole_sparse.png` |

---

## References

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
2. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
3. Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. *ICML 2017*.
4. Burda, Y. et al. (2019). Exploration by Random Network Distillation. *ICLR 2019*.
5. Lax, M. et al. (2023). Gymnax: A JAX-based Reinforcement Learning Environment Library. *arXiv:2311.16943*.
