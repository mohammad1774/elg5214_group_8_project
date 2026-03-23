# AI Prompt Template — ELG5214 Group 8

> **How to use this file:**
> 1. Find your section below (Student A/B/C/D)
> 2. Copy the prompt template
> 3. Fill in the `{{placeholders}}` with your specific values
> 4. Upload the required files listed below the prompt
> 5. Paste into Claude or ChatGPT and go

---

## Files You MUST Upload With Every Prompt

These files define the shared contracts. Without them, the AI will guess at function signatures and break compatibility.

### Required uploads (everyone, every time):

```
src/utils/reusable.py          ← RLMetricsDataset, setup_logger, Timer classes
src/training/rollout.py        ← run_one_episode_scan_simple() — the shared rollout
src/replay/replay_buffer.py    ← init_buffer, add_transition, sample_batch (DQN people)
src/networks/q_network.py      ← init_q_params, q_forward, q_forward_batch (DQN people)
src/networks/policy_network.py ← init_policy_params, policy_forward, log_prob (PPO people)
src/networks/value_network.py  ← init_value_params, value_forward (PPO people)
src/envs/env_utils.py          ← get_env() factory function
configs/sweep.yaml             ← Hyperparameter grid
README.md                      ← Folder structure reference
```

### Additional uploads depending on your task:

| If your task involves... | Also upload |
|--------------------------|-------------|
| RND | `src/networks/rnd_networks.py`, `src/exploration/rnd.py` |
| ICM | `src/networks/icm_networks.py`, `src/exploration/icm.py` |
| Entropy regularization | `src/exploration/entropy_reg.py` |
| DQN baseline (reference) | `src/agents/dqn_agent.py`, `src/training/train_dqn.py`, `src/test/test_dqn_agent.py` |
| PPO baseline (reference) | `src/agents/ppo_agent.py`, `src/training/train_ppo.py`, `src/test/test_ppo_agent.py` |

---

## The Universal Prompt Template

Copy everything below, fill in the `{{placeholders}}`, and paste along with the uploaded files.

---

```
I am working on a group reinforcement learning project in JAX/Gymnax for a graduate course (ELG5214). The project compares exploration mechanisms (Entropy Regularization, RND, ICM) across DQN and PPO under sparse and dense rewards.

I need you to implement **{{AGENT_NAME}}** — which is **{{BASE_ALGORITHM}}** combined with **{{EXPLORATION_MECHANISM}}**.

## My assignment

I am **Student {{LETTER}}** and I need to create these 3 files:
1. `src/agents/{{AGENT_FILENAME}}` — the agent class
2. `src/training/{{TRAINING_FILENAME}}` — the training loop
3. `src/test/{{TEST_FILENAME}}` — the test/run script that orchestrates the sweep

## Critical: follow these shared modules exactly

I have uploaded the shared source files from our project. You MUST:
- Import from these files exactly as they are (do not rewrite them)
- Use `from src.networks.{{NETWORK_MODULE}} import {{NETWORK_IMPORTS}}`
- Use `from src.training.rollout import run_one_episode_scan_simple`
- Use `from src.replay.replay_buffer import init_buffer, add_transition, sample_batch` (DQN only)
- Use `from src.exploration.{{EXPLORATION_MODULE}} import {{EXPLORATION_IMPORTS}}`
- Use `from src.utils.reusable import RLMetricsDataset, setup_logger, Timer`
- Use `from src.evaluate.{{EVAL_MODULE}} import {{EVAL_FUNCTION}}`
- Use `from src.envs.env_utils import get_env`

## Architecture and hyperparameters (from configs/sweep.yaml)

- Network: 2 hidden layers, 64 units, ReLU activation
- Sweep grid: learning_rates=[0.0001, 0.0005, 0.001], gammas=[0.99, 0.95], seeds=[0,1,2,3,4]
- Environments: CartPole-v1 and MountainCar-v0 (both dense and sparse reward variants)
- Max episodes: 2000, eval every 50 episodes, max_steps: 200
{{ALGO_SPECIFIC_HYPERPARAMS}}

## What each file should do

### 1. Agent class (`src/agents/{{AGENT_FILENAME}}`)

Follow the same pattern as the reference agent I uploaded. The agent class should have:
- `__init__(self, params)` — store network parameters
- `act(self, key, obs, {{EXTRA_ACT_ARGS}})` — select action ({{ACTION_STRATEGY}})
- `greedy_action(self, obs)` — deterministic action for evaluation
{{EXTRA_AGENT_METHODS}}

### 2. Training loop (`src/training/{{TRAINING_FILENAME}}`)

Follow the same pattern as the reference training file I uploaded. Must include:
- Episode collection via `lax.scan` (no Python while-loops) — use `run_one_episode_scan_simple` or write a custom `_run_{{ALGO}}_episode_scan` like the DQN reference
- `@jax.jit` decorated update function with `jax.value_and_grad`
- Manual SGD: `jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)`
- {{EXPLORATION_SPECIFIC_TRAINING_DETAILS}}
- Logging via `logger.info(...)` every episode
- `met_df.add_episode(...)` call every episode
- Greedy evaluation every `log_every` episodes using the evaluate function
- Return dict with: final params, episode_rewards, episode_lengths, losses, eval_success_rates

### 3. Test script (`src/test/{{TEST_FILENAME}}`)

Follow the same pattern as `test_dqn_agent.py` or `test_reinforce_agent.py` from the reference. Must include:
- A function `test_{{AGENT_SHORT_NAME}}(seed, gamma, lr, config, config_env_params, met_df)` 
- Create env using `get_env(env_name, reward_type)`
- Initialize network params with the appropriate init function
- Create `setup_logger(run_id, path="./logs/{{LOG_DIR}}")` 
- Call the training function
- Run final greedy evaluation
- Call `met_df.add_summary(...)` with final stats
- Return trained params and eval stats

## Constraints

- Pure JAX only — no PyTorch, no TensorFlow
- Use `jax.tree_util.tree_map` for SGD (no Optax)
- Use `lax.scan` for episode rollouts, `lax.fori_loop` for buffer insertion
- Use `jnp.where` for branchless logic (no Python if/else on JAX arrays)
- All training must be GPU-compatible
- No Python while-loops inside training — use lax.scan
- Functions that touch JAX arrays should be `@jax.jit` decorated where possible

## Output format

Give me the complete code for all 3 files, clearly separated with the filename as a header. Make sure the code is ready to run with no modifications needed.
```

---

## Pre-Filled Prompts Per Student

### Student A — DQN + Entropy Regularization

**Upload:** `reusable.py`, `rollout.py`, `replay_buffer.py`, `q_network.py`, `env_utils.py`, `sweep.yaml`, `entropy_reg.py`, `dqn_agent.py` (reference), `train_dqn.py` (reference), `test_dqn_agent.py` (reference), `evaluate_dqn.py`

```
{{AGENT_NAME}}              = DQN + Entropy Regularization
{{BASE_ALGORITHM}}          = DQN
{{EXPLORATION_MECHANISM}}   = Entropy Regularization
{{LETTER}}                  = A
{{AGENT_FILENAME}}          = dqn_entropy_agent.py
{{TRAINING_FILENAME}}       = train_dqn_entropy.py
{{TEST_FILENAME}}           = test_dqn_entropy_agent.py
{{NETWORK_MODULE}}          = q_network
{{NETWORK_IMPORTS}}         = init_q_params, q_forward, q_forward_batch
{{EXPLORATION_MODULE}}      = entropy_reg
{{EXPLORATION_IMPORTS}}     = entropy_bonus
{{EVAL_MODULE}}             = evaluate_dqn
{{EVAL_FUNCTION}}           = evaluate_dqn_greedy
{{EXTRA_ACT_ARGS}}          = epsilon
{{ACTION_STRATEGY}}         = epsilon-greedy with softmax entropy bonus in loss
{{EXTRA_AGENT_METHODS}}     = (none — entropy is added in the loss, not the agent)
{{ALGO_SPECIFIC_HYPERPARAMS}} = DQN: buffer_capacity=50000, batch_size=64, warmup_steps=100, target_update_freq=500, epsilon: 1.0→0.01 over 1000 episodes, updates_per_episode=4. Entropy coefficient alpha in {0.01, 0.05}.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = Add entropy bonus to DQN loss: total_loss = td_loss - alpha * entropy(q_values). Compute entropy from softmax over Q-values. Log entropy magnitude each episode.
{{AGENT_SHORT_NAME}}        = dqn_entropy
{{LOG_DIR}}                 = dqn_entropy
```

### Student A — DQN + RND

**Upload:** `reusable.py`, `rollout.py`, `replay_buffer.py`, `q_network.py`, `rnd_networks.py`, `rnd.py`, `env_utils.py`, `sweep.yaml`, `dqn_agent.py` (reference), `train_dqn.py` (reference), `test_dqn_agent.py` (reference), `evaluate_dqn.py`

```
{{AGENT_NAME}}              = DQN + RND
{{BASE_ALGORITHM}}          = DQN
{{EXPLORATION_MECHANISM}}   = Random Network Distillation (RND)
{{LETTER}}                  = A
{{AGENT_FILENAME}}          = dqn_rnd_agent.py
{{TRAINING_FILENAME}}       = train_dqn_rnd.py
{{TEST_FILENAME}}           = test_dqn_rnd_agent.py
{{NETWORK_MODULE}}          = q_network
{{NETWORK_IMPORTS}}         = init_q_params, q_forward, q_forward_batch
{{EXPLORATION_MODULE}}      = rnd
{{EXPLORATION_IMPORTS}}     = compute_rnd_reward, update_rnd_predictor
{{EVAL_MODULE}}             = evaluate_dqn
{{EVAL_FUNCTION}}           = evaluate_dqn_greedy
{{EXTRA_ACT_ARGS}}          = epsilon
{{ACTION_STRATEGY}}         = epsilon-greedy (same as baseline DQN)
{{EXTRA_AGENT_METHODS}}     = (none — RND augments reward, not the agent)
{{ALGO_SPECIFIC_HYPERPARAMS}} = DQN: buffer_capacity=50000, batch_size=64, warmup_steps=100, target_update_freq=500, epsilon: 1.0→0.01 over 1000 episodes. RND reward scale beta in {0.1, 1.0}. RND predictor_lr=0.001, hidden_dim=64. RND target network is FIXED (random, never trained).
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = Augment environment reward: r_total = r_extrinsic + beta * r_intrinsic. RND intrinsic reward = MSE between fixed target network and trained predictor network on state. Update predictor network to minimize prediction error. Store augmented reward in replay buffer. Log mean intrinsic reward magnitude each episode.
{{AGENT_SHORT_NAME}}        = dqn_rnd
{{LOG_DIR}}                 = dqn_rnd
```

### Student B — DQN Baseline

**Upload:** `reusable.py`, `rollout.py`, `replay_buffer.py`, `q_network.py`, `env_utils.py`, `sweep.yaml`, `evaluate_dqn.py`

```
{{AGENT_NAME}}              = DQN Baseline
{{BASE_ALGORITHM}}          = DQN
{{EXPLORATION_MECHANISM}}   = None (vanilla epsilon-greedy)
{{LETTER}}                  = B
{{AGENT_FILENAME}}          = dqn_agent.py
{{TRAINING_FILENAME}}       = train_dqn.py
{{TEST_FILENAME}}           = test_dqn_agent.py
{{NETWORK_MODULE}}          = q_network
{{NETWORK_IMPORTS}}         = init_q_params, q_forward, q_forward_batch
{{EXPLORATION_MODULE}}      = (none)
{{EXPLORATION_IMPORTS}}     = (none)
{{EVAL_MODULE}}             = evaluate_dqn
{{EVAL_FUNCTION}}           = evaluate_dqn_greedy
{{EXTRA_ACT_ARGS}}          = epsilon
{{ACTION_STRATEGY}}         = epsilon-greedy: with probability epsilon pick random, else argmax Q
{{EXTRA_AGENT_METHODS}}     = (none)
{{ALGO_SPECIFIC_HYPERPARAMS}} = DQN: buffer_capacity=50000, batch_size=64, warmup_steps=100, target_update_freq=500, epsilon: 1.0→0.01 over 1000 episodes, updates_per_episode=4.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = Standard DQN: collect episode with lax.scan, insert transitions into replay buffer with lax.fori_loop, sample batch, compute TD loss, update Q-network, periodically copy to target network.
{{AGENT_SHORT_NAME}}        = dqn
{{LOG_DIR}}                 = dqn
```

### Student B — DQN + ICM

**Upload:** `reusable.py`, `rollout.py`, `replay_buffer.py`, `q_network.py`, `icm_networks.py`, `icm.py`, `env_utils.py`, `sweep.yaml`, `dqn_agent.py` (reference), `train_dqn.py` (reference), `test_dqn_agent.py` (reference), `evaluate_dqn.py`

```
{{AGENT_NAME}}              = DQN + ICM
{{BASE_ALGORITHM}}          = DQN
{{EXPLORATION_MECHANISM}}   = Intrinsic Curiosity Module (ICM)
{{LETTER}}                  = B
{{AGENT_FILENAME}}          = dqn_icm_agent.py
{{TRAINING_FILENAME}}       = train_dqn_icm.py
{{TEST_FILENAME}}           = test_dqn_icm_agent.py
{{NETWORK_MODULE}}          = q_network
{{NETWORK_IMPORTS}}         = init_q_params, q_forward, q_forward_batch
{{EXPLORATION_MODULE}}      = icm
{{EXPLORATION_IMPORTS}}     = compute_icm_reward, update_icm
{{EVAL_MODULE}}             = evaluate_dqn
{{EVAL_FUNCTION}}           = evaluate_dqn_greedy
{{EXTRA_ACT_ARGS}}          = epsilon
{{ACTION_STRATEGY}}         = epsilon-greedy (same as baseline DQN)
{{EXTRA_AGENT_METHODS}}     = (none — ICM augments reward, not the agent)
{{ALGO_SPECIFIC_HYPERPARAMS}} = DQN: buffer_capacity=50000, batch_size=64, warmup_steps=100, target_update_freq=500, epsilon: 1.0→0.01 over 1000 episodes. ICM curiosity scale eta in {0.1, 1.0}. ICM forward_loss_weight=0.2, inverse_loss_weight=0.8, hidden_dim=64.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = ICM uses forward model (predict next state given state+action) and inverse model (predict action given state+next_state). Intrinsic reward = forward prediction error. Augment reward: r_total = r_extrinsic + eta * r_intrinsic. Update ICM networks alongside Q-network. Store augmented reward in replay buffer. Log mean intrinsic reward magnitude each episode.
{{AGENT_SHORT_NAME}}        = dqn_icm
{{LOG_DIR}}                 = dqn_icm
```

### Student C — PPO + Entropy Regularization

**Upload:** `reusable.py`, `rollout.py`, `policy_network.py`, `value_network.py`, `entropy_reg.py`, `env_utils.py`, `sweep.yaml`, `ppo_agent.py` (reference), `train_ppo.py` (reference), `test_ppo_agent.py` (reference), `evaluate_ppo.py`

```
{{AGENT_NAME}}              = PPO + Entropy Regularization
{{BASE_ALGORITHM}}          = PPO
{{EXPLORATION_MECHANISM}}   = Entropy Regularization
{{LETTER}}                  = C
{{AGENT_FILENAME}}          = ppo_entropy_agent.py
{{TRAINING_FILENAME}}       = train_ppo_entropy.py
{{TEST_FILENAME}}           = test_ppo_entropy_agent.py
{{NETWORK_MODULE}}          = policy_network
{{NETWORK_IMPORTS}}         = init_policy_params, policy_forward, log_prob, entropy
{{EXPLORATION_MODULE}}      = entropy_reg
{{EXPLORATION_IMPORTS}}     = entropy_bonus
{{EVAL_MODULE}}             = evaluate_ppo
{{EVAL_FUNCTION}}           = evaluate_ppo
{{EXTRA_ACT_ARGS}}          = (none — PPO samples from policy)
{{ACTION_STRATEGY}}         = sample from categorical policy: jax.random.categorical(key, logits)
{{EXTRA_AGENT_METHODS}}     = log_prob(obs, action), entropy(obs), get_value(obs) using value network
{{ALGO_SPECIFIC_HYPERPARAMS}} = PPO: n_steps=128, n_epochs=4, mini_batch_size=64, clip_epsilon=0.2, gae_lambda=0.95, gradient_clip_norm=10.0. Entropy coefficient alpha in {0.01, 0.05}.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = PPO clipped surrogate objective with entropy bonus: L = L_clip - alpha * H(pi). Collect n_steps of rollout, compute GAE advantages, run n_epochs of mini-batch updates. The entropy bonus is added directly to the policy loss. Log policy entropy each episode.
{{AGENT_SHORT_NAME}}        = ppo_entropy
{{LOG_DIR}}                 = ppo_entropy
```

### Student C — PPO + RND

**Upload:** `reusable.py`, `rollout.py`, `policy_network.py`, `value_network.py`, `rnd_networks.py`, `rnd.py`, `env_utils.py`, `sweep.yaml`, `ppo_agent.py` (reference), `train_ppo.py` (reference), `test_ppo_agent.py` (reference), `evaluate_ppo.py`

```
{{AGENT_NAME}}              = PPO + RND
{{BASE_ALGORITHM}}          = PPO
{{EXPLORATION_MECHANISM}}   = Random Network Distillation (RND)
{{LETTER}}                  = C
{{AGENT_FILENAME}}          = ppo_rnd_agent.py
{{TRAINING_FILENAME}}       = train_ppo_rnd.py
{{TEST_FILENAME}}           = test_ppo_rnd_agent.py
{{NETWORK_MODULE}}          = policy_network
{{NETWORK_IMPORTS}}         = init_policy_params, policy_forward, log_prob, entropy
{{EXPLORATION_MODULE}}      = rnd
{{EXPLORATION_IMPORTS}}     = compute_rnd_reward, update_rnd_predictor
{{EVAL_MODULE}}             = evaluate_ppo
{{EVAL_FUNCTION}}           = evaluate_ppo
{{EXTRA_ACT_ARGS}}          = (none — PPO samples from policy)
{{ACTION_STRATEGY}}         = sample from categorical policy
{{EXTRA_AGENT_METHODS}}     = log_prob(obs, action), entropy(obs), get_value(obs)
{{ALGO_SPECIFIC_HYPERPARAMS}} = PPO: n_steps=128, n_epochs=4, mini_batch_size=64, clip_epsilon=0.2, gae_lambda=0.95. RND reward scale beta in {0.1, 1.0}. RND predictor_lr=0.001, hidden_dim=64.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = Augment rollout rewards: r_total = r_extrinsic + beta * r_intrinsic. RND intrinsic reward = MSE between fixed target and trained predictor on state. Update predictor alongside policy. Some implementations use separate value heads for extrinsic and intrinsic returns — simpler approach: single value head on combined reward. Log mean intrinsic reward each episode.
{{AGENT_SHORT_NAME}}        = ppo_rnd
{{LOG_DIR}}                 = ppo_rnd
```

### Student D — PPO Baseline

**Upload:** `reusable.py`, `rollout.py`, `policy_network.py`, `value_network.py`, `env_utils.py`, `sweep.yaml`, `evaluate_ppo.py`

```
{{AGENT_NAME}}              = PPO Baseline
{{BASE_ALGORITHM}}          = PPO
{{EXPLORATION_MECHANISM}}   = None (vanilla PPO with clipped surrogate)
{{LETTER}}                  = D
{{AGENT_FILENAME}}          = ppo_agent.py
{{TRAINING_FILENAME}}       = train_ppo.py
{{TEST_FILENAME}}           = test_ppo_agent.py
{{NETWORK_MODULE}}          = policy_network
{{NETWORK_IMPORTS}}         = init_policy_params, policy_forward, log_prob, entropy
{{EXPLORATION_MODULE}}      = (none)
{{EXPLORATION_IMPORTS}}     = (none)
{{EVAL_MODULE}}             = evaluate_ppo
{{EVAL_FUNCTION}}           = evaluate_ppo
{{EXTRA_ACT_ARGS}}          = (none)
{{ACTION_STRATEGY}}         = sample from categorical policy: jax.random.categorical(key, logits)
{{EXTRA_AGENT_METHODS}}     = log_prob(obs, action), entropy(obs), get_value(obs) using value network
{{ALGO_SPECIFIC_HYPERPARAMS}} = PPO: n_steps=128, n_epochs=4, mini_batch_size=64, clip_epsilon=0.2, gae_lambda=0.95, gradient_clip_norm=10.0.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = Standard PPO: collect n_steps rollout, compute GAE advantages with value network, run n_epochs of clipped surrogate updates on mini-batches. Manual SGD via tree_map. Value loss = MSE(V(s) - returns). Total loss = policy_loss + 0.5*value_loss.
{{AGENT_SHORT_NAME}}        = ppo
{{LOG_DIR}}                 = ppo
```

### Student D — PPO + ICM

**Upload:** `reusable.py`, `rollout.py`, `policy_network.py`, `value_network.py`, `icm_networks.py`, `icm.py`, `env_utils.py`, `sweep.yaml`, `ppo_agent.py` (reference), `train_ppo.py` (reference), `test_ppo_agent.py` (reference), `evaluate_ppo.py`

```
{{AGENT_NAME}}              = PPO + ICM
{{BASE_ALGORITHM}}          = PPO
{{EXPLORATION_MECHANISM}}   = Intrinsic Curiosity Module (ICM)
{{LETTER}}                  = D
{{AGENT_FILENAME}}          = ppo_icm_agent.py
{{TRAINING_FILENAME}}       = train_ppo_icm.py
{{TEST_FILENAME}}           = test_ppo_icm_agent.py
{{NETWORK_MODULE}}          = policy_network
{{NETWORK_IMPORTS}}         = init_policy_params, policy_forward, log_prob, entropy
{{EXPLORATION_MODULE}}      = icm
{{EXPLORATION_IMPORTS}}     = compute_icm_reward, update_icm
{{EVAL_MODULE}}             = evaluate_ppo
{{EVAL_FUNCTION}}           = evaluate_ppo
{{EXTRA_ACT_ARGS}}          = (none)
{{ACTION_STRATEGY}}         = sample from categorical policy
{{EXTRA_AGENT_METHODS}}     = log_prob(obs, action), entropy(obs), get_value(obs)
{{ALGO_SPECIFIC_HYPERPARAMS}} = PPO: n_steps=128, n_epochs=4, mini_batch_size=64, clip_epsilon=0.2, gae_lambda=0.95. ICM curiosity scale eta in {0.1, 1.0}. ICM forward_loss_weight=0.2, inverse_loss_weight=0.8, hidden_dim=64.
{{EXPLORATION_SPECIFIC_TRAINING_DETAILS}} = ICM forward model predicts next state embedding given (state, action). Inverse model predicts action given (state, next_state). Intrinsic reward = forward prediction error. Augment rollout rewards: r_total = r_extrinsic + eta * r_intrinsic. Update ICM networks alongside policy. Log mean intrinsic reward each episode.
{{AGENT_SHORT_NAME}}        = ppo_icm
{{LOG_DIR}}                 = ppo_icm
```

---

## Tips for Getting Better AI Output

1. **Always upload the reference files.** The AI cannot follow your project structure without seeing the actual code. The more files you upload, the more consistent the output.

2. **If the AI generates code that imports modules you don't have,** tell it: "Use only imports from the files I uploaded. Do not create new utility modules."

3. **If the output is too long or cuts off,** split the request: first ask for the agent class, then the training loop, then the test script.

4. **After generating code, check these things before committing:**
   - Does it import from `src.*` paths (not relative imports)?
   - Does it call `met_df.add_episode()` every episode?
   - Does it call `met_df.add_summary()` at the end?
   - Does it use `setup_logger(run_id, path="./logs/...")` for logging?
   - Does it use `lax.scan` (not Python while-loops) for rollouts?
   - Does it use `jax.tree_util.tree_map` for SGD (not Optax)?
   - Does the test script loop over seeds, learning rates, and gammas?

5. **If you need to debug,** upload your generated code along with the error traceback and ask: "This code uses our project structure. Fix the error while keeping all imports and function signatures the same."
