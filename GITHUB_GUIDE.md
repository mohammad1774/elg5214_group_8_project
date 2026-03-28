# GitHub Collaboration Guide — ELG5214 Group 8

> **Repo:** `https://github.com/mohammad1774/elg5214_group_8_project`
> **Deadline:** March 26, 2026
> **Team:** Student A (Mohammad), Student B (Anthony), Student C (Mosarraf), Student D (Mariana)

---

## Part 1: One-Time Setup (Everyone Does This)

### 1.1 Install Git

```bash
# Ubuntu / WSL
sudo apt update && sudo apt install git -y

# macOS
brew install git

# Verify
git --version
```

### 1.2 Configure Your Identity

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your_uottawa_email@uottawa.ca"
```

### 1.3 Clone the Repo

```bash
# Using HTTPS (simplest — no SSH key needed)
git clone https://github.com/mohammad1774/elg5214_group_8_project.git
cd elg5214_group_8_project

# OR using SSH (if you have SSH keys configured on GitHub)
git clone git@github.com:mohammad1774/elg5214_group_8_project.git
cd elg5214_group_8_project
```

### 1.4 Verify You're on `main`

```bash
git branch          # Should show: * main
git status          # Should show: On branch main, nothing to commit
```

### 1.5 Install Dependencies

```bash
pip install -r requirements.txt

# OR with conda
conda env create -f environment.yml
conda activate elg5214
```

---

## Part 2: Branching Strategy

We use a **feature-branch workflow**. Nobody pushes directly to `main`. Every piece of work goes through a branch → pull request → review → merge.

### Branch Naming Convention

```
<your-letter>/<short-description>
```

Examples:

```
A/dqn-entropy            ← Student A working on DQN + entropy agent
A/dqn-rnd                ← Student A working on DQN + RND agent
B/dqn-baseline           ← Student B working on DQN baseline
B/dqn-icm                ← Student B working on DQN + ICM
C/ppo-entropy            ← Student C working on PPO + entropy
C/ppo-rnd                ← Student C working on PPO + RND
D/ppo-baseline           ← Student D working on PPO baseline
D/ppo-icm                ← Student D working on PPO + ICM
A/shared-envs            ← Student A setting up shared environment wrappers
A/visualizations         ← Student A doing plots and analysis
BCD/report               ← Shared branch for report writing
```

### Creating Your Branch

```bash
# 1. Make sure you're on main and up to date
git checkout main
git pull origin main

# 2. Create your branch
git checkout -b A/dqn-entropy

# 3. Verify
git branch    # Should show: * A/dqn-entropy
```

### Switching Between Branches

```bash
# Save your current work first
git add .
git commit -m "WIP: dqn entropy agent"

# Switch to another branch
git checkout main
git pull origin main
git checkout -b A/dqn-rnd    # new branch from updated main

# OR switch to an existing branch
git checkout A/dqn-entropy   # go back to your other branch
```

---

## Part 3: Daily Workflow

### The Golden Rule

**Always pull before you push.** This prevents merge conflicts.

### Step-by-Step: Working on Your Feature

```bash
# 1. Start your day — get latest main
git checkout main
git pull origin main

# 2. Switch to your branch (or create a new one)
git checkout A/dqn-entropy
git merge main                # Bring latest main changes into your branch

# 3. Do your work — edit files, run training, etc.
#    ... edit src/agents/dqn_entropy_agent.py ...
#    ... edit src/training/train_dqn_entropy.py ...
#    ... edit src/test/test_dqn_entropy_agent.py ...

# 4. Check what changed
git status                    # Shows modified/new files
git diff                      # Shows line-by-line changes

# 5. Stage your changes
git add src/agents/dqn_entropy_agent.py
git add src/training/train_dqn_entropy.py
git add src/test/test_dqn_entropy_agent.py

# OR stage everything at once
git add .

# 6. Commit with a clear message
git commit -m "feat: implement DQN + entropy regularization agent

- Added entropy bonus to DQN loss function (alpha=0.01)
- Training loop in train_dqn_entropy.py with lax.scan rollout
- Test script with sweep over 5 seeds x 3 LR x 2 gamma
- Tested on CartPole dense, reaches 450+ avg return"

# 7. Push to GitHub
git push origin A/dqn-entropy
```

### Commit Message Format

Use clear, descriptive messages:

```
feat: implement PPO + RND intrinsic reward
fix: replay buffer overflow when capacity < batch_size
docs: update README with PPO training instructions
refactor: extract shared MLP into networks/q_network.py
data: add CartPole sparse sweep results (5 seeds, lr=0.001)
```

---

## Part 4: What to Commit vs What NOT to Commit

### DO Commit (Track in Git)

```
src/                     ← All source code
configs/                 ← YAML config files
scripts/                 ← Sweep and analysis scripts
requirements.txt         ← Dependencies
environment.yml          ← Conda env
setup_structure.sh       ← Project setup
README.md                ← Documentation
report/                  ← Report source files
presentation/            ← Slide source files
visualizations/          ← Final generated plots (PNGs)
.gitignore               ← Ignore rules
```

### DO NOT Commit (Already in .gitignore)

```
__pycache__/             ← Python bytecode (auto-generated)
*.pyc                    ← Compiled Python files
logs/                    ← Training log files (large, per-run)
metrics/                 ← RLMetricsDataset CSVs (regenerated by running training)
results/*/seed_*         ← Raw per-seed result files (large)
checkpoints/             ← Saved model parameters (can be huge)
*.npy, *.npz             ← NumPy array dumps
venv/                    ← Virtual environment
.env                     ← Environment variables
```

### Why These Are Excluded

| Category | Why Excluded | How to Reproduce |
|----------|-------------|------------------|
| `logs/` | Per-run log files, hundreds of MB over a sweep | Re-run training — logs regenerate automatically |
| `metrics/` | CSV outputs from `RLMetricsDataset.save()` | Re-run training or `aggregate_results.py` |
| `checkpoints/` | Model parameter files, can be 100s of MB | Re-run training — or share via Google Drive if needed |
| `results/*/seed_*` | Raw per-seed CSVs from individual runs | Re-run the sweep |
| `__pycache__/` | Python auto-generates these, they cause conflicts | Python recreates them on import |

### Exception: Final Summary Files

You CAN commit aggregated summary files that others need:

```bash
# These are small and useful — OK to commit
git add results/summary.csv
git add results/significance_tests.csv
git add visualizations/learning_curves/*.png
git add visualizations/heatmaps/*.png
```

---

## Part 5: Creating a Pull Request (PR)

A pull request is how your code gets reviewed and merged into `main`.

### Step 1: Push Your Branch

```bash
git push origin A/dqn-entropy
```

### Step 2: Open PR on GitHub

1. Go to `https://github.com/mohammad1774/elg5214_group_8_project`
2. You'll see a yellow banner: **"A/dqn-entropy had recent pushes — Compare & pull request"**
3. Click **"Compare & pull request"**

### Step 3: Fill in the PR

**Title:** `feat: DQN + entropy regularization (Student A)`

**Description template:**

```markdown
## What this PR does
- Implements DQN + entropy regularization agent
- Adds entropy bonus (α coefficient) to the DQN loss
- Includes training loop, agent class, and test script

## Files changed
- `src/agents/dqn_entropy_agent.py` — Agent class with entropy-augmented action selection
- `src/training/train_dqn_entropy.py` — Training loop with entropy loss term
- `src/test/test_dqn_entropy_agent.py` — Test/run script for sweep
- `src/exploration/entropy_reg.py` — Entropy bonus computation

## Testing
- Ran on CartPole dense: 5 seeds × lr=0.001 × γ=0.99
- Mean eval reward: 480 ± 12
- Success rate: 96%

## Review checklist
- [ ] Code runs without errors
- [ ] Uses shared src/ modules (reusable.py, rollout.py, replay_buffer.py)
- [ ] CSV output format matches convention: lr{LR}_g{GAMMA}_seed{SEED}.csv
- [ ] No hardcoded paths — uses relative paths from project root
- [ ] Commit messages are descriptive
```

### Step 4: Request a Reviewer

- **Student A's PRs** → Request review from **Student B**
- **Student B's PRs** → Request review from **Student A**
- **Student C's PRs** → Request review from **Student D**
- **Student D's PRs** → Request review from **Student C**

Click the gear icon next to "Reviewers" on the right sidebar and select your reviewer.

### Step 5: Reviewer Reviews the Code

As a reviewer:

1. Go to the PR → click **"Files changed"** tab
2. Read through the code changes
3. Click the `+` icon on any line to leave a comment
4. When done, click **"Review changes"** at top-right:
   - **Comment** — general feedback, no approval
   - **Approve** — looks good, ready to merge
   - **Request changes** — needs fixes before merging

### Step 6: Merge the PR

Once approved:

1. Click **"Merge pull request"** (use "Squash and merge" for cleaner history)
2. Click **"Confirm merge"**
3. Optionally click **"Delete branch"** to clean up

### Step 7: Everyone Updates Their Local Main

After a PR is merged, everyone should:

```bash
git checkout main
git pull origin main
```

Then merge the new main into any active branches:

```bash
git checkout C/ppo-entropy
git merge main
# Resolve any conflicts if they appear (see Part 7)
```

---

## Part 6: Recommended PR Order

Since some code depends on shared modules, merge PRs in this order:

```
Week 1 (Mar 21-22): Foundation PRs — merge first
──────────────────────────────────────────────────
PR 1: A/shared-foundation
  → src/envs/*, src/networks/q_network.py, src/networks/policy_network.py,
    src/networks/value_network.py, src/replay/replay_buffer.py,
    src/training/rollout.py, src/utils/reusable.py, configs/sweep.yaml

PR 2: A/exploration-modules
  → src/exploration/entropy_reg.py, src/exploration/rnd.py, src/exploration/icm.py,
    src/networks/rnd_networks.py, src/networks/icm_networks.py

Week 1 (Mar 22-24): Agent PRs — can merge in parallel after foundation
──────────────────────────────────────────────────────────────────────
PR 3: B/dqn-baseline       → dqn_agent.py, train_dqn.py, test_dqn_agent.py
PR 4: A/dqn-entropy        → dqn_entropy_agent.py, train_dqn_entropy.py, test_...
PR 5: A/dqn-rnd            → dqn_rnd_agent.py, train_dqn_rnd.py, test_...
PR 6: B/dqn-icm            → dqn_icm_agent.py, train_dqn_icm.py, test_...
PR 7: D/ppo-baseline       → ppo_agent.py, train_ppo.py, test_ppo_agent.py
PR 8: C/ppo-entropy        → ppo_entropy_agent.py, train_ppo_entropy.py, test_...
PR 9: C/ppo-rnd            → ppo_rnd_agent.py, train_ppo_rnd.py, test_...
PR 10: D/ppo-icm           → ppo_icm_agent.py, train_ppo_icm.py, test_...

Week 2 (Mar 24-26): Analysis + Report PRs
──────────────────────────────────────────
PR 11: A/visualizations    → scripts/*, visualizations/*
PR 12: BCD/report          → report/*, presentation/*
```

---

## Part 7: Handling Merge Conflicts

Conflicts happen when two people edit the same lines. Git marks them like this:

```python
<<<<<<< HEAD
# Your version
learning_rate = 0.001
=======
# The other person's version
learning_rate = 0.0005
>>>>>>> main
```

### How to Resolve

```bash
# 1. Git tells you which files have conflicts
git merge main
# CONFLICT: src/utils/reusable.py

# 2. Open the file, find the <<<< ==== >>>> markers

# 3. Edit to keep what's correct (delete the markers)
learning_rate = 0.001  # keep your version, or combine both

# 4. Stage the resolved file
git add src/utils/reusable.py

# 5. Complete the merge
git commit -m "resolve merge conflict in reusable.py"
```

### How to AVOID Conflicts

- Each student works on **different files** (the folder structure is designed for this)
- Don't edit shared files (`reusable.py`, `rollout.py`) without telling the team
- Pull and merge `main` into your branch **at least once a day**
- Keep PRs small and focused — one feature per PR

---

## Part 8: Quick Reference Card

```bash
# ─── Start of day ─────────────────────────────
git checkout main && git pull origin main
git checkout A/my-feature && git merge main

# ─── During work ──────────────────────────────
git add .
git commit -m "feat: description of what I did"

# ─── End of day ───────────────────────────────
git push origin A/my-feature
# → Go to GitHub → Create PR if feature is done

# ─── After a PR is merged ────────────────────
git checkout main && git pull origin main
git branch -d A/old-feature    # delete local branch

# ─── Emergency: undo last commit ──────────────
git reset --soft HEAD~1        # undo commit, keep changes staged

# ─── Emergency: discard all local changes ─────
git checkout -- .              # reset all files to last commit
```

---

## Part 9: Team Communication Checklist

Before starting your branch:
- [ ] `git pull origin main` to get latest code
- [ ] Check if shared modules you depend on are merged yet

Before creating a PR:
- [ ] Code runs without errors on your machine
- [ ] `RLMetricsDataset` logs are saving correctly to `metrics/`
- [ ] Logger output goes to `logs/<your_agent>/`
- [ ] No absolute paths in your code (use relative from project root)
- [ ] Added yourself as author in the files you created

Before approving a PR (as reviewer):
- [ ] Code uses shared imports: `from src.networks.q_network import ...`
- [ ] Training loop follows reference pattern (lax.scan rollout → update → log)
- [ ] `met_df.add_episode()` and `met_df.add_summary()` calls are present
- [ ] No accidentally committed `__pycache__/`, `logs/`, or `checkpoints/`
