"""
train_dqn_rnd.py — DQN + Random Network Distillation training.

WHAT CHANGES VS BASELINE DQN (train_dqn.py):

    1. RND networks are initialized alongside Q-network
    2. After collecting an episode, compute intrinsic reward for each state:
           r_intrinsic = ||target(s) - predictor(s)||^2
    3. Augment the extrinsic reward before inserting into replay buffer:
           r_total = r_extrinsic + beta * r_intrinsic
    4. After Q-network update, also update RND predictor on the same batch
    5. Log mean intrinsic reward each episode

    The Q-network loss itself is unchanged — it still minimizes TD error.
    But the rewards it trains on now include a novelty bonus, which
    pushes the Q-values to favor visiting novel states.

    beta controls the balance:
        beta=0.1 → mild curiosity, mostly follows extrinsic reward
        beta=1.0 → strong curiosity, heavily explores novel states

Student A (Mohammad)
"""

from typing import Dict, Any, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from src.networks.q_network import q_forward, q_forward_batch
from src.networks.rnd_networks import init_rnd_params
from src.replay.replay_buffer import init_buffer, sample_batch, add_transitions_batch
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy
from src.exploration.rnd import compute_rnd_reward_batch, update_rnd_predictor


# ──────────────────────────────────────────────
#  Epsilon schedule
# ──────────────────────────────────────────────

def linear_epsilon_decay(episode, epsilon_start, epsilon_end, decay_episodes):
    if episode >= decay_episodes:
        return epsilon_end
    frac = episode / decay_episodes
    return epsilon_start + frac * (epsilon_end - epsilon_start)


# ──────────────────────────────────────────────
#  Standard DQN loss (unchanged — rewards are already augmented)
# ──────────────────────────────────────────────

def dqn_loss(q_params, target_params, batch, gamma=0.99):
    q_values = q_forward_batch(q_params, batch["obs"])
    q_sa = jnp.take_along_axis(
        q_values, batch["actions"][:, None], axis=1
    ).squeeze(1)

    next_q = q_forward_batch(target_params, batch["next_obs"])
    max_next_q = jnp.max(next_q, axis=1)

    dones = batch["dones"].astype(jnp.float32)
    targets = batch["rewards"] + gamma * max_next_q * (1.0 - dones)

    return jnp.mean((q_sa - targets) ** 2)


@partial(jax.jit, static_argnames=("gamma",))
def update_q_network(q_params, target_params, batch, learning_rate, gamma=0.99):
    loss_fn = lambda p: dqn_loss(p, target_params, batch, gamma)
    loss, grads = jax.value_and_grad(loss_fn)(q_params)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, q_params, grads
    )
    return new_params, loss


# ──────────────────────────────────────────────
#  lax.scan episode collection (same as baseline)
# ──────────────────────────────────────────────

def _run_dqn_episode_scan(env, env_params, q_params, key, epsilon, max_steps, num_actions):
    key, reset_key = jax.random.split(key)
    obs0, state0 = env.reset_env(reset_key, env_params)

    def step_fn(carry, _):
        key, obs, state, done = carry
        key, act_key, explore_key, step_key = jax.random.split(key, 4)

        q_values = q_forward(q_params, obs)
        greedy_act = jnp.argmax(q_values)
        random_act = jax.random.randint(act_key, shape=(), minval=0, maxval=num_actions)
        action = jnp.where(jax.random.uniform(explore_key) < epsilon, random_act, greedy_act)

        next_obs, next_state, reward, next_done, _ = env.step_env(step_key, state, action, env_params)

        reward = jnp.where(done, 0.0, reward)
        next_done = jnp.logical_or(done, next_done)
        safe_next_obs = jnp.where(done, obs, next_obs)
        safe_state = jax.tree_util.tree_map(
            lambda o, n: jnp.where(done, o, n), state, next_state
        )

        carry = (key, safe_next_obs, safe_state, next_done)
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": safe_next_obs,
            "done": next_done,
        }
        return carry, transition

    init_carry = (key, obs0, state0, jnp.array(False))
    _, transitions = lax.scan(step_fn, init_carry, xs=None, length=max_steps)

    done_cumsum = jnp.cumsum(transitions["done"].astype(jnp.int32))
    valid_mask = done_cumsum <= 1
    episode_length = jnp.sum(valid_mask)
    total_reward = jnp.sum(transitions["reward"])

    return {
        "obs": transitions["obs"],
        "actions": transitions["action"],
        "rewards": transitions["reward"],
        "next_obs": transitions["next_obs"],
        "dones": transitions["done"],
        "total_reward": total_reward,
        "episode_length": episode_length,
        "valid_mask": valid_mask,
    }


# ──────────────────────────────────────────────
#  Augment rewards with RND intrinsic bonus
# ──────────────────────────────────────────────

def _augment_rewards_rnd(rollout, rnd_params, beta):
    """Add RND intrinsic reward to extrinsic rewards.

    r_total[t] = r_extrinsic[t] + beta * ||target(s_t) - predictor(s_t)||^2

    Done BEFORE inserting into the replay buffer, so the Q-network
    trains on augmented rewards directly.
    """
    intrinsic_rewards = compute_rnd_reward_batch(rnd_params, rollout["obs"])
    augmented_rewards = rollout["rewards"] + beta * intrinsic_rewards
    mean_intrinsic = jnp.mean(intrinsic_rewards)

    augmented_rollout = {**rollout, "rewards": augmented_rewards}
    return augmented_rollout, mean_intrinsic


# ──────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────

def train_dqn_rnd(
    env,
    env_params,
    init_q_params: Dict,
    num_episodes: int = 2000,
    max_steps: int = 200,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 0,
    buffer_capacity: int = 50000,
    batch_size: int = 64,
    warmup_steps: int = 100,
    target_update_freq: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 1000,
    updates_per_episode: int = 4,
    log_every: int = 50,
    beta: float = 0.1,
    predictor_lr: float = 0.001,
    obs_dim: int = 4,
    num_actions: int = 2,
    env_name: str = "",
    reward_type: str = "",
    logger: Any = None,
    met_df: Any = None,
) -> Dict[str, Any]:

    key = jax.random.PRNGKey(seed)
    q_params = init_q_params
    target_params = init_q_params

    # Initialize RND networks
    key, rnd_key = jax.random.split(key)
    rnd_params = init_rnd_params(rnd_key, obs_dim=obs_dim)

    buffer = init_buffer(buffer_capacity, obs_dim)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[float] = []
    intrinsic_rewards: List[float] = []
    eval_success_rates: List[float] = []

    global_step = 0

    for episode in range(1, num_episodes + 1):
        epsilon = linear_epsilon_decay(
            episode, epsilon_start, epsilon_end, epsilon_decay_episodes
        )

        key, ep_key = jax.random.split(key)

        # 1. Collect episode on GPU
        rollout = _run_dqn_episode_scan(
            env, env_params, q_params, ep_key, epsilon, max_steps, num_actions
        )

        # 2. Augment rewards with RND intrinsic bonus
        rollout, mean_intrinsic = _augment_rewards_rnd(rollout, rnd_params, beta)

        ep_length = int(rollout["episode_length"])
        n_valid = ep_length

        # 3. Insert augmented transitions into buffer
        buffer = add_transitions_batch(buffer, rollout, n_valid)
        global_step += n_valid

        # 4. Q-network updates + RND predictor updates
        total_loss = jnp.array(0.0)
        n_updates = 0

        if int(buffer["size"]) >= max(warmup_steps, batch_size):
            n_updates = updates_per_episode
            for _ in range(n_updates):
                key, sample_key = jax.random.split(key)
                batch = sample_batch(buffer, sample_key, batch_size)

                # Update Q-network
                q_params, loss = update_q_network(
                    q_params=q_params,
                    target_params=target_params,
                    batch=batch,
                    learning_rate=learning_rate,
                    gamma=gamma,
                )
                total_loss = total_loss + loss

                # Update RND predictor on the same batch of states
                rnd_params, _ = update_rnd_predictor(
                    rnd_params, batch["obs"], predictor_lr
                )

            if global_step % target_update_freq < n_valid:
                target_params = q_params

        # 5. Logging
        ep_reward = float(rollout["total_reward"])
        avg_loss = float(total_loss / max(n_updates, 1))
        mean_intr = float(mean_intrinsic)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        losses.append(avg_loss)
        intrinsic_rewards.append(mean_intr)

        logger.info(
            f"Episode {episode:4d} - Reward: {ep_reward:.3f}, "
            f"Length: {ep_length}, Loss: {avg_loss:.4f}, "
            f"Intrinsic: {mean_intr:.4f}, Epsilon: {epsilon:.3f}"
        )
        met_df.add_episode(
            seed=seed, episode=episode, reward=ep_reward,
            episode_length=ep_length, loss=avg_loss,
            algorithm="DQN_RND", lr=learning_rate, gamma=gamma,
            intrinsic_reward=mean_intr,
            env_name=env_name, reward_type=reward_type,
        )

        if episode % log_every == 0:
            avg_r = sum(episode_rewards[-log_every:]) / log_every
            avg_l = sum(episode_lengths[-log_every:]) / log_every
            avg_lo = sum(losses[-log_every:]) / log_every

            eval_stats = evaluate_dqn_greedy(
                env=env, env_params=env_params,
                q_params=q_params, num_episodes=25,
                max_steps=max_steps, seed=seed + episode,
            )
            eval_success_rates.append(eval_stats["success_rate"])

            msg = (
                f"[Episode {episode:4d}] "
                f"eps={epsilon:.3f}  "
                f"avg_reward={avg_r:8.3f}  "
                f"avg_length={avg_l:6.2f}  "
                f"avg_loss={avg_lo:8.4f}  "
                f"avg_intrinsic={mean_intr:.4f}  "
                f"eval_success={eval_stats['success_rate']:.3f}"
            )
            print(msg)
            logger.info(msg)

            # Early stopping: if eval success is 1.0 for 3 consecutive evals, stop
            if (len(eval_success_rates) >= 3 and
                    eval_success_rates[-1] >= 0.99 and
                    eval_success_rates[-2] >= 0.99 and
                    eval_success_rates[-3] >= 0.99):
                logger.info(f"Early stopping at episode {episode} — "
                            f"eval_success >= 0.99 for 3 consecutive evals")
                print(f"  ** Early stop at episode {episode} **")
                break

    return {
        "final_q_params": q_params,
        "target_q_params": target_params,
        "rnd_params": rnd_params,
        "episode_rewards": jnp.array(episode_rewards, dtype=jnp.float32),
        "episode_lengths": jnp.array(episode_lengths, dtype=jnp.int32),
        "losses": jnp.array(losses, dtype=jnp.float32),
        "intrinsic_rewards": jnp.array(intrinsic_rewards, dtype=jnp.float32),
        "eval_success_rates": jnp.array(eval_success_rates, dtype=jnp.float32),
    }
