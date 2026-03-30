"""
train_ppo_entropy.py — Training loop for PPO + Entropy Regularization.

ALGORITHM OVERVIEW:
    1.  Collect one episode of transitions via lax.scan
    2.  Compute GAE (Generalized Advantage Estimation) advantages
    3.  For n_epochs:
            Shuffle the episode data
            Split into mini-batches of size mini_batch_size
            For each mini-batch:
                Compute clipped surrogate loss  L_clip
                Compute entropy bonus           alpha * H(pi)
                Compute value loss              (V(s) - R)^2
                Total loss = -L_clip - alpha * H(pi) + 0.5 * value_loss
                Update both policy and value params via manual SGD
    4.  Log metrics; repeat for num_episodes

USED BY: src/test/test_ppo_entropy_agent.py
"""

import jax
import jax.numpy as jnp
from jax import lax

from src.networks.policy_network import (
    policy_forward,
    log_prob as policy_log_prob,
    entropy as policy_entropy,
)
from src.networks.value_network import value_forward
from src.training.rollout import run_one_episode_scan_simple
from src.exploration.entropy_reg import entropy_bonus
from src.agents.ppo_entropy_agent import PPOEntropyAgent


# ──────────────────────────────────────────────
#  GAE computation
# ──────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma, gae_lambda, next_value):
    """Compute Generalized Advantage Estimation via reverse lax.scan.

    Args:
        rewards:    (T,)  per-step rewards
        values:     (T,)  V(s_t)
        dones:      (T,)  done flags (True after terminal step)
        gamma:      discount factor
        gae_lambda: GAE lambda
        next_value: V(s_T) bootstrap value for the last state

    Returns:
        advantages: (T,)
        returns:    (T,)  advantage + value (targets for the value network)
    """
    T = rewards.shape[0]

    def _gae_step(carry, t):
        gae, next_val = carry
        # Reverse index: process from last step to first
        idx = T - 1 - t
        done = dones[idx]
        r = rewards[idx]
        v = values[idx]

        not_done = 1.0 - done.astype(jnp.float32)
        delta = r + gamma * next_val * not_done - v
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, v), gae

    init_carry = (jnp.float32(0.0), next_value)
    _, advantages_rev = lax.scan(_gae_step, init_carry, jnp.arange(T))

    # Reverse back to chronological order
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────
#  PPO mini-batch update (policy + value)
# ──────────────────────────────────────────────

def _ppo_loss(policy_params, value_params, obs_b, act_b, old_log_probs_b,
              adv_b, ret_b, clip_epsilon, alpha):
    """Combined PPO clipped-surrogate + value + entropy loss for one mini-batch.

    Returns scalar loss and a dict of component metrics.
    """

    # --- policy loss (clipped surrogate) ---
    def _single_policy(obs, act, old_lp, adv):
        new_lp = policy_log_prob(policy_params, obs, act)
        ratio = jnp.exp(new_lp - old_lp)
        clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = jnp.minimum(ratio * adv, clipped * adv)
        ent = policy_entropy(policy_params, obs)
        return surrogate, ent

    surrogates, entropies = jax.vmap(_single_policy)(obs_b, act_b,
                                                     old_log_probs_b, adv_b)
    policy_loss = -jnp.mean(surrogates)

    # --- entropy bonus (maximise → subtract from loss) ---
    mean_entropy = jnp.mean(entropies)
    ent_bonus = entropy_bonus(mean_entropy, alpha)

    # --- value loss ---
    pred_values = jax.vmap(lambda o: value_forward(value_params, o))(obs_b)
    value_loss = jnp.mean((pred_values - ret_b) ** 2)

    total_loss = policy_loss - ent_bonus + 0.5 * value_loss

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": mean_entropy,
        "total_loss": total_loss,
    }
    return total_loss, metrics


@jax.jit
def _ppo_update_step(policy_params, value_params, obs_b, act_b,
                     old_log_probs_b, adv_b, ret_b,
                     clip_epsilon, alpha, lr):
    """One gradient step on a single mini-batch. Returns updated params + metrics."""

    def _loss_fn(params_tuple):
        pp, vp = params_tuple
        return _ppo_loss(pp, vp, obs_b, act_b, old_log_probs_b,
                         adv_b, ret_b, clip_epsilon, alpha)

    (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        (policy_params, value_params)
    )
    policy_grads, value_grads = grads

    # Manual SGD with gradient clipping
    def clip_and_step(params, grads):
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
        )
        scale = jnp.minimum(1.0, 10.0 / (grad_norm + 1e-8))
        return jax.tree_util.tree_map(
            lambda p, g: p - lr * g * scale, params, grads
        )

    new_policy_params = clip_and_step(policy_params, policy_grads)
    new_value_params = clip_and_step(value_params, value_grads)

    return new_policy_params, new_value_params, metrics


# ──────────────────────────────────────────────
#  Main training function
# ──────────────────────────────────────────────

def train_ppo_entropy(
    env,
    env_params,
    policy_params,
    value_params,
    config,
    logger,
    met_df,
    evaluate_fn,
    seed: int = 0,
):
    """Full PPO + Entropy training run.

    Args:
        env, env_params:  Gymnax environment (dense or sparse)
        policy_params:    initial policy network weights
        value_params:     initial value network weights
        config:           dict with hyperparameters (from sweep.yaml)
        logger:           Python logger from setup_logger
        met_df:           RLMetricsDataset instance
        evaluate_fn:      evaluate_ppo(env, env_params, agent, key, ...) callable
        seed:             base random seed

    Returns:
        dict with: policy_params, value_params, episode_rewards,
                   episode_lengths, losses, eval_success_rates
    """
    # ---- unpack config ----
    num_episodes = config.get("num_episodes", 2000)
    max_steps = config.get("max_steps", 200)
    lr = config.get("lr", 0.0005)
    gamma = config.get("gamma", 0.99)
    log_every = config.get("log_every", 50)
    algorithm = config.get("algorithm", "ppo_entropy")
    env_name = config.get("env_name", "")
    reward_type = config.get("reward_type", "")

    # PPO-specific
    n_epochs = config.get("n_epochs", 4)
    mini_batch_size = config.get("mini_batch_size", 64)
    clip_epsilon = config.get("clip_epsilon", 0.2)
    gae_lambda = config.get("gae_lambda", 0.95)
    alpha = config.get("alpha", 0.01)  # entropy coefficient

    key = jax.random.PRNGKey(seed)

    # ---- tracking lists ----
    episode_rewards = []
    episode_lengths = []
    losses = []
    eval_success_rates = []

    for ep in range(num_episodes):
        key, ep_key, eval_key = jax.random.split(key, 3)

        # ── 1. build agent with current params ──
        agent = PPOEntropyAgent(policy_params, value_params)

        # ── 2. collect one episode via lax.scan ──
        rollout = run_one_episode_scan_simple(
            env, env_params, agent, ep_key, max_steps=max_steps, greedy=False
        )

        obs_all = rollout["observations"]        # (max_steps, obs_dim)
        act_all = rollout["actions"]              # (max_steps,)
        rew_all = rollout["rewards"]              # (max_steps,)
        done_all = rollout["dones"]               # (max_steps,)
        next_obs_all = rollout["next_observations"]
        total_reward = float(rollout["total_reward"])
        ep_len = int(rollout["episode_length"])

        # ── 3. compute old log-probs & values ──
        old_log_probs = jax.vmap(
            lambda o, a: policy_log_prob(policy_params, o, a)
        )(obs_all, act_all)

        values = jax.vmap(
            lambda o: value_forward(value_params, o)
        )(obs_all)

        # Bootstrap value for the last state
        last_obs = next_obs_all[-1]
        next_value = value_forward(value_params, last_obs)
        # Zero out bootstrap if episode ended
        next_value = jnp.where(done_all[-1], 0.0, next_value)

        # ── 4. GAE ──
        advantages, returns = compute_gae(
            rew_all, values, done_all, gamma, gae_lambda, next_value
        )
        # Normalise advantages
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── 5. PPO epochs over mini-batches ──
        n_samples = max_steps
        ep_loss = 0.0
        ep_entropy = 0.0
        n_updates = 0

        for _epoch in range(n_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)

            # Number of full mini-batches
            n_batches = n_samples // mini_batch_size

            for b in range(n_batches):
                idx = perm[b * mini_batch_size : (b + 1) * mini_batch_size]
                obs_b = obs_all[idx]
                act_b = act_all[idx]
                old_lp_b = old_log_probs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]

                policy_params, value_params, metrics = _ppo_update_step(
                    policy_params, value_params,
                    obs_b, act_b, old_lp_b, adv_b, ret_b,
                    clip_epsilon, alpha, lr,
                )
                ep_loss += float(metrics["total_loss"])
                ep_entropy += float(metrics["entropy"])
                n_updates += 1

        avg_loss = ep_loss / max(n_updates, 1)
        avg_entropy = ep_entropy / max(n_updates, 1)

        episode_rewards.append(total_reward)
        episode_lengths.append(ep_len)
        losses.append(avg_loss)

        # ── 6. greedy evaluation ──
        eval_sr = -1.0
        if (ep + 1) % log_every == 0 or ep == 0:
            agent_eval = PPOEntropyAgent(policy_params, value_params)
            eval_sr = evaluate_fn(env, env_params, agent_eval, eval_key,
                                  num_episodes=config.get("eval_episodes", 100),
                                  max_steps=max_steps)
            eval_success_rates.append(eval_sr)

            logger.info(
                f"Episode {ep+1}/{num_episodes} | "
                f"reward={total_reward:.1f} | len={ep_len} | "
                f"loss={avg_loss:.4f} | entropy={avg_entropy:.4f} | "
                f"eval_sr={eval_sr:.3f}"
            )

        # ── 7. metrics logging ──
        met_df.add_episode(
            seed=seed,
            episode=ep + 1,
            reward=total_reward,
            episode_length=ep_len,
            algorithm=algorithm,
            lr=lr,
            gamma=gamma,
            loss=avg_loss,
            eval_success_rate=eval_sr,
            intrinsic_reward=0.0,
            policy_entropy=avg_entropy,
            env_name=env_name,
            reward_type=reward_type,
        )

    return {
        "policy_params": policy_params,
        "value_params": value_params,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "eval_success_rates": eval_success_rates,
    }
