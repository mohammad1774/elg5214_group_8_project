"""
train_ppo_rnd.py — Training loop for PPO + Random Network Distillation.

ALGORITHM OVERVIEW:
    1.  Collect one episode of transitions via lax.scan
    2.  Compute RND intrinsic reward for every observation:
            r_intrinsic = MSE(target(s) - predictor(s))
    3.  Augment rewards:  r_total = r_extrinsic + beta * r_intrinsic
    4.  Compute GAE advantages on augmented rewards
    5.  For n_epochs:
            For each mini-batch:
                - PPO clipped surrogate loss
                - Value loss
                - Update policy + value networks
    6.  Update RND predictor on the visited states (MSE vs target)
    7.  Log metrics; repeat for num_episodes
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
from src.exploration.rnd import (
    compute_rnd_reward_batch,
    update_rnd_predictor,
)
from src.agents.ppo_rnd_agent import PPORNDAgent


# ──────────────────────────────────────────────
#  GAE (reverse scan)
# ──────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma, gae_lambda, next_value):
    """Generalized Advantage Estimation via reverse lax.scan."""
    T = rewards.shape[0]

    def _step(carry, t):
        gae, _ = carry
        idx = T - 1 - t
        not_done = 1.0 - dones[idx].astype(jnp.float32)
        delta = rewards[idx] + gamma * carry[1] * not_done - values[idx]
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, values[idx]), gae

    init = (jnp.float32(0.0), next_value)
    _, adv_rev = lax.scan(_step, init, jnp.arange(T))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────
#  PPO loss (clipped surrogate + value)
# ──────────────────────────────────────────────

def _ppo_loss(policy_params, value_params, obs_b, act_b, old_lp_b,
              adv_b, ret_b, clip_epsilon):
    """Clipped surrogate + value loss for one mini-batch."""

    def _single(obs, act, old_lp, adv):
        new_lp = policy_log_prob(policy_params, obs, act)
        ratio = jnp.exp(new_lp - old_lp)
        clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = jnp.minimum(ratio * adv, clipped * adv)
        ent = policy_entropy(policy_params, obs)
        return surrogate, ent

    surrogates, entropies = jax.vmap(_single)(obs_b, act_b, old_lp_b, adv_b)
    policy_loss = -jnp.mean(surrogates)
    mean_entropy = jnp.mean(entropies)

    pred_values = jax.vmap(lambda o: value_forward(value_params, o))(obs_b)
    value_loss = jnp.mean((pred_values - ret_b) ** 2)

    total_loss = policy_loss + 0.5 * value_loss

    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": mean_entropy,
        "total_loss": total_loss,
    }


@jax.jit
def _ppo_update_step(policy_params, value_params, obs_b, act_b,
                     old_lp_b, adv_b, ret_b, clip_epsilon, lr):
    """One gradient step on policy + value (RND predictor is updated separately)."""

    def _loss_fn(params_tuple):
        pp, vp = params_tuple
        return _ppo_loss(pp, vp, obs_b, act_b, old_lp_b,
                         adv_b, ret_b, clip_epsilon)

    (_, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        (policy_params, value_params)
    )
    policy_grads, value_grads = grads

    def clip_and_step(params, grads):
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
        )
        scale = jnp.minimum(1.0, 10.0 / (grad_norm + 1e-8))
        return jax.tree_util.tree_map(
            lambda p, g: p - lr * g * scale, params, grads
        )

    new_policy = clip_and_step(policy_params, policy_grads)
    new_value = clip_and_step(value_params, value_grads)
    return new_policy, new_value, metrics


# ──────────────────────────────────────────────
#  Main training function
# ──────────────────────────────────────────────

def train_ppo_rnd(
    env,
    env_params,
    policy_params,
    value_params,
    rnd_params,
    config,
    logger,
    met_df,
    evaluate_fn,
    seed: int = 0,
    num_actions: int = 2,
):
    """Full PPO + RND training run.

    Args:
        env, env_params:  Gymnax environment
        policy_params:    policy network weights
        value_params:     value network weights
        rnd_params:       dict with "target" (fixed) and "predictor" (trained)
        config:           hyperparameters
        logger:           setup_logger output
        met_df:           RLMetricsDataset
        evaluate_fn:      evaluate_ppo(env, env_params, policy_params, ...)
        seed:             base random seed
        num_actions:      action space size

    Returns:
        Dict with policy_params, value_params, rnd_params, episode_rewards,
              episode_lengths, losses, eval_success_rates, intrinsic_rewards
    """
    # ---- unpack ----
    num_episodes = config.get("num_episodes", 2000)
    max_steps = config.get("max_steps", 200)
    lr = config.get("lr", 0.0005)
    gamma = config.get("gamma", 0.99)
    log_every = config.get("log_every", 50)
    algorithm = config.get("algorithm", "ppo_rnd")
    env_name = config.get("env_name", "")
    reward_type = config.get("reward_type", "")

    # PPO-specific
    n_epochs = config.get("n_epochs", 4)
    mini_batch_size = config.get("mini_batch_size", 64)
    clip_epsilon = config.get("clip_epsilon", 0.2)
    gae_lambda = config.get("gae_lambda", 0.95)

    # RND-specific
    beta = config.get("beta", 0.1)              # intrinsic reward scale
    predictor_lr = config.get("predictor_lr", 0.001)

    key = jax.random.PRNGKey(seed)

    # ---- tracking ----
    episode_rewards = []
    episode_lengths = []
    losses = []
    intrinsic_rewards_log = []
    eval_success_rates = []

    for ep in range(num_episodes):
        key, ep_key, eval_key = jax.random.split(key, 3)

        # ── 1. rollout ──
        agent = PPORNDAgent(policy_params, value_params, rnd_params)
        rollout = run_one_episode_scan_simple(
            env, env_params, agent, ep_key, max_steps=max_steps, greedy=False
        )

        obs_all = rollout["observations"]
        act_all = rollout["actions"]
        rew_ext = rollout["rewards"]
        done_all = rollout["dones"]
        next_obs_all = rollout["next_observations"]
        total_ext_reward = float(rollout["total_reward"])
        ep_len = int(rollout["episode_length"])

        # ── 2. intrinsic rewards ──
        intrinsic = compute_rnd_reward_batch(rnd_params, obs_all)
        mean_intrinsic = float(jnp.mean(intrinsic))
        intrinsic_rewards_log.append(mean_intrinsic)

        # Augmented reward: r_total = r_ext + beta * r_intrinsic
        rew_augmented = rew_ext + beta * intrinsic

        # ── 3. old log-probs and values ──
        old_log_probs = jax.vmap(
            lambda o, a: policy_log_prob(policy_params, o, a)
        )(obs_all, act_all)

        values = jax.vmap(lambda o: value_forward(value_params, o))(obs_all)

        last_obs = next_obs_all[-1]
        next_value = value_forward(value_params, last_obs)
        next_value = jnp.where(done_all[-1], 0.0, next_value)

        # ── 4. GAE on augmented rewards ──
        advantages, returns = compute_gae(
            rew_augmented, values, done_all, gamma, gae_lambda, next_value
        )
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # ── 5. PPO epochs over mini-batches ──
        n_samples = max_steps
        ep_loss = 0.0
        n_updates = 0

        for _epoch in range(n_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)
            n_batches = n_samples // mini_batch_size

            for b in range(n_batches):
                idx = perm[b * mini_batch_size:(b + 1) * mini_batch_size]
                policy_params, value_params, metrics = _ppo_update_step(
                    policy_params, value_params,
                    obs_all[idx], act_all[idx], old_log_probs[idx],
                    advantages[idx], returns[idx],
                    clip_epsilon, lr,
                )
                ep_loss += float(metrics["total_loss"])
                n_updates += 1

        avg_loss = ep_loss / max(n_updates, 1)

        # ── 6. RND predictor update (separate from policy/value) ──
        rnd_params, _rnd_loss = update_rnd_predictor(
            rnd_params, obs_all, predictor_lr
        )

        episode_rewards.append(total_ext_reward)
        episode_lengths.append(ep_len)
        losses.append(avg_loss)

        # ── 7. greedy eval ──
        eval_sr = -1.0
        if (ep + 1) % log_every == 0 or ep == 0:
            eval_res = evaluate_fn(
                env, env_params, policy_params,
                num_episodes=config.get("eval_episodes", 50),
                max_steps=max_steps,
                seed=seed + ep + 1000,
                greedy=True,
                num_actions=num_actions,
            )
            eval_sr = float(eval_res["success_rate"])
            eval_success_rates.append(eval_sr)

            logger.info(
                f"Ep {ep+1}/{num_episodes} | ext_rew={total_ext_reward:.1f} | "
                f"len={ep_len} | loss={avg_loss:.4f} | "
                f"intr_rew={mean_intrinsic:.4f} | eval_sr={eval_sr:.3f}"
            )

        # ── 8. metrics ──
        met_df.add_episode(
            seed=seed,
            episode=ep + 1,
            reward=total_ext_reward,
            episode_length=ep_len,
            algorithm=algorithm,
            lr=lr,
            gamma=gamma,
            loss=avg_loss,
            eval_success_rate=eval_sr,
            intrinsic_reward=mean_intrinsic,
            policy_entropy=0.0,
            env_name=env_name,
            reward_type=reward_type,
        )

    return {
        "policy_params": policy_params,
        "value_params": value_params,
        "rnd_params": rnd_params,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "eval_success_rates": eval_success_rates,
        "intrinsic_rewards": intrinsic_rewards_log,
    }
