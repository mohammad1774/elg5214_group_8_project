"""
train_ppo.py — PPO Baseline training loop.

TRAINING FLOW PER EPISODE:
    1. Collect n_steps of rollout via lax.scan
    2. Compute GAE advantages + discounted returns
    3. Run n_epochs of mini-batch updates on (policy + value)
    4. Log metrics every episode
    5. Greedy eval every log_every episodes

LOSS FUNCTION:
    policy_loss = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
    value_loss  = mean((V(s) - returns)^2)
    total_loss  = policy_loss + 0.5 * value_loss

    Where ratio = exp(new_log_prob - old_log_prob)
    The clip prevents catastrophically large policy updates.
"""

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax

from src.agents.ppo_agent import PPOAgent
from src.networks.policy_network import init_policy_params, policy_forward, log_prob, entropy
from src.networks.value_network import init_value_params, value_forward, value_forward_batch
from src.training.rollout import run_one_episode_scan_simple
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.utils.reusable import RLMetricsDataset, setup_logger, Timer

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    next_value: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:

    next_values = jnp.concatenate([values[1:], next_value[None]])
    not_done = 1.0 - dones.astype(jnp.float32)

    deltas = rewards + gamma * next_values * not_done - values

    def backward_scan(carry, inputs):
        delta, nd = inputs       # nd = not_done mask
        next_adv = carry
        adv = delta + gamma * gae_lambda * nd * next_adv
        return adv, adv

    _, advantages_reversed = lax.scan(
        backward_scan,
        init=jnp.float32(0.0),                           # A_{T+1} = 0
        xs=(deltas[::-1], not_done[::-1]),
    )

    advantages = advantages_reversed[::-1]   # flip back to forward order
    returns    = advantages + values         # Q(s,a) = A(s,a) + V(s)

    return advantages, returns

@partial(jax.jit, static_argnames=("clip_epsilon",))
def ppo_update_step(
    policy_params: dict,
    value_params: dict,
    obs_batch: jnp.ndarray,
    actions_batch: jnp.ndarray,
    old_log_probs_batch: jnp.ndarray,
    advantages_batch: jnp.ndarray,
    returns_batch: jnp.ndarray,
    lr: float,
    clip_epsilon: float = 0.2,
    grad_clip_norm: float = 10.0,
) -> tuple:
    

    def loss_fn(params):
        p_params, v_params = params

        new_log_probs = jax.vmap(
            lambda obs, act: log_prob(p_params, obs, act)
        )(obs_batch, actions_batch)

        log_ratio = new_log_probs - old_log_probs_batch
        ratio = jnp.exp(log_ratio)

        adv_mean = jnp.mean(advantages_batch)
        adv_std  = jnp.std(advantages_batch) + 1e-8
        norm_adv = (advantages_batch - adv_mean) / adv_std

        # Clipped surrogate objective
        surr1 = ratio * norm_adv
        surr2 = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * norm_adv
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        values_pred = jax.vmap(lambda obs: value_forward(v_params, obs))(obs_batch)
        value_loss  = jnp.mean((values_pred - returns_batch) ** 2)

        total_loss = policy_loss + 0.5 * value_loss
        return total_loss, (policy_loss, value_loss)

    (total_loss, _aux), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )((policy_params, value_params))

    policy_grads, value_grads = grads

    # Gradient clipping: prevents exploding gradients (clip_norm=10.0)
    def clip_grads(g_tree):  
        leaves = jax.tree_util.tree_leaves(g_tree)
        global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        scale = jnp.minimum(1.0, grad_clip_norm / (global_norm + 1e-8))
        return jax.tree_util.tree_map(lambda g: g * scale, g_tree)

    policy_grads = clip_grads(policy_grads)
    value_grads  = clip_grads(value_grads)

    # Manual SGD — no Optax per project rules
    new_policy_params = jax.tree_util.tree_map(
        lambda p, g: p - lr * g, policy_params, policy_grads
    )
    new_value_params = jax.tree_util.tree_map(
        lambda p, g: p - lr * g, value_params, value_grads
    )

    return new_policy_params, new_value_params, total_loss


#  Main Training Function
def train_ppo(
    env,
    env_params,
    obs_dim: int,
    num_actions: int,
    config: dict,
    seed: int = 0,
    lr: float = 0.001,
    gamma: float = 0.99,
    logger=None,
    met_df: RLMetricsDataset = None,
    algorithm_name: str = "PPO",
    env_name: str = "cartpole",
    reward_type: str = "dense",
) -> dict:
    
    ### Hyperparameters
    num_episodes    = config.get("num_episodes", 2000)
    n_steps         = config.get("ppo", {}).get("n_steps", 128)
    n_epochs        = config.get("ppo", {}).get("n_epochs", 4)
    mini_batch_size = config.get("ppo", {}).get("mini_batch_size", 64)
    clip_epsilon    = config.get("ppo", {}).get("clip_epsilon", 0.2)
    gae_lambda      = config.get("ppo", {}).get("gae_lambda", 0.95)
    grad_clip_norm  = config.get("ppo", {}).get("gradient_clip_norm", 10.0)
    max_steps       = config.get("max_steps", 200)
    log_every       = config.get("log_every", 50)
    eval_episodes   = config.get("eval_episodes", 100)

    ### Agent and networks initialization
    key = jax.random.PRNGKey(seed)
    key, pk, vk = jax.random.split(key, 3)

    policy_params = init_policy_params(pk, obs_dim=obs_dim, num_actions=num_actions)
    value_params  = init_value_params(vk, obs_dim=obs_dim)
    agent         = PPOAgent(policy_params, num_actions=num_actions)

    episode_rewards    = []
    episode_lengths    = []
    losses             = []
    eval_success_rates = []
    wall_time_s        = 0.0

    if logger:
        logger.info(
            f"PPO training start | seed={seed} lr={lr} gamma={gamma} "
            f"env={env_name}/{reward_type}"
        )
        logger.info(
            f"  n_steps={n_steps} n_epochs={n_epochs} "
            f"mini_batch={mini_batch_size} clip={clip_epsilon} lambda={gae_lambda}"
        )

    ###  Main Episode Loop
    with Timer("PPO training") as timer:
        for episode in range(num_episodes):
            key, rollout_key = jax.random.split(key)

            # Collect rollout 
            agent.update_params(policy_params)

            rollout = run_one_episode_scan_simple(
                env=env,
                env_params=env_params,
                agent=agent,
                key=rollout_key,
                max_steps=n_steps,   # collect exactly n_steps transitions
                greedy=False,        # stochastic during training
            )

            obs      = rollout["observations"]        # (n_steps, obs_dim)
            actions  = rollout["actions"]             # (n_steps,)
            rewards  = rollout["rewards"]             # (n_steps,)
            dones    = rollout["dones"]               # (n_steps,)
            next_obs = rollout["next_observations"]   # (n_steps, obs_dim)

            # V(s) and GAE 
            values     = value_forward_batch(value_params, obs)     # (n_steps,)
            next_value = value_forward(value_params, next_obs[-1])  # bootstrap scalar

            advantages, returns = compute_gae(
                rewards, values, next_value, dones, gamma, gae_lambda
            )
            old_log_probs = jax.vmap(
                lambda o, a: log_prob(policy_params, o, a)
            )(obs, actions)
            epoch_losses = []

            for _epoch in range(n_epochs):
                key, shuffle_key = jax.random.split(key)
                perm        = jax.random.permutation(shuffle_key, n_steps)
                num_batches = n_steps // mini_batch_size

                for b in range(num_batches):
                    idx = perm[b * mini_batch_size : (b + 1) * mini_batch_size]

                    policy_params, value_params, batch_loss = ppo_update_step(
                        policy_params        = policy_params,
                        value_params         = value_params,
                        obs_batch            = obs[idx],
                        actions_batch        = actions[idx],
                        old_log_probs_batch  = old_log_probs[idx],
                        advantages_batch     = advantages[idx],
                        returns_batch        = returns[idx],
                        lr                   = lr,
                        clip_epsilon         = clip_epsilon,
                        grad_clip_norm       = grad_clip_norm,
                    )
                    epoch_losses.append(float(batch_loss))

            mean_loss      = float(jnp.mean(jnp.array(epoch_losses)))
            episode_reward = float(rollout["total_reward"])
            episode_length = int(rollout["episode_length"])

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            losses.append(mean_loss)

            eval_success = -1.0
            if (episode + 1) % log_every == 0:
                eval_stats = evaluate_ppo(
                    env           = env,
                    env_params    = env_params,
                    policy_params = policy_params,
                    num_episodes  = eval_episodes,
                    max_steps     = max_steps,
                    seed          = seed + episode,
                    greedy        = True,
                    num_actions   = num_actions,
                )
                eval_success = eval_stats["success_rate"]
                eval_success_rates.append(eval_success)

            # Episode Log
            if logger:
                logger.info(
                    f"Ep {episode+1:4d} | "
                    f"reward={episode_reward:7.2f} | "
                    f"length={episode_length:3d} | "
                    f"loss={mean_loss:.4f} | "
                    f"eval_success={eval_success:.3f}"
                )

            if met_df is not None:
                met_df.add_episode(
                    seed              = seed,
                    episode           = episode,
                    reward            = episode_reward,
                    episode_length    = episode_length,
                    algorithm         = algorithm_name,
                    lr                = lr,
                    gamma             = gamma,
                    loss              = mean_loss,
                    eval_success_rate = eval_success,
                    env_name          = env_name,
                    reward_type       = reward_type,
                )

    wall_time_s = timer.elapsed

    # Final evaluation after all episodes 
    final_eval = evaluate_ppo(
        env           = env,
        env_params    = env_params,
        policy_params = policy_params,
        num_episodes  = eval_episodes,
        max_steps     = max_steps,
        seed          = seed + 99999,
        greedy        = True,
        num_actions   = num_actions,
    )

    if logger:
        logger.info(
            f"DONE | final_mean_reward={final_eval['mean_reward']:.2f} | "
            f"final_success_rate={final_eval['success_rate']:.3f}"
        )

    return {
        "policy_params":      policy_params,
        "value_params":       value_params,
        "episode_rewards":    episode_rewards,
        "episode_lengths":    episode_lengths,
        "losses":             losses,
        "eval_success_rates": eval_success_rates,
        "final_eval":         final_eval,
        "wall_time_s":        wall_time_s,
    }