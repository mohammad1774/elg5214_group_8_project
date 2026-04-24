"""
train_ppo_entropy.py — PPO with entropy regularization training.

Student C (Md Mosarraf)
"""

from typing import Dict, Any, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from src.agents.ppo_entropy_agent import PPOEntropyAgent
from src.networks.policy_network import policy_forward
from src.networks.value_network import value_forward, value_forward_batch
from src.exploration.entropy_reg import entropy_from_logits_batch
from src.evaluate.evaluate_ppo import evaluate_ppo
from src.training.rollout import run_one_episode_scan_simple


def _discounted_returns(rewards: jnp.ndarray,
                        dones: jnp.ndarray,
                        gamma: float) -> jnp.ndarray:
    T = rewards.shape[0]
    returns = jnp.zeros_like(rewards)

    def body(step, carry):
        next_return, returns = carry
        idx = T - 1 - step
        reward = rewards[idx]
        done = dones[idx].astype(jnp.float32)
        current_return = reward + gamma * next_return * (1.0 - done)
        returns = returns.at[idx].set(current_return)
        return current_return, returns

    _, returns = lax.fori_loop(0, T, body, (0.0, returns))
    return returns


def _compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> jnp.ndarray:
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)

    def body(step, carry):
        next_adv, next_value, advantages = carry
        idx = T - 1 - step
        reward = rewards[idx]
        done = dones[idx].astype(jnp.float32)
        value = values[idx]
        delta = reward + gamma * next_value * (1.0 - done) - value
        adv = delta + gamma * lam * (1.0 - done) * next_adv
        advantages = advantages.at[idx].set(adv)
        return adv, value, advantages

    _, _, advantages = lax.fori_loop(
        0,
        T,
        body,
        (0.0, last_value, advantages),
    )
    return advantages


def _compute_ppo_loss(
    policy_params: Dict,
    value_params: Dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    mask: jnp.ndarray,
    clip_epsilon: float,
    value_coef: float,
    entropy_coeff: float,
) -> tuple:
    logits = jax.vmap(lambda x: policy_forward(policy_params, x))(obs)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[:, None], axis=1
    ).squeeze(-1)

    ratios = jnp.exp(action_log_probs - old_log_probs)
    clipped_ratios = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate1 = ratios * advantages
    surrogate2 = clipped_ratios * advantages
    policy_loss = -jnp.sum(jnp.minimum(surrogate1, surrogate2) * mask) / jnp.sum(mask)

    values = value_forward_batch(value_params, obs)
    value_loss = jnp.sum(((returns - values) ** 2) * mask) / jnp.sum(mask)

    entropy = jnp.sum(entropy_from_logits_batch(logits) * mask) / jnp.sum(mask)

    total_loss = policy_loss + value_coef * value_loss - entropy_coeff * entropy
    return total_loss, policy_loss, value_loss, entropy


@partial(
    jax.jit,
    static_argnames=(
        "clip_epsilon",
        "value_coef",
        "entropy_coeff",
    ),
)
def _update_ppo_params(
    policy_params: Dict,
    value_params: Dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    mask: jnp.ndarray,
    learning_rate: float,
    clip_epsilon: float,
    value_coef: float,
    entropy_coeff: float,
) -> tuple:
    params = {"policy": policy_params, "value": value_params}

    def loss_fn(params):
        total_loss, _, _, _ = _compute_ppo_loss(
            params["policy"],
            params["value"],
            obs,
            actions,
            old_log_probs,
            returns,
            advantages,
            mask,
            clip_epsilon,
            value_coef,
            entropy_coeff,
        )
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    total_loss, policy_loss, value_loss, entropy = _compute_ppo_loss(
        new_params["policy"],
        new_params["value"],
        obs,
        actions,
        old_log_probs,
        returns,
        advantages,
        mask,
        clip_epsilon,
        value_coef,
        entropy_coeff,
    )
    return (
        new_params["policy"],
        new_params["value"],
        loss,
        policy_loss,
        value_loss,
        entropy,
    )


def train_ppo_entropy(
    env,
    env_params,
    init_policy_params: Dict,
    init_value_params: Dict,
    num_episodes: int = 2000,
    max_steps: int = 200,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    seed: int = 0,
    log_every: int = 50,
    clip_epsilon: float = 0.2,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    entropy_coeff: float = 0.01,
    obs_dim: int = 4,
    num_actions: int = 2,
    env_name: str = "",
    reward_type: str = "",
    logger: Any = None,
    met_df: Any = None,
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    policy_params = init_policy_params
    value_params = init_value_params

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    losses: List[float] = []
    entropies: List[float] = []
    eval_success_rates: List[float] = []

    for episode in range(1, num_episodes + 1):
        key, ep_key = jax.random.split(key)
        agent = PPOEntropyAgent(policy_params, num_actions=num_actions)

        rollout = run_one_episode_scan_simple(
            env=env,
            env_params=env_params,
            agent=agent,
            key=ep_key,
            max_steps=max_steps,
            greedy=False,
        )

        episode_length = int(rollout["episode_length"])
        obs = rollout["observations"][:episode_length]
        actions = rollout["actions"][:episode_length]
        rewards = rollout["rewards"][:episode_length]
        dones = rollout["dones"][:episode_length]

        values = value_forward_batch(value_params, obs)
        last_value = value_forward(value_params, rollout["next_observations"][-1])
        advantages = _compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)
        returns = advantages + values
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        logits = jax.vmap(lambda x: policy_forward(policy_params, x))(obs)
        log_probs = jax.nn.log_softmax(logits)
        old_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1
        ).squeeze(-1)
        old_log_probs = jax.lax.stop_gradient(old_log_probs)

        mask = jnp.ones_like(rewards, dtype=jnp.float32)

        policy_params, value_params, loss, policy_loss, value_loss, entropy = _update_ppo_params(
            policy_params,
            value_params,
            obs,
            actions,
            old_log_probs,
            returns,
            advantages,
            mask,
            learning_rate,
            clip_epsilon,
            value_coef,
            entropy_coeff,
        )

        ep_reward = float(jnp.sum(rewards))
        episode_rewards.append(ep_reward)
        episode_lengths.append(episode_length)
        losses.append(float(loss))
        entropies.append(float(entropy))

        if logger is not None:
            logger.info(
                f"Episode {episode:4d} - Reward: {ep_reward:.3f}, "
                f"Length: {episode_length}, Loss: {loss:.4f}, "
                f"PolicyLoss: {policy_loss:.4f}, ValueLoss: {value_loss:.4f}, "
                f"Entropy: {entropy:.4f}"
            )

        if met_df is not None:
            met_df.add_episode(
                seed=seed,
                episode=episode,
                reward=ep_reward,
                episode_length=episode_length,
                loss=float(loss),
                algorithm="PPO_Entropy",
                lr=learning_rate,
                gamma=gamma,
                policy_entropy=float(entropy),
                env_name=env_name,
                reward_type=reward_type,
            )

        if episode % log_every == 0:
            eval_stats = evaluate_ppo(
                env=env,
                env_params=env_params,
                policy_params=policy_params,
                num_episodes=25,
                max_steps=max_steps,
                seed=seed + episode,
                greedy=True,
                num_actions=num_actions,
            )
            eval_success_rates.append(eval_stats["success_rate"])
            if logger is not None:
                logger.info(
                    f"Eval @ episode {episode}: success={eval_stats['success_rate']:.3f}, "
                    f"mean_reward={eval_stats['mean_reward']:.3f}, "
                    f"mean_length={eval_stats['mean_length']:.2f}"
                )

    return {
        "final_policy_params": policy_params,
        "final_value_params": value_params,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "policy_entropies": entropies,
        "eval_success_rates": eval_success_rates,
    }
