from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.agents.dqn_icm_agent import (
    DQNICMAgent,
    create_dqn_icm_agent,
    act,
    greedy_action,
    sync_target_network,
    update_epsilon,
    update_icm_params,
    update_q_params,
)
from src.envs.env_utils import get_env
from src.exploration.icm import icm_intrinsic_reward_batch, icm_loss
from src.networks.q_network import q_forward_batch
from src.replay.replay_buffer import init_buffer, add_transition, sample_batch


def dqn_td_loss(
    q_params: Any,
    target_q_params: Any,
    batch: Dict[str, jnp.ndarray],
    rewards_for_td: jnp.ndarray,
    gamma: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    q_all = q_forward_batch(q_params, batch["obs"])
    q_selected = jnp.take_along_axis(
        q_all, batch["actions"][:, None], axis=1
    ).squeeze(axis=1)

    next_q_all_target = q_forward_batch(target_q_params, batch["next_obs"])
    next_q_max = jnp.max(next_q_all_target, axis=1)

    td_target = rewards_for_td + gamma * (1.0 - batch["dones"].astype(jnp.float32)) * next_q_max
    td_error = q_selected - jax.lax.stop_gradient(td_target)

    loss = jnp.mean(optax.huber_loss(td_error, delta=1.0))

    metrics = {
        "loss": loss,
        "q_mean": jnp.mean(q_selected),
        "target_mean": jnp.mean(td_target),
        "td_error_mean": jnp.mean(jnp.abs(td_error)),
    }
    return loss, metrics


@jax.jit
def q_train_step(
    q_params: Any,
    target_q_params: Any,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    rewards_for_td: jnp.ndarray,
    gamma: float,
    learning_rate: float,
):
    optimizer = optax.adam(learning_rate)

    def loss_fn(params):
        return dqn_td_loss(params, target_q_params, batch, rewards_for_td, gamma)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, q_params)
    new_q_params = optax.apply_updates(q_params, updates)

    return new_q_params, new_opt_state, metrics


@jax.jit(static_argnames=("num_actions",))
def icm_train_step(
    icm_params: Any,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    num_actions: int,
    beta: float,
    learning_rate: float,
):
    optimizer = optax.adam(learning_rate)

    def loss_fn(params):
        return icm_loss(
            icm_params=params,
            obs_batch=batch["obs"],
            action_batch=batch["actions"],
            next_obs_batch=batch["next_obs"],
            num_actions=num_actions,
            beta=beta,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(icm_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, icm_params)
    new_icm_params = optax.apply_updates(icm_params, updates)

    return new_icm_params, new_opt_state, metrics


def evaluate_dqn_icm_agent(
    agent: DQNICMAgent,
    env_name: str,
    reward_type: str,
    seed: int,
    num_eval_episodes: int = 5,
    max_steps_per_episode: int = 1000,
) -> Dict[str, float]:
    env, env_params, _, _ = get_env(env_name, reward_type)

    returns = []
    lengths = []

    eval_key = jax.random.PRNGKey(seed + 9999)

    for _ in range(num_eval_episodes):
        eval_key, reset_key = jax.random.split(eval_key)
        obs, env_state = env.reset_env(reset_key, env_params)

        ep_return = 0.0
        ep_len = 0
        done = False

        while (not bool(done)) and ep_len < max_steps_per_episode:
            action = int(greedy_action(agent, obs))
            eval_key, step_key = jax.random.split(eval_key)
            obs, env_state, reward, done, _ = env.step_env(step_key, env_state, action, env_params)

            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }


def train_dqn_icm(
    env_name: str,
    reward_type: str,
    seed: int = 0,
    num_episodes: int = 300,
    max_steps_per_episode: int = 1000,
    q_hidden_dim: int = 64,
    icm_hidden_dim: int = 64,
    icm_feat_dim: int = 64,
    q_learning_rate: float = 1e-3,
    icm_learning_rate: float = 1e-3,
    gamma: float = 0.99,
    buffer_capacity: int = 50000,
    batch_size: int = 64,
    min_buffer_size_before_training: int = 1000,
    train_freq: int = 1,
    target_update_freq: int = 250,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 200,
    num_eval_episodes: int = 5,
    eval_every: int = 25,
    icm_eta: float = 0.01,
    icm_beta: float = 0.2,
    icm_update_freq: int = 1, 
) -> Tuple[DQNICMAgent, Dict[str, List[float]]]:
    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    key = jax.random.PRNGKey(seed)
    key, agent_key = jax.random.split(key)

    agent = create_dqn_icm_agent(
        key=agent_key,
        obs_dim=obs_dim,
        num_actions=num_actions,
        q_hidden_dim=q_hidden_dim,
        icm_hidden_dim=icm_hidden_dim,
        icm_feat_dim=icm_feat_dim,
        epsilon=epsilon_start,
    )

    q_optimizer = optax.adam(q_learning_rate)
    icm_optimizer = optax.adam(icm_learning_rate)

    q_opt_state = q_optimizer.init(agent.q_params)
    icm_opt_state = icm_optimizer.init(agent.icm_params)

    replay_buffer = init_buffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
    )

    history: Dict[str, List[float]] = {
        "episode_return": [],
        "episode_length": [],
        "epsilon": [],
        "train_q_loss": [],
        "train_q_mean": [],
        "train_target_mean": [],
        "train_td_error_mean": [],
        "train_icm_loss": [],
        "train_inverse_loss": [],
        "train_forward_loss": [],
        "train_intrinsic_reward_mean": [],
        "eval_return_mean": [],
        "eval_return_std": [],
        "eval_length_mean": [],
        "eval_episode_idx": [],
    }

    total_env_steps = 0
    total_gradient_steps = 0

    for episode_idx in range(num_episodes):
        frac = min(1.0, episode_idx / max(1, epsilon_decay_episodes))
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        agent = update_epsilon(agent, epsilon)

        key, reset_key = jax.random.split(key)
        obs, env_state = env.reset_env(reset_key, env_params)

        ep_return = 0.0
        ep_len = 0
        done = False

        last_q_metrics = None
        last_icm_metrics = None

        while (not bool(done)) and ep_len < max_steps_per_episode:
            key, action_key = jax.random.split(key)
            action = int(act(agent, action_key, obs))

            key, step_key = jax.random.split(key)
            next_obs, next_env_state, reward, done, _ = env.step_env(
                step_key, env_state, action, env_params
            )

            replay_buffer = add_transition(
                replay_buffer,
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            env_state = next_env_state
            ep_return += float(reward)
            ep_len += 1
            total_env_steps += 1

            can_train = (
                int(replay_buffer["size"]) >= min_buffer_size_before_training
                and total_env_steps % train_freq == 0
            )

            if can_train:
                key, sample_key = jax.random.split(key)
                batch = sample_batch(replay_buffer, sample_key, batch_size)

                intrinsic_rewards = icm_intrinsic_reward_batch(
                    icm_params=agent.icm_params,
                    obs_batch=batch["obs"],
                    action_batch=batch["actions"],
                    next_obs_batch=batch["next_obs"],
                    num_actions=num_actions,
                    eta=icm_eta,
                )

                rewards_for_td = batch["rewards"] + intrinsic_rewards

                new_q_params, q_opt_state, q_metrics = q_train_step(
                    q_params=agent.q_params,
                    target_q_params=agent.target_q_params,
                    opt_state=q_opt_state,
                    batch=batch,
                    rewards_for_td=rewards_for_td,
                    gamma=gamma,
                    learning_rate=q_learning_rate,
                )
                agent = update_q_params(agent, new_q_params)

                if total_gradient_steps % icm_update_freq == 0:
                    new_icm_params, icm_opt_state, icm_metrics = icm_train_step(
                        icm_params=agent.icm_params,
                        opt_state=icm_opt_state,
                        batch=batch,
                        num_actions=num_actions,
                        beta=icm_beta,
                        learning_rate=icm_learning_rate,
                    )
                    agent = update_icm_params(agent, new_icm_params)

                total_gradient_steps += 1
                last_q_metrics = q_metrics
                last_icm_metrics = icm_metrics

                if total_gradient_steps % target_update_freq == 0:
                    agent = sync_target_network(agent)

        history["episode_return"].append(ep_return)
        history["episode_length"].append(ep_len)
        history["epsilon"].append(float(agent.epsilon))

        if last_q_metrics is not None:
            history["train_q_loss"].append(float(last_q_metrics["loss"]))
            history["train_q_mean"].append(float(last_q_metrics["q_mean"]))
            history["train_target_mean"].append(float(last_q_metrics["target_mean"]))
            history["train_td_error_mean"].append(float(last_q_metrics["td_error_mean"]))
        else:
            history["train_q_loss"].append(np.nan)
            history["train_q_mean"].append(np.nan)
            history["train_target_mean"].append(np.nan)
            history["train_td_error_mean"].append(np.nan)

        if last_icm_metrics is not None:
            history["train_icm_loss"].append(float(last_icm_metrics["icm_loss"]))
            history["train_inverse_loss"].append(float(last_icm_metrics["inverse_loss"]))
            history["train_forward_loss"].append(float(last_icm_metrics["forward_loss"]))
            history["train_intrinsic_reward_mean"].append(float(last_icm_metrics["intrinsic_reward_mean"]))
        else:
            history["train_icm_loss"].append(np.nan)
            history["train_inverse_loss"].append(np.nan)
            history["train_forward_loss"].append(np.nan)
            history["train_intrinsic_reward_mean"].append(np.nan)

        if (episode_idx + 1) % eval_every == 0:
            eval_metrics = evaluate_dqn_icm_agent(
                agent=agent,
                env_name=env_name,
                reward_type=reward_type,
                seed=seed,
                num_eval_episodes=num_eval_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )
            history["eval_return_mean"].append(eval_metrics["eval_return_mean"])
            history["eval_return_std"].append(eval_metrics["eval_return_std"])
            history["eval_length_mean"].append(eval_metrics["eval_length_mean"])
            history["eval_episode_idx"].append(episode_idx + 1)

            print(
                f"[DQN+ICM] Env={env_name} Reward={reward_type} "
                f"Ep={episode_idx + 1}/{num_episodes} "
                f"TrainReturn={ep_return:.2f} "
                f"EvalReturn={eval_metrics['eval_return_mean']:.2f} "
                f"IntrinsicMean={history['train_intrinsic_reward_mean'][-1]:.4f} "
                f"Eps={agent.epsilon:.3f}"
            )
        else:
            print(
                f"[DQN+ICM] Env={env_name} Reward={reward_type} "
                f"Ep={episode_idx + 1}/{num_episodes} "
                f"Return={ep_return:.2f} "
                f"Len={ep_len} "
                f"IntrinsicMean={history['train_intrinsic_reward_mean'][-1]:.4f} "
                f"Eps={agent.epsilon:.3f}"
            )

    agent = sync_target_network(agent)
    return agent, history