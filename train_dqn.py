from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.agents.dqn_agent import (
    DQNAgent,
    create_dqn_agent,
    act,
    greedy_action,
    sync_target_network,
    update_epsilon,
    update_q_params,
)
from src.envs.env_utils import get_env
from src.networks.q_network import q_forward_batch
from src.replay.replay_buffer import init_buffer, add_transition, sample_batch


def dqn_td_loss(
    q_params: Any,
    target_q_params: Any,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    q_all = q_forward_batch(q_params, batch["obs"])
    q_selected = jnp.take_along_axis(
        q_all, batch["actions"][:, None], axis=1
    ).squeeze(axis=1)

    next_q_all_target = q_forward_batch(target_q_params, batch["next_obs"])
    next_q_max = jnp.max(next_q_all_target, axis=1)

    td_target = batch["rewards"] + gamma * (1.0 - batch["dones"].astype(jnp.float32)) * next_q_max
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
def train_step(
    q_params: Any,
    target_q_params: Any,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    gamma: float,
    learning_rate: float,
):
    optimizer = optax.adam(learning_rate)

    def loss_fn(params):
        return dqn_td_loss(params, target_q_params, batch, gamma)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, q_params)
    new_q_params = optax.apply_updates(q_params, updates)

    return new_q_params, new_opt_state, metrics


def evaluate_dqn_agent(
    agent: DQNAgent,
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


def train_dqn(
    env_name: str,
    reward_type: str,
    seed: int = 0,
    num_episodes: int = 300,
    max_steps_per_episode: int = 1000,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
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
) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    key = jax.random.PRNGKey(seed)
    key, agent_key = jax.random.split(key)

    agent = create_dqn_agent(
        key=agent_key,
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        epsilon=epsilon_start,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(agent.q_params)

    replay_buffer = init_buffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
    )

    history: Dict[str, List[float]] = {
        "episode_return": [],
        "episode_length": [],
        "epsilon": [],
        "train_loss": [],
        "train_q_mean": [],
        "train_target_mean": [],
        "train_td_error_mean": [],
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
        last_train_metrics = None

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

                new_q_params, opt_state, train_metrics = train_step(
                    q_params=agent.q_params,
                    target_q_params=agent.target_q_params,
                    opt_state=opt_state,
                    batch=batch,
                    gamma=gamma,
                    learning_rate=learning_rate,
                )

                agent = update_q_params(agent, new_q_params)

                total_gradient_steps += 1
                last_train_metrics = train_metrics

                if total_gradient_steps % target_update_freq == 0:
                    agent = sync_target_network(agent)

        history["episode_return"].append(ep_return)
        history["episode_length"].append(ep_len)
        history["epsilon"].append(float(agent.epsilon))

        if last_train_metrics is not None:
            history["train_loss"].append(float(last_train_metrics["loss"]))
            history["train_q_mean"].append(float(last_train_metrics["q_mean"]))
            history["train_target_mean"].append(float(last_train_metrics["target_mean"]))
            history["train_td_error_mean"].append(float(last_train_metrics["td_error_mean"]))
        else:
            history["train_loss"].append(np.nan)
            history["train_q_mean"].append(np.nan)
            history["train_target_mean"].append(np.nan)
            history["train_td_error_mean"].append(np.nan)

        if (episode_idx + 1) % eval_every == 0:
            eval_metrics = evaluate_dqn_agent(
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
                f"[DQN] Env={env_name} Reward={reward_type} "
                f"Ep={episode_idx + 1}/{num_episodes} "
                f"TrainReturn={ep_return:.2f} "
                f"EvalReturn={eval_metrics['eval_return_mean']:.2f} "
                f"Eps={agent.epsilon:.3f}"
            )
        else:
            print(
                f"[DQN] Env={env_name} Reward={reward_type} "
                f"Ep={episode_idx + 1}/{num_episodes} "
                f"Return={ep_return:.2f} "
                f"Len={ep_len} "
                f"Eps={agent.epsilon:.3f}"
            )

    agent = sync_target_network(agent)
    return agent, history