"""
DQN + ICM Training Loop — Student B.

Extends the vanilla DQN training loop with ICM curiosity rewards:
  1. Collect transitions via ε-greedy
  2. Compute ICM intrinsic reward for each transition
  3. Store (extrinsic + intrinsic) reward in replay buffer
  4. Sample mini-batch → update Q-network AND ICM networks
  5. Periodically evaluate with greedy rollouts

The key difference from vanilla DQN: rewards in the buffer include the
curiosity bonus, so the Q-network learns to value states that are novel
(high forward prediction error) in addition to states that yield extrinsic reward.
"""

import jax
import jax.numpy as jnp
import optax
import time
from typing import Dict, Optional

from src.networks.q_network import init_q_params
from src.networks.icm_networks import init_icm_params
from src.agents.dqn_agent import (
    create_dqn_agent, act, get_epsilon, update_dqn,
)
from src.agents.dqn_icm_agent import DQNICMConfig
from src.exploration.icm import icm_intrinsic_reward, update_icm
from src.replay.replay_buffer import init_buffer, add_transition, sample_batch, can_sample
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy


def train_dqn_icm(
    key: jax.random.PRNGKey,
    env,
    env_params,
    config: DQNICMConfig,
    num_episodes: int = 2000,
    max_steps_per_episode: int = 500,
    buffer_capacity: int = 50000,
    batch_size: int = 64,
    warmup_steps: int = 500,
    eval_interval: int = 50,
    eval_episodes: int = 10,
    metrics_tracker=None,
    logger=None,
) -> Dict:
    """Train a DQN agent with ICM curiosity-driven exploration.

    Args:
        key: JAX PRNG key.
        env: Gymnax environment (dense or sparse).
        env_params: Environment parameters.
        config: DQN + ICM hyperparameters.
        num_episodes: Total training episodes.
        max_steps_per_episode: Max steps per episode.
        buffer_capacity: Replay buffer size.
        batch_size: Mini-batch size for updates.
        warmup_steps: Random steps before training starts.
        eval_interval: Episodes between greedy evaluations.
        eval_episodes: Number of evaluation rollouts.
        metrics_tracker: Optional RLMetricsDataset for CSV logging.
        logger: Optional Python logger.

    Returns:
        Dict with final params, ICM params, training history, and timing info.
    """
    start_time = time.time()

    # Initialize DQN agent
    dqn_config = config.to_dqn_config()
    key, init_key = jax.random.split(key)
    params, target_params, opt_state, optimizer = create_dqn_agent(init_key, dqn_config)

    # Initialize ICM networks
    key, icm_key = jax.random.split(key)
    icm_params = init_icm_params(
        icm_key, config.obs_dim, config.act_dim,
        feature_dim=config.feature_dim, hidden_dim=config.icm_hidden_dim,
    )
    icm_optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.icm_lr),
    )
    icm_opt_state = icm_optimizer.init(icm_params)

    # Initialize replay buffer
    buffer = init_buffer(buffer_capacity, config.obs_dim)

    # Training tracking
    global_step = 0
    episode_rewards = []
    episode_intrinsic_rewards = []
    eval_history = []
    loss_history = []
    icm_loss_history = []

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)
        episode_reward = 0.0
        episode_intrinsic = 0.0
        episode_loss = 0.0
        episode_icm_loss = 0.0
        episode_updates = 0
        done = False

        for t in range(max_steps_per_episode):
            if done:
                break

            # Select action (ε-greedy)
            epsilon = get_epsilon(global_step, dqn_config)
            key, act_key = jax.random.split(key)
            action = int(act(act_key, params, obs, epsilon, config.act_dim))

            # Step environment
            key, step_key = jax.random.split(key)
            next_obs, state, extrinsic_reward, done, info = env.step(
                step_key, state, action, env_params
            )

            # Compute ICM intrinsic reward
            intrinsic_reward = float(icm_intrinsic_reward(
                icm_params, obs, action, next_obs, config.act_dim, eta=config.eta,
            ))

            # Total reward = extrinsic + intrinsic
            total_reward = float(extrinsic_reward) + intrinsic_reward

            # Store in replay buffer with augmented reward
            buffer = add_transition(buffer, obs, action, total_reward, next_obs, bool(done))

            episode_reward += float(extrinsic_reward)
            episode_intrinsic += intrinsic_reward
            obs = next_obs
            global_step += 1

            # Update Q-network + ICM (after warmup)
            if can_sample(buffer, batch_size) and global_step > warmup_steps:
                key, sample_key = jax.random.split(key)
                batch = sample_batch(sample_key, buffer, batch_size)

                # Update Q-network
                params, target_params, opt_state, q_loss = update_dqn(
                    params, target_params, opt_state, optimizer,
                    batch, config.gamma, global_step, config.target_update_freq,
                )

                # Update ICM networks on the same batch
                icm_params, icm_opt_state, icm_info = update_icm(
                    icm_params, icm_opt_state, icm_optimizer,
                    batch["obs"], batch["actions"], batch["next_obs"],
                    config.act_dim, beta_icm=config.beta_icm,
                )

                episode_loss += float(q_loss)
                episode_icm_loss += float(icm_info["total_loss"])
                episode_updates += 1

            done = bool(done)

        # Track episode metrics
        avg_loss = episode_loss / max(episode_updates, 1)
        avg_icm_loss = episode_icm_loss / max(episode_updates, 1)
        episode_rewards.append(episode_reward)
        episode_intrinsic_rewards.append(episode_intrinsic)
        loss_history.append(avg_loss)
        icm_loss_history.append(avg_icm_loss)

        # Periodic evaluation (greedy — no intrinsic reward, pure extrinsic)
        eval_success_rate = None
        if (episode + 1) % eval_interval == 0:
            key, eval_key = jax.random.split(key)
            mean_r, std_r, success_rate = evaluate_dqn_greedy(
                eval_key, params, env, env_params,
                num_episodes=eval_episodes, max_steps=max_steps_per_episode,
            )
            eval_success_rate = success_rate
            eval_history.append({
                "episode": episode + 1,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "success_rate": success_rate,
                "epsilon": epsilon,
                "mean_intrinsic_reward": jnp.mean(jnp.array(
                    episode_intrinsic_rewards[-eval_interval:]
                )),
            })
            if logger:
                logger.info(
                    f"[Eval ep {episode+1}] mean_reward={mean_r:.2f} ± {std_r:.2f} | "
                    f"success={success_rate:.2f} | ε={epsilon:.4f} | "
                    f"intrinsic_r={episode_intrinsic:.4f}"
                )

        # Log to metrics tracker
        if metrics_tracker is not None:
            metrics_tracker.add_episode(
                seed=0,
                episode=episode + 1,
                reward=episode_reward,
                episode_length=t + 1,
                loss=avg_loss,
                eval_success_rate=eval_success_rate,
                algorithm="dqn_icm",
                learning_rate=config.lr,
                gamma=config.gamma,
            )

        # Console progress
        if (episode + 1) % 100 == 0 and logger:
            recent_mean = jnp.mean(jnp.array(episode_rewards[-100:]))
            recent_intrinsic = jnp.mean(jnp.array(episode_intrinsic_rewards[-100:]))
            logger.info(
                f"[Train ep {episode+1}/{num_episodes}] "
                f"recent_100_mean={recent_mean:.2f} | q_loss={avg_loss:.4f} | "
                f"icm_loss={avg_icm_loss:.4f} | intrinsic_r={recent_intrinsic:.4f}"
            )

    wall_time = time.time() - start_time

    return {
        "params": params,
        "target_params": target_params,
        "icm_params": icm_params,
        "episode_rewards": episode_rewards,
        "episode_intrinsic_rewards": episode_intrinsic_rewards,
        "eval_history": eval_history,
        "loss_history": loss_history,
        "icm_loss_history": icm_loss_history,
        "global_steps": global_step,
        "wall_time_s": wall_time,
    }
