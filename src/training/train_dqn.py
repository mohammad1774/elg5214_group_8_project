"""
DQN Baseline Training Loop — Student B.

Implements the standard DQN training pipeline:
  1. Collect transitions via ε-greedy
  2. Store in replay buffer
  3. Sample mini-batch and update Q-network
  4. Periodically update target network
  5. Evaluate every eval_interval episodes with greedy rollouts
  6. Log metrics to CSV

Usage:
    Called from test/test_dqn_agent.py with specific hyperparameters.
"""

import jax
import jax.numpy as jnp
import time
from typing import Dict, Optional

from src.networks.q_network import q_forward, init_q_params
from src.agents.dqn_agent import (
    DQNConfig, create_dqn_agent, act, greedy_action,
    update_dqn, get_epsilon,
)
from src.replay.replay_buffer import init_buffer, add_transition, sample_batch, can_sample
from src.evaluate.evaluate_dqn import evaluate_dqn_greedy


def train_dqn(
    key: jax.random.PRNGKey,
    env,
    env_params,
    config: DQNConfig,
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
    """Train a vanilla DQN agent.

    Args:
        key: JAX PRNG key.
        env: Gymnax environment (dense or sparse).
        env_params: Environment parameters.
        config: DQN hyperparameters.
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
        Dict with final params, training history, and timing info.
    """
    start_time = time.time()

    # Initialize agent
    key, init_key = jax.random.split(key)
    params, target_params, opt_state, optimizer = create_dqn_agent(init_key, config)

    # Initialize replay buffer
    buffer = init_buffer(buffer_capacity, config.obs_dim)

    # Training tracking
    global_step = 0
    episode_rewards = []
    eval_history = []
    loss_history = []

    for episode in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)
        episode_reward = 0.0
        episode_loss = 0.0
        episode_updates = 0
        done = False

        for t in range(max_steps_per_episode):
            if done:
                break

            # Select action (ε-greedy)
            epsilon = get_epsilon(global_step, config)
            key, act_key = jax.random.split(key)
            action = int(act(act_key, params, obs, epsilon, config.act_dim))

            # Step environment
            key, step_key = jax.random.split(key)
            next_obs, state, reward, done, info = env.step(
                step_key, state, action, env_params
            )

            # Store transition
            buffer = add_transition(buffer, obs, action, float(reward), next_obs, bool(done))

            episode_reward += float(reward)
            obs = next_obs
            global_step += 1

            # Update Q-network (after warmup, every step)
            if can_sample(buffer, batch_size) and global_step > warmup_steps:
                key, sample_key = jax.random.split(key)
                batch = sample_batch(sample_key, buffer, batch_size)

                params, target_params, opt_state, loss = update_dqn(
                    params, target_params, opt_state, optimizer,
                    batch, config.gamma, global_step, config.target_update_freq,
                )
                episode_loss += float(loss)
                episode_updates += 1

            done = bool(done)

        # Track episode metrics
        avg_loss = episode_loss / max(episode_updates, 1)
        episode_rewards.append(episode_reward)
        loss_history.append(avg_loss)

        # Periodic evaluation
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
            })
            if logger:
                logger.info(
                    f"[Eval ep {episode+1}] mean_reward={mean_r:.2f} ± {std_r:.2f} | "
                    f"success_rate={success_rate:.2f} | ε={epsilon:.4f}"
                )

        # Log to metrics tracker
        if metrics_tracker is not None:
            metrics_tracker.add_episode(
                seed=0,  # Overridden by caller
                episode=episode + 1,
                reward=episode_reward,
                episode_length=t + 1,
                loss=avg_loss,
                eval_success_rate=eval_success_rate,
                algorithm="dqn_baseline",
                learning_rate=config.lr,
                gamma=config.gamma,
            )

        # Console progress
        if (episode + 1) % 100 == 0 and logger:
            recent_mean = jnp.mean(jnp.array(episode_rewards[-100:]))
            logger.info(
                f"[Train ep {episode+1}/{num_episodes}] "
                f"recent_100_mean={recent_mean:.2f} | loss={avg_loss:.4f} | "
                f"ε={epsilon:.4f} | buffer={buffer['size']}"
            )

    wall_time = time.time() - start_time

    return {
        "params": params,
        "target_params": target_params,
        "episode_rewards": episode_rewards,
        "eval_history": eval_history,
        "loss_history": loss_history,
        "global_steps": global_step,
        "wall_time_s": wall_time,
    }
