"""
Environment utilities for Gymnax CartPole-v1 and MountainCar-v0.
Provides dense and sparse reward variants.

Usage:
    env, env_params = get_env("CartPole-v1", sparse=True)
    obs, state = env.reset(key, env_params)
    obs, state, reward, done, info = env.step(key, state, action, env_params)
"""

import jax
import jax.numpy as jnp
import gymnax


# Environment metadata
ENV_CONFIGS = {
    "CartPole-v1": {"obs_dim": 4, "act_dim": 2, "max_steps": 500},
    "MountainCar-v0": {"obs_dim": 2, "act_dim": 3, "max_steps": 200},
}


def get_env(name: str, sparse: bool = False):
    """Create a Gymnax environment with optional sparse reward wrapper.

    Args:
        name: Environment name ("CartPole-v1" or "MountainCar-v0").
        sparse: If True, replace dense reward with sparse (binary at episode end).

    Returns:
        (env, env_params) tuple. If sparse=True, returns a SparseWrapper.
    """
    env, env_params = gymnax.make(name)
    if sparse:
        env = SparseRewardWrapper(env, name)
    return env, env_params


def get_env_config(name: str) -> dict:
    """Get observation/action dimensions and max steps for an environment."""
    if name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env: {name}. Choose from {list(ENV_CONFIGS.keys())}")
    return ENV_CONFIGS[name]


class SparseRewardWrapper:
    """Wraps a Gymnax environment to provide sparse (binary) rewards.

    Dense rewards are zeroed out during the episode. At termination:
      - CartPole: reward = 1.0 if survived max_steps, else 0.0
      - MountainCar: reward = 1.0 if reached the goal (position >= 0.5), else 0.0
    """

    def __init__(self, env, env_name: str):
        self.env = env
        self.env_name = env_name

    @property
    def default_params(self):
        return self.env.default_params

    def reset(self, key, params=None):
        return self.env.reset(key, params)

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self.env.step(key, state, action, params)

        if self.env_name == "CartPole-v1":
            # Sparse: 1.0 only if episode ends by reaching max steps (success)
            # done=True from falling over → reward=0.0
            # done=True from max steps → reward=1.0
            # not done → reward=0.0
            # In CartPole, the dense reward is +1 per step. If done and
            # the reward was still +1, the pole didn't fall (time limit).
            sparse_reward = jnp.where(
                done,
                jnp.where(reward > 0, 1.0, 0.0),  # success vs failure at termination
                0.0  # zero reward during episode
            )
        elif self.env_name == "MountainCar-v0":
            # Sparse: 1.0 only if car reached the goal (position >= 0.5)
            # obs[0] is position
            sparse_reward = jnp.where(
                done,
                jnp.where(obs[0] >= 0.5, 1.0, 0.0),
                0.0
            )
        else:
            # Fallback: binary at episode end based on positive reward
            sparse_reward = jnp.where(done, jnp.where(reward > 0, 1.0, 0.0), 0.0)

        return obs, state, sparse_reward, done, info

    # Delegate attribute access to wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)
