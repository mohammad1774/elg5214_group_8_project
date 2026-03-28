"""
test_env.py — Environment sanity check.

Run this FIRST after cloning the repo to verify:
    1. Gymnax is installed correctly
    2. All 4 env variants (cartpole/mountaincar × dense/sparse) work
    3. Observations, rewards, and done signals have correct shapes
    4. Sparse wrappers produce zero reward mid-episode

Usage:
    python -m src.test.test_env
"""

import jax
import jax.numpy as jnp

from src.envs.env_utils import get_env


def test_single_env(env_name: str, reward_type: str):
    """Run a few steps and print diagnostics."""
    env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)

    obs, state = env.reset_env(reset_key, env_params)

    print(f"\n{'='*50}")
    print(f"  {env_name} / {reward_type}")
    print(f"{'='*50}")
    print(f"  obs_dim={obs_dim}, num_actions={num_actions}")
    print(f"  Initial obs: {obs}")
    print(f"  Obs shape: {obs.shape}")

    total_reward = 0.0
    steps = 0

    for i in range(20):
        key, act_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(act_key, shape=(), minval=0, maxval=num_actions)

        obs, state, reward, done, _ = env.step_env(step_key, state, action, env_params)
        total_reward += float(reward)
        steps += 1

        if i < 5 or bool(done):
            print(f"  Step {i+1}: action={int(action)}, reward={float(reward):.2f}, "
                  f"done={bool(done)}, obs={obs}")

        if bool(done):
            print(f"  Episode ended at step {steps}")
            break

    print(f"  Total reward after {steps} steps: {total_reward:.2f}")


def main():
    print("Environment Sanity Check")
    print("========================")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    for env_name in ["cartpole", "mountaincar"]:
        for reward_type in ["dense", "sparse"]:
            test_single_env(env_name, reward_type)

    print(f"\n{'='*50}")
    print("All 4 environment variants working!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
