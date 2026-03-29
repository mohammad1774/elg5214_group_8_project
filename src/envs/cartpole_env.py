"""
cartpole_env.py — CartPole-v1 environment setup and sparse reward wrapper.

WHAT IS CARTPOLE:
    A pole is balanced on a cart that moves left or right.
    The agent gets +1 reward for every timestep the pole stays upright.
    Episode ends when:
        - Pole angle exceeds ±12 degrees (failure)
        - Cart moves out of bounds (failure)
        - 500 timesteps reached (success)

    Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions:     0 = push left, 1 = push right

DENSE vs SPARSE REWARDS:
    Dense:  +1.0 every single timestep the pole is balanced.
            Total return can reach up to 500.
            Easy for the agent — constant feedback signal.

    Sparse: 0.0 during the episode, +1.0 ONLY if the pole survived
            to max_steps (success), 0.0 if the pole fell (failure).
            Much harder — the agent gets no signal until the episode ends.
            This is where exploration mechanisms (RND, ICM) should help.

WHY A SEPARATE FILE:
    Following the reference project pattern where gridworld.py was its own file.
    Keeps environment-specific logic isolated. env_utils.py imports from here.
"""

import gymnax
import jax.numpy as jnp


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

CARTPOLE_OBS_DIM = 4         # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
CARTPOLE_NUM_ACTIONS = 2     # 0=push_left, 1=push_right
CARTPOLE_MAX_STEPS = 500     # Gymnax CartPole-v1 default max episode length


# ──────────────────────────────────────────────
#  Dense (original) CartPole
# ──────────────────────────────────────────────

def make_cartpole_dense():
    """Create the standard CartPole-v1 environment.

    Returns:
        env:        Gymnax CartPole environment
        env_params: Gymnax CartPole parameters
    """
    env, env_params = gymnax.make("CartPole-v1")
    return env, env_params


# ──────────────────────────────────────────────
#  Sparse CartPole Wrapper
# ──────────────────────────────────────────────

class CartPoleSparseWrapper:
    """Wraps CartPole-v1 to provide sparse end-of-episode rewards.

    Reward logic:
        During episode:  0.0 (no signal)
        At done:
            - If dense_reward > 0 at moment of done → pole was still balanced
              → success → +1.0
            - If dense_reward = 0 at moment of done → pole fell
              → failure → 0.0

    Why check dense_reward instead of step count?
        Gymnax sets reward=0 on the step where done=True due to failure,
        but reward=1 when done=True due to reaching max_steps. So the
        dense reward at the moment of termination tells us WHY it ended.

    Interface matches Gymnax exactly:
        .reset_env(key, params) → (obs, state)
        .step_env(key, state, action, params) → (obs, state, reward, done, info)
    """

    def __init__(self, env):
        self.env = env

    def reset_env(self, key, params):
        return self.env.reset_env(key, params)

    def step_env(self, key, state, action, params):
        obs, next_state, dense_reward, done, info = self.env.step_env(
            key, state, action, params
        )

        # Success = pole survived to max steps (time is 0-indexed, so
        # next_state.time >= params.max_steps_in_episode means we hit the limit)
        survived = next_state.time >= params.max_steps_in_episode

        # Sparse: 0 during episode, +1 if survived full episode, 0 if fell
        sparse_reward = jnp.where(
            done,
            jnp.where(survived, 1.0, 0.0),
            0.0,
        )

        return obs, next_state, sparse_reward, done, info


def make_cartpole_sparse():
    """Create CartPole-v1 with sparse reward wrapper.

    Returns:
        env:        CartPoleSparseWrapper
        env_params: Gymnax CartPole parameters (unchanged)
    """
    env, env_params = gymnax.make("CartPole-v1")
    sparse_env = CartPoleSparseWrapper(env)
    return sparse_env, env_params
