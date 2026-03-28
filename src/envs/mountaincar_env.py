"""
mountaincar_env.py — MountainCar-v0 environment setup and sparse reward wrapper.

WHAT IS MOUNTAINCAR:
    A car sits in a valley between two hills. The engine is too weak to
    drive directly up the right hill. The agent must learn to swing back
    and forth (build momentum) to reach the goal at the top of the right hill.

    Observation: [position, velocity]
        position: -1.2 to 0.6 (goal is at position >= 0.5)
        velocity: -0.07 to 0.07

    Actions: 0 = push left, 1 = no push, 2 = push right

    Episode ends when:
        - Car reaches position >= 0.5 (success)
        - 200 timesteps reached (timeout / failure)

DENSE vs SPARSE REWARDS:
    Dense:  -1.0 every timestep (penalty for being slow).
            Best possible return is around -110 (reach goal in ~110 steps).
            The constant -1 pushes the agent to find the goal faster.

    Sparse: 0.0 during the episode.
            +1.0 if car reaches goal (position >= 0.5) at done.
            -1.0 if episode timed out without reaching goal.

            This is the HARDEST condition in the project. MountainCar
            is already difficult with dense rewards because the agent
            must learn counter-intuitive behavior (go left first to
            build momentum). With sparse rewards, there's zero signal
            unless the agent accidentally reaches the goal.

            This is exactly where we expect RND and ICM to shine —
            they provide intrinsic reward for visiting novel states,
            which encourages the agent to explore the full valley
            even before finding the goal.

WHY A SEPARATE FILE:
    Following the reference project pattern. MountainCar's sparse logic
    differs from CartPole's — we need to check position >= 0.5 to know
    if the goal was reached, while CartPole checks the dense reward value.
"""

import gymnax
import jax.numpy as jnp


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

MOUNTAINCAR_OBS_DIM = 2          # [position, velocity]
MOUNTAINCAR_NUM_ACTIONS = 3      # 0=push_left, 1=no_push, 2=push_right
MOUNTAINCAR_MAX_STEPS = 200      # Gymnax MountainCar-v0 default
MOUNTAINCAR_GOAL_POSITION = 0.5  # position >= this means goal reached


# ──────────────────────────────────────────────
#  Dense (original) MountainCar
# ──────────────────────────────────────────────

def make_mountaincar_dense():
    """Create the standard MountainCar-v0 environment.

    Returns:
        env:        Gymnax MountainCar environment
        env_params: Gymnax MountainCar parameters
    """
    env, env_params = gymnax.make("MountainCar-v0")
    return env, env_params


# ──────────────────────────────────────────────
#  Sparse MountainCar Wrapper
# ──────────────────────────────────────────────

class MountainCarSparseWrapper:
    """Wraps MountainCar-v0 to provide sparse end-of-episode rewards.

    Reward logic:
        During episode:  0.0 (no signal at all)
        At done:
            - If car position >= 0.5 → reached goal → +1.0
            - Otherwise → timed out → -1.0

    Why check obs[0] instead of done reason?
        Gymnax MountainCar sets done=True for BOTH reaching the goal
        and timing out. The only way to distinguish is checking the
        car's position in the returned observation.

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

        # Check if car reached the goal position
        goal_reached = obs[0] >= MOUNTAINCAR_GOAL_POSITION

        # Sparse: zero during episode, +1 if goal reached at done, -1 if timed out
        sparse_reward = jnp.where(
            done,
            jnp.where(goal_reached, 1.0, -1.0),
            0.0,
        )

        return obs, next_state, sparse_reward, done, info

    def action_space(self, params):
        return self.env.action_space(params)

    def observation_space(self, params):
        return self.env.observation_space(params)


def make_mountaincar_sparse():
    """Create MountainCar-v0 with sparse reward wrapper.

    Returns:
        env:        MountainCarSparseWrapper
        env_params: Gymnax MountainCar parameters (unchanged)
    """
    env, env_params = gymnax.make("MountainCar-v0")
    sparse_env = MountainCarSparseWrapper(env)
    return sparse_env, env_params
