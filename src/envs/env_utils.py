"""
env_utils.py — Environment factory.

This is the ONLY file teammates need to call. It imports from
cartpole_env.py and mountaincar_env.py and provides one clean interface:

    env, env_params, obs_dim, num_actions = get_env("cartpole", "sparse")

WHY THIS FILE EXISTS:
    - cartpole_env.py defines CartPole-specific logic (dense + sparse)
    - mountaincar_env.py defines MountainCar-specific logic (dense + sparse)
    - This file is the router: you tell it WHAT you want, it gives you
      the right env without you importing anything else.

    This means a test script like test_dqn_rnd_agent.py just does:
        from src.envs.env_utils import get_env
        env, env_params, obs_dim, num_actions = get_env(env_name, reward_type)

    And it works for all 4 combinations without any if/else logic in the test file.

USED BY: Every test_*.py file in src/test/
"""

from src.envs.cartpole_env import (
    make_cartpole_dense,
    make_cartpole_sparse,
    CARTPOLE_OBS_DIM,
    CARTPOLE_NUM_ACTIONS,
)

from src.envs.mountaincar_env import (
    make_mountaincar_dense,
    make_mountaincar_sparse,
    MOUNTAINCAR_OBS_DIM,
    MOUNTAINCAR_NUM_ACTIONS,
)


def get_env(env_name: str, reward_type: str = "dense"):
    """Create an environment for the given name and reward variant.

    Args:
        env_name:    "cartpole" or "mountaincar"
        reward_type: "dense" (original Gymnax) or "sparse" (binary end-of-episode)

    Returns:
        env:         Gymnax environment or sparse wrapper
        env_params:  Gymnax environment parameters
        obs_dim:     int — observation vector size (4 or 2)
        num_actions: int — number of discrete actions (2 or 3)

    Example:
        # In your test script:
        env, env_params, obs_dim, num_actions = get_env("cartpole", "sparse")

        # Initialize network with the right dimensions:
        q_params = init_q_params(key, obs_dim=obs_dim, num_actions=num_actions)

        # Run the env:
        obs, state = env.reset_env(key, env_params)
        obs, state, reward, done, info = env.step_env(key, state, action, env_params)
    """

    if env_name == "cartpole":
        if reward_type == "dense":
            env, env_params = make_cartpole_dense()
        elif reward_type == "sparse":
            env, env_params = make_cartpole_sparse()
        else:
            raise ValueError(f"Unknown reward_type: '{reward_type}'")
        return env, env_params, CARTPOLE_OBS_DIM, CARTPOLE_NUM_ACTIONS

    elif env_name == "mountaincar":
        if reward_type == "dense":
            env, env_params = make_mountaincar_dense()
        elif reward_type == "sparse":
            env, env_params = make_mountaincar_sparse()
        else:
            raise ValueError(f"Unknown reward_type: '{reward_type}'")
        return env, env_params, MOUNTAINCAR_OBS_DIM, MOUNTAINCAR_NUM_ACTIONS

    else:
        raise ValueError(
            f"Unknown env_name: '{env_name}'. Choose 'cartpole' or 'mountaincar'"
        )
