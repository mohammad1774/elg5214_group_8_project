"""
rnd.py — Random Network Distillation intrinsic reward and training.

This file uses the networks from rnd_networks.py to:
    1. Compute intrinsic reward for a state
    2. Update the predictor network to minimize prediction error

HOW IT FITS INTO THE TRAINING LOOP:

    For DQN+RND (Student A):
        1. Agent collects episode with lax.scan
        2. For each transition (s, a, r, s'):
           - Compute r_intrinsic = compute_rnd_reward(rnd_params, s)
           - Store r_total = r_extrinsic + beta * r_intrinsic in replay buffer
        3. Sample batch from replay buffer
        4. Update Q-network on the augmented rewards
        5. Update RND predictor on the same batch of states
        6. Repeat

    For PPO+RND (Student C):
        1. Agent collects n_steps of rollout
        2. Augment rewards: r_total = r_extrinsic + beta * r_intrinsic
        3. Compute GAE advantages using the augmented rewards
        4. Update policy on the augmented advantages
        5. Update RND predictor on the rollout states
        6. Repeat

    beta (reward scale) controls how much the agent cares about novelty
    vs the actual task reward. beta=0.1 is subtle, beta=1.0 is aggressive.

WHY WE UPDATE THE PREDICTOR:
    As the predictor gets better at matching the target for visited states,
    the intrinsic reward for those states drops. This naturally shifts
    exploration toward truly novel states. Without updating, the agent
    would keep getting rewarded for the same states forever.

USED BY: Student A (DQN+RND), Student C (PPO+RND)
"""

from functools import partial

import jax
import jax.numpy as jnp

from src.networks.rnd_networks import rnd_forward


def compute_rnd_reward(rnd_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Compute intrinsic reward for a single observation.

    r_intrinsic = mean((target(s) - predictor(s))^2)

    High when the state is novel (predictor hasn't learned it yet).
    Low when the state is familiar (predictor matches target well).

    Args:
        rnd_params: dict with "target" and "predictor" sub-dicts
        obs:        single observation

    Returns:
        Scalar intrinsic reward >= 0
    """
    target_embed = rnd_forward(rnd_params["target"], obs)
    pred_embed = rnd_forward(rnd_params["predictor"], obs)
    return jnp.mean((target_embed - pred_embed) ** 2)


def compute_rnd_reward_batch(rnd_params: dict,
                             obs_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute intrinsic reward for a batch of observations.

    Args:
        rnd_params: dict with "target" and "predictor" sub-dicts
        obs_batch:  shape (batch_size, obs_dim)

    Returns:
        Intrinsic rewards, shape (batch_size,)
    """
    return jax.vmap(lambda obs: compute_rnd_reward(rnd_params, obs))(obs_batch)


def rnd_predictor_loss(predictor_params: dict,
                       target_params: dict,
                       obs_batch: jnp.ndarray) -> jnp.ndarray:
    """MSE loss for training the predictor to match the target.

    We differentiate through predictor_params only.
    target_params are fixed (jax.lax.stop_gradient is implicit because
    we only pass predictor_params to value_and_grad).

    Args:
        predictor_params: predictor network weights (what we're training)
        target_params:    target network weights (fixed, never trained)
        obs_batch:        batch of observations to train on

    Returns:
        Scalar MSE loss
    """
    # Forward pass through both networks
    target_embeds = jax.vmap(lambda obs: rnd_forward(target_params, obs))(obs_batch)
    pred_embeds = jax.vmap(lambda obs: rnd_forward(predictor_params, obs))(obs_batch)

    return jnp.mean((target_embeds - pred_embeds) ** 2)


@jax.jit
def update_rnd_predictor(rnd_params: dict,
                         obs_batch: jnp.ndarray,
                         predictor_lr: float) -> tuple:
    """One gradient step on the RND predictor network.

    Only the predictor is updated. The target stays fixed.

    Args:
        rnd_params:    dict with "target" and "predictor"
        obs_batch:     batch of observations
        predictor_lr:  learning rate for predictor updates

    Returns:
        (updated_rnd_params, predictor_loss)

    Usage in training loop:
        rnd_params, rnd_loss = update_rnd_predictor(rnd_params, batch_obs, 0.001)
    """
    # Only differentiate w.r.t. predictor params
    loss_fn = lambda pred_p: rnd_predictor_loss(
        pred_p, rnd_params["target"], obs_batch
    )
    loss, grads = jax.value_and_grad(loss_fn)(rnd_params["predictor"])

    # Manual SGD on predictor only
    new_predictor = jax.tree_util.tree_map(
        lambda p, g: p - predictor_lr * g,
        rnd_params["predictor"],
        grads,
    )

    # Return updated params (target unchanged, predictor updated)
    updated_rnd_params = {
        "target": rnd_params["target"],
        "predictor": new_predictor,
    }

    return updated_rnd_params, loss
