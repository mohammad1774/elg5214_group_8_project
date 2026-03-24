"""
ICM Exploration Module.

Computes curiosity-driven intrinsic reward as forward model prediction error:
    r_intrinsic = η * ||f_forward(φ(s), a) - φ(s')||²

Also provides a combined loss for training the ICM networks:
    L_icm = (1 - β_icm) * L_inverse + β_icm * L_forward

where L_inverse trains the encoder to produce action-relevant features,
and L_forward trains the forward model to predict next-state features.

Reference: Pathak et al. (2017) "Curiosity-driven Exploration by Self-Supervised Prediction"
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple

from src.networks.icm_networks import icm_forward_batch


def icm_intrinsic_reward(
    params: Dict,
    obs: jnp.ndarray,
    action: int,
    next_obs: jnp.ndarray,
    act_dim: int,
    eta: float = 1.0,
) -> float:
    """Compute ICM intrinsic reward for a single transition.

    Args:
        params: ICM network parameters.
        obs: Current observation, shape (obs_dim,).
        action: Action taken (integer).
        next_obs: Next observation, shape (obs_dim,).
        act_dim: Number of discrete actions.
        eta: Curiosity reward scale (η).

    Returns:
        Scalar intrinsic reward.
    """
    from src.networks.icm_networks import encode, forward_model

    phi_s = encode(params, obs)
    phi_s_next = encode(params, next_obs)
    pred_phi_s_next = forward_model(params, phi_s, action, act_dim)

    # Forward prediction error = intrinsic reward
    forward_error = jnp.sum((pred_phi_s_next - phi_s_next) ** 2)
    return eta * forward_error


def icm_loss_fn(
    params: Dict,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    act_dim: int,
    beta_icm: float = 0.2,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute combined ICM loss for a batch.

    Args:
        params: ICM network parameters.
        obs_batch: Observations, shape (batch, obs_dim).
        action_batch: Actions, shape (batch,) integer.
        next_obs_batch: Next observations, shape (batch, obs_dim).
        act_dim: Number of discrete actions.
        beta_icm: Weight for forward vs inverse loss (higher = more forward model emphasis).

    Returns:
        (total_loss, info_dict) where info_dict has forward_loss and inverse_loss.
    """
    phi_s, phi_s_next, pred_phi_s_next, action_logits = icm_forward_batch(
        params, obs_batch, action_batch, next_obs_batch, act_dim
    )

    # Forward loss: prediction error in feature space
    forward_loss = jnp.mean(jnp.sum((pred_phi_s_next - phi_s_next) ** 2, axis=-1))

    # Inverse loss: cross-entropy on action prediction
    inverse_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(action_logits, action_batch)
    )

    total_loss = beta_icm * forward_loss + (1.0 - beta_icm) * inverse_loss

    return total_loss, {
        "forward_loss": forward_loss,
        "inverse_loss": inverse_loss,
        "total_loss": total_loss,
    }


def update_icm(
    icm_params: Dict,
    icm_opt_state: optax.OptState,
    icm_optimizer: optax.GradientTransformation,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    act_dim: int,
    beta_icm: float = 0.2,
) -> Tuple[Dict, optax.OptState, Dict]:
    """Single ICM update step.

    Returns:
        (updated_icm_params, updated_opt_state, loss_info)
    """
    (loss, info), grads = jax.value_and_grad(icm_loss_fn, has_aux=True)(
        icm_params, obs_batch, action_batch, next_obs_batch, act_dim, beta_icm
    )

    updates, icm_opt_state = icm_optimizer.update(grads, icm_opt_state, icm_params)
    icm_params = optax.apply_updates(icm_params, updates)

    return icm_params, icm_opt_state, info
