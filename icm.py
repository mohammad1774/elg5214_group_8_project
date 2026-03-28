from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from src.networks.icm_networks import icm_forward_all


def icm_intrinsic_reward_batch(
    icm_params: dict,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    num_actions: int,
    eta: float = 1.0,
) -> jnp.ndarray:
    """
    Intrinsic reward = eta * 0.5 * || predicted_phi(s') - phi(s') ||^2
    returned per sample in the batch.
    """
    out = icm_forward_all(
        icm_params=icm_params,
        obs_batch=obs_batch,
        action_batch=action_batch,
        next_obs_batch=next_obs_batch,
        num_actions=num_actions,
    )

    pred_feat = out["pred_feat_s_next"]
    true_feat = jax.lax.stop_gradient(out["feat_s_next"])

    intrinsic = 0.5 * jnp.sum(jnp.square(pred_feat - true_feat), axis=1)
    return eta * intrinsic


def icm_loss(
    icm_params: dict,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    num_actions: int,
    beta: float = 0.2,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Total ICM loss:
        (1 - beta) * inverse_loss + beta * forward_loss
    """
    out = icm_forward_all(
        icm_params=icm_params,
        obs_batch=obs_batch,
        action_batch=action_batch,
        next_obs_batch=next_obs_batch,
        num_actions=num_actions,
    )

    inverse_logits = out["inverse_logits"]
    pred_feat = out["pred_feat_s_next"]
    true_feat = jax.lax.stop_gradient(out["feat_s_next"])

    inverse_loss = optax.softmax_cross_entropy_with_integer_labels(
        inverse_logits,
        action_batch.astype(jnp.int32),
    ).mean()

    forward_loss = 0.5 * jnp.sum(jnp.square(pred_feat - true_feat), axis=1).mean()

    total_loss = (1.0 - beta) * inverse_loss + beta * forward_loss

    metrics = {
        "icm_loss": total_loss,
        "inverse_loss": inverse_loss,
        "forward_loss": forward_loss,
        "intrinsic_reward_mean": (0.5 * jnp.sum(jnp.square(pred_feat - true_feat), axis=1)).mean(),
    }
    return total_loss, metrics