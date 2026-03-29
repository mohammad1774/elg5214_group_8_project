from typing import Dict

import jax
import jax.numpy as jnp


def init_icm_params(
    key: jax.Array,
    obs_dim: int,
    num_actions: int,
    feat_dim: int = 64,
    hidden_dim: int = 64,
    init_scale: float = 0.1,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Initialize ICM parameters.

    Components:
    - encoder: obs -> feature vector phi(s)
    - inverse model: [phi(s), phi(s')] -> action logits
    - forward model: [phi(s), one_hot(a)] -> predicted phi(s')
    """
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    return {
        "encoder": {
            "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * init_scale,
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "W2": jax.random.normal(k2, (hidden_dim, feat_dim)) * init_scale,
            "b2": jnp.zeros((feat_dim,), dtype=jnp.float32),
        },
        "inverse": {
            "W1": jax.random.normal(k3, (2 * feat_dim, hidden_dim)) * init_scale,
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "W2": jax.random.normal(k4, (hidden_dim, num_actions)) * init_scale,
            "b2": jnp.zeros((num_actions,), dtype=jnp.float32),
        },
        "forward": {
            "W1": jax.random.normal(k5, (feat_dim + num_actions, hidden_dim)) * init_scale,
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "W2": jax.random.normal(k6, (hidden_dim, feat_dim)) * init_scale,
            "b2": jnp.zeros((feat_dim,), dtype=jnp.float32),
        },
    }


def icm_encode(encoder_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(obs, dtype=jnp.float32)
    h = jax.nn.relu(x @ encoder_params["W1"] + encoder_params["b1"])
    return h @ encoder_params["W2"] + encoder_params["b2"]


def icm_encode_batch(encoder_params: dict, obs_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda obs: icm_encode(encoder_params, obs))(obs_batch)


def icm_inverse_logits(
    inverse_params: dict,
    feat_s: jnp.ndarray,
    feat_s_next: jnp.ndarray,
) -> jnp.ndarray:
    x = jnp.concatenate([feat_s, feat_s_next], axis=-1)
    h = jax.nn.relu(x @ inverse_params["W1"] + inverse_params["b1"])
    return h @ inverse_params["W2"] + inverse_params["b2"]


def icm_inverse_logits_batch(
    inverse_params: dict,
    feat_s_batch: jnp.ndarray,
    feat_s_next_batch: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        lambda fs, fsn: icm_inverse_logits(inverse_params, fs, fsn)
    )(feat_s_batch, feat_s_next_batch)


def icm_forward_features(
    forward_params: dict,
    feat_s: jnp.ndarray,
    action: jnp.ndarray,
    num_actions: int,
) -> jnp.ndarray:
    action_one_hot = jax.nn.one_hot(action.astype(jnp.int32), num_actions, dtype=jnp.float32)
    x = jnp.concatenate([feat_s, action_one_hot], axis=-1)
    h = jax.nn.relu(x @ forward_params["W1"] + forward_params["b1"])
    return h @ forward_params["W2"] + forward_params["b2"]


def icm_forward_features_batch(
    forward_params: dict,
    feat_s_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    num_actions: int,
) -> jnp.ndarray:
    return jax.vmap(
        lambda fs, a: icm_forward_features(forward_params, fs, a, num_actions)
    )(feat_s_batch, action_batch)


def icm_forward_all(
    icm_params: dict,
    obs_batch: jnp.ndarray,
    action_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    num_actions: int,
):
    """
    Convenience function returning all major intermediate tensors.
    """
    feat_s = icm_encode_batch(icm_params["encoder"], obs_batch)
    feat_s_next = icm_encode_batch(icm_params["encoder"], next_obs_batch)
    pred_feat_s_next = icm_forward_features_batch(
        icm_params["forward"], feat_s, action_batch, num_actions
    )
    inverse_logits = icm_inverse_logits_batch(
        icm_params["inverse"], feat_s, feat_s_next
    )

    return {
        "feat_s": feat_s,
        "feat_s_next": feat_s_next,
        "pred_feat_s_next": pred_feat_s_next,
        "inverse_logits": inverse_logits,
    }