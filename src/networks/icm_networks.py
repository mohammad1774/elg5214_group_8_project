"""
icm_networks.py — Intrinsic Curiosity Module neural networks.

HOW ICM WORKS (the intuition):
    ICM asks: "Can I predict what happens next?"
    If a transition (s, a, s') is SURPRISING — the agent couldn't predict
    what state it would end up in — then that transition gets high intrinsic
    reward. This drives the agent to seek out unpredictable experiences.

    Unlike RND (which only looks at states), ICM looks at TRANSITIONS.
    It considers the action taken and asks "given where I was and what I did,
    could I predict where I ended up?"

THE THREE COMPONENTS:

    1. Feature Encoder: phi(s)
       obs → 64 (ReLU) → feature_dim
       Compresses raw observations into a learned feature space.
       Both current and next observations are encoded before comparison.
       This is important because comparing raw pixels/states is noisy —
       the encoder learns to ignore irrelevant details.

    2. Forward Model: predict phi(s') from (phi(s), action)
       [phi(s), one_hot(a)] → 64 (ReLU) → feature_dim
       "Given where I was and what I did, where do I expect to end up?"
       The intrinsic reward = prediction error of this model.

    3. Inverse Model: predict action from (phi(s), phi(s'))
       [phi(s), phi(s')] → 64 (ReLU) → num_actions (logits)
       "Given where I was and where I ended up, what action did I take?"
       This is an auxiliary task that helps the encoder learn useful features.
       Without it, the encoder might ignore action-relevant information.

TRAINING OBJECTIVE (from Pathak et al. 2017):
    L_icm = forward_weight * L_forward + inverse_weight * L_inverse
    where:
        L_forward = MSE(phi(s'), forward_model(phi(s), a))
        L_inverse = cross_entropy(inverse_model(phi(s), phi(s')), a)

    The paper uses forward_weight=0.2, inverse_weight=0.8.
    The inverse loss dominates because learning good features is
    more important than the forward prediction itself.

USED BY: src/exploration/icm.py (which computes rewards and updates)
         Students B (DQN+ICM) and D (PPO+ICM)
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_icm_params(key, obs_dim: int, num_actions: int = 2,
                    hidden_dim: int = 64, feature_dim: int = 32) -> Dict:
    """Initialize all three ICM networks.

    Args:
        key:         JAX random key
        obs_dim:     observation dimension
        num_actions: number of actions (for one-hot encoding in forward model)
        hidden_dim:  hidden layer size
        feature_dim: encoder output dimension

    Returns:
        Dict with three sub-dicts: "encoder", "forward", "inverse"
        ALL of these are trained (unlike RND where target is fixed).
    """
    k_enc, k_fwd, k_inv = jax.random.split(key, 3)

    # ── Feature encoder: obs → feature_dim ──
    k1, k2 = jax.random.split(k_enc)
    encoder = {
        "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, feature_dim)) * 0.1,
        "b2": jnp.zeros((feature_dim,)),
    }

    # ── Forward model: (feature_dim + num_actions) → feature_dim ──
    # Input is concatenation of phi(s) and one_hot(action)
    forward_input_dim = feature_dim + num_actions
    k1, k2 = jax.random.split(k_fwd)
    forward_model = {
        "W1": jax.random.normal(k1, (forward_input_dim, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, feature_dim)) * 0.1,
        "b2": jnp.zeros((feature_dim,)),
    }

    # ── Inverse model: (feature_dim * 2) → num_actions ──
    # Input is concatenation of phi(s) and phi(s')
    k1, k2 = jax.random.split(k_inv)
    inverse_model = {
        "W1": jax.random.normal(k1, (feature_dim * 2, hidden_dim)) * 0.1,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": jax.random.normal(k2, (hidden_dim, num_actions)) * 0.1,
        "b2": jnp.zeros((num_actions,)),
    }

    return {
        "encoder": encoder,
        "forward": forward_model,
        "inverse": inverse_model,
    }


def icm_encode(encoder_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Feature encoder: obs → phi(s).

    Compresses the raw observation into a learned feature space.
    """
    x = jnp.asarray(obs, dtype=jnp.float32)
    h = jax.nn.relu(x @ encoder_params["W1"] + encoder_params["b1"])
    phi = h @ encoder_params["W2"] + encoder_params["b2"]
    return phi


def icm_forward_model(forward_params: dict,
                      phi_s: jnp.ndarray,
                      action_onehot: jnp.ndarray) -> jnp.ndarray:
    """Forward model: predict phi(s') from (phi(s), action).

    "Given my current state features and the action I took,
     what do I expect the next state features to look like?"
    """
    x = jnp.concatenate([phi_s, action_onehot])
    h = jax.nn.relu(x @ forward_params["W1"] + forward_params["b1"])
    phi_s_next_pred = h @ forward_params["W2"] + forward_params["b2"]
    return phi_s_next_pred


def icm_inverse_model(inverse_params: dict,
                      phi_s: jnp.ndarray,
                      phi_s_next: jnp.ndarray) -> jnp.ndarray:
    """Inverse model: predict action from (phi(s), phi(s')).

    "Given where I was and where I ended up,
     which action must I have taken?"

    Returns logits — apply softmax for probabilities or
    use cross-entropy loss directly.
    """
    x = jnp.concatenate([phi_s, phi_s_next])
    h = jax.nn.relu(x @ inverse_params["W1"] + inverse_params["b1"])
    action_logits = h @ inverse_params["W2"] + inverse_params["b2"]
    return action_logits
