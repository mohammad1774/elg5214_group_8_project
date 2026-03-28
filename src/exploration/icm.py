"""
icm.py — Intrinsic Curiosity Module reward and training logic.

This file uses the networks from icm_networks.py to:
    1. Compute intrinsic reward for a transition (s, a, s')
    2. Update ALL three ICM networks (encoder, forward, inverse)

HOW IT FITS INTO THE TRAINING LOOP:

    For DQN+ICM (Student B):
        1. Agent collects episode with lax.scan
        2. For each transition (s, a, r, s'):
           - Compute r_intrinsic = compute_icm_reward(icm_params, s, a, s')
           - Store r_total = r_extrinsic + eta * r_intrinsic in replay buffer
        3. Sample batch from replay buffer
        4. Update Q-network on the augmented rewards
        5. Update ICM networks on the same batch: (obs, actions, next_obs)
        6. Repeat

    For PPO+ICM (Student D):
        1. Agent collects n_steps of rollout → (obs, actions, rewards, next_obs)
        2. Augment rewards: r_total = r_extrinsic + eta * r_intrinsic
        3. Compute GAE advantages using augmented rewards
        4. Update policy on augmented advantages
        5. Update ICM networks on the rollout transitions
        6. Repeat

KEY DIFFERENCE FROM RND:
    RND computes reward from state only:     r_i = f(s)
    ICM computes reward from transition:     r_i = f(s, a, s')

    This means ICM considers the ACTION taken. A transition that's
    surprising given the action (e.g., pushing right but ending up left)
    gets more reward. RND would give the same reward regardless of action.

    ICM's inverse model also helps learn features that capture
    action-relevant dynamics, which can be more useful for exploration
    than RND's arbitrary random features.

    eta (curiosity scale) controls how much the agent cares about
    curiosity vs the actual task reward. eta=0.1 is subtle, eta=1.0 is aggressive.

USED BY: Student B (DQN+ICM), Student D (PPO+ICM)
"""

from functools import partial

import jax
import jax.numpy as jnp

from src.networks.icm_networks import icm_encode, icm_forward_model, icm_inverse_model


def compute_icm_reward(icm_params: dict,
                       obs: jnp.ndarray,
                       action: jnp.ndarray,
                       next_obs: jnp.ndarray,
                       num_actions: int = 2) -> jnp.ndarray:
    """Compute ICM intrinsic reward for a single transition.

    r_intrinsic = MSE(phi(s'), forward_model(phi(s), one_hot(a)))

    "How surprised am I by what happened after I took this action?"

    Args:
        icm_params:  dict with "encoder", "forward", "inverse"
        obs:         current observation s
        action:      action taken a (integer)
        next_obs:    next observation s'
        num_actions: action space size (for one-hot encoding)

    Returns:
        Scalar intrinsic reward >= 0
    """
    # Encode both states into feature space
    phi_s = icm_encode(icm_params["encoder"], obs)
    phi_s_next = icm_encode(icm_params["encoder"], next_obs)

    # Predict next state features from (current features, action)
    action_onehot = jax.nn.one_hot(action, num_actions)
    phi_s_next_pred = icm_forward_model(icm_params["forward"], phi_s, action_onehot)

    # Intrinsic reward = forward prediction error
    return jnp.mean((phi_s_next - phi_s_next_pred) ** 2)


def compute_icm_reward_batch(icm_params: dict,
                             obs_batch: jnp.ndarray,
                             actions_batch: jnp.ndarray,
                             next_obs_batch: jnp.ndarray,
                             num_actions: int = 2) -> jnp.ndarray:
    """Compute ICM intrinsic reward for a batch of transitions.

    Args:
        icm_params:     dict with "encoder", "forward", "inverse"
        obs_batch:      shape (batch_size, obs_dim)
        actions_batch:  shape (batch_size,) integers
        next_obs_batch: shape (batch_size, obs_dim)
        num_actions:    action space size

    Returns:
        Intrinsic rewards, shape (batch_size,)
    """
    return jax.vmap(
        lambda o, a, no: compute_icm_reward(icm_params, o, a, no, num_actions)
    )(obs_batch, actions_batch, next_obs_batch)


def icm_loss(icm_params: dict,
             obs_batch: jnp.ndarray,
             actions_batch: jnp.ndarray,
             next_obs_batch: jnp.ndarray,
             num_actions: int = 2,
             forward_weight: float = 0.2,
             inverse_weight: float = 0.8) -> jnp.ndarray:
    """Combined ICM training loss.

    L = forward_weight * L_forward + inverse_weight * L_inverse

    L_forward: MSE between predicted and actual next-state features.
               "Can I predict what happens next?"

    L_inverse: Cross-entropy loss for predicting the action.
               "Can I tell what action was taken from the state pair?"

    The inverse loss is weighted higher (0.8 vs 0.2) because learning
    good features is more important than the forward prediction itself.
    The encoder is trained through BOTH losses — inverse loss ensures
    the features capture action-relevant information.

    Args:
        icm_params:     dict with "encoder", "forward", "inverse"
        obs_batch:      shape (batch_size, obs_dim)
        actions_batch:  shape (batch_size,) integers
        next_obs_batch: shape (batch_size, obs_dim)
        num_actions:    action space size
        forward_weight: weight for forward model loss (default 0.2)
        inverse_weight: weight for inverse model loss (default 0.8)

    Returns:
        Scalar combined loss
    """
    def single_loss(obs, action, next_obs):
        # Encode both states
        phi_s = icm_encode(icm_params["encoder"], obs)
        phi_s_next = icm_encode(icm_params["encoder"], next_obs)

        # Forward loss: predict phi(s') from (phi(s), action)
        action_onehot = jax.nn.one_hot(action, num_actions)
        phi_pred = icm_forward_model(icm_params["forward"], phi_s, action_onehot)
        forward_loss = jnp.mean((phi_s_next - phi_pred) ** 2)

        # Inverse loss: predict action from (phi(s), phi(s'))
        action_logits = icm_inverse_model(icm_params["inverse"], phi_s, phi_s_next)
        inverse_loss = -jax.nn.log_softmax(action_logits)[action]

        return forward_weight * forward_loss + inverse_weight * inverse_loss

    losses = jax.vmap(single_loss)(obs_batch, actions_batch, next_obs_batch)
    return jnp.mean(losses)


@partial(jax.jit, static_argnames=("num_actions",))
def update_icm(icm_params: dict,
               obs_batch: jnp.ndarray,
               actions_batch: jnp.ndarray,
               next_obs_batch: jnp.ndarray,
               icm_lr: float,
               num_actions: int = 2,
               forward_weight: float = 0.2,
               inverse_weight: float = 0.8) -> tuple:
    """One gradient step on ALL ICM networks.

    Unlike RND (where only the predictor is updated), ICM updates
    all three components: encoder, forward model, and inverse model.
    The gradient flows through all of them.

    Args:
        icm_params:     dict with "encoder", "forward", "inverse"
        obs_batch:      shape (batch_size, obs_dim)
        actions_batch:  shape (batch_size,) integers
        next_obs_batch: shape (batch_size, obs_dim)
        icm_lr:         learning rate
        num_actions:    action space size
        forward_weight: weight for forward loss
        inverse_weight: weight for inverse loss

    Returns:
        (updated_icm_params, icm_loss_value)

    Usage in training loop:
        icm_params, icm_loss = update_icm(
            icm_params, batch_obs, batch_actions, batch_next_obs,
            icm_lr=0.001, num_actions=2
        )
    """
    loss_fn = lambda p: icm_loss(
        p, obs_batch, actions_batch, next_obs_batch,
        num_actions, forward_weight, inverse_weight
    )
    loss, grads = jax.value_and_grad(loss_fn)(icm_params)

    # Manual SGD on all three sub-networks
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - icm_lr * g,
        icm_params,
        grads,
    )

    return new_params, loss
