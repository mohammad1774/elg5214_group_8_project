"""
rnd_networks.py — Random Network Distillation neural networks.

HOW RND WORKS (the intuition):
    Imagine you have two students in a class:
    - Student A (TARGET): was given random answers to a test and NEVER studies.
      Their answers are fixed and random forever.
    - Student B (PREDICTOR): tries to memorize Student A's random answers
      by studying states the agent visits.

    When the agent visits a state it's been to many times before,
    the predictor has seen Student A's answer for that state many times,
    so it can predict it well → LOW prediction error → LOW intrinsic reward.

    When the agent visits a NOVEL state it's never been to,
    the predictor has never seen Student A's answer for that state,
    so it can't predict it → HIGH prediction error → HIGH intrinsic reward.

    This makes the agent seek out new states = exploration!

THE TWO NETWORKS:
    Target (fixed):
        obs → 64 (ReLU) → 64 (ReLU) → embed_dim
        Random weights, NEVER updated. Acts as a "random oracle".

    Predictor (trained):
        obs → 64 (ReLU) → 64 (ReLU) → embed_dim
        Trained to match target's output. Gets better over time for
        states it has seen, stays bad for states it hasn't seen.

    Intrinsic reward = MSE(target(s), predictor(s))

WHY embed_dim=32:
    The output embedding doesn't need to be as wide as the hidden layers.
    32 dimensions is enough to create a diverse random mapping while
    keeping the MSE computation cheap. This matches common practice.

USED BY: src/exploration/rnd.py (which computes the actual rewards)
         Students A (DQN+RND) and C (PPO+RND)
"""

from typing import Dict

import jax
import jax.numpy as jnp


def init_rnd_params(key, obs_dim: int, hidden_dim: int = 64,
                    embed_dim: int = 32) -> Dict:
    """Initialize both target and predictor network parameters.

    IMPORTANT: After calling this, NEVER update rnd_params["target"].
    Only update rnd_params["predictor"] during training.

    Args:
        key:        JAX random key
        obs_dim:    observation dimension (4 for CartPole, 2 for MountainCar)
        hidden_dim: hidden layer size (64 per proposal)
        embed_dim:  output embedding dimension (32)

    Returns:
        Dict with two sub-dicts:
            "target":    fixed network params (never update these!)
            "predictor": trainable network params
    """
    k_target, k_predictor = jax.random.split(key)

    def _init_single(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "W1": jax.random.normal(k1, (obs_dim, hidden_dim)) * 0.1,
            "b1": jnp.zeros((hidden_dim,)),
            "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1,
            "b2": jnp.zeros((hidden_dim,)),
            "W3": jax.random.normal(k3, (hidden_dim, embed_dim)) * 0.1,
            "b3": jnp.zeros((embed_dim,)),
        }

    return {
        "target": _init_single(k_target),       # FIXED — never train
        "predictor": _init_single(k_predictor),  # TRAINED — update each step
    }


def rnd_forward(params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for ONE RND network (either target or predictor).

    Args:
        params: a single network's weight dict (not the full rnd_params)
        obs:    single observation

    Returns:
        Embedding vector, shape (embed_dim,)
    """
    x = jnp.asarray(obs, dtype=jnp.float32)
    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])
    embed = h2 @ params["W3"] + params["b3"]
    return embed


def rnd_target_forward(rnd_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Target network output (fixed random mapping, never trained)."""
    return rnd_forward(rnd_params["target"], obs)


def rnd_predictor_forward(rnd_params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Predictor network output (trained to match target)."""
    return rnd_forward(rnd_params["predictor"], obs)
