"""
entropy_reg.py — Entropy regularization bonus.

WHAT IT DOES:
    Computes the entropy of a probability distribution:
        H(pi) = -sum(pi(a) * log(pi(a)))

    High entropy = policy is uncertain (spread across actions) = more exploration
    Low entropy  = policy is confident (one action dominates) = less exploration

    By ADDING entropy to the objective (or subtracting from loss), we
    discourage the policy from becoming too confident too fast.

HOW IT'S USED:
    PPO + entropy:  L_total = L_policy - alpha * H(pi)
        (subtract from loss = maximize entropy = explore more)
        The alpha coefficient controls the strength.
        alpha=0.01 is a gentle nudge, alpha=0.05 is more aggressive.

    DQN + entropy:  This is trickier because DQN doesn't have an explicit policy.
        We compute a "soft policy" from Q-values via softmax:
            pi(a|s) = softmax(Q(s,a) / temperature)
        Then compute entropy of that distribution and add it to the loss.
        This encourages the Q-values to stay spread out rather than
        converging to one dominant action too quickly.

WHY IT'S THE SIMPLEST:
    - No extra neural networks needed
    - No extra forward passes
    - Just one line of math on the existing logits/Q-values
    - But also the weakest — it doesn't actively seek novelty,
      it just slows down convergence to a deterministic policy

WHERE EACH FUNCTION IS CALLED:
    entropy_from_logits()    → PPO+entropy training loop (Student C)
    entropy_from_q_values()  → DQN+entropy training loop (Student A)

USED BY: Student A (DQN+entropy), Student C (PPO+entropy)
"""

import jax
import jax.numpy as jnp


def entropy_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a categorical distribution from logits.

    This is what PPO+entropy uses. The policy network outputs logits,
    we compute H(pi) = -sum(softmax(logits) * log_softmax(logits)).

    Args:
        logits: unnormalized log-probabilities, shape (num_actions,)

    Returns:
        Scalar entropy H(pi) >= 0.
        Maximum entropy = log(num_actions) when all actions equally likely.
        Minimum entropy = 0 when one action has probability 1.
    """
    log_probs = jax.nn.log_softmax(logits)  # numerically stable
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs)


def entropy_from_logits_batch(logits_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy for a batch of logit vectors.

    Args:
        logits_batch: shape (batch_size, num_actions)

    Returns:
        Entropy per sample, shape (batch_size,)
    """
    return jax.vmap(entropy_from_logits)(logits_batch)


def entropy_from_q_values(q_values: jnp.ndarray,
                          temperature: float = 1.0) -> jnp.ndarray:
    """Compute entropy from Q-values via Boltzmann softmax.

    This is what DQN+entropy uses. DQN doesn't have an explicit policy,
    so we CREATE one by treating Q-values as logits with a temperature:

        pi(a|s) = softmax(Q(s,a) / temperature)

    Higher temperature → flatter distribution → higher entropy.
    Lower temperature → sharper distribution → lower entropy.

    At temperature=1.0, this is equivalent to treating Q-values as logits directly.

    Args:
        q_values:    Q-values for all actions, shape (num_actions,)
        temperature: Boltzmann temperature (default 1.0)

    Returns:
        Scalar entropy of the induced soft policy.
    """
    logits = q_values / temperature
    return entropy_from_logits(logits)
