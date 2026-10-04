"""Numerically stable educational RLHF/reward/PPO utilities."""

from __future__ import annotations

import numpy as np


def sigmoid_stable(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    output = np.empty_like(values)

    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    negative_values = values[~positive]
    exponential = np.exp(negative_values)
    output[~positive] = exponential / (1.0 + exponential)
    return output


def bradley_terry_probability(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
) -> np.ndarray:
    chosen = np.asarray(chosen_rewards, dtype=float)
    rejected = np.asarray(rejected_rewards, dtype=float)

    if chosen.shape != rejected.shape:
        raise ValueError("reward shapes must match")

    return sigmoid_stable(chosen - rejected)


def reward_model_loss(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
) -> float:
    chosen = np.asarray(chosen_rewards, dtype=float)
    rejected = np.asarray(rejected_rewards, dtype=float)

    if chosen.shape != rejected.shape:
        raise ValueError("reward shapes must match")
    if chosen.size == 0:
        raise ValueError("rewards must be non-empty")

    margin = chosen - rejected
    return float(np.mean(np.logaddexp(0.0, -margin)))


def pairwise_accuracy(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
) -> float:
    chosen = np.asarray(chosen_rewards, dtype=float)
    rejected = np.asarray(rejected_rewards, dtype=float)

    if chosen.shape != rejected.shape or chosen.size == 0:
        raise ValueError("matching non-empty rewards required")

    return float(np.mean(chosen > rejected))


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )


def categorical_kl(
    policy_logits: np.ndarray,
    reference_logits: np.ndarray,
) -> np.ndarray:
    policy = np.asarray(policy_logits, dtype=float)
    reference = np.asarray(reference_logits, dtype=float)

    if policy.shape != reference.shape:
        raise ValueError("logit shapes must match")
    if policy.ndim < 1:
        raise ValueError("logits need a class dimension")

    log_p = _log_softmax(policy)
    log_q = _log_softmax(reference)
    p = np.exp(log_p)

    return np.sum(p * (log_p - log_q), axis=-1)


def sampled_log_ratio(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
) -> np.ndarray:
    policy = np.asarray(policy_log_probs, dtype=float)
    reference = np.asarray(reference_log_probs, dtype=float)

    if policy.shape != reference.shape:
        raise ValueError("log-probability shapes must match")

    return policy - reference


def kl_shaped_reward(
    external_reward: np.ndarray,
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    if beta < 0:
        raise ValueError("beta must be non-negative")

    reward = np.asarray(external_reward, dtype=float)
    ratio = sampled_log_ratio(
        policy_log_probs,
        reference_log_probs,
    )

    if reward.shape != ratio.shape:
        raise ValueError("reward/log-probability shapes must match")

    return reward - beta * ratio


def normalize_advantages(
    advantages: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(advantages, dtype=float)

    if values.size == 0:
        raise ValueError("advantages must be non-empty")
    if eps <= 0:
        raise ValueError("eps must be positive")

    std = np.std(values)
    return (values - np.mean(values)) / max(std, eps)


def ppo_ratio(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
) -> np.ndarray:
    new = np.asarray(new_log_probs, dtype=float)
    old = np.asarray(old_log_probs, dtype=float)

    if new.shape != old.shape:
        raise ValueError("log-probability shapes must match")

    return np.exp(new - old)


def ppo_clipped_policy_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    *,
    clip_range: float = 0.2,
) -> float:
    if not 0.0 < clip_range < 1.0:
        raise ValueError("clip_range must lie in (0,1)")

    ratio = ppo_ratio(new_log_probs, old_log_probs)
    advantage = np.asarray(advantages, dtype=float)

    if ratio.shape != advantage.shape or ratio.size == 0:
        raise ValueError("matching non-empty arrays required")

    clipped_ratio = np.clip(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )

    surrogate = np.minimum(
        ratio * advantage,
        clipped_ratio * advantage,
    )

    return float(-np.mean(surrogate))
