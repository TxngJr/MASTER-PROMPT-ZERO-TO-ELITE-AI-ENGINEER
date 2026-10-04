"""Reinforcement-learning mathematical primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def discounted_returns(
    rewards: list[float] | np.ndarray,
    gamma: float,
) -> np.ndarray:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")

    r = np.asarray(rewards, dtype=float)
    returns = np.zeros_like(r)
    running = 0.0

    for index in range(len(r) - 1, -1, -1):
        running = r[index] + gamma * running
        returns[index] = running

    return returns


def bellman_optimality_backup(
    reward: float,
    next_q_values: np.ndarray,
    *,
    gamma: float,
    done: bool,
) -> float:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")

    next_q = np.asarray(next_q_values, dtype=float)

    if done:
        return float(reward)

    if next_q.ndim != 1 or len(next_q) == 0:
        raise ValueError("next_q_values must be a non-empty vector")

    return float(reward + gamma * np.max(next_q))


def q_learning_update(
    q_value: float,
    reward: float,
    next_q_values: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    done: bool,
) -> tuple[float, float]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0,1]")

    target = bellman_optimality_backup(
        reward,
        next_q_values,
        gamma=gamma,
        done=done,
    )
    td_error = target - q_value
    updated = q_value + alpha * td_error
    return float(updated), float(td_error)


def epsilon_greedy(
    q_values: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    q = np.asarray(q_values, dtype=float)

    if q.ndim != 1 or len(q) == 0:
        raise ValueError("q_values must be a non-empty vector")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")

    if rng.random() < epsilon:
        return int(rng.integers(0, len(q)))

    maxima = np.flatnonzero(q == np.max(q))
    return int(rng.choice(maxima))


def ppo_clipped_surrogate(
    new_log_prob: np.ndarray,
    old_log_prob: np.ndarray,
    advantage: np.ndarray,
    *,
    clip_epsilon: float = 0.2,
) -> np.ndarray:
    new_lp = np.asarray(new_log_prob, dtype=float)
    old_lp = np.asarray(old_log_prob, dtype=float)
    adv = np.asarray(advantage, dtype=float)

    if new_lp.shape != old_lp.shape or new_lp.shape != adv.shape:
        raise ValueError("inputs must have equal shapes")
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")

    ratio = np.exp(new_lp - old_lp)
    clipped = np.clip(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )

    return np.minimum(
        ratio * adv,
        clipped * adv,
    )
