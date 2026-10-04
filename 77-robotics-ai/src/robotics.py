"""Simulation-first robotics kinematics, control and imitation-learning utilities."""

from __future__ import annotations

import math
import numpy as np


def planar_two_link_fk(
    theta1: float,
    theta2: float,
    *,
    link1: float,
    link2: float,
) -> tuple[float, float]:
    if link1 <= 0 or link2 <= 0:
        raise ValueError("link lengths must be positive")

    x = (
        link1 * math.cos(theta1)
        + link2 * math.cos(theta1 + theta2)
    )
    y = (
        link1 * math.sin(theta1)
        + link2 * math.sin(theta1 + theta2)
    )
    return float(x), float(y)


def clamp_action(
    action: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    u = np.asarray(action, dtype=float)
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)

    if not (u.shape == lo.shape == hi.shape):
        raise ValueError("action/limits shape mismatch")
    if np.any(lo > hi):
        raise ValueError("low must be <= high")

    return np.clip(u, lo, hi)


def proportional_control(
    target: np.ndarray,
    current: np.ndarray,
    *,
    gain: float,
    max_abs_action: float | None = None,
) -> np.ndarray:
    if gain < 0:
        raise ValueError("gain must be non-negative")

    target_arr = np.asarray(target, dtype=float)
    current_arr = np.asarray(current, dtype=float)

    if target_arr.shape != current_arr.shape:
        raise ValueError("target/current shape mismatch")

    action = gain * (target_arr - current_arr)

    if max_abs_action is not None:
        if max_abs_action <= 0:
            raise ValueError("max_abs_action must be positive")
        action = np.clip(
            action,
            -max_abs_action,
            max_abs_action,
        )

    return action


def behavior_cloning_mse(
    predicted_actions: np.ndarray,
    expert_actions: np.ndarray,
) -> float:
    pred = np.asarray(predicted_actions, dtype=float)
    expert = np.asarray(expert_actions, dtype=float)

    if pred.shape != expert.shape or pred.size == 0:
        raise ValueError("matching non-empty actions required")

    return float(np.mean((pred - expert) ** 2))


def action_chunk(
    actions: np.ndarray,
    *,
    start: int,
    chunk_size: int,
) -> np.ndarray:
    sequence = np.asarray(actions, dtype=float)

    if sequence.ndim != 2:
        raise ValueError("actions must be [time,action_dim]")
    if start < 0:
        raise ValueError("start must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    return sequence[start : start + chunk_size].copy()


def trajectory_path_length(
    positions: np.ndarray,
) -> float:
    points = np.asarray(positions, dtype=float)

    if points.ndim != 2 or len(points) == 0:
        raise ValueError("positions must be non-empty [time,dim]")
    if len(points) == 1:
        return 0.0

    deltas = np.diff(points, axis=0)
    return float(
        np.sum(np.linalg.norm(deltas, axis=1))
    )


def action_smoothness(
    actions: np.ndarray,
) -> float:
    values = np.asarray(actions, dtype=float)

    if values.ndim != 2 or len(values) == 0:
        raise ValueError("actions must be non-empty [time,dim]")
    if len(values) == 1:
        return 0.0

    deltas = np.diff(values, axis=0)
    return float(np.mean(np.linalg.norm(deltas, axis=1)))


def goal_success(
    final_position: np.ndarray,
    goal_position: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    final = np.asarray(final_position, dtype=float)
    goal = np.asarray(goal_position, dtype=float)

    if final.shape != goal.shape:
        raise ValueError("position shape mismatch")

    return bool(
        np.linalg.norm(final - goal) <= tolerance
    )


def control_period_ms(
    *,
    frequency_hz: float,
) -> float:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return float(1000.0 / frequency_hz)


def latency_budget_ok(
    *,
    observation_ms: float,
    inference_ms: float,
    action_dispatch_ms: float,
    control_frequency_hz: float,
) -> bool:
    values = [
        observation_ms,
        inference_ms,
        action_dispatch_ms,
    ]
    if any(value < 0 for value in values):
        raise ValueError("latencies must be non-negative")

    period = control_period_ms(
        frequency_hz=control_frequency_hz,
    )
    return sum(values) <= period
