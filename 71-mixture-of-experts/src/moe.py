"""Sparse Mixture-of-Experts routing and capacity utilities."""

from __future__ import annotations

import math
import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    if z.ndim < 1 or z.size == 0:
        raise ValueError("logits must be non-empty")
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def top_k_router(
    router_logits: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(router_logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")

    num_experts = logits.shape[1]
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must lie in [1,num_experts]")

    probabilities = stable_softmax(logits)
    indices = np.argsort(
        probabilities,
        axis=1,
        kind="stable",
    )[:, -top_k:][:, ::-1]

    selected = np.take_along_axis(
        probabilities,
        indices,
        axis=1,
    )
    selected_sum = np.sum(
        selected,
        axis=1,
        keepdims=True,
    )
    weights = selected / selected_sum

    return indices, weights, probabilities


def expert_load(
    expert_indices: np.ndarray,
    *,
    num_experts: int,
) -> np.ndarray:
    indices = np.asarray(expert_indices, dtype=int)
    if indices.ndim != 2:
        raise ValueError("expert_indices must be [tokens,top_k]")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if np.any(indices < 0) or np.any(indices >= num_experts):
        raise ValueError("expert index out of range")

    return np.bincount(
        indices.reshape(-1),
        minlength=num_experts,
    )


def expert_capacity(
    *,
    tokens: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float = 1.0,
) -> int:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if num_experts <= 0 or top_k <= 0:
        raise ValueError("expert counts must be positive")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    if tokens == 0:
        return 0

    average = tokens * top_k / num_experts
    return int(math.ceil(capacity_factor * average))


def apply_capacity(
    expert_indices: np.ndarray,
    routing_weights: np.ndarray,
    *,
    num_experts: int,
    capacity: int,
) -> np.ndarray:
    indices = np.asarray(expert_indices, dtype=int)
    weights = np.asarray(routing_weights, dtype=float)

    if indices.shape != weights.shape or indices.ndim != 2:
        raise ValueError("indices/weights must be matching 2D arrays")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    accepted = np.zeros(indices.shape, dtype=bool)

    for expert in range(num_experts):
        positions = np.argwhere(indices == expert)
        if len(positions) == 0 or capacity == 0:
            continue

        scored = []
        for token_pos, slot_pos in positions:
            scored.append(
                (
                    float(weights[token_pos, slot_pos]),
                    int(token_pos),
                    int(slot_pos),
                )
            )

        scored.sort(
            key=lambda item: (-item[0], item[1], item[2])
        )

        for _, token_pos, slot_pos in scored[:capacity]:
            accepted[token_pos, slot_pos] = True

    return accepted


def dropped_assignment_fraction(
    accepted_mask: np.ndarray,
) -> float:
    mask = np.asarray(accepted_mask, dtype=bool)
    if mask.size == 0:
        raise ValueError("mask must be non-empty")
    return float(1.0 - np.mean(mask))


def switch_load_balance_loss(
    router_probabilities: np.ndarray,
    top1_experts: np.ndarray,
) -> float:
    """Educational Switch-style auxiliary load-balancing loss.

    E * sum_i(f_i * p_i), where f_i is the top-1 assignment
    fraction and p_i is the mean router probability for expert i.
    """
    probabilities = np.asarray(router_probabilities, dtype=float)
    assignments = np.asarray(top1_experts, dtype=int).reshape(-1)

    if probabilities.ndim != 2:
        raise ValueError("router_probabilities must be [tokens,experts]")
    if len(assignments) != len(probabilities):
        raise ValueError("assignment length mismatch")

    tokens, experts = probabilities.shape
    if tokens == 0:
        raise ValueError("at least one token is required")
    if np.any(assignments < 0) or np.any(assignments >= experts):
        raise ValueError("expert index out of range")

    f = np.bincount(
        assignments,
        minlength=experts,
    ).astype(float) / tokens
    p = np.mean(probabilities, axis=0)

    return float(experts * np.sum(f * p))


def active_parameter_fraction(
    *,
    shared_parameters: int,
    expert_parameters_each: int,
    num_experts: int,
    active_experts: int,
) -> float:
    values = [
        shared_parameters,
        expert_parameters_each,
        num_experts,
        active_experts,
    ]
    if any(value < 0 for value in values):
        raise ValueError("parameter counts must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if not 1 <= active_experts <= num_experts:
        raise ValueError("active_experts out of range")

    total = (
        shared_parameters
        + expert_parameters_each * num_experts
    )
    active = (
        shared_parameters
        + expert_parameters_each * active_experts
    )
    if total <= 0:
        raise ValueError("model must have parameters")

    return float(active / total)
