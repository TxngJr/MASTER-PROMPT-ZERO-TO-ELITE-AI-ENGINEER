"""Educational pruning and knowledge-distillation utilities."""

from __future__ import annotations

import numpy as np


def sparsity(values: np.ndarray) -> float:
    x = np.asarray(values)
    if x.size == 0:
        raise ValueError("values must be non-empty")
    return float(np.count_nonzero(x == 0) / x.size)


def nonzero_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(values)))


def magnitude_prune(
    values: np.ndarray,
    *,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0,1]")

    x = np.asarray(values, dtype=float)
    flat = x.reshape(-1)

    prune_count = int(np.floor(flat.size * fraction))
    mask = np.ones(flat.size, dtype=bool)

    if prune_count > 0:
        order = np.argsort(np.abs(flat), kind="stable")
        mask[order[:prune_count]] = False

    pruned = flat.copy()
    pruned[~mask] = 0.0

    return pruned.reshape(x.shape), mask.reshape(x.shape)


def structured_row_prune(
    matrix: np.ndarray,
    *,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("matrix must be 2D")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0,1]")

    rows = x.shape[0]
    prune_count = int(np.floor(rows * fraction))
    norms = np.linalg.norm(x, axis=1)

    keep = np.ones(rows, dtype=bool)
    if prune_count > 0:
        order = np.argsort(norms, kind="stable")
        keep[order[:prune_count]] = False

    return x[keep].copy(), keep


def softmax_temperature(
    logits: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = np.asarray(logits, dtype=float) / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def kl_divergence(
    teacher_prob: np.ndarray,
    student_prob: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    p = np.asarray(teacher_prob, dtype=float)
    q = np.asarray(student_prob, dtype=float)

    if p.shape != q.shape or p.size == 0:
        raise ValueError("matching non-empty probabilities required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    return float(np.mean(np.sum(p * np.log(p / q), axis=-1)))


def distillation_loss(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    *,
    temperature: float = 2.0,
) -> float:
    teacher_prob = softmax_temperature(
        teacher_logits,
        temperature=temperature,
    )
    student_prob = softmax_temperature(
        student_logits,
        temperature=temperature,
    )
    return float(
        temperature**2
        * kl_divergence(teacher_prob, student_prob)
    )


def compression_ratio(
    *,
    original_bytes: int,
    compressed_bytes: int,
) -> float:
    if original_bytes <= 0 or compressed_bytes <= 0:
        raise ValueError("sizes must be positive")
    return float(original_bytes / compressed_bytes)
