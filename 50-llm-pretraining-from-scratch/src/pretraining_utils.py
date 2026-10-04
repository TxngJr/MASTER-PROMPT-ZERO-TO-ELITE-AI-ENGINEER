"""Framework-independent utilities for understanding LLM pretraining."""

from __future__ import annotations

import math
import numpy as np


def make_causal_windows(
    token_ids: list[int] | np.ndarray,
    *,
    sequence_length: int,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride is None:
        stride = sequence_length
    if stride <= 0:
        raise ValueError("stride must be positive")

    needed = sequence_length + 1
    inputs = []
    targets = []

    for start in range(0, len(tokens) - needed + 1, stride):
        window = tokens[start : start + needed]
        inputs.append(window[:-1])
        targets.append(window[1:])

    if not inputs:
        return (
            np.empty((0, sequence_length), dtype=np.int64),
            np.empty((0, sequence_length), dtype=np.int64),
        )

    return np.stack(inputs), np.stack(targets)


def stable_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=np.int64)

    if scores.ndim != y.ndim + 1:
        raise ValueError("logits must add one vocabulary dimension")
    if scores.shape[:-1] != y.shape:
        raise ValueError("logit/target shape mismatch")
    if np.any((y < 0) | (y >= scores.shape[-1])):
        raise ValueError("target out of range")

    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_targets = y.reshape(-1)

    maximum = np.max(flat_scores, axis=1, keepdims=True)
    logsumexp = (
        maximum[:, 0]
        + np.log(np.sum(np.exp(flat_scores - maximum), axis=1))
    )
    correct = flat_scores[
        np.arange(len(flat_targets)),
        flat_targets,
    ]

    return float(np.mean(logsumexp - correct))


def perplexity_from_loss(loss: float) -> float:
    if not np.isfinite(loss):
        raise ValueError("loss must be finite")
    if loss < 0:
        raise ValueError("cross-entropy loss cannot be negative")
    return float(math.exp(loss))


def warmup_cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("invalid step counts")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    if not 0.0 <= min_lr <= max_lr:
        raise ValueError("require 0 <= min_lr <= max_lr")

    if step < warmup_steps:
        if warmup_steps == 0:
            return float(max_lr)
        return float(max_lr * (step + 1) / warmup_steps)

    if step >= total_steps - 1:
        return float(min_lr)

    progress = (
        step - warmup_steps
    ) / max(1, total_steps - warmup_steps - 1)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        min_lr + (max_lr - min_lr) * cosine
    )


def effective_tokens_per_update(
    *,
    micro_batch_size: int,
    accumulation_steps: int,
    devices: int,
    sequence_length: int,
) -> int:
    values = [
        micro_batch_size,
        accumulation_steps,
        devices,
        sequence_length,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("all values must be positive")

    return int(np.prod(values))


def global_norm(
    gradients: list[np.ndarray],
) -> float:
    if not gradients:
        return 0.0

    squared = sum(
        float(np.sum(np.asarray(gradient, dtype=float) ** 2))
        for gradient in gradients
    )
    return float(math.sqrt(squared))


def clip_by_global_norm(
    gradients: list[np.ndarray],
    *,
    max_norm: float,
) -> tuple[list[np.ndarray], float]:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    norm = global_norm(gradients)

    if norm == 0.0 or norm <= max_norm:
        return [
            np.asarray(gradient, dtype=float).copy()
            for gradient in gradients
        ], norm

    scale = max_norm / norm

    return [
        np.asarray(gradient, dtype=float) * scale
        for gradient in gradients
    ], norm
