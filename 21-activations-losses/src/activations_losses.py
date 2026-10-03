"""Stable activation and loss functions with analytic gradients."""

from __future__ import annotations

import numpy as np


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.where(values >= 0.0, values, alpha * values)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.where(values > 0.0, values, alpha * np.expm1(values))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * values * (
        1.0
        + np.tanh(
            coefficient * (values + 0.044715 * values**3)
        )
    )


def silu(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return values * stable_sigmoid(values)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def mse_loss_and_grad(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("prediction and target must have equal non-empty shape")
    residual = p - y
    return float(np.mean(residual**2)), 2.0 * residual / residual.size


def mae_loss_and_grad(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("prediction and target must have equal non-empty shape")
    residual = p - y
    # Convention: subgradient at zero is 0.
    return float(np.mean(np.abs(residual))), np.sign(residual) / residual.size


def huber_loss_and_grad(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    delta: float = 1.0,
) -> tuple[float, np.ndarray]:
    if delta <= 0:
        raise ValueError("delta must be positive")

    p = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("prediction and target must have equal non-empty shape")

    residual = p - y
    absolute = np.abs(residual)
    quadratic = absolute <= delta

    element_loss = np.where(
        quadratic,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )
    element_grad = np.where(
        quadratic,
        residual,
        delta * np.sign(residual),
    )

    return float(np.mean(element_loss)), element_grad / residual.size


def bce_with_logits_loss_and_grad(
    logits: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    x = np.asarray(logits, dtype=float)
    y = np.asarray(target, dtype=float)

    if x.shape != y.shape or x.size == 0:
        raise ValueError("logits and target must have equal non-empty shape")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("binary targets must be 0/1")

    element_loss = (
        np.maximum(x, 0.0)
        - x * y
        + np.log1p(np.exp(-np.abs(x)))
    )
    gradient = (stable_sigmoid(x) - y) / x.size

    return float(np.mean(element_loss)), gradient


def cross_entropy_with_logits_loss_and_grad(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    label_smoothing: float = 0.0,
) -> tuple[float, np.ndarray]:
    z = np.asarray(logits, dtype=float)
    labels = np.asarray(target, dtype=int).reshape(-1)

    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError("logits must have shape (N,C) with C>=2")
    if labels.shape[0] != z.shape[0]:
        raise ValueError("target length must match batch")
    if np.any(labels < 0) or np.any(labels >= z.shape[1]):
        raise ValueError("target class index out of range")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0,1)")

    shifted = z - np.max(z, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_prob = shifted - logsumexp
    probabilities = np.exp(log_prob)

    n_samples, n_classes = z.shape
    target_distribution = np.full(
        (n_samples, n_classes),
        label_smoothing / n_classes,
        dtype=float,
    )
    target_distribution[np.arange(n_samples), labels] += 1.0 - label_smoothing

    loss = -np.sum(target_distribution * log_prob) / n_samples
    gradient = (probabilities - target_distribution) / n_samples

    return float(loss), gradient
