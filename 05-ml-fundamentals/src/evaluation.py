"""Evaluation primitives for learning ML fundamentals."""

from __future__ import annotations

import numpy as np


def _validate_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if true.size == 0:
        raise ValueError("targets must not be empty")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("targets and predictions must be finite")
    return true, pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _validate_targets(y_true, y_pred)
    residual = true - pred
    return float(np.mean(residual**2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _validate_targets(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _validate_targets(y_true, y_pred)
    total = float(np.sum((true - np.mean(true)) ** 2))
    residual = float(np.sum((true - pred) ** 2))

    if total == 0.0:
        if residual == 0.0:
            return 1.0
        raise ValueError("R2 is undefined for constant y_true with imperfect predictions")
    return 1.0 - residual / total


def mean_baseline(train_targets: np.ndarray, n_predictions: int) -> np.ndarray:
    train = np.asarray(train_targets, dtype=float).reshape(-1)
    if train.size == 0:
        raise ValueError("train_targets must not be empty")
    if n_predictions < 0:
        raise ValueError("n_predictions must be non-negative")
    return np.full(n_predictions, np.mean(train), dtype=float)


def kfold_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")
    if not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits must be between 2 and n_samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        validation = indices[start:stop]
        training = np.concatenate([indices[:start], indices[stop:]])
        folds.append((training, validation))
        current = stop

    return folds


def gradient_descent_quadratic(
    *,
    initial_x: float,
    learning_rate: float,
    steps: int,
) -> list[float]:
    """Minimize f(x)=(x-3)^2 to expose gradient descent dynamics."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    x = float(initial_x)
    history = [x]

    for _ in range(steps):
        gradient = 2.0 * (x - 3.0)
        x -= learning_rate * gradient
        history.append(x)

    return history
