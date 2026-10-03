"""Binary logistic regression from scratch using NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    output = np.empty_like(values)

    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def binary_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)

    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must have equal non-empty shape")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("binary targets must contain only 0 and 1")
    if eps <= 0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    clipped = np.clip(p, eps, 1.0 - eps)
    return float(
        -np.mean(
            y * np.log(clipped)
            + (1.0 - y) * np.log(1.0 - clipped)
        )
    )


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.1
    steps: int = 2_000
    l2: float = 0.0
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    loss_history_: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must be finite")
        if not np.all(np.isin(y_array, [0.0, 1.0])):
            raise ValueError("y must contain only 0 and 1")
        if self.learning_rate <= 0 or self.steps <= 0:
            raise ValueError("learning_rate and steps must be positive")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")

        n_samples, n_features = X_array.shape
        w = np.zeros(n_features, dtype=float)
        b = 0.0
        self.loss_history_ = []

        for _ in range(self.steps):
            logits = X_array @ w + b
            probabilities = sigmoid(logits)
            error = probabilities - y_array

            loss = binary_log_loss(y_array, probabilities)
            loss += self.l2 * float(w @ w)
            self.loss_history_.append(loss)

            grad_w = (X_array.T @ error) / n_samples + 2.0 * self.l2 * w
            grad_b = float(np.mean(error))

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

            if not np.all(np.isfinite(w)) or not np.isfinite(b):
                raise FloatingPointError("optimization diverged")

        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.coef_.shape[0]:
            raise ValueError("X feature shape differs from fitted model")
        return X_array @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
