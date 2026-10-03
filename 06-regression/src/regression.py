"""Regression algorithms implemented for learning, not production speed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_X(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"X must be non-empty 2D data, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values")
    return array


def _as_y(y: np.ndarray, n_samples: int) -> np.ndarray:
    array = np.asarray(y, dtype=float).reshape(-1)
    if array.shape[0] != n_samples:
        raise ValueError("X and y have different sample counts")
    if not np.all(np.isfinite(array)):
        raise ValueError("y must contain only finite values")
    return array


def _augment_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


@dataclass
class LinearRegressionClosedForm:
    intercept_: float | None = None
    coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionClosedForm":
        X_array = _as_X(X)
        y_array = _as_y(y, X_array.shape[0])
        design = _augment_intercept(X_array)

        theta, *_ = np.linalg.lstsq(design, y_array, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("model must be fit before predict")
        X_array = _as_X(X)
        if X_array.shape[1] != self.coef_.shape[0]:
            raise ValueError("feature count differs from fitted model")
        return X_array @ self.coef_ + self.intercept_


@dataclass
class RidgeRegressionClosedForm:
    alpha: float = 1.0
    intercept_: float | None = None
    coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionClosedForm":
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        X_array = _as_X(X)
        y_array = _as_y(y, X_array.shape[0])
        design = _augment_intercept(X_array)

        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0

        system = design.T @ design + self.alpha * penalty
        rhs = design.T @ y_array
        theta = np.linalg.solve(system, rhs)

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("model must be fit before predict")
        X_array = _as_X(X)
        return X_array @ self.coef_ + self.intercept_


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.05
    steps: int = 2_000
    l2: float = 0.0
    intercept_: float | None = None
    coef_: np.ndarray | None = None
    loss_history_: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")

        X_array = _as_X(X)
        y_array = _as_y(y, X_array.shape[0])
        n_samples, n_features = X_array.shape

        w = np.zeros(n_features, dtype=float)
        b = 0.0
        self.loss_history_ = []

        for _ in range(self.steps):
            prediction = X_array @ w + b
            error = prediction - y_array

            data_loss = float(np.mean(error**2))
            objective = data_loss + self.l2 * float(w @ w)
            self.loss_history_.append(objective)

            grad_w = (2.0 / n_samples) * (X_array.T @ error) + 2.0 * self.l2 * w
            grad_b = 2.0 * float(np.mean(error))

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

            if not np.all(np.isfinite(w)) or not np.isfinite(b):
                raise FloatingPointError(
                    "gradient descent diverged; scale data or reduce learning_rate"
                )

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("model must be fit before predict")
        X_array = _as_X(X)
        return X_array @ self.coef_ + self.intercept_


@dataclass
class LassoRegressionSubgradient:
    alpha: float = 0.01
    learning_rate: float = 0.01
    steps: int = 5_000
    intercept_: float | None = None
    coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressionSubgradient":
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.learning_rate <= 0 or self.steps <= 0:
            raise ValueError("learning_rate and steps must be positive")

        X_array = _as_X(X)
        y_array = _as_y(y, X_array.shape[0])
        n_samples, n_features = X_array.shape

        w = np.zeros(n_features, dtype=float)
        b = 0.0

        for _ in range(self.steps):
            error = X_array @ w + b - y_array
            grad_w = (1.0 / n_samples) * (X_array.T @ error)
            grad_w += self.alpha * np.sign(w)
            grad_b = float(np.mean(error))

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("model must be fit before predict")
        X_array = _as_X(X)
        return X_array @ self.coef_ + self.intercept_


def polynomial_features_1d(
    x: np.ndarray,
    degree: int,
    *,
    include_bias: bool = False,
) -> np.ndarray:
    if degree < 1:
        raise ValueError("degree must be >= 1")

    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("x must not be empty")

    powers = range(0 if include_bias else 1, degree + 1)
    return np.column_stack([values**power for power in powers])
