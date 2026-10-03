"""Educational binary linear SVM using primal subgradient descent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hinge_loss(
    y_signed: np.ndarray,
    scores: np.ndarray,
) -> float:
    y = np.asarray(y_signed, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.shape != s.shape or y.size == 0:
        raise ValueError("inputs must have equal non-empty shape")
    if not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("labels must be -1/+1")
    return float(np.mean(np.maximum(0.0, 1.0 - y * s)))


@dataclass
class LinearSVMFromScratch:
    C: float = 1.0
    learning_rate: float = 0.02
    steps: int = 3_000
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    classes_: np.ndarray | None = None
    objective_history_: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be non-empty 2D")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X/y sample counts differ")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must be finite")
        if self.C <= 0 or self.learning_rate <= 0 or self.steps <= 0:
            raise ValueError("C, learning_rate and steps must be positive")

        classes = np.unique(y_array)
        if classes.size != 2:
            raise ValueError("educational implementation supports binary labels")

        self.classes_ = classes
        signed = np.where(y_array == classes[0], -1.0, 1.0)

        n_samples, n_features = X_array.shape
        w = np.zeros(n_features, dtype=float)
        b = 0.0
        self.objective_history_ = []

        for step in range(self.steps):
            scores = X_array @ w + b
            margins = signed * scores
            active = margins < 1.0

            grad_w = w.copy()
            grad_b = 0.0

            if np.any(active):
                grad_w -= (
                    self.C / n_samples
                ) * (X_array[active].T @ signed[active])
                grad_b -= (
                    self.C / n_samples
                ) * float(np.sum(signed[active]))

            # Mild inverse-time decay stabilizes the simple subgradient solver.
            eta = self.learning_rate / (1.0 + 0.0005 * step)
            w -= eta * grad_w
            b -= eta * grad_b

            hinge = hinge_loss(signed, X_array @ w + b)
            objective = 0.5 * float(w @ w) + self.C * hinge
            self.objective_history_.append(objective)

        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model must be fit before prediction")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.coef_.shape[0]:
            raise ValueError("feature shape differs from fitted model")
        return X_array @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model must be fit before prediction")
        signed = np.where(self.decision_function(X) >= 0.0, 1, -1)
        return np.where(signed == -1, self.classes_[0], self.classes_[1])
