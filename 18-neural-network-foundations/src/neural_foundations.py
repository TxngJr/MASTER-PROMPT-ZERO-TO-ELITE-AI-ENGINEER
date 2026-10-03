"""Neural-network forward primitives before backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(x, dtype=float))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("equal non-empty shapes required")
    return float(np.mean((y - p) ** 2))


def binary_cross_entropy(
    y_true: np.ndarray,
    probability: np.ndarray,
    eps: float = 1e-12,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("equal non-empty shapes required")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be 0/1")
    clipped = np.clip(p, eps, 1.0 - eps)
    return float(
        -np.mean(
            y * np.log(clipped)
            + (1.0 - y) * np.log(1.0 - clipped)
        )
    )


@dataclass
class Linear:
    in_features: int
    out_features: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("feature counts must be positive")
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / self.in_features)
        self.weight = rng.normal(
            0.0,
            scale,
            size=(self.in_features, self.out_features),
        )
        self.bias = np.zeros(self.out_features, dtype=float)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.in_features:
            raise ValueError("input shape mismatch")
        return array @ self.weight + self.bias

    @property
    def parameter_count(self) -> int:
        return self.in_features * self.out_features + self.out_features


class PerceptronClassifier:
    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        epochs: int = 50,
    ) -> None:
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronClassifier":
        array = np.asarray(X, dtype=float)
        labels = np.asarray(y).reshape(-1)

        if array.ndim != 2 or array.shape[0] != labels.shape[0]:
            raise ValueError("invalid X/y shapes")

        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("binary classes required")

        signed = np.where(labels == classes[0], -1.0, 1.0)
        w = np.zeros(array.shape[1], dtype=float)
        b = 0.0

        for _ in range(self.epochs):
            mistakes = 0
            for row, target in zip(array, signed):
                if target * (float(row @ w) + b) <= 0.0:
                    w += self.learning_rate * target * row
                    b += self.learning_rate * target
                    mistakes += 1
            if mistakes == 0:
                break

        self.coef_ = w
        self.intercept_ = b
        self.classes_ = classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.classes_ is None
        ):
            raise RuntimeError("fit before predict")

        array = np.asarray(X, dtype=float)
        score = array @ self.coef_ + self.intercept_
        signed = np.where(score >= 0.0, 1.0, -1.0)
        return np.where(signed < 0, self.classes_[0], self.classes_[1])


class TinyMLP:
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        seed: int = 42,
    ) -> None:
        self.layer1 = Linear(in_features, hidden_features, seed=seed)
        self.layer2 = Linear(hidden_features, out_features, seed=seed + 1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        hidden = relu(self.layer1(X))
        return self.layer2(hidden)

    @property
    def parameter_count(self) -> int:
        return self.layer1.parameter_count + self.layer2.parameter_count
