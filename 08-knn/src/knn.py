"""Educational KNN classifier implemented with NumPy brute-force search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def minkowski_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    p: float = 2.0,
) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)

    if x.shape != y.shape or x.size == 0:
        raise ValueError("vectors must have equal non-empty shape")
    if p < 1:
        raise ValueError("p must be >= 1")

    return float(np.sum(np.abs(x - y) ** p) ** (1.0 / p))


@dataclass
class KNNClassifier:
    n_neighbors: int = 5
    weights: str = "uniform"
    p: float = 2.0
    X_train_: np.ndarray | None = None
    y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if self.n_neighbors <= 0 or self.n_neighbors > X_array.shape[0]:
            raise ValueError("n_neighbors must be in [1, n_train]")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.p < 1:
            raise ValueError("p must be >= 1")

        self.X_train_ = X_array.copy()
        self.y_train_ = y_array.copy()
        return self

    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("model must be fit before prediction")
        difference = np.abs(self.X_train_ - query)
        return np.sum(difference**self.p, axis=1) ** (1.0 / self.p)

    def _predict_one(self, query: np.ndarray):
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("model must be fit before prediction")

        distances = self._distances(query)
        indices = np.argsort(distances, kind="stable")[: self.n_neighbors]
        labels = self.y_train_[indices]
        neighbor_distances = distances[indices]

        classes = np.unique(labels)
        best_label = None
        best_primary = -np.inf
        best_secondary = -np.inf

        for label in classes:
            mask = labels == label
            count = int(np.sum(mask))

            zero_mask = mask & (neighbor_distances == 0)
            if np.any(zero_mask):
                distance_score = float("inf")
            else:
                distance_score = float(
                    np.sum(1.0 / (neighbor_distances[mask] + 1e-12))
                )

            primary = (
                float(count)
                if self.weights == "uniform"
                else distance_score
            )
            secondary = distance_score

            if (
                primary > best_primary
                or (
                    primary == best_primary
                    and secondary > best_secondary
                )
                or (
                    primary == best_primary
                    and secondary == best_secondary
                    and (best_label is None or str(label) < str(best_label))
                )
            ):
                best_label = label
                best_primary = primary
                best_secondary = secondary

        return best_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("model must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X feature shape differs from fitted data")

        return np.asarray([self._predict_one(row) for row in X_array])
