"""Educational greedy classification tree implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    labels = np.asarray(y).reshape(-1)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def entropy(y: np.ndarray) -> float:
    labels = np.asarray(y).reshape(-1)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


@dataclass
class Node:
    prediction: object
    probabilities: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifierFromScratch:
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.classes_: np.ndarray | None = None
        self.root_: Node | None = None

    def _impurity(self, y: np.ndarray) -> float:
        return gini_impurity(y) if self.criterion == "gini" else entropy(y)

    def _leaf(self, y: np.ndarray) -> Node:
        assert self.classes_ is not None
        counts = np.array([np.sum(y == label) for label in self.classes_], dtype=float)
        probabilities = counts / counts.sum()
        prediction = self.classes_[int(np.argmax(counts))]
        return Node(prediction=prediction, probabilities=probabilities)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int | None, float | None, float]:
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if values.size < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue

                left_y = y[left_mask]
                right_y = y[~left_mask]

                child_impurity = (
                    (n_left / n_samples) * self._impurity(left_y)
                    + (n_right / n_samples) * self._impurity(right_y)
                )
                gain = parent_impurity - child_impurity

                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        leaf = self._leaf(y)

        pure = np.unique(y).size == 1
        too_small = y.size < self.min_samples_split
        too_deep = self.max_depth is not None and depth >= self.max_depth

        if pure or too_small or too_deep:
            return leaf

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or threshold is None or gain <= 0.0:
            return leaf

        left_mask = X[:, feature] <= threshold
        leaf.feature_index = feature
        leaf.threshold = threshold
        leaf.left = self._build(X[left_mask], y[left_mask], depth + 1)
        leaf.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return leaf

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DecisionTreeClassifierFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("from-scratch tree requires finite numeric X")

        self.classes_ = np.unique(y_array)
        self.root_ = self._build(X_array, y_array, depth=0)
        return self

    def _traverse(self, row: np.ndarray) -> Node:
        if self.root_ is None:
            raise RuntimeError("tree must be fit before prediction")

        node = self.root_
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            assert node.left is not None
            assert node.right is not None

            node = (
                node.left
                if row[node.feature_index] <= node.threshold
                else node.right
            )
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be 2D")
        return np.asarray([self._traverse(row).prediction for row in X_array])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be 2D")
        return np.vstack([self._traverse(row).probabilities for row in X_array])
