"""Educational Random Forest classifier implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


@dataclass
class _Node:
    probabilities: np.ndarray
    prediction_index: int
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class _RandomizedTree:
    def __init__(
        self,
        *,
        classes: np.ndarray,
        max_depth: int | None,
        min_samples_leaf: int,
        max_features: int,
        rng: np.random.Generator,
    ) -> None:
        self.classes = classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.root: _Node | None = None

    def _leaf(self, y: np.ndarray) -> _Node:
        counts = np.array([np.sum(y == label) for label in self.classes], dtype=float)
        probabilities = counts / counts.sum()
        return _Node(
            probabilities=probabilities,
            prediction_index=int(np.argmax(counts)),
        )

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        candidate_features = self.rng.choice(
            n_features,
            size=min(self.max_features, n_features),
            replace=False,
        )

        parent = gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in candidate_features:
            values = np.unique(X[:, feature])
            if values.size < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                n_left = int(np.sum(left))
                n_right = n_samples - n_left

                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue

                child = (
                    (n_left / n_samples) * gini(y[left])
                    + (n_right / n_samples) * gini(y[~left])
                )
                gain = parent - child

                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        leaf = self._leaf(y)

        if np.unique(y).size == 1:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        if y.size < 2 * self.min_samples_leaf:
            return leaf

        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return leaf

        mask = X[:, feature] <= threshold
        leaf.feature = feature
        leaf.threshold = threshold
        leaf.left = self._build(X[mask], y[mask], depth + 1)
        leaf.right = self._build(X[~mask], y[~mask], depth + 1)
        return leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomizedTree":
        self.root = self._build(X, y, 0)
        return self

    def _traverse(self, row: np.ndarray) -> _Node:
        if self.root is None:
            raise RuntimeError("tree is not fit")
        node = self.root

        while not node.is_leaf:
            assert node.feature is not None
            assert node.threshold is not None
            assert node.left is not None
            assert node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right

        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self._traverse(row).probabilities for row in X])


class RandomForestClassifierFromScratch:
    def __init__(
        self,
        *,
        n_estimators: int = 25,
        max_depth: int | None = 6,
        min_samples_leaf: int = 1,
        max_features: str | int = "sqrt",
        bootstrap: bool = True,
        seed: int = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

        self.classes_: np.ndarray | None = None
        self.trees_: list[_RandomizedTree] = []
        self.oob_score_: float | None = None

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            if not 1 <= self.max_features <= n_features:
                raise ValueError("integer max_features must be in [1, n_features]")
            return self.max_features
        raise ValueError("max_features must be 'sqrt', 'log2', or an integer")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "RandomForestClassifierFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must be finite")

        n_samples, n_features = X_array.shape
        self.classes_ = np.unique(y_array)
        n_feature_candidates = self._resolve_max_features(n_features)

        master_rng = np.random.default_rng(self.seed)
        self.trees_ = []

        oob_probability_sum = np.zeros((n_samples, len(self.classes_)), dtype=float)
        oob_counts = np.zeros(n_samples, dtype=int)

        for _ in range(self.n_estimators):
            tree_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            tree_rng = np.random.default_rng(tree_seed)

            if self.bootstrap:
                train_indices = tree_rng.integers(0, n_samples, size=n_samples)
                selected = np.zeros(n_samples, dtype=bool)
                selected[np.unique(train_indices)] = True
                oob_indices = np.flatnonzero(~selected)
            else:
                train_indices = np.arange(n_samples)
                oob_indices = np.array([], dtype=int)

            tree = _RandomizedTree(
                classes=self.classes_,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=n_feature_candidates,
                rng=tree_rng,
            ).fit(X_array[train_indices], y_array[train_indices])

            self.trees_.append(tree)

            if oob_indices.size:
                oob_probability_sum[oob_indices] += tree.predict_proba(
                    X_array[oob_indices]
                )
                oob_counts[oob_indices] += 1

        valid_oob = oob_counts > 0
        if np.any(valid_oob):
            averaged = (
                oob_probability_sum[valid_oob]
                / oob_counts[valid_oob, None]
            )
            predictions = self.classes_[np.argmax(averaged, axis=1)]
            self.oob_score_ = float(
                np.mean(predictions == y_array[valid_oob])
            )
        else:
            self.oob_score_ = None

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_ or self.classes_ is None:
            raise RuntimeError("forest must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be 2D")

        probabilities = np.stack(
            [tree.predict_proba(X_array) for tree in self.trees_],
            axis=0,
        )
        return probabilities.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("forest must be fit before prediction")
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
