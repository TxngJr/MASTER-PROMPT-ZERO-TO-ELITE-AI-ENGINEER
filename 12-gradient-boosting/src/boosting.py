"""Educational AdaBoost classifier and Gradient Boosting regressor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _RegressionStump:
    feature: int | None = None
    threshold: float | None = None
    left_value: float = 0.0
    right_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionStump":
        n_samples, n_features = X.shape
        best_loss = float("inf")

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if values.size < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left
                if not np.any(left) or not np.any(right):
                    continue

                left_value = float(np.mean(y[left]))
                right_value = float(np.mean(y[right]))
                prediction = np.where(left, left_value, right_value)
                loss = float(np.sum((y - prediction) ** 2))

                if loss < best_loss:
                    best_loss = loss
                    self.feature = feature
                    self.threshold = float(threshold)
                    self.left_value = left_value
                    self.right_value = right_value

        if self.feature is None:
            self.feature = 0
            self.threshold = float("inf")
            self.left_value = float(np.mean(y))
            self.right_value = self.left_value

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.feature is not None
        assert self.threshold is not None
        return np.where(
            X[:, self.feature] <= self.threshold,
            self.left_value,
            self.right_value,
        )


class GradientBoostingRegressorFromScratch:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_: float | None = None
        self.estimators_: list[_RegressionStump] = []
        self.loss_history_: list[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GradientBoostingRegressorFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be non-empty 2D")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
            raise ValueError("X and y must be finite")

        self.initial_ = float(np.mean(y_array))
        prediction = np.full_like(y_array, self.initial_, dtype=float)
        self.estimators_ = []
        self.loss_history_ = []

        for _ in range(self.n_estimators):
            residual = y_array - prediction
            stump = _RegressionStump().fit(X_array, residual)
            update = stump.predict(X_array)
            prediction += self.learning_rate * update

            self.estimators_.append(stump)
            self.loss_history_.append(
                float(np.mean((y_array - prediction) ** 2))
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.initial_ is None:
            raise RuntimeError("model must be fit before prediction")
        X_array = np.asarray(X, dtype=float)
        prediction = np.full(X_array.shape[0], self.initial_, dtype=float)
        for stump in self.estimators_:
            prediction += self.learning_rate * stump.predict(X_array)
        return prediction


@dataclass
class _BinaryStump:
    feature: int
    threshold: float
    polarity: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = np.ones(X.shape[0], dtype=int)
        if self.polarity == 1:
            prediction[X[:, self.feature] <= self.threshold] = -1
        else:
            prediction[X[:, self.feature] > self.threshold] = -1
        return prediction


class AdaBoostBinaryClassifierFromScratch:
    def __init__(self, *, n_estimators: int = 50) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = n_estimators
        self.classes_: np.ndarray | None = None
        self.estimators_: list[_BinaryStump] = []
        self.alphas_: list[float] = []

    def _best_stump(
        self,
        X: np.ndarray,
        y_signed: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[_BinaryStump, float]:
        n_features = X.shape[1]
        best_error = float("inf")
        best_stump: _BinaryStump | None = None

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if values.size == 1:
                thresholds = values
            else:
                mids = (values[:-1] + values[1:]) / 2.0
                thresholds = np.concatenate(
                    [[values[0] - 1e-12], mids, [values[-1] + 1e-12]]
                )

            for threshold in thresholds:
                for polarity in (1, -1):
                    stump = _BinaryStump(
                        feature=feature,
                        threshold=float(threshold),
                        polarity=polarity,
                    )
                    prediction = stump.predict(X)
                    error = float(
                        np.sum(weights[prediction != y_signed])
                    )

                    if error < best_error:
                        best_error = error
                        best_stump = stump

        assert best_stump is not None
        return best_stump, best_error

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "AdaBoostBinaryClassifierFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be non-empty 2D")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")

        classes = np.unique(y_array)
        if classes.size != 2:
            raise ValueError("educational AdaBoost supports binary classification")

        self.classes_ = classes
        y_signed = np.where(y_array == classes[0], -1, 1)
        weights = np.full(X_array.shape[0], 1.0 / X_array.shape[0])

        self.estimators_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            stump, error = self._best_stump(X_array, y_signed, weights)

            error = float(np.clip(error, 1e-12, 1.0 - 1e-12))
            if error >= 0.5:
                break

            alpha = 0.5 * np.log((1.0 - error) / error)
            prediction = stump.predict(X_array)

            weights *= np.exp(-alpha * y_signed * prediction)
            weights /= weights.sum()

            self.estimators_.append(stump)
            self.alphas_.append(float(alpha))

            if error <= 1e-12:
                break

        if not self.estimators_:
            raise RuntimeError("no weak learner better than random was found")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("model must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        score = np.zeros(X_array.shape[0], dtype=float)
        for alpha, stump in zip(self.alphas_, self.estimators_):
            score += alpha * stump.predict(X_array)
        return score

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model must be fit before prediction")
        signed = np.where(self.decision_function(X) >= 0.0, 1, -1)
        return np.where(signed == -1, self.classes_[0], self.classes_[1])
