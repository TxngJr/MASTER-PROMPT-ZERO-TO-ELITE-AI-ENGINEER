"""Gaussian and Multinomial Naive Bayes from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianNBFromScratch:
    var_smoothing: float = 1e-9
    classes_: np.ndarray | None = None
    class_log_prior_: np.ndarray | None = None
    theta_: np.ndarray | None = None
    var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be non-empty 2D")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must be finite")
        if self.var_smoothing <= 0:
            raise ValueError("var_smoothing must be positive")

        classes, counts = np.unique(y_array, return_counts=True)
        means = []
        variances = []

        global_variance = float(np.var(X_array, axis=0).max())
        epsilon = self.var_smoothing * max(global_variance, 1.0)

        for label in classes:
            subset = X_array[y_array == label]
            means.append(subset.mean(axis=0))
            variances.append(subset.var(axis=0, ddof=0) + epsilon)

        self.classes_ = classes
        self.class_log_prior_ = np.log(counts / counts.sum())
        self.theta_ = np.asarray(means)
        self.var_ = np.asarray(variances)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if (
            self.classes_ is None
            or self.class_log_prior_ is None
            or self.theta_ is None
            or self.var_ is None
        ):
            raise RuntimeError("model must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.theta_.shape[1]:
            raise ValueError("X feature shape differs from fitted model")

        scores = []
        for index in range(len(self.classes_)):
            mean = self.theta_[index]
            variance = self.var_[index]
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * variance)
                + ((X_array - mean) ** 2) / variance,
                axis=1,
            )
            scores.append(self.class_log_prior_[index] + log_likelihood)
        return np.column_stack(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model must be fit before prediction")
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]


@dataclass
class MultinomialNBFromScratch:
    alpha: float = 1.0
    classes_: np.ndarray | None = None
    class_log_prior_: np.ndarray | None = None
    feature_log_prob_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNBFromScratch":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y).reshape(-1)

        if X_array.ndim != 2 or X_array.shape[0] == 0:
            raise ValueError("X must be non-empty 2D")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y sample counts differ")
        if np.any(X_array < 0) or not np.all(np.isfinite(X_array)):
            raise ValueError("MultinomialNB requires finite non-negative features")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        classes, counts = np.unique(y_array, return_counts=True)
        feature_log_prob = []

        for label in classes:
            class_feature_counts = X_array[y_array == label].sum(axis=0)
            smoothed = class_feature_counts + self.alpha
            denominator = smoothed.sum()

            if denominator <= 0:
                raise ValueError("feature counts plus smoothing must sum positive")
            feature_log_prob.append(np.log(smoothed / denominator))

        self.classes_ = classes
        self.class_log_prior_ = np.log(counts / counts.sum())
        self.feature_log_prob_ = np.asarray(feature_log_prob)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if (
            self.classes_ is None
            or self.class_log_prior_ is None
            or self.feature_log_prob_ is None
        ):
            raise RuntimeError("model must be fit before prediction")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X feature shape differs from fitted model")
        if np.any(X_array < 0):
            raise ValueError("MultinomialNB requires non-negative features")

        return X_array @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model must be fit before prediction")
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]
