"""Educational anomaly detectors implemented from statistical primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("X must be non-empty 2D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must be finite")
    return arr


@dataclass
class ZScoreDetector:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        arr = _as_2d(X)
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit before scoring")
        arr = _as_2d(X)
        z = np.abs((arr - self.mean_) / self.scale_)
        return np.max(z, axis=1)


@dataclass
class RobustZScoreDetector:
    median_: np.ndarray | None = None
    mad_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustZScoreDetector":
        arr = _as_2d(X)
        self.median_ = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - self.median_), axis=0)
        self.mad_ = np.where(mad == 0.0, 1.0, mad)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("fit before scoring")
        arr = _as_2d(X)
        score = 0.67448975 * np.abs(arr - self.median_) / self.mad_
        return np.max(score, axis=1)


@dataclass
class MahalanobisDetector:
    regularization: float = 1e-6
    mean_: np.ndarray | None = None
    precision_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        arr = _as_2d(X)
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if arr.shape[0] < 2:
            raise ValueError("need at least 2 samples")

        self.mean_ = arr.mean(axis=0)
        covariance = np.atleast_2d(np.cov(arr, rowvar=False, ddof=1))
        covariance += self.regularization * np.eye(arr.shape[1])
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("fit before scoring")
        arr = _as_2d(X)
        delta = arr - self.mean_
        squared = np.einsum("ni,ij,nj->n", delta, self.precision_, delta)
        return np.sqrt(np.maximum(squared, 0.0))


def threshold_from_quantile(
    training_scores: np.ndarray,
    *,
    contamination: float,
) -> float:
    scores = np.asarray(training_scores, dtype=float).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite and non-empty")
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be in (0, 0.5)")
    return float(np.quantile(scores, 1.0 - contamination))


def predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    return (values >= threshold).astype(int)
