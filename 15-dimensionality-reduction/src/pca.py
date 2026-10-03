"""Principal Component Analysis from scratch using NumPy SVD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAFromScratch:
    n_components: int | None = None
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    singular_values_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[0] < 2 or X_array.shape[1] == 0:
            raise ValueError("X must be 2D with at least 2 samples")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must be finite")

        n_samples, n_features = X_array.shape
        max_components = min(n_samples, n_features)

        if self.n_components is None:
            k = max_components
        else:
            if not 1 <= self.n_components <= max_components:
                raise ValueError("invalid n_components")
            k = self.n_components

        self.mean_ = X_array.mean(axis=0)
        centered = X_array - self.mean_

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        all_variance = (singular_values**2) / (n_samples - 1)
        total_variance = float(np.sum(all_variance))

        self.components_ = vt[:k].copy()
        self.singular_values_ = singular_values[:k].copy()
        self.explained_variance_ = all_variance[:k].copy()
        self.explained_variance_ratio_ = (
            all_variance[:k] / total_variance
            if total_variance > 0
            else np.zeros(k, dtype=float)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fit before transform")
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != self.mean_.shape[0]:
            raise ValueError("feature shape differs from fitted PCA")
        return (X_array - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fit before inverse_transform")
        Z_array = np.asarray(Z, dtype=float)
        if Z_array.ndim != 2 or Z_array.shape[1] != self.components_.shape[0]:
            raise ValueError("component shape differs from fitted PCA")
        return Z_array @ self.components_ + self.mean_


def reconstruction_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    a = np.asarray(original, dtype=float)
    b = np.asarray(reconstructed, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return float(np.mean((a - b) ** 2))
