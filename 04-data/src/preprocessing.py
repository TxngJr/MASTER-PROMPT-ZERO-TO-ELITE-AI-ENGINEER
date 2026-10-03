"""Educational preprocessing utilities implemented from first principles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


def split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    train_end = int(n_samples * train_fraction)
    validation_end = train_end + int(n_samples * validation_fraction)

    return (
        indices[:train_end],
        indices[train_end:validation_end],
        indices[validation_end:],
    )


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = _as_2d_float(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        X = _as_2d_float(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("feature count differs from fitted data")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class MedianImputer:
    median_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MedianImputer":
        X = _as_2d_float(X)
        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("cannot fit a column containing only NaN")
        self.median_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("MedianImputer must be fit before transform")
        X = _as_2d_float(X).copy()
        if X.shape[1] != self.median_.shape[0]:
            raise ValueError("feature count differs from fitted data")
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.median_[cols]
        return X


@dataclass
class SimpleOneHotEncoder:
    categories_: list[list[Hashable]] | None = None

    def fit(self, X: Sequence[Sequence[Hashable]]) -> "SimpleOneHotEncoder":
        rows = [list(row) for row in X]
        if not rows:
            raise ValueError("X must not be empty")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("X must be a non-empty rectangular table")

        self.categories_ = []
        for column in range(width):
            values = sorted({row[column] for row in rows}, key=str)
            self.categories_.append(values)
        return self

    def transform(self, X: Sequence[Sequence[Hashable]]) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("encoder must be fit before transform")

        rows = [list(row) for row in X]
        width = len(self.categories_)
        if any(len(row) != width for row in rows):
            raise ValueError("feature count differs from fitted data")

        output_width = sum(len(cats) for cats in self.categories_)
        output = np.zeros((len(rows), output_width), dtype=float)

        offsets = np.cumsum([0] + [len(cats) for cats in self.categories_[:-1]])
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                categories = self.categories_[column_index]
                if value in categories:
                    category_index = categories.index(value)
                    output[row_index, offsets[column_index] + category_index] = 1.0
                # Unknown categories intentionally become all-zero for this column.
        return output


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("array must have at least one row and one column")
    return array
