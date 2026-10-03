"""Educational K-Means and DBSCAN implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _validate_X(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain finite values")
    return array


@dataclass
class KMeansFromScratch:
    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    seed: int = 42
    cluster_centers_: np.ndarray | None = None
    labels_: np.ndarray | None = None
    inertia_: float | None = None
    inertia_history_: list[float] = field(default_factory=list)

    def _init_kmeans_plus_plus(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_samples = X.shape[0]
        centers = [X[int(rng.integers(n_samples))].copy()]

        while len(centers) < self.n_clusters:
            existing = np.vstack(centers)
            squared = np.sum(
                (X[:, None, :] - existing[None, :, :]) ** 2,
                axis=2,
            )
            nearest_squared = np.min(squared, axis=1)
            total = float(nearest_squared.sum())

            if total == 0.0:
                remaining = rng.integers(n_samples)
                centers.append(X[int(remaining)].copy())
                continue

            probabilities = nearest_squared / total
            index = int(rng.choice(n_samples, p=probabilities))
            centers.append(X[index].copy())

        return np.vstack(centers)

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        X_array = _validate_X(X)
        if not 1 <= self.n_clusters <= X_array.shape[0]:
            raise ValueError("n_clusters must be in [1, n_samples]")
        if self.max_iter <= 0 or self.tol < 0:
            raise ValueError("invalid max_iter/tol")

        rng = np.random.default_rng(self.seed)
        centers = self._init_kmeans_plus_plus(X_array, rng)
        self.inertia_history_ = []

        for _ in range(self.max_iter):
            squared = np.sum(
                (X_array[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            )
            labels = np.argmin(squared, axis=1)
            inertia = float(np.sum(squared[np.arange(len(X_array)), labels]))
            self.inertia_history_.append(inertia)

            new_centers = centers.copy()
            for cluster in range(self.n_clusters):
                members = X_array[labels == cluster]
                if members.size:
                    new_centers[cluster] = members.mean(axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                break

        squared = np.sum(
            (X_array[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(squared, axis=1)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(
            np.sum(squared[np.arange(len(X_array)), labels])
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("model must be fit before predict")
        X_array = _validate_X(X)
        squared = np.sum(
            (X_array[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared, axis=1)


@dataclass
class DBSCANFromScratch:
    eps: float = 0.5
    min_samples: int = 5
    labels_: np.ndarray | None = None
    core_sample_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCANFromScratch":
        X_array = _validate_X(X)
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")

        n = X_array.shape[0]
        distances = np.linalg.norm(
            X_array[:, None, :] - X_array[None, :, :],
            axis=2,
        )
        neighbors = [
            np.flatnonzero(distances[i] <= self.eps)
            for i in range(n)
        ]
        core = np.array(
            [len(indexes) >= self.min_samples for indexes in neighbors]
        )

        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for point in range(n):
            if visited[point]:
                continue
            visited[point] = True

            if not core[point]:
                continue

            labels[point] = cluster_id
            queue = list(neighbors[point])
            queued = set(queue)

            while queue:
                current = queue.pop(0)

                if not visited[current]:
                    visited[current] = True
                    if core[current]:
                        for candidate in neighbors[current]:
                            c = int(candidate)
                            if c not in queued:
                                queue.append(c)
                                queued.add(c)

                if labels[current] == -1:
                    labels[current] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()
