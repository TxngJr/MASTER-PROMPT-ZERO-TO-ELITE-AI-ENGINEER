"""Educational exact and IVF vector retrieval implemented with NumPy."""

from __future__ import annotations

import numpy as np


def l2_normalize(
    vectors: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def cosine_scores(
    query: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query, dtype=float).reshape(1, -1)
    x = np.asarray(vectors, dtype=float)

    if x.ndim != 2 or q.shape[1] != x.shape[1]:
        raise ValueError("dimension mismatch")

    return (
        l2_normalize(q)
        @ l2_normalize(x).T
    )[0]


def exact_top_k(
    query: np.ndarray,
    vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")

    scores = cosine_scores(query, vectors)
    k = min(k, len(scores))

    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]


def kmeans(
    vectors: np.ndarray,
    clusters: int,
    *,
    iterations: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(vectors, dtype=float)

    if x.ndim != 2:
        raise ValueError("vectors must be a matrix")
    if not 1 <= clusters <= len(x):
        raise ValueError("invalid cluster count")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    centroids = x[
        rng.choice(len(x), size=clusters, replace=False)
    ].copy()

    assignments = np.zeros(len(x), dtype=np.int64)

    for _ in range(iterations):
        distances = np.sum(
            (x[:, None, :] - centroids[None, :, :]) ** 2,
            axis=-1,
        )
        assignments = np.argmin(distances, axis=1)

        for cluster_id in range(clusters):
            members = x[assignments == cluster_id]
            if len(members):
                centroids[cluster_id] = members.mean(axis=0)
            else:
                centroids[cluster_id] = x[
                    int(rng.integers(0, len(x)))
                ]

    return centroids, assignments


class IVFIndex:
    def __init__(
        self,
        clusters: int = 8,
        *,
        iterations: int = 20,
        seed: int = 42,
    ) -> None:
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed

    def fit(self, vectors: np.ndarray) -> "IVFIndex":
        x = np.asarray(vectors, dtype=float)

        if x.ndim != 2 or len(x) == 0:
            raise ValueError("vectors must be a non-empty matrix")

        self.vectors_ = x.copy()
        self.centroids_, assignments = kmeans(
            x,
            self.clusters,
            iterations=self.iterations,
            seed=self.seed,
        )
        self.lists_ = [
            np.flatnonzero(assignments == cluster_id)
            for cluster_id in range(self.clusters)
        ]
        return self

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        nprobe: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "vectors_"):
            raise RuntimeError("fit before search")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 1 <= nprobe <= self.clusters:
            raise ValueError("nprobe out of range")

        q = np.asarray(query, dtype=float).reshape(-1)
        if q.shape[0] != self.vectors_.shape[1]:
            raise ValueError("query dimension mismatch")

        centroid_distances = np.sum(
            (self.centroids_ - q[None, :]) ** 2,
            axis=1,
        )
        probes = np.argsort(centroid_distances)[:nprobe]

        candidates = np.concatenate(
            [self.lists_[int(cluster)] for cluster in probes]
        )

        if len(candidates) == 0:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=float),
            )

        local_ids, scores = exact_top_k(
            q,
            self.vectors_[candidates],
            k,
        )
        return candidates[local_ids], scores


def ann_recall_at_k(
    exact_ids: np.ndarray,
    approximate_ids: np.ndarray,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    exact = set(np.asarray(exact_ids).reshape(-1)[:k].tolist())
    approximate = set(
        np.asarray(approximate_ids).reshape(-1)[:k].tolist()
    )

    if not exact:
        return 0.0
    return float(len(exact & approximate) / len(exact))


def metadata_filter(
    ids: np.ndarray,
    metadata: list[dict[str, object]],
    **conditions: object,
) -> np.ndarray:
    selected = []

    for item_id in np.asarray(ids, dtype=np.int64).reshape(-1):
        if not 0 <= item_id < len(metadata):
            raise ValueError("id out of metadata range")

        item = metadata[int(item_id)]
        if all(item.get(key) == value for key, value in conditions.items()):
            selected.append(int(item_id))

    return np.asarray(selected, dtype=np.int64)
