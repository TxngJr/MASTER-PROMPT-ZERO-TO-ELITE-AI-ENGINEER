from __future__ import annotations
import numpy as np


def weighted_federated_average(client_vectors, counts):
    vectors = [np.asarray(v, dtype=float) for v in client_vectors]
    counts = np.asarray(counts, dtype=float)
    if len(vectors) == 0 or counts.shape != (len(vectors),):
        raise ValueError("counts must align with non-empty clients")
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("counts must be non-negative with positive total")
    shape = vectors[0].shape
    if any(v.shape != shape for v in vectors):
        raise ValueError("all client vectors must share shape")
    weights = counts / counts.sum()
    out = np.zeros(shape, dtype=float)
    for w, v in zip(weights, vectors):
        out += w * v
    return out


def l2_clip(vector, max_norm: float):
    x = np.asarray(vector, dtype=float)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = float(np.linalg.norm(x))
    if norm == 0 or norm <= max_norm:
        return x.copy()
    return x * (max_norm / norm)


def add_gaussian_noise(vector, std: float, seed: int):
    if std < 0:
        raise ValueError("std must be non-negative")
    x = np.asarray(vector, dtype=float)
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, std, size=x.shape)
