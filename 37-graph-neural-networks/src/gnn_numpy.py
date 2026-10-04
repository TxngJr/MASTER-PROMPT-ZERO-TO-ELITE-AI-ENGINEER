"""Educational graph neural network primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def adjacency_from_edges(
    num_nodes: int,
    edges: list[tuple[int, int]],
    *,
    directed: bool = False,
) -> np.ndarray:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)

    for source, target in edges:
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError("edge node id out of range")
        adjacency[source, target] = 1.0
        if not directed:
            adjacency[target, source] = 1.0

    return adjacency


def add_self_loops(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    result = a.copy()
    np.fill_diagonal(result, 1.0)
    return result


def symmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    if np.any(a < 0):
        raise ValueError("adjacency weights must be non-negative")

    degree = a.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])

    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def gcn_layer(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(node_features, dtype=float)
    a = np.asarray(adjacency, dtype=float)
    w = np.asarray(weight, dtype=float)

    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("X and W must be matrices")
    if a.shape != (x.shape[0], x.shape[0]):
        raise ValueError("adjacency shape mismatch")
    if x.shape[1] != w.shape[0]:
        raise ValueError("feature/weight shape mismatch")

    normalized = symmetric_normalize(add_self_loops(a))
    output = normalized @ x @ w

    if bias is not None:
        b = np.asarray(bias, dtype=float)
        if b.shape != (w.shape[1],):
            raise ValueError("bias shape mismatch")
        output = output + b

    return output


def mean_neighbor_aggregate(
    node_features: np.ndarray,
    adjacency: np.ndarray,
    *,
    include_self: bool = False,
) -> np.ndarray:
    x = np.asarray(node_features, dtype=float)
    a = np.asarray(adjacency, dtype=float)

    if a.shape != (x.shape[0], x.shape[0]):
        raise ValueError("adjacency shape mismatch")

    weights = add_self_loops(a) if include_self else a.copy()
    degree = weights.sum(axis=1, keepdims=True)

    aggregate = weights @ x
    nonzero = degree[:, 0] > 0
    aggregate[nonzero] /= degree[nonzero]
    aggregate[~nonzero] = 0.0
    return aggregate


def graph_mean_pool(
    node_features: np.ndarray,
    graph_ids: np.ndarray,
) -> np.ndarray:
    x = np.asarray(node_features, dtype=float)
    ids = np.asarray(graph_ids, dtype=np.int64)

    if x.ndim != 2 or ids.ndim != 1 or len(ids) != len(x):
        raise ValueError("shape mismatch")
    if len(ids) == 0 or np.any(ids < 0):
        raise ValueError("graph_ids must be non-negative and non-empty")

    graph_count = int(ids.max()) + 1
    pooled = np.zeros((graph_count, x.shape[1]), dtype=float)
    counts = np.zeros(graph_count, dtype=int)

    for row, graph_id in zip(x, ids):
        pooled[graph_id] += row
        counts[graph_id] += 1

    if np.any(counts == 0):
        raise ValueError("graph ids must be contiguous with no empty graph")

    return pooled / counts[:, None]
