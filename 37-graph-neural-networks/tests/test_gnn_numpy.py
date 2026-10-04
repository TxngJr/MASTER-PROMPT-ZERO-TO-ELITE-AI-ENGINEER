from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "gnn_numpy.py"
SPEC = importlib.util.spec_from_file_location("gnn_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_undirected_adjacency_is_symmetric() -> None:
    a = mod.adjacency_from_edges(3, [(0, 1), (1, 2)])
    np.testing.assert_array_equal(a, a.T)


def test_symmetric_normalization_is_symmetric() -> None:
    a = mod.add_self_loops(
        mod.adjacency_from_edges(3, [(0, 1), (1, 2)])
    )
    normalized = mod.symmetric_normalize(a)
    np.testing.assert_allclose(normalized, normalized.T)


def test_gcn_shape() -> None:
    x = np.arange(12, dtype=float).reshape(4, 3)
    a = mod.adjacency_from_edges(
        4,
        [(0, 1), (1, 2), (2, 3)],
    )
    w = np.ones((3, 5))

    out = mod.gcn_layer(x, a, w)
    assert out.shape == (4, 5)
    assert np.all(np.isfinite(out))


def test_mean_aggregate_is_order_independent() -> None:
    x = np.array([[1.0], [3.0], [5.0]])
    a = mod.adjacency_from_edges(3, [(0, 1), (0, 2)])

    aggregate = mod.mean_neighbor_aggregate(x, a)
    np.testing.assert_allclose(aggregate[0], [4.0])


def test_graph_mean_pool() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    ids = np.array([0, 0, 1])

    pooled = mod.graph_mean_pool(x, ids)
    np.testing.assert_allclose(pooled, [[2.0, 3.0], [10.0, 20.0]])
