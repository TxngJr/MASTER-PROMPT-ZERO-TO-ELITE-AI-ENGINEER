from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


MODULE_PATH = Path(__file__).parents[1] / "src" / "knn.py"
SPEC = importlib.util.spec_from_file_location("knn", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_minkowski_distance() -> None:
    assert mod.minkowski_distance([0, 0], [3, 4], p=2) == 5.0
    assert mod.minkowski_distance([0, 0], [3, 4], p=1) == 7.0


def test_knn_predicts_simple_clusters() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = mod.KNNClassifier(n_neighbors=3).fit(X, y)
    pred = model.predict(np.array([[0.2, 0.2], [10.2, 10.1]]))

    np.testing.assert_array_equal(pred, [0, 1])


def test_matches_sklearn_on_non_tied_queries() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    queries = rng.normal(size=(20, 3))

    ours = mod.KNNClassifier(n_neighbors=5, weights="uniform", p=2).fit(X, y)
    sk = KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2).fit(X, y)

    np.testing.assert_array_equal(ours.predict(queries), sk.predict(queries))


def test_distance_weighting_honors_exact_match() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 0, 0])
    model = mod.KNNClassifier(n_neighbors=3, weights="distance").fit(X, y)

    assert model.predict(np.array([[0.0]]))[0] == 1
