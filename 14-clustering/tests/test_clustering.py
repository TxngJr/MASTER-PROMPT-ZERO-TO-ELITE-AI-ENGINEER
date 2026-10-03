from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


MODULE_PATH = Path(__file__).parents[1] / "src" / "clustering.py"
SPEC = importlib.util.spec_from_file_location("clustering_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_kmeans_finds_clear_blobs() -> None:
    X, y = make_blobs(
        n_samples=240,
        centers=[[-5, -5], [0, 5], [5, -4]],
        cluster_std=0.45,
        random_state=7,
    )

    model = mod.KMeansFromScratch(
        n_clusters=3,
        seed=7,
    ).fit(X)

    assert adjusted_rand_score(y, model.labels_) > 0.98
    assert model.inertia_ is not None


def test_kmeans_reference_has_similar_inertia_scale() -> None:
    X, _ = make_blobs(
        n_samples=180,
        centers=3,
        cluster_std=0.7,
        random_state=11,
    )

    ours = mod.KMeansFromScratch(n_clusters=3, seed=11).fit(X)
    ref = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        random_state=11,
    ).fit(X)

    assert ours.inertia_ <= ref.inertia_ * 1.20


def test_dbscan_matches_reference_structure_on_simple_data() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [20.0, 20.0],
        ]
    )

    ours = mod.DBSCANFromScratch(eps=0.25, min_samples=2).fit_predict(X)
    ref = DBSCAN(eps=0.25, min_samples=2).fit_predict(X)

    assert adjusted_rand_score(ref, ours) == 1.0
    assert ours[-1] == -1
