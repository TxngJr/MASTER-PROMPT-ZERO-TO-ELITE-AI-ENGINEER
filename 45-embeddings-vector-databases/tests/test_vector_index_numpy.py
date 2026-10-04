from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "vector_index_numpy.py"
SPEC = importlib.util.spec_from_file_location("vector_index_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_exact_top_k_returns_nearest_cosine() -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
    )
    ids, scores = mod.exact_top_k(
        np.array([1.0, 0.0]),
        vectors,
        2,
    )

    assert ids.tolist() == [0, 2]
    assert scores[0] >= scores[1]


def test_kmeans_shapes() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(30, 4))

    centroids, assignments = mod.kmeans(
        vectors,
        clusters=3,
        iterations=3,
        seed=3,
    )

    assert centroids.shape == (3, 4)
    assert assignments.shape == (30,)


def test_ivf_with_all_probes_matches_exact() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(80, 6))
    query = rng.normal(size=6)

    exact_ids, _ = mod.exact_top_k(query, vectors, 5)

    index = mod.IVFIndex(
        clusters=4,
        iterations=4,
        seed=4,
    ).fit(vectors)

    approx_ids, _ = index.search(
        query,
        5,
        nprobe=4,
    )

    assert mod.ann_recall_at_k(
        exact_ids,
        approx_ids,
        5,
    ) == 1.0


def test_metadata_filter() -> None:
    ids = np.array([0, 1, 2])
    metadata = [
        {"lang": "th"},
        {"lang": "en"},
        {"lang": "th"},
    ]

    filtered = mod.metadata_filter(
        ids,
        metadata,
        lang="th",
    )

    assert filtered.tolist() == [0, 2]
