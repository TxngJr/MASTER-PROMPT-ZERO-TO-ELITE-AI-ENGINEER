from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "geometry_lab.py"
SPEC = importlib.util.spec_from_file_location("geometry_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_dataset_reproducible() -> None:
    X1, y1 = lab.make_dataset(500, seed=2)
    X2, y2 = lab.make_dataset(500, seed=2)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_pca_reduces_dimensions_and_keeps_requested_variance() -> None:
    X, y = lab.make_dataset(900, seed=7)
    report = lab.run_experiment(X, y, seed=7)

    assert report["pca_features"] < report["original_features"]
    assert report["pca_explained_variance_ratio_sum"] >= 0.95


def test_selected_svm_is_useful() -> None:
    X, y = lab.make_dataset(1000, seed=42)
    report = lab.run_experiment(X, y, seed=42)

    assert report["selected_classifier"] in {
        "scaled_linear_svm",
        "pca_linear_svm",
    }
    assert report["test_metrics"]["accuracy"] > 0.75
    assert report["test_metrics"]["f1"] > 0.75


def test_clustering_metrics_are_finite() -> None:
    X, y = lab.make_dataset(700, seed=5)
    report = lab.run_experiment(X, y, seed=5)

    assert np.isfinite(report["kmeans"]["silhouette"])
    assert np.isfinite(report["kmeans"]["adjusted_rand_diagnostic"])
    assert report["dbscan"]["noise_points"] >= 0
