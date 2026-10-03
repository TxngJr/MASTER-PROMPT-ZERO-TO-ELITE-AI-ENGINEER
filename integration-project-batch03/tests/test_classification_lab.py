from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "classification_lab.py"
SPEC = importlib.util.spec_from_file_location("classification_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_dataset_is_reproducible() -> None:
    X1, y1 = lab.make_dataset(400, seed=8)
    X2, y2 = lab.make_dataset(400, seed=8)

    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_stratified_split_preserves_rough_positive_rate() -> None:
    X, y = lab.make_dataset(1000, seed=3)
    X_train, X_val, X_test, y_train, y_val, y_test = lab.split_dataset(
        X,
        y,
        seed=3,
    )

    overall = np.mean(y)
    for subset in [y_train, y_val, y_test]:
        assert abs(np.mean(subset) - overall) < 0.03

    assert len(X_train) + len(X_val) + len(X_test) == len(X)


def test_threshold_selection_returns_valid_threshold() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.3, 0.4, 0.7, 0.9, 0.2])

    threshold, metrics = lab.choose_threshold(y, p)

    assert 0.05 <= threshold <= 0.95
    assert 0.0 <= metrics["f1"] <= 1.0


def test_full_experiment_produces_useful_classifier() -> None:
    X, y = lab.make_dataset(1200, seed=42)
    report = lab.run_experiment(X, y, seed=42)

    assert report["selected_model"] in {"logistic", "knn_7", "gaussian_nb"}
    assert report["test_metrics"]["f1"] > 0.45
    assert report["test_metrics"]["roc_auc"] > 0.70
