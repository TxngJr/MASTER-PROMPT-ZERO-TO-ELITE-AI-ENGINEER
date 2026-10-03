from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "tree_ensemble_lab.py"
SPEC = importlib.util.spec_from_file_location("tree_ensemble_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_dataset_reproducible() -> None:
    X1, y1 = lab.make_dataset(500, seed=10)
    X2, y2 = lab.make_dataset(500, seed=10)

    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_split_is_disjoint_and_complete() -> None:
    X, y = lab.make_dataset(500, seed=3)

    split = lab.split_dataset(X, y, seed=3)
    X_train, X_val, X_test, y_train, y_val, y_test = split

    assert len(X_train) + len(X_val) + len(X_test) == 500
    assert len(y_train) + len(y_val) + len(y_test) == 500


def test_experiment_produces_strong_tree_family_model() -> None:
    X, y = lab.make_dataset(1200, seed=42)
    report = lab.run_experiment(X, y, seed=42)

    assert report["selected_model"] in {
        "decision_tree",
        "random_forest",
        "adaboost",
        "gradient_boosting",
        "hist_gradient_boosting",
    }
    assert report["test_metrics"]["f1"] > 0.65
    assert report["test_metrics"]["roc_auc"] > 0.80


def test_validation_metrics_are_finite() -> None:
    X, y = lab.make_dataset(800, seed=9)
    report = lab.run_experiment(X, y, seed=9)

    for metrics in report["validation_results"].values():
        for key in ["accuracy", "f1", "roc_auc", "predict_seconds", "fit_seconds"]:
            assert np.isfinite(metrics[key])
