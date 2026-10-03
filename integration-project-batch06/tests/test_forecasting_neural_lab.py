from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "forecasting_neural_lab.py"
SPEC = importlib.util.spec_from_file_location("forecasting_neural_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_series_reproducible() -> None:
    a, la = lab.make_series(500, seed=3)
    b, lb = lab.make_series(500, seed=3)
    np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(la, lb)


def test_lagged_features_use_only_past_values() -> None:
    series = np.arange(10, dtype=float)
    X, y, indices = lab.build_lagged(series, 3)

    np.testing.assert_allclose(X[0], [2.0, 1.0, 0.0])
    assert y[0] == 3.0
    assert indices[0] == 3


def test_experiment_selects_known_candidate_and_returns_finite_rmse() -> None:
    series, labels = lab.make_series(700, seed=42)
    report = lab.run_experiment(series, labels, seed=42)

    assert report["selected_model"] in {
        "seasonal_naive",
        "linear_ar",
        "mlp",
    }
    assert np.isfinite(report["test_rmse"])
    assert report["test_rmse"] > 0.0


def test_anomaly_detector_flags_some_points() -> None:
    series, labels = lab.make_series(700, seed=11)
    report = lab.run_experiment(series, labels, seed=11)

    total_flags = (
        report["anomaly_detection"]["flagged_train"]
        + report["anomaly_detection"]["flagged_validation"]
        + report["anomaly_detection"]["flagged_test"]
    )
    assert total_flags > 0
    assert report["anomaly_detection"]["true_anomalies_total"] > 0
