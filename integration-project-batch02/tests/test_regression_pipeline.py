from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "regression_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("regression_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
project = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(project)


def test_synthetic_dataset_is_reproducible() -> None:
    first = project.make_synthetic_dataset(100, seed=5)
    second = project.make_synthetic_dataset(100, seed=5)
    pd.testing.assert_frame_equal(first, second)


def test_split_is_disjoint_by_index() -> None:
    frame = project.make_synthetic_dataset(100, seed=10)
    X_train, X_val, X_test, *_ = project.split_dataset(
        frame,
        target_column="target",
        seed=10,
    )

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_val) + len(X_test) == len(frame)


def test_unknown_category_at_prediction_does_not_crash() -> None:
    frame = project.make_synthetic_dataset(120, seed=4)
    X_train, X_val, _, y_train, _, _ = project.split_dataset(
        frame,
        target_column="target",
        seed=4,
    )

    candidates = project.build_candidates(X_train)
    model = candidates["ridge_1.0"].fit(X_train, y_train)

    example = X_val.iloc[[0]].copy()
    example["group"] = "UNSEEN"
    prediction = model.predict(example)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_experiment_beats_mean_baseline_on_synthetic_signal() -> None:
    frame = project.make_synthetic_dataset(600, seed=42)
    report = project.run_experiment(frame, seed=42)

    selected = report["selected_model"]
    model_mse = report["test_results"][selected]["mse"]
    baseline_mse = report["test_results"]["mean_baseline"]["mse"]

    assert model_mse < baseline_mse
