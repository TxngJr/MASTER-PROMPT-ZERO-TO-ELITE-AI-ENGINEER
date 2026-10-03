from pathlib import Path
import importlib.util

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def test_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert ev.mean_squared_error(y_true, y_pred) == pytest.approx(1 / 3)
    assert ev.mean_absolute_error(y_true, y_pred) == pytest.approx(1 / 3)
    assert ev.r2_score(y_true, y_pred) == pytest.approx(0.5)


def test_r2_can_be_negative() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 10.0, 10.0])
    assert ev.r2_score(y_true, y_pred) < 0.0


def test_mean_baseline_uses_training_targets() -> None:
    prediction = ev.mean_baseline(np.array([1.0, 3.0]), 3)
    np.testing.assert_allclose(prediction, [2.0, 2.0, 2.0])


def test_kfold_covers_each_validation_sample_once() -> None:
    folds = ev.kfold_indices(11, n_splits=4, seed=9)
    validation = np.concatenate([val for _, val in folds])

    assert sorted(validation.tolist()) == list(range(11))
    for train, val in folds:
        assert len(set(train) & set(val)) == 0


def test_gradient_descent_converges_with_reasonable_learning_rate() -> None:
    history = ev.gradient_descent_quadratic(
        initial_x=-10.0,
        learning_rate=0.1,
        steps=100,
    )
    assert history[-1] == pytest.approx(3.0, abs=1e-8)
