from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MODULE_PATH = Path(__file__).parents[1] / "src" / "logistic_regression.py"
SPEC = importlib.util.spec_from_file_location("logistic_regression", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_sigmoid_extreme_values_are_finite() -> None:
    values = mod.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(values))
    assert values[0] < 1e-10
    assert values[1] == pytest.approx(0.5)
    assert values[2] > 1.0 - 1e-10


def test_log_loss_prefers_better_probabilities() -> None:
    y = np.array([0, 1, 1, 0])
    good = np.array([0.1, 0.9, 0.8, 0.2])
    bad = np.array([0.9, 0.1, 0.2, 0.8])
    assert mod.binary_log_loss(y, good) < mod.binary_log_loss(y, bad)


def test_from_scratch_learns_separable_data() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 2))
    y = (2.0 * X[:, 0] - X[:, 1] > 0.2).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = mod.LogisticRegressionGD(
        learning_rate=0.1,
        steps=2500,
    ).fit(X_scaled, y)

    accuracy = np.mean(model.predict(X_scaled) == y)
    assert accuracy > 0.97
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_from_scratch_agrees_with_sklearn_classifications() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 2))
    y = (1.5 * X[:, 0] + 0.7 * X[:, 1] > -0.1).astype(int)

    X_scaled = StandardScaler().fit_transform(X)

    ours = mod.LogisticRegressionGD(
        learning_rate=0.1,
        steps=3000,
    ).fit(X_scaled, y)

    sk = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=3000,
    ).fit(X_scaled, y)

    agreement = np.mean(ours.predict(X_scaled) == sk.predict(X_scaled))
    assert agreement > 0.98
