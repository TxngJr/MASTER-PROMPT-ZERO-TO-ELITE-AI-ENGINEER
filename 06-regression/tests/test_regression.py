from pathlib import Path
import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


MODULE_PATH = Path(__file__).parents[1] / "src" / "regression.py"
SPEC = importlib.util.spec_from_file_location("regression", MODULE_PATH)
assert SPEC and SPEC.loader
reg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reg)


def test_closed_form_recovers_exact_line() -> None:
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2.5 * X[:, 0] - 4.0

    model = reg.LinearRegressionClosedForm().fit(X, y)

    assert model.intercept_ == pytest.approx(-4.0)
    assert model.coef_[0] == pytest.approx(2.5)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)


def test_closed_form_matches_sklearn() -> None:
    X = np.array(
        [[0.0, 1.0], [1.0, 1.5], [2.0, -1.0], [3.0, 2.0], [4.0, 0.0]]
    )
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 3.0

    ours = reg.LinearRegressionClosedForm().fit(X, y)
    sklearn_model = LinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.coef_, sklearn_model.coef_, atol=1e-10)
    assert ours.intercept_ == pytest.approx(sklearn_model.intercept_, abs=1e-10)


def test_gradient_descent_recovers_scaled_line() -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = 3.0 * X[:, 0] + 2.0

    model = reg.LinearRegressionGD(
        learning_rate=0.1,
        steps=1_000,
    ).fit(X, y)

    assert model.coef_[0] == pytest.approx(3.0, abs=1e-5)
    assert model.intercept_ == pytest.approx(2.0, abs=1e-5)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_ridge_shrinks_correlated_coefficients() -> None:
    x = np.linspace(-1.0, 1.0, 50)
    X = np.column_stack([x, x])
    y = 4.0 * x

    ols = reg.LinearRegressionClosedForm().fit(X, y)
    ridge = reg.RidgeRegressionClosedForm(alpha=10.0).fit(X, y)

    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)


def test_polynomial_features() -> None:
    result = reg.polynomial_features_1d(np.array([2.0, 3.0]), degree=3)
    np.testing.assert_allclose(
        result,
        [[2.0, 4.0, 8.0], [3.0, 9.0, 27.0]],
    )
