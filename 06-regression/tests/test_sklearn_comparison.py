from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "sklearn_comparison.py"
SPEC = importlib.util.spec_from_file_location("sklearn_comparison", MODULE_PATH)
assert SPEC and SPEC.loader
sk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sk)


def test_linear_pipeline_fits_exact_relation() -> None:
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 5.0 * X[:, 0] + 1.0

    model = sk.build_linear_pipeline().fit(X, y)
    pred = model.predict(X)
    metrics = sk.regression_metrics(y, pred)

    assert metrics["mse"] < 1e-20
    assert metrics["r2"] > 0.999999999


def test_polynomial_pipeline_fits_quadratic() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    X = x.reshape(-1, 1)
    y = 2.0 * x**2 - 3.0 * x + 1.0

    model = sk.build_polynomial_pipeline(degree=2).fit(X, y)
    pred = model.predict(X)

    assert sk.regression_metrics(y, pred)["mse"] < 1e-20
