from pathlib import Path
import importlib.util
import sys

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


MODULE_PATH = Path(__file__).parents[1] / "src" / "autoregression.py"
SPEC = importlib.util.spec_from_file_location("autoregression_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_lag_matrix_has_expected_order() -> None:
    X, y = mod.make_lag_matrix(np.array([1, 2, 3, 4, 5], dtype=float), 2)
    np.testing.assert_allclose(X, [[2, 1], [3, 2], [4, 3]])
    np.testing.assert_allclose(y, [3, 4, 5])


def test_autoregression_learns_ar1_process() -> None:
    rng = np.random.default_rng(4)
    values = [0.5]
    for _ in range(600):
        values.append(0.8 * values[-1] + rng.normal(0, 0.03))

    series = np.asarray(values)
    model = mod.AutoRegressorOLS(n_lags=1).fit(series)

    assert abs(model.coef_[0] - 0.8) < 0.08


def test_forecast_is_recursive_and_finite() -> None:
    series = np.sin(np.linspace(0, 8, 200))
    model = mod.AutoRegressorOLS(n_lags=8).fit(series)
    forecast = model.forecast(12)

    assert forecast.shape == (12,)
    assert np.all(np.isfinite(forecast))


def test_statsmodels_arima_current_api_runs() -> None:
    rng = np.random.default_rng(2)
    values = [0.0]
    for _ in range(150):
        values.append(0.6 * values[-1] + rng.normal(0, 0.2))
    series = np.asarray(values)

    fit = ARIMA(series, order=(1, 0, 0)).fit()
    forecast = fit.forecast(steps=5)

    assert len(forecast) == 5
    assert np.all(np.isfinite(forecast))
