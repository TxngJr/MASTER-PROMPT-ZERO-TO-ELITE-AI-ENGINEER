"""Compare regression families using a leakage-safe scikit-learn workflow."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_linear_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_polynomial_pipeline(degree: int = 2) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_lasso_pipeline(alpha: float = 0.01) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20_000)),
        ]
    )


def build_elastic_net_pipeline(
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=20_000,
                ),
            ),
        ]
    )
