"""Autoregressive forecasting primitives implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_lag_matrix(
    series: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(series, dtype=float).reshape(-1)
    if y.size <= n_lags:
        raise ValueError("series must be longer than n_lags")
    if n_lags <= 0:
        raise ValueError("n_lags must be positive")
    if not np.all(np.isfinite(y)):
        raise ValueError("series must be finite")

    X = np.array(
        [
            y[index - n_lags : index][::-1]
            for index in range(n_lags, len(y))
        ],
        dtype=float,
    )
    target = y[n_lags:].copy()
    return X, target


@dataclass
class AutoRegressorOLS:
    n_lags: int = 5
    intercept_: float | None = None
    coef_: np.ndarray | None = None
    history_: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "AutoRegressorOLS":
        X, y = make_lag_matrix(series, self.n_lags)
        design = np.column_stack([np.ones(len(X)), X])
        theta, *_ = np.linalg.lstsq(design, y, rcond=None)

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        self.history_ = np.asarray(series, dtype=float).reshape(-1).copy()
        return self

    def predict_from_lags(self, lag_rows: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("fit before prediction")
        X = np.asarray(lag_rows, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_lags:
            raise ValueError("lag feature shape mismatch")
        return X @ self.coef_ + self.intercept_

    def forecast(self, horizon: int) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("fit before forecast")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        history = self.history_.tolist()
        output: list[float] = []

        for _ in range(horizon):
            lags = np.asarray(history[-self.n_lags :][::-1], dtype=float)
            value = float(self.predict_from_lags(lags)[0])
            history.append(value)
            output.append(value)

        return np.asarray(output)


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    y = np.asarray(history, dtype=float).reshape(-1)
    if y.size == 0 or horizon <= 0:
        raise ValueError("history non-empty and horizon positive required")
    return np.full(horizon, y[-1], dtype=float)


def seasonal_naive_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    y = np.asarray(history, dtype=float).reshape(-1)
    if season_length <= 0 or y.size < season_length:
        raise ValueError("invalid season_length")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    pattern = y[-season_length:]
    return np.asarray(
        [pattern[index % season_length] for index in range(horizon)],
        dtype=float,
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("equal non-empty shapes required")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("equal non-empty shapes required")
    return float(np.sqrt(np.mean((a - b) ** 2)))
