"""Batch 06 integration: anomaly detection, time series, and an MLP baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_series(
    n_points: int = 900,
    *,
    seed: int = 42,
    season_length: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    if n_points < 300:
        raise ValueError("n_points must be at least 300")

    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    trend = 0.003 * t
    seasonal = 1.5 * np.sin(2.0 * np.pi * t / season_length)
    noise = rng.normal(0.0, 0.18, n_points)

    values = trend + seasonal + noise

    # Inject rare point anomalies for evaluation only.
    anomaly_labels = np.zeros(n_points, dtype=int)
    candidates = np.arange(60, n_points - 20)
    anomaly_indices = rng.choice(candidates, size=max(8, n_points // 90), replace=False)
    values[anomaly_indices] += rng.choice([-4.0, 4.0], size=len(anomaly_indices))
    anomaly_labels[anomaly_indices] = 1

    return values.astype(float), anomaly_labels


def chronological_boundaries(n_points: int) -> tuple[int, int]:
    train_end = int(n_points * 0.70)
    validation_end = int(n_points * 0.85)
    return train_end, validation_end


def anomaly_features(series: np.ndarray, season_length: int = 24) -> np.ndarray:
    y = np.asarray(series, dtype=float).reshape(-1)
    diff1 = np.zeros_like(y)
    diff1[1:] = y[1:] - y[:-1]

    seasonal_diff = np.zeros_like(y)
    seasonal_diff[season_length:] = y[season_length:] - y[:-season_length]

    return np.column_stack([y, diff1, seasonal_diff])


def build_lagged(
    series: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(series, dtype=float).reshape(-1)
    if n_lags <= 0 or len(y) <= n_lags:
        raise ValueError("invalid n_lags")

    X = []
    target = []
    indices = []

    for t in range(n_lags, len(y)):
        X.append(y[t - n_lags : t][::-1])
        target.append(y[t])
        indices.append(t)

    return np.asarray(X), np.asarray(target), np.asarray(indices)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def seasonal_naive_predictions(
    series: np.ndarray,
    target_indices: np.ndarray,
    season_length: int,
) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    indices = np.asarray(target_indices, dtype=int)
    if np.any(indices < season_length):
        raise ValueError("not enough seasonal history")
    return y[indices - season_length]


def run_experiment(
    series: np.ndarray,
    anomaly_labels: np.ndarray,
    *,
    seed: int = 42,
    season_length: int = 24,
    n_lags: int = 24,
) -> dict[str, Any]:
    train_end, validation_end = chronological_boundaries(len(series))

    features = anomaly_features(series, season_length=season_length)
    anomaly_model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(features[:train_end])

    anomaly_scores = -anomaly_model.decision_function(features)
    anomaly_flags = (anomaly_model.predict(features) == -1).astype(int)

    X, y, indices = build_lagged(series, n_lags)

    train_mask = indices < train_end
    validation_mask = (indices >= train_end) & (indices < validation_end)
    test_mask = indices >= validation_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_validation, y_validation = X[validation_mask], y[validation_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    validation_indices = indices[validation_mask]
    test_indices = indices[test_mask]

    linear = LinearRegression().fit(X_train, y_train)
    mlp = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.005,
                    max_iter=600,
                    random_state=seed,
                ),
            ),
        ]
    ).fit(X_train, y_train)

    validation_predictions = {
        "seasonal_naive": seasonal_naive_predictions(
            series,
            validation_indices,
            season_length,
        ),
        "linear_ar": linear.predict(X_validation),
        "mlp": mlp.predict(X_validation),
    }

    validation_rmse = {
        name: rmse(y_validation, prediction)
        for name, prediction in validation_predictions.items()
    }

    selected = min(validation_rmse, key=validation_rmse.get)

    if selected == "seasonal_naive":
        test_prediction = seasonal_naive_predictions(
            series,
            test_indices,
            season_length,
        )
    elif selected == "linear_ar":
        test_prediction = linear.predict(X_test)
    else:
        test_prediction = mlp.predict(X_test)

    return {
        "seed": seed,
        "rows": int(len(series)),
        "split_points": {
            "train_end": train_end,
            "validation_end": validation_end,
        },
        "anomaly_detection": {
            "true_anomalies_total": int(np.sum(anomaly_labels)),
            "flagged_train": int(np.sum(anomaly_flags[:train_end])),
            "flagged_validation": int(
                np.sum(anomaly_flags[train_end:validation_end])
            ),
            "flagged_test": int(np.sum(anomaly_flags[validation_end:])),
            "score_mean_train": float(np.mean(anomaly_scores[:train_end])),
            "score_mean_test": float(np.mean(anomaly_scores[validation_end:])),
        },
        "validation_rmse": {
            name: float(value) for name, value in validation_rmse.items()
        },
        "selected_model": selected,
        "test_rmse": float(rmse(y_test, test_prediction)),
        "lag_features": int(n_lags),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 06 anomaly-aware forecasting/neural lab."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch06"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    series, labels = make_series(args.points, seed=args.seed)
    report = run_experiment(series, labels, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"selected model: {report['selected_model']}")
    print(f"test RMSE: {report['test_rmse']:.4f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
