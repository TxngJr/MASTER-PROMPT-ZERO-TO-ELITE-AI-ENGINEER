"""Batch 02 integration: leakage-safe regression model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_synthetic_dataset(
    n_samples: int = 600,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if n_samples < 50:
        raise ValueError("n_samples must be at least 50")

    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, n_samples)
    x2 = rng.normal(3.0, 2.0, n_samples)
    group = rng.choice(["A", "B", "C"], size=n_samples, p=[0.5, 0.3, 0.2])

    group_effect = np.select(
        [group == "A", group == "B", group == "C"],
        [0.0, 2.0, -1.5],
    )
    noise = rng.normal(0.0, 0.8, n_samples)

    target = 4.0 * x1 - 1.5 * x2 + group_effect + noise

    frame = pd.DataFrame(
        {
            "feature_a": x1,
            "feature_b": x2,
            "group": group,
            "target": target,
        }
    )

    missing_a = rng.random(n_samples) < 0.06
    missing_b = rng.random(n_samples) < 0.04
    frame.loc[missing_a, "feature_a"] = np.nan
    frame.loc[missing_b, "feature_b"] = np.nan

    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if target_column not in frame.columns:
        raise ValueError(f"missing target column: {target_column}")

    X = frame.drop(columns=[target_column])
    y = frame[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )


def build_candidates(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    specs = {
        "linear": LinearRegression(),
        "ridge_0.1": Ridge(alpha=0.1),
        "ridge_1.0": Ridge(alpha=1.0),
        "ridge_10.0": Ridge(alpha=10.0),
        "lasso_0.001": Lasso(alpha=0.001, max_iter=20_000),
        "lasso_0.01": Lasso(alpha=0.01, max_iter=20_000),
        "lasso_0.1": Lasso(alpha=0.1, max_iter=20_000),
    }

    return {
        name: Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        for name, model in specs.items()
    }


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, prediction)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def run_experiment(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(frame, target_column="target", seed=seed)

    baseline_validation = np.full(len(y_validation), y_train.mean())
    baseline_test = np.full(len(y_test), y_train.mean())

    validation_results: dict[str, dict[str, float]] = {
        "mean_baseline": metrics(y_validation, baseline_validation)
    }

    fitted_candidates: dict[str, Pipeline] = {}

    for name, pipeline in build_candidates(X_train).items():
        pipeline.fit(X_train, y_train)
        fitted_candidates[name] = pipeline
        validation_results[name] = metrics(
            y_validation,
            pipeline.predict(X_validation),
        )

    best_name = min(
        fitted_candidates,
        key=lambda name: validation_results[name]["mse"],
    )
    best_model = fitted_candidates[best_name]

    test_results = {
        "mean_baseline": metrics(y_test, baseline_test),
        best_name: metrics(y_test, best_model.predict(X_test)),
    }

    return {
        "seed": seed,
        "rows": int(frame.shape[0]),
        "features": [column for column in frame.columns if column != "target"],
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
        },
        "training_missing": {
            column: int(count)
            for column, count in X_train.isna().sum().items()
        },
        "validation_results": validation_results,
        "selected_model": best_name,
        "test_results": test_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Batch 02 leakage-safe regression experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch02"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = make_synthetic_dataset(args.samples, seed=args.seed)
    report = run_experiment(frame, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"selected model: {report['selected_model']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
