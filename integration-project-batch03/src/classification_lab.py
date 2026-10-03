"""Batch 03 integration: compare classifiers and tune threshold safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_dataset(
    n_samples: int = 1200,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200")

    return make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000)),
            ]
        ),
        "knn_7": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "gaussian_nb": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
    }


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
    }


def choose_threshold(
    y_validation: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scored = [
        binary_metrics(y_validation, probability, threshold=float(t))
        for t in thresholds
    ]
    best = max(
        scored,
        key=lambda item: (
            item["f1"],
            item["recall"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return best["threshold"], best


def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
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
    ) = split_dataset(X, y, seed=seed)

    validation_results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}

    for name, model in candidates().items():
        model.fit(X_train, y_train)
        fitted[name] = model

        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold, metrics = choose_threshold(
            y_validation,
            validation_probability,
        )
        thresholds[name] = threshold
        validation_results[name] = metrics

    selected = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f1"],
            validation_results[name]["pr_auc"],
        ),
    )

    selected_model = fitted[selected]
    selected_threshold = thresholds[selected]
    test_probability = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = binary_metrics(
        y_test,
        test_probability,
        threshold=selected_threshold,
    )

    return {
        "seed": seed,
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "split_sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_validation)),
            "test": int(len(y_test)),
        },
        "validation_results": validation_results,
        "selected_model": selected,
        "selected_threshold": float(selected_threshold),
        "test_metrics": test_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 03 classification model lab."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch03"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    X, y = make_dataset(args.samples, seed=args.seed)
    report = run_experiment(X, y, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"selected: {report['selected_model']}")
    print(f"threshold: {report['selected_threshold']:.2f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
