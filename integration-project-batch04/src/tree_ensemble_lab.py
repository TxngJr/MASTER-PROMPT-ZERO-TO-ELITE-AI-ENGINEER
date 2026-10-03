"""Batch 04 integration benchmark for tree-based ensemble families."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def make_dataset(
    n_samples: int = 1800,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 300:
        raise ValueError("n_samples must be at least 300")

    return make_classification(
        n_samples=n_samples,
        n_features=14,
        n_informative=8,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        weights=[0.72, 0.28],
        class_sep=1.0,
        flip_y=0.03,
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


def candidates(seed: int) -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1,
                random_state=seed,
            ),
            n_estimators=100,
            learning_rate=0.5,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=15,
            random_state=seed,
        ),
    }


def evaluate(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    started = time.perf_counter()
    prediction = model.predict(X)
    predict_seconds = time.perf_counter() - started

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:, 1]
        roc_auc = float(roc_auc_score(y, probability))
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
        roc_auc = float(roc_auc_score(y, score))
    else:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "roc_auc": roc_auc,
        "predict_seconds": float(predict_seconds),
    }


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
    fitted: dict[str, object] = {}

    for name, model in candidates(seed).items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        metrics = evaluate(model, X_validation, y_validation)
        metrics["fit_seconds"] = float(fit_seconds)
        validation_results[name] = metrics
        fitted[name] = model

    selected = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f1"],
            validation_results[name]["roc_auc"],
        ),
    )

    test_metrics = evaluate(fitted[selected], X_test, y_test)

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
        "test_metrics": test_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 04 tree ensemble benchmark."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch04"),
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
    print(f"report: {output}")


if __name__ == "__main__":
    main()
