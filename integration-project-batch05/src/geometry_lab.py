"""Batch 05 integration: SVM, PCA and clustering on shared geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_dataset(
    n_samples: int = 1400,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 300:
        raise ValueError("n_samples must be at least 300")

    return make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=7,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.025,
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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
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

    baseline_svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearSVC(C=1.0, max_iter=10_000, random_state=seed)),
        ]
    )
    baseline_svm.fit(X_train, y_train)

    pca_svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("model", LinearSVC(C=1.0, max_iter=10_000, random_state=seed)),
        ]
    )
    pca_svm.fit(X_train, y_train)

    validation_results = {
        "scaled_linear_svm": classification_metrics(
            y_validation,
            baseline_svm.predict(X_validation),
        ),
        "pca_linear_svm": classification_metrics(
            y_validation,
            pca_svm.predict(X_validation),
        ),
    }

    selected_name = max(
        validation_results,
        key=lambda name: validation_results[name]["f1"],
    )
    selected_model = baseline_svm if selected_name == "scaled_linear_svm" else pca_svm

    test_metrics = classification_metrics(
        y_test,
        selected_model.predict(X_test),
    )

    # Unsupervised branch: fit representation on TRAIN only.
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)

    pca = PCA(n_components=0.95, svd_solver="full").fit(train_scaled)
    train_reduced = pca.transform(train_scaled)

    kmeans = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init="auto",
        random_state=seed,
    ).fit(train_reduced)

    silhouette = float(silhouette_score(train_reduced, kmeans.labels_))
    ari = float(adjusted_rand_score(y_train, kmeans.labels_))

    # eps is a fixed lab setting, not label-tuned.
    dbscan_labels = DBSCAN(eps=1.6, min_samples=8).fit_predict(train_reduced)
    non_noise = dbscan_labels[dbscan_labels != -1]
    n_clusters = int(len(np.unique(non_noise))) if non_noise.size else 0

    return {
        "seed": seed,
        "rows": int(len(y)),
        "original_features": int(X.shape[1]),
        "pca_features": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "validation_results": validation_results,
        "selected_classifier": selected_name,
        "test_metrics": test_metrics,
        "kmeans": {
            "silhouette": silhouette,
            "adjusted_rand_diagnostic": ari,
            "inertia": float(kmeans.inertia_),
        },
        "dbscan": {
            "clusters": n_clusters,
            "noise_points": int(np.sum(dbscan_labels == -1)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 05 geometry and representation lab."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1400)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch05"),
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

    print(f"selected classifier: {report['selected_classifier']}")
    print(f"PCA dimensions: {report['pca_features']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
