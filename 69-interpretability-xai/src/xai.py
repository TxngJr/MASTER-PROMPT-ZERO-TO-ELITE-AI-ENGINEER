"""Small, framework-independent explainability algorithms."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Callable

import numpy as np


ArrayFunction = Callable[[np.ndarray], float]


def finite_difference_gradient(
    function: ArrayFunction,
    x: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    point = np.asarray(x, dtype=float)
    gradient = np.zeros_like(point)

    for index in range(point.size):
        plus = point.copy().reshape(-1)
        minus = point.copy().reshape(-1)
        plus[index] += epsilon
        minus[index] -= epsilon

        gradient.reshape(-1)[index] = (
            function(plus.reshape(point.shape))
            - function(minus.reshape(point.shape))
        ) / (2.0 * epsilon)

    return gradient


def integrated_gradients(
    function: ArrayFunction,
    x: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    steps: int = 100,
    epsilon: float = 1e-5,
) -> np.ndarray:
    point = np.asarray(x, dtype=float)
    base = (
        np.zeros_like(point)
        if baseline is None
        else np.asarray(baseline, dtype=float)
    )

    if point.shape != base.shape:
        raise ValueError("x/baseline shape mismatch")
    if steps <= 0:
        raise ValueError("steps must be positive")

    delta = point - base
    gradient_sum = np.zeros_like(point)

    for step in range(steps):
        alpha = (step + 0.5) / steps
        sample = base + alpha * delta
        gradient_sum += finite_difference_gradient(
            function,
            sample,
            epsilon=epsilon,
        )

    average_gradient = gradient_sum / steps
    return delta * average_gradient


def completeness_gap(
    function: ArrayFunction,
    x: np.ndarray,
    attributions: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> float:
    point = np.asarray(x, dtype=float)
    base = (
        np.zeros_like(point)
        if baseline is None
        else np.asarray(baseline, dtype=float)
    )
    attrs = np.asarray(attributions, dtype=float)

    if point.shape != base.shape or point.shape != attrs.shape:
        raise ValueError("shape mismatch")

    target_difference = function(point) - function(base)
    return float(
        target_difference - np.sum(attrs)
    )


def occlusion_importance(
    function: ArrayFunction,
    x: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    point = np.asarray(x, dtype=float)
    base = (
        np.zeros_like(point)
        if baseline is None
        else np.asarray(baseline, dtype=float)
    )

    if point.shape != base.shape:
        raise ValueError("shape mismatch")

    original = function(point)
    importance = np.zeros_like(point)

    flat_point = point.reshape(-1)
    flat_base = base.reshape(-1)
    flat_output = importance.reshape(-1)

    for index in range(flat_point.size):
        occluded = flat_point.copy()
        occluded[index] = flat_base[index]
        flat_output[index] = (
            original
            - function(occluded.reshape(point.shape))
        )

    return importance


def exact_shapley_values(
    value_function: Callable[[frozenset[int]], float],
    *,
    num_features: int,
) -> np.ndarray:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if num_features > 12:
        raise ValueError(
            "exact Shapley is exponential; keep num_features <= 12"
        )

    features = list(range(num_features))
    factorial_n = math.factorial(num_features)
    values = np.zeros(num_features, dtype=float)

    for feature in features:
        others = [item for item in features if item != feature]

        for subset_size in range(len(others) + 1):
            weight = (
                math.factorial(subset_size)
                * math.factorial(
                    num_features - subset_size - 1
                )
                / factorial_n
            )

            for subset_tuple in combinations(
                others,
                subset_size,
            ):
                subset = frozenset(subset_tuple)
                with_feature = subset | {feature}
                marginal = (
                    value_function(with_feature)
                    - value_function(subset)
                )
                values[feature] += weight * marginal

    return values


def permutation_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    repeats: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x)
    targets = np.asarray(y)

    if features.ndim != 2:
        raise ValueError("x must be a matrix")
    if len(features) != len(targets):
        raise ValueError("x/y length mismatch")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    baseline_score = metric(
        targets,
        np.asarray(predict(features)),
    )
    rng = np.random.default_rng(seed)
    all_drops = np.zeros(
        (repeats, features.shape[1]),
        dtype=float,
    )

    for repeat in range(repeats):
        for feature in range(features.shape[1]):
            permuted = features.copy()
            order = rng.permutation(len(features))
            permuted[:, feature] = features[order, feature]
            score = metric(
                targets,
                np.asarray(predict(permuted)),
            )
            all_drops[repeat, feature] = (
                baseline_score - score
            )

    return (
        np.mean(all_drops, axis=0),
        np.std(all_drops, axis=0),
    )


def cosine_attribution_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)

    if a.shape != b.shape or a.size == 0:
        raise ValueError("matching non-empty vectors required")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)

    # Numerical roundoff can produce tiny excursions outside the
    # mathematical cosine range; clamp them away.
    return float(np.clip(similarity, -1.0, 1.0))
