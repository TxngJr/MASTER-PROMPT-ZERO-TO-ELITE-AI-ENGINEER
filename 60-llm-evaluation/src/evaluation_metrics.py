"""Reproducible LLM evaluation metrics and bootstrap utilities."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
import string

import numpy as np


def normalize_answer(text: str) -> str:
    value = text.lower().strip()
    value = value.translate(
        str.maketrans("", "", string.punctuation)
    )
    value = re.sub(r"\s+", " ", value)
    return value


def exact_match(
    predictions: list[str],
    references: list[str],
) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("matching non-empty lists required")

    return float(
        np.mean(
            [
                normalize_answer(prediction)
                == normalize_answer(reference)
                for prediction, reference
                in zip(predictions, references)
            ]
        )
    )


def _token_overlap_f1(
    prediction: str,
    reference: str,
) -> float:
    predicted = normalize_answer(prediction).split()
    target = normalize_answer(reference).split()

    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0

    common = Counter(predicted) & Counter(target)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(target)

    return float(
        2.0 * precision * recall
        / (precision + recall)
    )


def token_f1(
    predictions: list[str],
    references: list[str],
) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("matching non-empty lists required")

    return float(
        np.mean(
            [
                _token_overlap_f1(prediction, reference)
                for prediction, reference
                in zip(predictions, references)
            ]
        )
    )


def binary_brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if p.shape != y.shape or p.size == 0:
        raise ValueError("matching non-empty arrays required")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("outcomes must be binary")

    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if p.shape != y.shape or p.size == 0:
        raise ValueError("matching non-empty arrays required")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    total = len(p)
    ece = 0.0

    for index in range(num_bins):
        left = boundaries[index]
        right = boundaries[index + 1]

        if index == num_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)

        if not np.any(mask):
            continue

        confidence = float(np.mean(p[mask]))
        accuracy = float(np.mean(y[mask]))
        ece += (
            np.sum(mask) / total
            * abs(accuracy - confidence)
        )

    return float(ece)


def pairwise_rates(
    outcomes: list[str],
) -> dict[str, float]:
    if not outcomes:
        raise ValueError("outcomes must be non-empty")

    allowed = {"a", "b", "tie"}
    if any(outcome not in allowed for outcome in outcomes):
        raise ValueError("outcomes must be a, b or tie")

    total = len(outcomes)
    return {
        "a_win_rate": outcomes.count("a") / total,
        "b_win_rate": outcomes.count("b") / total,
        "tie_rate": outcomes.count("tie") / total,
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float).reshape(-1)

    if x.size == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(x),
        size=(num_resamples, len(x)),
    )
    means = np.mean(x[indices], axis=1)

    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return float(np.mean(x)), float(lower), float(upper)


def paired_bootstrap_difference(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    *,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    a = np.asarray(model_a_scores, dtype=float).reshape(-1)
    b = np.asarray(model_b_scores, dtype=float).reshape(-1)

    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired non-empty score arrays required")

    return bootstrap_mean_interval(
        a - b,
        confidence=confidence,
        num_resamples=num_resamples,
        seed=seed,
    )


def normalized_text_hashes(
    texts: list[str],
) -> set[str]:
    return {
        hashlib.sha256(
            normalize_answer(text).encode("utf-8")
        ).hexdigest()
        for text in texts
    }


def exact_contamination_rate(
    evaluation_texts: list[str],
    training_texts: list[str],
) -> float:
    if not evaluation_texts:
        raise ValueError("evaluation_texts must be non-empty")

    training_hashes = normalized_text_hashes(training_texts)
    contaminated = sum(
        hashlib.sha256(
            normalize_answer(text).encode("utf-8")
        ).hexdigest()
        in training_hashes
        for text in evaluation_texts
    )

    return float(contaminated / len(evaluation_texts))
