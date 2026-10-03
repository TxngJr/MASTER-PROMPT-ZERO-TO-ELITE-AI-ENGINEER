"""Binary classification metrics implemented from definitions."""

from __future__ import annotations

import numpy as np


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)

    if y.shape != p.shape or y.size == 0:
        raise ValueError("inputs must have equal non-empty shape")
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isin(p, [0, 1])):
        raise ValueError("binary metrics require labels 0/1")

    return {
        "tn": int(np.sum((y == 0) & (p == 0))),
        "fp": int(np.sum((y == 0) & (p == 1))),
        "fn": int(np.sum((y == 1) & (p == 0))),
        "tp": int(np.sum((y == 1) & (p == 1))),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    total = c["tn"] + c["fp"] + c["fn"] + c["tp"]

    precision = _safe_ratio(c["tp"], c["tp"] + c["fp"])
    recall = _safe_ratio(c["tp"], c["tp"] + c["fn"])
    specificity = _safe_ratio(c["tn"], c["tn"] + c["fp"])
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

    return {
        "accuracy": (c["tp"] + c["tn"]) / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }
