"""Framework-independent MLOps tracking, drift and promotion utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def experiment_fingerprint(
    *,
    code_revision: str,
    dataset_revision: str,
    config: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> str:
    payload = {
        "code_revision": code_revision,
        "dataset_revision": dataset_revision,
        "config": config,
        "environment": environment or {},
    }
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def population_stability_index(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
    *,
    smoothing: float = 1e-6,
) -> float:
    ref = np.asarray(reference_counts, dtype=float)
    cur = np.asarray(current_counts, dtype=float)

    if ref.shape != cur.shape or ref.size == 0:
        raise ValueError("matching non-empty counts required")
    if np.any(ref < 0) or np.any(cur < 0):
        raise ValueError("counts must be non-negative")
    if ref.sum() <= 0 or cur.sum() <= 0:
        raise ValueError("each histogram needs positive mass")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    p = ref / ref.sum()
    q = cur / cur.sum()

    p = np.clip(p, smoothing, None)
    q = np.clip(q, smoothing, None)
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum((q - p) * np.log(q / p)))


def missing_rate(values: list[Any]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    missing = sum(
        value is None
        or (
            isinstance(value, float)
            and np.isnan(value)
        )
        for value in values
    )
    return float(missing / len(values))


def promotion_gate(
    metrics: dict[str, float],
    requirements: dict[str, tuple[str, float]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    for name, (operator, threshold) in requirements.items():
        if name not in metrics:
            failures.append(f"missing metric: {name}")
            continue

        value = metrics[name]

        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        else:
            raise ValueError(
                f"unsupported operator for {name}: {operator}"
            )

        if not passed:
            failures.append(
                f"{name}={value} fails {operator} {threshold}"
            )

    return len(failures) == 0, failures


def registry_alias_update(
    aliases: dict[str, str],
    *,
    alias: str,
    version: str,
) -> dict[str, str]:
    if not alias or not version:
        raise ValueError("alias and version must be non-empty")

    updated = dict(aliases)
    updated[alias] = version
    return updated


def rolling_error_rate(
    outcomes: list[bool],
    *,
    window: int,
) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    if not outcomes:
        return []

    rates = []
    for end in range(1, len(outcomes) + 1):
        start = max(0, end - window)
        values = outcomes[start:end]
        errors = sum(not value for value in values)
        rates.append(errors / len(values))

    return rates
