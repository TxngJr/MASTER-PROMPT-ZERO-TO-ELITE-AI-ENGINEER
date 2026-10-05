"""Framework-independent utilities for rigorous paper reproduction."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


def relative_error(reference: float, reproduced: float, *, epsilon: float = 1e-12) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    denominator = max(abs(float(reference)), epsilon)
    return abs(float(reproduced) - float(reference)) / denominator


def result_within_tolerance(
    reference: float,
    reproduced: float,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> bool:
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    difference = abs(float(reproduced) - float(reference))
    allowed = max(float(absolute_tolerance), abs(float(reference)) * float(relative_tolerance))
    return difference <= allowed


def normalized_improvement_recovery(
    *,
    baseline: float,
    claimed: float,
    reproduced: float,
) -> float:
    denominator = float(claimed) - float(baseline)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        raise ValueError("claimed result must differ from baseline")
    return (float(reproduced) - float(baseline)) / denominator


def ablation_effect(
    *,
    full_metric: float,
    ablated_metric: float,
    higher_is_better: bool = True,
) -> float:
    if higher_is_better:
        return float(full_metric) - float(ablated_metric)
    return float(ablated_metric) - float(full_metric)


def seed_statistics(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    mean = float(np.mean(array))
    if array.size == 1:
        std = 0.0
        sem = 0.0
    else:
        std = float(np.std(array, ddof=1))
        sem = std / math.sqrt(array.size)

    margin = 1.96 * sem
    return {
        "count": int(array.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def accelerator_hours(*, accelerator_count: int, wall_clock_hours: float) -> float:
    if accelerator_count < 0:
        raise ValueError("accelerator_count must be non-negative")
    if wall_clock_hours < 0:
        raise ValueError("wall_clock_hours must be non-negative")
    return float(accelerator_count) * float(wall_clock_hours)


def experiment_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


REQUIRED_REPRODUCIBILITY_FIELDS = (
    "paper",
    "code_commit",
    "dataset",
    "dataset_version",
    "seed",
    "hardware",
    "python",
    "dependencies",
    "model_config",
    "optimizer",
    "metric",
)


def audit_reproducibility_metadata(
    metadata: dict[str, Any],
    *,
    required_fields: Iterable[str] = REQUIRED_REPRODUCIBILITY_FIELDS,
) -> dict[str, Any]:
    required = tuple(required_fields)
    missing = [
        field
        for field in required
        if field not in metadata or metadata[field] in (None, "", [], {})
    ]
    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "present_count": len(required) - len(missing),
        "required_count": len(required),
    }
