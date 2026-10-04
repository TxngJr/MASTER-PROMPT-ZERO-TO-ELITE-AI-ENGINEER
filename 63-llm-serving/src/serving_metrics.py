"""Serving SLO, capacity, and rollout calculations."""

from __future__ import annotations

import math
import numpy as np


def percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must lie in [0,100]")
    if any(value < 0 for value in values):
        raise ValueError("latencies must be non-negative")

    return float(np.percentile(np.asarray(values, dtype=float), q))


def request_latency(
    *,
    queue_seconds: float,
    prefill_seconds: float,
    decode_seconds: float,
    network_seconds: float = 0.0,
) -> float:
    values = [
        queue_seconds,
        prefill_seconds,
        decode_seconds,
        network_seconds,
    ]
    if any(value < 0 for value in values):
        raise ValueError("latencies must be non-negative")
    return float(sum(values))


def throughput(
    *,
    completed_units: int,
    elapsed_seconds: float,
) -> float:
    if completed_units < 0:
        raise ValueError("completed_units must be non-negative")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return float(completed_units / elapsed_seconds)


def success_rate(
    *,
    successes: int,
    total_requests: int,
) -> float:
    if total_requests <= 0:
        raise ValueError("total_requests must be positive")
    if not 0 <= successes <= total_requests:
        raise ValueError("successes out of range")
    return float(successes / total_requests)


def little_law_concurrency(
    *,
    arrival_rate_per_second: float,
    average_latency_seconds: float,
) -> float:
    if arrival_rate_per_second < 0:
        raise ValueError("arrival rate must be non-negative")
    if average_latency_seconds < 0:
        raise ValueError("latency must be non-negative")
    return float(
        arrival_rate_per_second * average_latency_seconds
    )


def required_replicas(
    *,
    offered_requests_per_second: float,
    sustainable_requests_per_second_per_replica: float,
    target_utilization: float = 0.7,
) -> int:
    if offered_requests_per_second < 0:
        raise ValueError("offered load must be non-negative")
    if sustainable_requests_per_second_per_replica <= 0:
        raise ValueError("replica capacity must be positive")
    if not 0.0 < target_utilization <= 1.0:
        raise ValueError("target_utilization must lie in (0,1]")

    usable = (
        sustainable_requests_per_second_per_replica
        * target_utilization
    )
    if offered_requests_per_second == 0:
        return 0
    return int(math.ceil(offered_requests_per_second / usable))


def token_rate_limit_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    output_weight: float = 1.0,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if output_weight < 0:
        raise ValueError("output_weight must be non-negative")
    return float(
        input_tokens + output_weight * output_tokens
    )


def canary_split(
    total_requests: int,
    *,
    canary_fraction: float,
) -> tuple[int, int]:
    if total_requests < 0:
        raise ValueError("total_requests must be non-negative")
    if not 0.0 <= canary_fraction <= 1.0:
        raise ValueError("canary_fraction must lie in [0,1]")

    canary = int(round(total_requests * canary_fraction))
    stable = total_requests - canary
    return stable, canary
