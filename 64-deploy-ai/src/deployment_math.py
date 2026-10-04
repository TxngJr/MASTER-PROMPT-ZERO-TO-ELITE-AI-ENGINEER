"""Deployment, rollout, autoscaling and GPU-capacity calculations."""

from __future__ import annotations

import math


def desired_replicas_from_metric(
    *,
    current_replicas: int,
    current_metric: float,
    target_metric: float,
    min_replicas: int = 1,
    max_replicas: int = 100,
) -> int:
    if current_replicas < 0:
        raise ValueError("current_replicas must be non-negative")
    if current_metric < 0:
        raise ValueError("current_metric must be non-negative")
    if target_metric <= 0:
        raise ValueError("target_metric must be positive")
    if min_replicas < 0 or max_replicas < min_replicas:
        raise ValueError("invalid replica bounds")

    raw = math.ceil(
        current_replicas * current_metric / target_metric
    )
    return int(min(max(raw, min_replicas), max_replicas))


def rolling_update_bounds(
    *,
    desired_replicas: int,
    max_surge: int,
    max_unavailable: int,
) -> dict[str, int]:
    if desired_replicas < 0:
        raise ValueError("desired_replicas must be non-negative")
    if max_surge < 0 or max_unavailable < 0:
        raise ValueError("rollout values must be non-negative")
    if (
        desired_replicas > 0
        and max_surge == 0
        and max_unavailable == 0
    ):
        raise ValueError(
            "rollout cannot have zero surge and zero unavailable"
        )

    return {
        "max_total_pods": desired_replicas + max_surge,
        "min_available_pods": max(
            0,
            desired_replicas - max_unavailable,
        ),
    }


def canary_request_counts(
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


def availability(
    *,
    successful_requests: int,
    total_requests: int,
) -> float:
    if total_requests <= 0:
        raise ValueError("total_requests must be positive")
    if not 0 <= successful_requests <= total_requests:
        raise ValueError("successful_requests out of range")

    return float(successful_requests / total_requests)


def capacity_with_headroom(
    *,
    replicas: int,
    sustainable_rps_per_replica: float,
    target_utilization: float,
) -> float:
    if replicas < 0:
        raise ValueError("replicas must be non-negative")
    if sustainable_rps_per_replica < 0:
        raise ValueError("capacity must be non-negative")
    if not 0.0 < target_utilization <= 1.0:
        raise ValueError("target_utilization must lie in (0,1]")

    return float(
        replicas
        * sustainable_rps_per_replica
        * target_utilization
    )


def gpu_pool_capacity(
    *,
    nodes: int,
    gpus_per_node: int,
    gpus_per_replica: int,
    reserved_gpus: int = 0,
) -> int:
    values = [
        nodes,
        gpus_per_node,
        gpus_per_replica,
        reserved_gpus,
    ]
    if any(value < 0 for value in values):
        raise ValueError("counts must be non-negative")
    if gpus_per_replica <= 0:
        raise ValueError("gpus_per_replica must be positive")

    available = nodes * gpus_per_node - reserved_gpus
    if available < 0:
        raise ValueError("reserved_gpus exceeds pool")

    return int(available // gpus_per_replica)
