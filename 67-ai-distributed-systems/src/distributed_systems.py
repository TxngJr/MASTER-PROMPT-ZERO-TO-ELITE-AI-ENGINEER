"""Distributed-systems reliability and placement utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics


def exponential_backoff(
    attempt: int,
    *,
    base_seconds: float = 0.5,
    cap_seconds: float = 30.0,
) -> float:
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base_seconds <= 0 or cap_seconds <= 0:
        raise ValueError("delays must be positive")
    return float(
        min(cap_seconds, base_seconds * (2**attempt))
    )


def queue_utilization(
    *,
    arrival_rate: float,
    service_rate_per_worker: float,
    workers: int,
) -> float:
    if arrival_rate < 0:
        raise ValueError("arrival_rate must be non-negative")
    if service_rate_per_worker <= 0 or workers <= 0:
        raise ValueError("service capacity must be positive")

    return float(
        arrival_rate
        / (service_rate_per_worker * workers)
    )


def bounded_admission(
    *,
    queue_size: int,
    queue_capacity: int,
    incoming: int = 1,
) -> tuple[int, int]:
    if queue_size < 0 or queue_capacity < 0 or incoming < 0:
        raise ValueError("counts must be non-negative")
    if queue_size > queue_capacity:
        raise ValueError("queue_size exceeds capacity")

    available = queue_capacity - queue_size
    admitted = min(available, incoming)
    rejected = incoming - admitted
    return int(admitted), int(rejected)


def modulo_shard(key: str, *, shards: int) -> int:
    if shards <= 0:
        raise ValueError("shards must be positive")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big")
    return int(number % shards)


def rendezvous_nodes(
    key: str,
    nodes: list[str],
    *,
    replicas: int = 1,
) -> list[str]:
    if not nodes:
        raise ValueError("nodes must be non-empty")
    if not 1 <= replicas <= len(nodes):
        raise ValueError("invalid replicas")

    unique_nodes = list(dict.fromkeys(nodes))
    if len(unique_nodes) != len(nodes):
        raise ValueError("node IDs must be unique")
    if replicas > len(unique_nodes):
        raise ValueError("replicas exceed unique nodes")

    scored = []
    for node in unique_nodes:
        payload = f"{key}\0{node}".encode("utf-8")
        score = int.from_bytes(
            hashlib.sha256(payload).digest(),
            "big",
        )
        scored.append((score, node))

    scored.sort(reverse=True)
    return [node for _, node in scored[:replicas]]


def quorum_overlap(
    *,
    replicas: int,
    read_quorum: int,
    write_quorum: int,
) -> bool:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if not 1 <= read_quorum <= replicas:
        raise ValueError("invalid read_quorum")
    if not 1 <= write_quorum <= replicas:
        raise ValueError("invalid write_quorum")

    return read_quorum + write_quorum > replicas


def straggler_flags(
    durations: list[float],
    *,
    median_multiplier: float = 2.0,
) -> list[bool]:
    if not durations:
        raise ValueError("durations must be non-empty")
    if any(value < 0 for value in durations):
        raise ValueError("durations must be non-negative")
    if median_multiplier <= 0:
        raise ValueError("median_multiplier must be positive")

    median = statistics.median(durations)
    threshold = median * median_multiplier

    if math.isclose(median, 0.0):
        return [value > 0.0 for value in durations]

    return [value > threshold for value in durations]


@dataclass(frozen=True)
class CircuitBreakerState:
    state: str = "closed"
    consecutive_failures: int = 0


def circuit_breaker_record(
    state: CircuitBreakerState,
    *,
    success: bool,
    failure_threshold: int = 3,
    half_open_probe: bool = False,
) -> CircuitBreakerState:
    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be positive")
    if state.state not in {"closed", "open", "half_open"}:
        raise ValueError("invalid circuit state")

    if state.state == "open":
        if not half_open_probe:
            return state
        return CircuitBreakerState(
            state="closed" if success else "open",
            consecutive_failures=0 if success else state.consecutive_failures,
        )

    if state.state == "half_open":
        return CircuitBreakerState(
            state="closed" if success else "open",
            consecutive_failures=0 if success else state.consecutive_failures + 1,
        )

    if success:
        return CircuitBreakerState(
            state="closed",
            consecutive_failures=0,
        )

    failures = state.consecutive_failures + 1
    if failures >= failure_threshold:
        return CircuitBreakerState(
            state="open",
            consecutive_failures=failures,
        )

    return CircuitBreakerState(
        state="closed",
        consecutive_failures=failures,
    )
