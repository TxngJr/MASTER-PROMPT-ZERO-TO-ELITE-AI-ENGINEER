"""Educational distributed-training math and memory estimators."""

from __future__ import annotations

import math
import numpy as np


def shard_indices(
    num_samples: int,
    *,
    world_size: int,
    rank: int,
) -> np.ndarray:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank out of range")

    return np.arange(rank, num_samples, world_size, dtype=np.int64)


def all_reduce_mean(
    rank_gradients: list[np.ndarray],
) -> np.ndarray:
    if not rank_gradients:
        raise ValueError("rank_gradients must be non-empty")

    arrays = [
        np.asarray(gradient, dtype=float)
        for gradient in rank_gradients
    ]

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all gradients must have equal shape")

    return np.mean(np.stack(arrays, axis=0), axis=0)


def ring_allreduce_per_rank_bytes(
    tensor_bytes: int,
    *,
    world_size: int,
) -> float:
    if tensor_bytes < 0:
        raise ValueError("tensor_bytes must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return 0.0

    return float(
        2.0
        * (world_size - 1)
        / world_size
        * tensor_bytes
    )


def effective_global_batch(
    *,
    local_batch: int,
    world_size: int,
    accumulation_steps: int = 1,
) -> int:
    values = [local_batch, world_size, accumulation_steps]
    if any(value <= 0 for value in values):
        raise ValueError("batch/world/accumulation must be positive")

    return int(local_batch * world_size * accumulation_steps)


def zero_style_state_bytes(
    *,
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_bytes: int,
    world_size: int,
    stage: int,
) -> float:
    values = [
        parameter_bytes,
        gradient_bytes,
        optimizer_bytes,
    ]
    if any(value < 0 for value in values):
        raise ValueError("byte counts must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if stage not in {0, 1, 2, 3}:
        raise ValueError("stage must be 0, 1, 2 or 3")

    parameters = float(parameter_bytes)
    gradients = float(gradient_bytes)
    optimizer = float(optimizer_bytes)

    if stage >= 1:
        optimizer /= world_size
    if stage >= 2:
        gradients /= world_size
    if stage >= 3:
        parameters /= world_size

    return parameters + gradients + optimizer


def pipeline_efficiency(
    *,
    stages: int,
    microbatches: int,
) -> float:
    if stages <= 0 or microbatches <= 0:
        raise ValueError("stages/microbatches must be positive")

    return float(
        microbatches / (microbatches + stages - 1)
    )


def scaling_efficiency(
    *,
    single_device_time: float,
    parallel_time: float,
    devices: int,
) -> float:
    if single_device_time <= 0 or parallel_time <= 0:
        raise ValueError("times must be positive")
    if devices <= 0:
        raise ValueError("devices must be positive")

    speedup = single_device_time / parallel_time
    return float(speedup / devices)


def communication_time_seconds(
    payload_bytes: float,
    *,
    bandwidth_bytes_per_second: float,
    latency_seconds: float = 0.0,
) -> float:
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be non-negative")
    if bandwidth_bytes_per_second <= 0:
        raise ValueError("bandwidth must be positive")
    if latency_seconds < 0:
        raise ValueError("latency must be non-negative")

    return float(
        latency_seconds
        + payload_bytes / bandwidth_bytes_per_second
    )
