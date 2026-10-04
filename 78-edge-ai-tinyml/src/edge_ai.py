"""Edge AI / TinyML memory, operator, quantization and energy utilities."""

from __future__ import annotations

import math
import numpy as np


_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "int4_packed": 0.5,
}


def tensor_storage_bytes(
    shape: tuple[int, ...],
    *,
    dtype: str,
) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if not shape or any(dim < 0 for dim in shape):
        raise ValueError("shape must be non-empty and non-negative")

    elements = math.prod(shape)
    raw = elements * _DTYPE_BYTES[dtype]
    return int(math.ceil(raw))


def model_parameter_bytes(
    parameter_shapes: list[tuple[int, ...]],
    *,
    dtype: str,
) -> int:
    return int(
        sum(
            tensor_storage_bytes(shape, dtype=dtype)
            for shape in parameter_shapes
        )
    )


def peak_memory_bytes(
    *,
    model_bytes: int,
    peak_activation_bytes: int,
    arena_bytes: int = 0,
    runtime_overhead_bytes: int = 0,
) -> int:
    values = [
        model_bytes,
        peak_activation_bytes,
        arena_bytes,
        runtime_overhead_bytes,
    ]
    if any(value < 0 for value in values):
        raise ValueError("memory values must be non-negative")

    return int(sum(values))


def fits_memory_budget(
    required_bytes: int,
    *,
    available_bytes: int,
    safety_fraction: float = 0.9,
) -> bool:
    if required_bytes < 0 or available_bytes < 0:
        raise ValueError("memory values must be non-negative")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must lie in (0,1]")

    safe_budget = available_bytes * safety_fraction
    return bool(required_bytes <= safe_budget)


def operator_coverage(
    model_operators: list[str],
    supported_operators: set[str],
) -> dict[str, object]:
    unique = sorted(set(model_operators))
    unsupported = [
        op for op in unique
        if op not in supported_operators
    ]

    coverage = (
        1.0
        if not unique
        else (len(unique) - len(unsupported)) / len(unique)
    )

    return {
        "coverage": float(coverage),
        "unsupported": unsupported,
        "unique_operator_count": len(unique),
    }


def affine_int8_quantize(
    values: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("values must be non-empty")

    minimum = min(float(np.min(x)), 0.0)
    maximum = max(float(np.max(x)), 0.0)

    qmin, qmax = -128, 127

    if math.isclose(minimum, maximum):
        scale = 1.0
        zero_point = 0
        return np.zeros_like(x, dtype=np.int8), scale, zero_point

    scale = (maximum - minimum) / (qmax - qmin)
    zero_point_real = qmin - minimum / scale
    zero_point = int(
        np.clip(
            round(zero_point_real),
            qmin,
            qmax,
        )
    )

    quantized = np.clip(
        np.round(x / scale + zero_point),
        qmin,
        qmax,
    ).astype(np.int8)

    return quantized, float(scale), zero_point


def affine_int8_dequantize(
    quantized: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")

    q = np.asarray(quantized)
    return scale * (
        q.astype(float) - zero_point
    )


def energy_per_inference_mj(
    *,
    average_power_watts: float,
    latency_ms: float,
) -> float:
    if average_power_watts < 0 or latency_ms < 0:
        raise ValueError("power/latency must be non-negative")

    joules = average_power_watts * latency_ms / 1000.0
    return float(joules * 1000.0)


def inferences_per_joule(
    *,
    energy_mj: float,
) -> float:
    if energy_mj <= 0:
        raise ValueError("energy_mj must be positive")
    return float(1000.0 / energy_mj)


def sensor_window_samples(
    *,
    sample_rate_hz: float,
    window_ms: float,
) -> int:
    if sample_rate_hz <= 0 or window_ms <= 0:
        raise ValueError("sample rate/window must be positive")

    return int(
        round(sample_rate_hz * window_ms / 1000.0)
    )


def duty_cycle(
    *,
    active_ms: float,
    period_ms: float,
) -> float:
    if active_ms < 0 or period_ms <= 0:
        raise ValueError("invalid timing")
    if active_ms > period_ms:
        raise ValueError("active_ms exceeds period_ms")

    return float(active_ms / period_ms)
