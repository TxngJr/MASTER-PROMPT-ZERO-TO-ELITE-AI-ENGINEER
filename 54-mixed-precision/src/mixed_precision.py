"""Mixed-precision format, quantization and loss-scaling utilities."""

from __future__ import annotations

import math
import numpy as np


_FORMATS = {
    "fp32": {
        "bits": 32,
        "exponent_bits": 8,
        "fraction_bits": 23,
        "bytes": 4,
        "min_normal": 2.0 ** -126,
        "max_finite": (2.0 - 2.0 ** -23) * 2.0 ** 127,
    },
    "fp16": {
        "bits": 16,
        "exponent_bits": 5,
        "fraction_bits": 10,
        "bytes": 2,
        "min_normal": 2.0 ** -14,
        "max_finite": 65504.0,
    },
    "bf16": {
        "bits": 16,
        "exponent_bits": 8,
        "fraction_bits": 7,
        "bytes": 2,
        "min_normal": 2.0 ** -126,
        "max_finite": (2.0 - 2.0 ** -7) * 2.0 ** 127,
    },
}


def floating_format_info(name: str) -> dict[str, float | int]:
    key = name.lower()
    if key not in _FORMATS:
        raise ValueError("format must be fp32, fp16 or bf16")
    return dict(_FORMATS[key])


def bfloat16_roundtrip(values: np.ndarray) -> np.ndarray:
    """Round float32 to nearest-even BF16 precision, returned as float32."""
    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32)

    # Round-to-nearest-even before discarding the low 16 bits.
    least_significant_kept_bit = (bits >> 16) & 1
    rounding_bias = np.uint32(0x7FFF) + least_significant_kept_bit
    rounded = (bits + rounding_bias) & np.uint32(0xFFFF0000)

    return rounded.view(np.float32)


def scale_gradients(
    gradients: list[np.ndarray],
    scale: float,
) -> list[np.ndarray]:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    return [
        np.asarray(gradient, dtype=float) * scale
        for gradient in gradients
    ]


def unscale_gradients(
    gradients: list[np.ndarray],
    scale: float,
) -> list[np.ndarray]:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    return [
        np.asarray(gradient, dtype=float) / scale
        for gradient in gradients
    ]


def contains_nonfinite(
    arrays: list[np.ndarray],
) -> bool:
    return any(
        not np.all(np.isfinite(np.asarray(array)))
        for array in arrays
    )


def update_dynamic_loss_scale(
    current_scale: float,
    *,
    found_nonfinite: bool,
    stable_steps: int,
    growth_interval: int = 2000,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
) -> tuple[float, int]:
    if current_scale <= 0:
        raise ValueError("current_scale must be positive")
    if stable_steps < 0:
        raise ValueError("stable_steps must be non-negative")
    if growth_interval <= 0:
        raise ValueError("growth_interval must be positive")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must exceed one")
    if not 0.0 < backoff_factor < 1.0:
        raise ValueError("backoff_factor must lie in (0,1)")

    if found_nonfinite:
        return (
            max(current_scale * backoff_factor, 1.0),
            0,
        )

    next_stable = stable_steps + 1

    if next_stable >= growth_interval:
        return current_scale * growth_factor, 0

    return current_scale, next_stable


def tensor_storage_bytes(
    num_elements: int,
    *,
    dtype: str,
) -> int:
    if num_elements < 0:
        raise ValueError("num_elements must be non-negative")
    info = floating_format_info(dtype)
    return int(num_elements * int(info["bytes"]))


def relative_error(
    reference: np.ndarray,
    approximation: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    if eps <= 0:
        raise ValueError("eps must be positive")

    ref = np.asarray(reference, dtype=float)
    approx = np.asarray(approximation, dtype=float)

    if ref.shape != approx.shape:
        raise ValueError("shape mismatch")

    numerator = np.linalg.norm(ref - approx)
    denominator = max(np.linalg.norm(ref), eps)
    return float(numerator / denominator)
