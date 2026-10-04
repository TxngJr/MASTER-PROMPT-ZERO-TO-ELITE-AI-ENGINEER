"""Educational uniform quantization and storage utilities."""

from __future__ import annotations

import math
import numpy as np


def signed_integer_range(bits: int) -> tuple[int, int]:
    if bits < 2 or bits > 16:
        raise ValueError("bits must lie in [2,16]")
    qmax = 2 ** (bits - 1) - 1
    return -qmax, qmax


def symmetric_quantize(
    values: np.ndarray,
    *,
    bits: int = 8,
) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=float)
    qmin, qmax = signed_integer_range(bits)

    maximum = float(np.max(np.abs(x))) if x.size else 0.0
    if maximum == 0.0:
        return np.zeros_like(x, dtype=np.int64), 1.0

    scale = maximum / qmax
    quantized = np.clip(
        np.rint(x / scale),
        qmin,
        qmax,
    ).astype(np.int64)

    return quantized, float(scale)


def symmetric_dequantize(
    quantized: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    return np.asarray(quantized, dtype=float) * scale


def asymmetric_quantize(
    values: np.ndarray,
    *,
    bits: int = 8,
) -> tuple[np.ndarray, float, int]:
    if bits < 2 or bits > 16:
        raise ValueError("bits must lie in [2,16]")

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.asarray([], dtype=np.int64), 1.0, 0

    qmin = 0
    qmax = 2**bits - 1

    # Affine quantization should make real zero representable.
    # If the observed tensor is entirely positive or entirely negative,
    # extend the calibration interval to include 0 before solving for
    # scale and zero point. Otherwise clipping the zero point can shrink
    # the actual dequantized range.
    minimum = min(float(np.min(x)), 0.0)
    maximum = max(float(np.max(x)), 0.0)

    if maximum == minimum:
        return np.zeros_like(x, dtype=np.int64), 1.0, 0

    scale = (maximum - minimum) / (qmax - qmin)
    zero_point = int(
        np.clip(
            np.rint(qmin - minimum / scale),
            qmin,
            qmax,
        )
    )
    quantized = np.clip(
        np.rint(x / scale + zero_point),
        qmin,
        qmax,
    ).astype(np.int64)

    return quantized, float(scale), zero_point


def asymmetric_dequantize(
    quantized: np.ndarray,
    *,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")

    q = np.asarray(quantized, dtype=float)
    return scale * (q - int(zero_point))


def per_channel_symmetric_quantize(
    values: np.ndarray,
    *,
    axis: int = 0,
    bits: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.ndim == 0:
        raise ValueError("values must have at least one dimension")

    axis = axis % x.ndim
    moved = np.moveaxis(x, axis, 0)

    quantized = np.empty_like(moved, dtype=np.int64)
    scales = np.empty(moved.shape[0], dtype=float)

    for index, channel in enumerate(moved):
        q, scale = symmetric_quantize(channel, bits=bits)
        quantized[index] = q
        scales[index] = scale

    return np.moveaxis(quantized, 0, axis), scales


def grouped_symmetric_quantize(
    values: np.ndarray,
    *,
    group_size: int,
    bits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).reshape(-1)

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    quantized = np.empty_like(x, dtype=np.int64)
    scales = []

    for start in range(0, len(x), group_size):
        end = min(start + group_size, len(x))
        q, scale = symmetric_quantize(
            x[start:end],
            bits=bits,
        )
        quantized[start:end] = q
        scales.append(scale)

    return quantized, np.asarray(scales, dtype=float)


def quantization_mse(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> float:
    ref = np.asarray(reference, dtype=float)
    approx = np.asarray(approximation, dtype=float)

    if ref.shape != approx.shape:
        raise ValueError("shape mismatch")
    if ref.size == 0:
        raise ValueError("arrays must be non-empty")

    return float(np.mean((ref - approx) ** 2))


def ideal_storage_bytes(
    num_values: int,
    *,
    bits_per_value: int,
) -> int:
    if num_values < 0:
        raise ValueError("num_values must be non-negative")
    if bits_per_value <= 0:
        raise ValueError("bits_per_value must be positive")

    return int(math.ceil(num_values * bits_per_value / 8))


def compression_ratio(
    original_bits: int,
    quantized_bits: int,
) -> float:
    if original_bits <= 0 or quantized_bits <= 0:
        raise ValueError("bit widths must be positive")
    return float(original_bits / quantized_bits)
