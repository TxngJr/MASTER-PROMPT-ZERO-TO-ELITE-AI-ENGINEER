"""Educational LoRA math and generic codebook quantization."""

from __future__ import annotations

import numpy as np


def lora_parameter_count(
    input_features: int,
    output_features: int,
    rank: int,
) -> int:
    if min(input_features, output_features, rank) <= 0:
        raise ValueError("dimensions must be positive")
    return int(rank * (input_features + output_features))


def lora_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)

    if a_values.ndim != 2 or b_values.ndim != 2:
        raise ValueError("A and B must be matrices")
    if b_values.shape[1] != a_values.shape[0]:
        raise ValueError("rank dimension mismatch")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    rank = a_values.shape[0]
    if rank <= 0:
        raise ValueError("rank must be positive")

    return (alpha / rank) * (b_values @ a_values)


def lora_forward(
    x: np.ndarray,
    weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    base = np.asarray(weight, dtype=float)

    delta = lora_delta(a, b, alpha=alpha)

    if base.shape != delta.shape:
        raise ValueError("base and LoRA delta shape mismatch")
    if values.shape[-1] != base.shape[1]:
        raise ValueError("input feature mismatch")

    return values @ (base + delta).T


def merge_lora(
    weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    base = np.asarray(weight, dtype=float)
    delta = lora_delta(a, b, alpha=alpha)

    if base.shape != delta.shape:
        raise ValueError("shape mismatch")
    return base + delta


def unmerge_lora(
    merged_weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    merged = np.asarray(merged_weight, dtype=float)
    delta = lora_delta(a, b, alpha=alpha)

    if merged.shape != delta.shape:
        raise ValueError("shape mismatch")
    return merged - delta


def trainable_fraction(
    *,
    base_parameters: int,
    adapter_parameters: int,
) -> float:
    if base_parameters <= 0:
        raise ValueError("base_parameters must be positive")
    if adapter_parameters < 0:
        raise ValueError("adapter_parameters must be non-negative")

    return float(
        adapter_parameters
        / (base_parameters + adapter_parameters)
    )


def nearest_codebook_quantize(
    values: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each value to nearest codebook level.

    This generic educational routine is not an NF4 implementation.
    """
    x = np.asarray(values, dtype=float)
    levels = np.asarray(codebook, dtype=float).reshape(-1)

    if len(levels) < 2:
        raise ValueError("codebook needs at least two levels")
    if not np.all(np.isfinite(levels)):
        raise ValueError("codebook levels must be finite")

    distances = np.abs(
        x[..., None] - levels.reshape((1,) * x.ndim + (-1,))
    )
    indices = np.argmin(distances, axis=-1)
    quantized = levels[indices]
    return quantized, indices.astype(np.int64)
