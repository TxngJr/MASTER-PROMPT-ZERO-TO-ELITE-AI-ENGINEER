"""Modern decoder-only LLM primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if eps <= 0:
        raise ValueError("eps must be positive")

    rms = np.sqrt(
        np.mean(values**2, axis=-1, keepdims=True)
        + eps
    )
    normalized = values / rms

    if weight is None:
        return normalized

    scale = np.asarray(weight, dtype=float)
    if scale.shape != (values.shape[-1],):
        raise ValueError("weight shape mismatch")

    return normalized * scale


def rope_angles(
    sequence_length: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    start_position: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be positive and even")
    if base <= 1.0:
        raise ValueError("base must be greater than one")
    if start_position < 0:
        raise ValueError("start_position must be non-negative")

    pair_dim = head_dim // 2
    inverse_frequency = base ** (
        -np.arange(pair_dim, dtype=float) / pair_dim
    )
    positions = np.arange(
        start_position,
        start_position + sequence_length,
        dtype=float,
    )

    angles = positions[:, None] * inverse_frequency[None, :]
    return np.cos(angles), np.sin(angles)


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    cosine = np.asarray(cos, dtype=float)
    sine = np.asarray(sin, dtype=float)

    if values.ndim != 4:
        raise ValueError("x must have shape (B,H,T,D)")
    if values.shape[-1] % 2 != 0:
        raise ValueError("head dimension must be even")

    expected = (
        values.shape[2],
        values.shape[3] // 2,
    )
    if cosine.shape != expected or sine.shape != expected:
        raise ValueError("cos/sin shape mismatch")

    even = values[..., 0::2]
    odd = values[..., 1::2]

    cos_b = cosine[None, None, :, :]
    sin_b = sine[None, None, :, :]

    rotated_even = even * cos_b - odd * sin_b
    rotated_odd = even * sin_b + odd * cos_b

    output = np.empty_like(values)
    output[..., 0::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output


def repeat_kv(
    key_or_value: np.ndarray,
    query_heads: int,
) -> np.ndarray:
    values = np.asarray(key_or_value, dtype=float)

    if values.ndim != 4:
        raise ValueError("expected shape (B,Hkv,T,D)")

    kv_heads = values.shape[1]

    if query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")

    repetitions = query_heads // kv_heads
    return np.repeat(values, repetitions, axis=1)


def causal_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=float)
    k = np.asarray(key, dtype=float)
    v = np.asarray(value, dtype=float)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected B,H,T,D tensors")
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("Q/K/V shapes must match after GQA expansion")

    head_dim = q.shape[-1]
    scores = (
        q @ np.swapaxes(k, -1, -2)
    ) / np.sqrt(head_dim)

    query_length = q.shape[2]
    key_length = k.shape[2]

    if query_length != key_length:
        raise ValueError(
            "educational full-sequence attention expects equal T"
        )

    future = np.triu(
        np.ones((query_length, key_length), dtype=bool),
        k=1,
    )
    scores = np.where(
        future[None, None, :, :],
        -np.inf,
        scores,
    )

    maximum = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - maximum)
    weights = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True,
    )

    return weights @ v, weights


def swiglu(
    gate_projection: np.ndarray,
    value_projection: np.ndarray,
) -> np.ndarray:
    gate = np.asarray(gate_projection, dtype=float)
    value = np.asarray(value_projection, dtype=float)

    if gate.shape != value.shape:
        raise ValueError("gate/value shapes must match")

    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    silu = gate * sigmoid
    return silu * value


def kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    kv_heads: int,
    sequence_length: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    values = [
        layers,
        batch_size,
        kv_heads,
        sequence_length,
        head_dim,
        bytes_per_element,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("all dimensions must be positive")

    # 2 for K and V.
    return int(
        2
        * layers
        * batch_size
        * kv_heads
        * sequence_length
        * head_dim
        * bytes_per_element
    )
