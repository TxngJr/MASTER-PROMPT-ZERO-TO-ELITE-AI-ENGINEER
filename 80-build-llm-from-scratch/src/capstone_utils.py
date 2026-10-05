"""Framework-independent utilities for the final LLM capstone."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


def causal_windows(
    token_ids: Iterable[int],
    *,
    sequence_length: int,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(list(token_ids), dtype=np.int64).reshape(-1)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride is None:
        stride = sequence_length
    if stride <= 0:
        raise ValueError("stride must be positive")

    needed = sequence_length + 1
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for start in range(0, len(tokens) - needed + 1, stride):
        window = tokens[start : start + needed]
        inputs.append(window[:-1])
        targets.append(window[1:])

    if not inputs:
        empty = np.empty((0, sequence_length), dtype=np.int64)
        return empty.copy(), empty

    return np.stack(inputs), np.stack(targets)


def model_weight_bytes(parameter_count: int, *, bytes_per_parameter: int) -> int:
    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be positive")
    return int(parameter_count) * int(bytes_per_parameter)


def adam_training_state_bytes(
    parameter_count: int,
    *,
    weight_bytes: int = 4,
    gradient_bytes: int = 4,
    first_moment_bytes: int = 4,
    second_moment_bytes: int = 4,
) -> int:
    values = (weight_bytes, gradient_bytes, first_moment_bytes, second_moment_bytes)
    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    if any(value < 0 for value in values):
        raise ValueError("byte counts must be non-negative")
    return int(parameter_count) * int(sum(values))


def kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    sequence_length: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    values = (layers, batch_size, sequence_length, kv_heads, head_dim, bytes_per_element)
    if any(value <= 0 for value in values):
        raise ValueError("all KV-cache dimensions must be positive")
    return int(
        2 * layers * batch_size * sequence_length
        * kv_heads * head_dim * bytes_per_element
    )


def symmetric_int8_quantize(values: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    max_abs = float(np.max(np.abs(array)))
    if max_abs == 0.0:
        return np.zeros_like(array, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def symmetric_int8_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    return np.asarray(quantized, dtype=np.float32) * np.float32(scale)


def dpo_loss_from_log_ratios(
    *,
    policy_chosen_logp: float,
    policy_rejected_logp: float,
    reference_chosen_logp: float,
    reference_rejected_logp: float,
    beta: float,
) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = float(policy_chosen_logp) - float(policy_rejected_logp)
    reference_margin = float(reference_chosen_logp) - float(reference_rejected_logp)
    z = float(beta) * (policy_margin - reference_margin)
    return float(np.logaddexp(0.0, -z))


def document_fingerprint(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
