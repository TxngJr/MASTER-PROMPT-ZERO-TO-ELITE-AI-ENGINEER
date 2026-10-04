"""Long-context attention, cache, chunking, and memory utilities."""

from __future__ import annotations

import math
import numpy as np


def causal_dense_attention_pairs(sequence_length: int) -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    return int(sequence_length * (sequence_length + 1) // 2)


def sliding_window_attention_pairs(
    sequence_length: int,
    *,
    window: int,
) -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if window <= 0:
        raise ValueError("window must be positive")

    return int(
        sum(
            min(position + 1, window)
            for position in range(sequence_length)
        )
    )


def attention_pair_reduction(
    sequence_length: int,
    *,
    window: int,
) -> float:
    dense = causal_dense_attention_pairs(sequence_length)
    if dense == 0:
        return 1.0
    sliding = sliding_window_attention_pairs(
        sequence_length,
        window=window,
    )
    return float(dense / sliding)


def kv_cache_bytes(
    *,
    layers: int,
    sequence_length: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
    batch_size: int = 1,
) -> int:
    values = [
        layers,
        sequence_length,
        kv_heads,
        head_dim,
        bytes_per_element,
        batch_size,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("all values must be positive")

    return int(
        2
        * layers
        * sequence_length
        * kv_heads
        * head_dim
        * bytes_per_element
        * batch_size
    )


def bounded_sliding_cache_tokens(
    sequence_length: int,
    *,
    window: int,
) -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if window <= 0:
        raise ValueError("window must be positive")
    return int(min(sequence_length, window))


def linear_rope_scaled_position(
    position: float,
    *,
    factor: float,
) -> float:
    if position < 0:
        raise ValueError("position must be non-negative")
    if factor <= 0:
        raise ValueError("factor must be positive")
    return float(position / factor)


def chunk_ranges(
    sequence_length: int,
    *,
    chunk_size: int,
    overlap: int = 0,
) -> list[tuple[int, int]]:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must lie in [0,chunk_size)")

    if sequence_length == 0:
        return []

    step = chunk_size - overlap
    ranges = []

    start = 0
    while start < sequence_length:
        end = min(start + chunk_size, sequence_length)
        ranges.append((start, end))
        if end == sequence_length:
            break
        start += step

    return ranges


def select_context_segments(
    token_lengths: list[int],
    priorities: list[float],
    *,
    token_budget: int,
) -> list[int]:
    if len(token_lengths) != len(priorities):
        raise ValueError("length/priorities mismatch")
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if any(length < 0 for length in token_lengths):
        raise ValueError("token lengths must be non-negative")

    order = sorted(
        range(len(token_lengths)),
        key=lambda index: (
            -priorities[index],
            token_lengths[index],
            index,
        ),
    )

    chosen = []
    used = 0

    for index in order:
        length = token_lengths[index]
        if used + length <= token_budget:
            chosen.append(index)
            used += length

    return chosen


def memory_score(
    *,
    similarity: float,
    recency: float,
    importance: float,
    similarity_weight: float = 0.6,
    recency_weight: float = 0.2,
    importance_weight: float = 0.2,
) -> float:
    values = [similarity, recency, importance]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("signals must lie in [0,1]")

    weights = [
        similarity_weight,
        recency_weight,
        importance_weight,
    ]
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ValueError("weights must sum to 1")

    return float(
        similarity_weight * similarity
        + recency_weight * recency
        + importance_weight * importance
    )


def retrieval_recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")

    retrieved = set(ranked_ids[:k])
    return float(
        len(retrieved & relevant_ids)
        / len(relevant_ids)
    )
