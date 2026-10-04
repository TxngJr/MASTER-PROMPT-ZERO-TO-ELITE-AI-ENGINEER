"""LLM inference memory, batching, paging, and speculation math."""

from __future__ import annotations

import math


def kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    sequence_length: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    values = [
        layers,
        batch_size,
        sequence_length,
        kv_heads,
        head_dim,
        bytes_per_element,
    ]
    if any(value <= 0 for value in values):
        raise ValueError("all dimensions must be positive")

    return int(
        2
        * layers
        * batch_size
        * sequence_length
        * kv_heads
        * head_dim
        * bytes_per_element
    )


def cache_reduction_ratio(
    *,
    mha_heads: int,
    kv_heads: int,
) -> float:
    if mha_heads <= 0 or kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if kv_heads > mha_heads:
        raise ValueError("kv_heads cannot exceed mha_heads")

    return float(mha_heads / kv_heads)


def padded_batch_efficiency(
    sequence_lengths: list[int],
) -> float:
    if not sequence_lengths:
        raise ValueError("sequence_lengths must be non-empty")
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError("lengths must be positive")

    maximum = max(sequence_lengths)
    useful = sum(sequence_lengths)
    allocated = len(sequence_lengths) * maximum
    return float(useful / allocated)


def paged_block_usage(
    sequence_length: int,
    *,
    block_size: int,
) -> dict[str, int | float]:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    blocks = math.ceil(sequence_length / block_size)
    allocated_slots = blocks * block_size
    wasted_slots = allocated_slots - sequence_length
    utilization = (
        sequence_length / allocated_slots
        if allocated_slots
        else 1.0
    )

    return {
        "blocks": int(blocks),
        "allocated_slots": int(allocated_slots),
        "wasted_slots": int(wasted_slots),
        "utilization": float(utilization),
    }


def prefix_cache_saved_tokens(
    prompt_length: int,
    *,
    cached_prefix_length: int,
) -> int:
    if prompt_length < 0 or cached_prefix_length < 0:
        raise ValueError("lengths must be non-negative")
    if cached_prefix_length > prompt_length:
        raise ValueError("cached prefix exceeds prompt")

    return int(cached_prefix_length)


def speculative_acceptance_rate(
    *,
    proposed_tokens: int,
    accepted_tokens: int,
) -> float:
    if proposed_tokens <= 0:
        raise ValueError("proposed_tokens must be positive")
    if not 0 <= accepted_tokens <= proposed_tokens:
        raise ValueError("accepted_tokens out of range")

    return float(accepted_tokens / proposed_tokens)


def average_tokens_per_target_verification(
    accepted_per_verification: list[int],
) -> float:
    if not accepted_per_verification:
        raise ValueError("values must be non-empty")
    if any(value < 0 for value in accepted_per_verification):
        raise ValueError("accepted token counts must be non-negative")

    return float(
        sum(accepted_per_verification)
        / len(accepted_per_verification)
    )


def bandwidth_lower_bound_seconds(
    *,
    bytes_read: float,
    bandwidth_bytes_per_second: float,
) -> float:
    if bytes_read < 0:
        raise ValueError("bytes_read must be non-negative")
    if bandwidth_bytes_per_second <= 0:
        raise ValueError("bandwidth must be positive")

    return float(bytes_read / bandwidth_bytes_per_second)
