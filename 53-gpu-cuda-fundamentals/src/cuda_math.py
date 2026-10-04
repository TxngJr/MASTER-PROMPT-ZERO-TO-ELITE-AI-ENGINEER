"""Small calculations for CUDA execution and performance intuition."""

from __future__ import annotations


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0:
        raise ValueError("numerator must be non-negative")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def grid_size_1d(
    num_elements: int,
    *,
    block_size: int,
) -> int:
    if num_elements < 0:
        raise ValueError("num_elements must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return ceil_div(num_elements, block_size)


def global_thread_index(
    *,
    block_index: int,
    block_size: int,
    thread_index: int,
) -> int:
    if min(block_index, thread_index) < 0:
        raise ValueError("indices must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if thread_index >= block_size:
        raise ValueError("thread_index must be smaller than block_size")
    return block_index * block_size + thread_index


def warp_and_lane(
    thread_index: int,
    *,
    warp_size: int = 32,
) -> tuple[int, int]:
    if thread_index < 0:
        raise ValueError("thread_index must be non-negative")
    if warp_size <= 0:
        raise ValueError("warp_size must be positive")
    return (
        thread_index // warp_size,
        thread_index % warp_size,
    )


def transaction_segment_count(
    addresses: list[int],
    *,
    transaction_bytes: int = 32,
) -> int:
    if transaction_bytes <= 0:
        raise ValueError("transaction_bytes must be positive")
    if any(address < 0 for address in addresses):
        raise ValueError("addresses must be non-negative")

    return len(
        {
            address // transaction_bytes
            for address in addresses
        }
    )


def theoretical_resident_blocks(
    *,
    threads_per_block: int,
    registers_per_thread: int,
    shared_bytes_per_block: int,
    max_threads_per_sm: int,
    registers_per_sm: int,
    shared_bytes_per_sm: int,
    max_blocks_per_sm: int,
) -> int:
    positive = [
        threads_per_block,
        max_threads_per_sm,
        registers_per_sm,
        shared_bytes_per_sm,
        max_blocks_per_sm,
    ]
    if any(value <= 0 for value in positive):
        raise ValueError("capacity/thread values must be positive")
    if registers_per_thread < 0 or shared_bytes_per_block < 0:
        raise ValueError("resource usage must be non-negative")

    thread_limit = max_threads_per_sm // threads_per_block

    if registers_per_thread == 0:
        register_limit = max_blocks_per_sm
    else:
        registers_per_block = (
            registers_per_thread * threads_per_block
        )
        register_limit = registers_per_sm // registers_per_block

    if shared_bytes_per_block == 0:
        shared_limit = max_blocks_per_sm
    else:
        shared_limit = (
            shared_bytes_per_sm // shared_bytes_per_block
        )

    return max(
        0,
        min(
            thread_limit,
            register_limit,
            shared_limit,
            max_blocks_per_sm,
        ),
    )


def arithmetic_intensity(
    *,
    floating_point_operations: float,
    bytes_moved: float,
) -> float:
    if floating_point_operations < 0:
        raise ValueError("FLOPs must be non-negative")
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return float(floating_point_operations / bytes_moved)


def roofline_bound(
    *,
    peak_flops_per_second: float,
    bandwidth_bytes_per_second: float,
    arithmetic_intensity_flops_per_byte: float,
) -> float:
    if peak_flops_per_second <= 0:
        raise ValueError("peak compute must be positive")
    if bandwidth_bytes_per_second <= 0:
        raise ValueError("bandwidth must be positive")
    if arithmetic_intensity_flops_per_byte < 0:
        raise ValueError("arithmetic intensity cannot be negative")

    memory_bound = (
        bandwidth_bytes_per_second
        * arithmetic_intensity_flops_per_byte
    )
    return float(
        min(peak_flops_per_second, memory_bound)
    )
