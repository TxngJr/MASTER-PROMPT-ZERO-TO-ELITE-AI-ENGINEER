from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "cuda_math.py"
SPEC = importlib.util.spec_from_file_location("cuda_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_grid_and_global_index() -> None:
    assert mod.grid_size_1d(1000, block_size=256) == 4
    assert mod.global_thread_index(
        block_index=2,
        block_size=256,
        thread_index=5,
    ) == 517


def test_warp_and_lane() -> None:
    assert mod.warp_and_lane(0) == (0, 0)
    assert mod.warp_and_lane(31) == (0, 31)
    assert mod.warp_and_lane(32) == (1, 0)


def test_contiguous_float_addresses_touch_few_segments() -> None:
    addresses = [lane * 4 for lane in range(32)]
    contiguous = mod.transaction_segment_count(
        addresses,
        transaction_bytes=32,
    )
    assert contiguous == 4

    strided = [lane * 128 for lane in range(32)]
    assert (
        mod.transaction_segment_count(strided)
        > contiguous
    )


def test_resident_blocks_are_resource_limited() -> None:
    blocks = mod.theoretical_resident_blocks(
        threads_per_block=256,
        registers_per_thread=32,
        shared_bytes_per_block=16 * 1024,
        max_threads_per_sm=2048,
        registers_per_sm=65536,
        shared_bytes_per_sm=64 * 1024,
        max_blocks_per_sm=16,
    )
    assert blocks == 4


def test_roofline_bound() -> None:
    intensity = mod.arithmetic_intensity(
        floating_point_operations=200,
        bytes_moved=100,
    )
    assert intensity == 2.0

    result = mod.roofline_bound(
        peak_flops_per_second=1000,
        bandwidth_bytes_per_second=100,
        arithmetic_intensity_flops_per_byte=intensity,
    )
    assert result == 200.0
