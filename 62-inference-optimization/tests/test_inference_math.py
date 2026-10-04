from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "inference_math.py"
SPEC = importlib.util.spec_from_file_location("inference_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_kv_cache_grows_linearly_with_sequence() -> None:
    short = mod.kv_cache_bytes(
        layers=2,
        batch_size=1,
        sequence_length=128,
        kv_heads=4,
        head_dim=16,
        bytes_per_element=2,
    )
    long = mod.kv_cache_bytes(
        layers=2,
        batch_size=1,
        sequence_length=256,
        kv_heads=4,
        head_dim=16,
        bytes_per_element=2,
    )
    assert long == 2 * short


def test_gqa_cache_reduction() -> None:
    assert mod.cache_reduction_ratio(
        mha_heads=32,
        kv_heads=8,
    ) == 4.0


def test_padding_efficiency() -> None:
    efficiency = mod.padded_batch_efficiency([10, 100])
    np.testing.assert_allclose(efficiency, 110 / 200)


def test_paged_block_usage() -> None:
    report = mod.paged_block_usage(
        33,
        block_size=16,
    )
    assert report["blocks"] == 3
    assert report["allocated_slots"] == 48
    assert report["wasted_slots"] == 15
    assert 0.0 < report["utilization"] <= 1.0


def test_speculative_acceptance() -> None:
    assert mod.speculative_acceptance_rate(
        proposed_tokens=10,
        accepted_tokens=7,
    ) == 0.7
    assert mod.average_tokens_per_target_verification(
        [2, 3, 1]
    ) == 2.0


def test_bandwidth_lower_bound() -> None:
    seconds = mod.bandwidth_lower_bound_seconds(
        bytes_read=1_000_000_000,
        bandwidth_bytes_per_second=500_000_000_000,
    )
    np.testing.assert_allclose(seconds, 0.002)
