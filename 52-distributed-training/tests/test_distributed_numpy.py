from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "distributed_numpy.py"
SPEC = importlib.util.spec_from_file_location("distributed_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_shards_cover_samples_without_overlap() -> None:
    shards = [
        mod.shard_indices(
            10,
            world_size=3,
            rank=rank,
        )
        for rank in range(3)
    ]

    combined = np.concatenate(shards)
    assert sorted(combined.tolist()) == list(range(10))
    assert len(set(combined.tolist())) == 10


def test_all_reduce_mean() -> None:
    result = mod.all_reduce_mean(
        [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
    )
    np.testing.assert_allclose(result, [2.0, 3.0])


def test_ring_payload_approaches_twice_tensor_size() -> None:
    tensor_bytes = 1_000_000
    payload = mod.ring_allreduce_per_rank_bytes(
        tensor_bytes,
        world_size=8,
    )
    np.testing.assert_allclose(
        payload,
        2 * 7 / 8 * tensor_bytes,
    )


def test_zero_stages_reduce_idealized_state_memory() -> None:
    totals = [
        mod.zero_style_state_bytes(
            parameter_bytes=100,
            gradient_bytes=100,
            optimizer_bytes=200,
            world_size=4,
            stage=stage,
        )
        for stage in range(4)
    ]

    assert totals == [400.0, 250.0, 175.0, 100.0]


def test_pipeline_efficiency_improves_with_microbatches() -> None:
    low = mod.pipeline_efficiency(stages=4, microbatches=2)
    high = mod.pipeline_efficiency(stages=4, microbatches=16)

    assert 0.0 < low < high < 1.0


def test_scaling_efficiency() -> None:
    efficiency = mod.scaling_efficiency(
        single_device_time=100.0,
        parallel_time=30.0,
        devices=4,
    )
    np.testing.assert_allclose(efficiency, 100 / 30 / 4)
