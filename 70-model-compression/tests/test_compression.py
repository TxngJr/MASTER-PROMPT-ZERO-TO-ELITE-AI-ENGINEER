from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "compression.py"
SPEC = importlib.util.spec_from_file_location("compression_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_magnitude_pruning_hits_requested_floor_count() -> None:
    x = np.array([1.0, -0.1, 3.0, 0.2, 5.0])
    pruned, mask = mod.magnitude_prune(x, fraction=0.4)

    assert np.count_nonzero(~mask) == 2
    assert pruned[1] == 0.0
    assert pruned[3] == 0.0
    assert mod.sparsity(pruned) == 0.4


def test_structured_row_pruning_removes_small_norm_rows() -> None:
    x = np.array(
        [
            [10.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [0.2, 0.0],
        ]
    )
    compressed, keep = mod.structured_row_prune(
        x,
        fraction=0.5,
    )

    assert compressed.shape == (2, 2)
    assert keep.tolist() == [True, False, True, False]


def test_temperature_softens_distribution() -> None:
    logits = np.array([[5.0, 1.0, -1.0]])

    cold = mod.softmax_temperature(
        logits,
        temperature=1.0,
    )
    warm = mod.softmax_temperature(
        logits,
        temperature=4.0,
    )

    assert warm.max() < cold.max()
    np.testing.assert_allclose(warm.sum(axis=-1), 1.0)


def test_distillation_loss_zero_for_identical_logits() -> None:
    logits = np.array([[2.0, 1.0], [0.5, -0.5]])
    loss = mod.distillation_loss(
        logits,
        logits.copy(),
        temperature=3.0,
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-12)


def test_compression_ratio() -> None:
    assert mod.compression_ratio(
        original_bytes=400,
        compressed_bytes=100,
    ) == 4.0
