from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "pretraining_utils.py"
SPEC = importlib.util.spec_from_file_location("pretraining_utils_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_causal_windows_are_shifted() -> None:
    inputs, targets = mod.make_causal_windows(
        np.arange(10),
        sequence_length=4,
        stride=4,
    )
    np.testing.assert_array_equal(inputs[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(targets[0], [1, 2, 3, 4])


def test_cross_entropy_prefers_correct_logit() -> None:
    good = np.array([[[5.0, 0.0]]])
    bad = np.array([[[0.0, 5.0]]])
    target = np.array([[0]])

    assert (
        mod.stable_cross_entropy(good, target)
        < mod.stable_cross_entropy(bad, target)
    )


def test_perplexity_uniform_distribution() -> None:
    vocab = 10
    loss = np.log(vocab)
    np.testing.assert_allclose(
        mod.perplexity_from_loss(loss),
        vocab,
    )


def test_lr_schedule_warmup_and_decay() -> None:
    values = [
        mod.warmup_cosine_lr(
            step,
            warmup_steps=2,
            total_steps=8,
            max_lr=1e-3,
            min_lr=1e-4,
        )
        for step in range(8)
    ]

    assert values[0] < values[1]
    assert values[1] == 1e-3
    assert values[-1] == 1e-4
    assert all(1e-4 <= value <= 1e-3 for value in values)


def test_effective_tokens_per_update() -> None:
    assert mod.effective_tokens_per_update(
        micro_batch_size=4,
        accumulation_steps=8,
        devices=2,
        sequence_length=1024,
    ) == 65536


def test_global_norm_clipping() -> None:
    gradients = [
        np.array([3.0, 4.0]),
        np.array([0.0]),
    ]
    clipped, original_norm = mod.clip_by_global_norm(
        gradients,
        max_norm=2.5,
    )

    np.testing.assert_allclose(original_norm, 5.0)
    np.testing.assert_allclose(
        mod.global_norm(clipped),
        2.5,
    )
