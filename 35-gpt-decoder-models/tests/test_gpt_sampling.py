from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "gpt_sampling.py"
SPEC = importlib.util.spec_from_file_location("gpt_sampling_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_causal_lm_pairs() -> None:
    x, y = mod.make_causal_lm_pairs([1, 2, 3, 4, 5], 3)
    np.testing.assert_array_equal(x, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y, [[2, 3, 4], [3, 4, 5]])


def test_temperature_changes_distribution_sharpness() -> None:
    logits = np.array([0.0, 1.0, 2.0])
    cold = mod.stable_softmax(mod.apply_temperature(logits, 0.5))
    hot = mod.stable_softmax(mod.apply_temperature(logits, 2.0))

    assert cold[-1] > hot[-1]


def test_top_k_keeps_exactly_k() -> None:
    filtered = mod.top_k_filter(
        np.array([1.0, 4.0, 3.0, 2.0]),
        2,
    )
    assert np.isfinite(filtered).sum() == 2
    assert np.isfinite(filtered[1])
    assert np.isfinite(filtered[2])


def test_top_p_keeps_threshold_crossing_token() -> None:
    logits = np.array([5.0, 4.0, 1.0, 0.0])
    filtered = mod.top_p_filter(logits, 0.75)

    assert np.isfinite(filtered[0])
    assert np.isfinite(filtered[1])
    assert not np.isfinite(filtered[3])


def test_sampling_reproducible() -> None:
    logits = np.array([0.1, 0.2, 2.0, 0.3])
    a = mod.sample_from_logits(
        logits,
        rng=np.random.default_rng(7),
        top_k=3,
        top_p=0.9,
    )
    b = mod.sample_from_logits(
        logits,
        rng=np.random.default_rng(7),
        top_k=3,
        top_p=0.9,
    )
    assert a == b
