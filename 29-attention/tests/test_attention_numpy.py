from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "attention_numpy.py"
SPEC = importlib.util.spec_from_file_location("attention_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_softmax_rows_sum_to_one() -> None:
    x = np.array([[1000.0, 1001.0, 1002.0]])
    p = mod.stable_softmax(x)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)


def test_attention_known_equal_scores_average_values() -> None:
    q = np.zeros((1, 2, 3))
    k = np.zeros((1, 2, 3))
    v = np.array([[[1.0, 0.0], [3.0, 2.0]]])

    output, weights = mod.scaled_dot_product_attention(q, k, v)

    expected = np.array([[[2.0, 1.0], [2.0, 1.0]]])
    np.testing.assert_allclose(output, expected)
    np.testing.assert_allclose(weights, 0.5)


def test_causal_mask_blocks_future() -> None:
    q = np.ones((1, 3, 2))
    k = np.ones((1, 3, 2))
    v = np.array([[[1.0], [2.0], [100.0]]])
    keep = mod.causal_mask(3)[None, :, :]

    output, weights = mod.scaled_dot_product_attention(
        q,
        k,
        v,
        keep_mask=keep,
    )

    assert weights[0, 0, 1] == 0.0
    assert weights[0, 0, 2] == 0.0
    np.testing.assert_allclose(output[0, 0, 0], 1.0)


def test_split_combine_round_trip() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 5, 12))
    heads = mod.split_heads(x, 3)
    rebuilt = mod.combine_heads(heads)
    np.testing.assert_allclose(rebuilt, x)


def test_multihead_causal_shapes() -> None:
    model = mod.MultiHeadSelfAttentionNumPy(
        model_dim=12,
        num_heads=3,
        seed=4,
    )
    x = np.ones((2, 6, 12))

    output, weights = model.forward(x, causal=True)

    assert output.shape == (2, 6, 12)
    assert weights.shape == (2, 3, 6, 6)
    assert np.allclose(np.triu(weights, k=1), 0.0)
