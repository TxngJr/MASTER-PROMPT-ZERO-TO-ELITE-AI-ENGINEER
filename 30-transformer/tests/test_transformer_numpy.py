from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "transformer_numpy.py"
SPEC = importlib.util.spec_from_file_location("transformer_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_position_encoding_shape_and_origin() -> None:
    pe = mod.sinusoidal_position_encoding(5, 6)

    assert pe.shape == (5, 6)
    np.testing.assert_allclose(pe[0, 0::2], 0.0)
    np.testing.assert_allclose(pe[0, 1::2], 1.0)


def test_layer_norm_zero_mean_unit_variance_approximately() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 5, 8))
    y = mod.layer_norm(x, eps=1e-8)

    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-7)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=1e-6)


def test_transformer_block_preserves_shape() -> None:
    model = mod.TransformerBlockNumPy(
        model_dim=12,
        num_heads=3,
        ff_dim=24,
        seed=5,
    )
    x = np.ones((2, 7, 12))

    output = model.forward(x)

    assert output.shape == x.shape
    assert np.all(np.isfinite(output))


def test_future_token_does_not_affect_previous_positions() -> None:
    model = mod.TransformerBlockNumPy(
        model_dim=8,
        num_heads=2,
        ff_dim=16,
        seed=8,
    )

    x1 = np.zeros((1, 4, 8))
    x2 = x1.copy()
    x2[:, 3, :] = 100.0

    y1 = model.forward(x1)
    y2 = model.forward(x2)

    np.testing.assert_allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-10)
