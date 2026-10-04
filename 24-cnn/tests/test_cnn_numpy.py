from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "cnn_numpy.py"
SPEC = importlib.util.spec_from_file_location("cnn_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_output_size_formula() -> None:
    assert mod.conv_output_size(8, 3, stride=1, padding=1) == 8
    assert mod.conv_output_size(8, 3, stride=2, padding=1) == 4
    assert mod.conv_output_size(7, 3, stride=1, padding=0) == 5


def test_conv2d_known_values() -> None:
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    weight = np.ones((1, 1, 2, 2), dtype=float)

    output = mod.conv2d_nchw(x, weight)

    expected = np.array(
        [[[[12.0, 16.0], [24.0, 28.0]]]]
    )
    np.testing.assert_allclose(output, expected)


def test_multichannel_convolution_shape() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 3, 8, 8))
    weight = rng.normal(size=(5, 3, 3, 3))
    bias = np.zeros(5)

    output = mod.conv2d_nchw(x, weight, bias, padding=1)
    assert output.shape == (2, 5, 8, 8)


def test_max_pool_known_values() -> None:
    x = np.array(
        [[[[1.0, 2.0, 3.0, 4.0],
           [5.0, 6.0, 7.0, 8.0],
           [9.0, 10.0, 11.0, 12.0],
           [13.0, 14.0, 15.0, 16.0]]]]
    )

    output = mod.max_pool2d_nchw(x, kernel_size=2, stride=2)

    expected = np.array([[[[6.0, 8.0], [14.0, 16.0]]]])
    np.testing.assert_allclose(output, expected)
