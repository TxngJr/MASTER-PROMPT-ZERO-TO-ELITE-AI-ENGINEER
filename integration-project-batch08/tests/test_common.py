from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "cnn_framework_lab.py"
SPEC = importlib.util.spec_from_file_location("cnn_framework_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_digits_split_and_normalization() -> None:
    splits = lab.load_digits_splits(seed=5, limit=600)

    assert sum(len(y) for _, y in splits.values()) == 600

    for images, labels in splits.values():
        assert images.ndim == 3
        assert images.shape[1:] == (8, 8)
        assert labels.ndim == 1
        assert images.min() >= 0.0
        assert images.max() <= 1.0


def test_layout_conversions() -> None:
    images = np.zeros((7, 8, 8), dtype=np.float32)

    assert lab.to_nchw(images).shape == (7, 1, 8, 8)
    assert lab.to_nhwc(images).shape == (7, 8, 8, 1)


def test_split_is_reproducible() -> None:
    a = lab.load_digits_splits(seed=7, limit=500)
    b = lab.load_digits_splits(seed=7, limit=500)

    for name in a:
        np.testing.assert_allclose(a[name][0], b[name][0])
        np.testing.assert_array_equal(a[name][1], b[name][1])
