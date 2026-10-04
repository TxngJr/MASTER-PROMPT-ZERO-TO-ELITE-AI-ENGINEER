from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "sequence_latent_lab.py"
SPEC = importlib.util.spec_from_file_location("sequence_latent_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_memory_dataset_reproducible() -> None:
    a_x, a_y = lab.make_memory_dataset(300, seed=7)
    b_x, b_y = lab.make_memory_dataset(300, seed=7)

    np.testing.assert_allclose(a_x, b_x)
    np.testing.assert_array_equal(a_y, b_y)


def test_memory_dataset_shapes() -> None:
    x, y = lab.make_memory_dataset(
        250,
        sequence_length=20,
        seed=3,
    )

    assert x.shape == (250, 20, 1)
    assert y.shape == (250,)
    assert set(np.unique(y)).issubset({0, 1})
