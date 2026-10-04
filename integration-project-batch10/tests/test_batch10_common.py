from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "generative_transformer_lab.py"
SPEC = importlib.util.spec_from_file_location("batch10_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_ring_mixture_reproducible_and_finite() -> None:
    a = lab.make_ring_mixture(300, seed=7)
    b = lab.make_ring_mixture(300, seed=7)

    np.testing.assert_allclose(a, b)
    assert a.shape == (300, 2)
    assert np.all(np.isfinite(a))
