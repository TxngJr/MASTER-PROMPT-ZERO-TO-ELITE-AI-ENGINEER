from pathlib import Path
import importlib.util
import numpy as np

P = Path(__file__).parents[1] / "src" / "federated.py"
S = importlib.util.spec_from_file_location("federated", P)
m = importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_weighted_average():
    out = m.weighted_federated_average([[0, 2], [2, 4]], [1, 3])
    assert np.allclose(out, [1.5, 3.5])


def test_clip_and_noise_reproducible():
    clipped = m.l2_clip([3.0, 4.0], 2.5)
    assert np.isclose(np.linalg.norm(clipped), 2.5)
    a = m.add_gaussian_noise([0, 0], 1.0, 7)
    b = m.add_gaussian_noise([0, 0], 1.0, 7)
    assert np.allclose(a, b)
