from pathlib import Path
import importlib.util
import numpy as np

P = Path(__file__).parents[1] / "src" / "causal.py"
S = importlib.util.spec_from_file_location("causal", P)
m = importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_difference_and_smd():
    y = np.array([2, 4, 1, 1], float)
    t = np.array([1, 1, 0, 0])
    assert m.difference_in_means(y, t) == 2.0
    assert np.isfinite(m.standardized_mean_difference(y, t))


def test_ipw_randomized_propensity():
    y = np.array([3, 5, 1, 1], float)
    t = np.array([1, 1, 0, 0])
    p = np.full(4, 0.5)
    assert np.isclose(m.ipw_ate(y, t, p), 3.0)
