from pathlib import Path
import importlib.util
import numpy as np

P=Path(__file__).parents[1]/"src"/"scientific_ml.py"
S=importlib.util.spec_from_file_location("scientific_ml",P)
m=importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_decay_residual_small_with_fine_grid():
    t=np.linspace(0,1,1001)
    u=np.exp(-2*t)
    r=m.exponential_decay_residual(t,u,2.0)
    assert np.max(np.abs(r)) < 1e-4


def test_relative_l2():
    assert m.relative_l2([1,2],[1,2]) == 0.0
